/*
  Copyright (C) 2024 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

/**
 * fault_viscosity.cc
 *
 * Particle property that weakly imposes a target slip rate on a curved
 * dip-slip fault using two complementary mechanisms:
 *
 *   Correct-sense slip  (edot_signed > 0):
 *     Adaptive viscosity feedback:
 *       eta_new = eta_old * (edot_signed / edot_target)^alpha
 *     capped by max_correction_factor per step.
 *     Lower eta -> weaker fault -> more strain: this is the correct sign.
 *
 *   Wrong-sense slip  (edot_signed < 0):
 *     Reset viscosity to initial_viscosity (neutral, not extreme) and
 *     store a body force in the correct slip direction proportional to
 *     how wrong the current slip is.  The force is picked up by
 *     visco_plastic.cc and written into AdditionalMaterialOutputsStokesRHS,
 *     which the Stokes assembler in stokes.cc already handles when
 *     "set Enable additional Stokes RHS = true" is set in the .prm.
 *
 * Local fault geometry (curved fault support):
 *     The fault normal is computed from the gradient of the fault
 *     composition field at each particle -- no fixed global direction.
 *     The slip direction is derived by rotating the normal 90 degrees
 *     in the vertical plane.  "Hanging wall side = positive" means the
 *     side the normal points toward moves down (normal fault).
 *     "negative" means it moves up (thrust).
 *
 * Stored particle properties  (1 + dim scalars):
 *   [0]       fault_viscosity      -- adaptive viscosity [Pa s]
 *   [1..dim]  fault_force_x/y(/z)  -- body force components [N/m^3]
 *             (zero when slip sense is correct)
 */

#include <aspect/particle/property/interface.h>
#include <aspect/simulator_access.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      template <int dim>
      class FaultViscosity : public Interface<dim>,
                             public SimulatorAccess<dim>
      {
        public:

          void
          initialize_one_particle_property (const Point<dim> & /*position*/,
                                            std::vector<double> &data) const override
          {
            data.push_back (initial_viscosity);      // [0] fault_viscosity
            for (unsigned int d = 0; d < dim; ++d)
              data.push_back (0.0);                  // [1..dim] fault_force_*
          }


          void
          update_particle_properties (const ParticleUpdateInputs<dim> &inputs,
                                      typename ParticleHandler<dim>::particle_iterator_range &particles) const override
          {
            const unsigned int fault_cidx =
              this->introspection().compositional_index_for_name (fault_composition_name);

            const unsigned int fault_comp_component =
              this->introspection().component_indices.compositional_fields[fault_cidx];

            unsigned int p = 0;
            for (auto &particle : particles)
              {
                ArrayView<double> props = particle.get_properties ();
                const unsigned int base = this->data_position;

                // ----------------------------------------------------------
                // 1. Is this particle inside the fault zone?
                // ----------------------------------------------------------
                const double comp_val = inputs.solution[p][fault_comp_component];

                if (comp_val < fault_composition_threshold)
                  {
                    // Outside fault: zero force, leave viscosity unchanged.
                    for (unsigned int d = 0; d < dim; ++d)
                      props[base + 1 + d] = 0.0;
                    ++p;
                    continue;
                  }

                // ----------------------------------------------------------
                // 2. Local fault normal from composition gradient.
                //    The gradient of the fault composition field points
                //    across the fault (low -> high), giving the normal.
                // ----------------------------------------------------------
                Tensor<1,dim> normal = inputs.gradients[p][fault_comp_component];
                const double normal_norm = normal.norm ();

                if (normal_norm < 1e-30)
                  {
                    // Particle deep inside fault zone, gradient too small.
                    // Skip this step -- properties carry over from last step.
                    ++p;
                    continue;
                  }
                normal /= normal_norm;

                // ----------------------------------------------------------
                // 3. Local down-dip slip direction.
                //    In 2D: rotate normal 90 degrees CCW then apply sign.
                //    In 3D: project gravity direction onto the fault plane.
                //    hanging_wall_sign: +1 = normal fault, -1 = thrust.
                // ----------------------------------------------------------
                Tensor<1,dim> slip_dir;
                if (dim == 2)
                  {
                    slip_dir[0] = -normal[1] * hanging_wall_sign;
                    slip_dir[1] =  normal[0] * hanging_wall_sign;
                  }
                else
                  {
                    Tensor<1,dim> down;
                    down[dim-1] = -1.0;
                    slip_dir = down - (down * normal) * normal;
                    const double sd_norm = slip_dir.norm ();
                    if (sd_norm < 1e-10)
                      { ++p; continue; }   // horizontal fault: skip
                    slip_dir /= sd_norm;
                    slip_dir *= hanging_wall_sign;
                  }

                // ----------------------------------------------------------
                // 4. Strain rate and signed shear projection.
                //    Pattern from integrated_strain.cc.
                // ----------------------------------------------------------
                Tensor<2,dim> grad_u;
                for (unsigned int d = 0; d < dim; ++d)
                  grad_u[d] = inputs.gradients[p][
                    this->introspection().component_indices.velocities[d]];

                const SymmetricTensor<2,dim> eps = deviator (symmetrize (grad_u));

                // Signed shear strain rate on the fault plane.
                // Positive = correct-sense, negative = wrong-sense.
                double edot_signed = 0.0;
                for (unsigned int i = 0; i < dim; ++i)
                  for (unsigned int j = 0; j < dim; ++j)
                    edot_signed += normal[i] * eps[i][j] * slip_dir[j];

                // ----------------------------------------------------------
                // 5. Branch on slip sense.
                // ----------------------------------------------------------
                if (edot_signed >= 0.0)
                  {
                    // ---- Correct-sense: viscosity feedback ----
                    // Zero the body force.
                    for (unsigned int d = 0; d < dim; ++d)
                      props[base + 1 + d] = 0.0;

                    // Lower eta -> more strain.  Correct feedback direction:
                    //   ratio < 1  (too slow)  ->  eta decreases
                    //   ratio > 1  (too fast)  ->  eta increases
                    const double edot_clamped = std::max (edot_signed, min_strain_rate);
                    double ratio = edot_clamped / target_edot;
                    ratio = std::max (ratio, 1.0 / max_correction_factor);
                    ratio = std::min (ratio,        max_correction_factor);

                    double new_eta = props[base] * std::pow (ratio, relaxation_exponent);
                    new_eta = std::max (min_viscosity, std::min (max_viscosity, new_eta));
                    props[base] = new_eta;
                  }
                else
                  {
                    // ---- Wrong-sense: body force + neutral viscosity ----
                    // Reset to neutral viscosity rather than ramping it up,
                    // which would hurt solver conditioning.
                    props[base] = initial_viscosity;

                    // Force magnitude: stress needed to drive target strain
                    // rate at neutral viscosity, plus a term proportional to
                    // how wrong the current slip is.
                    // Units: Pa/m = N/m^3.
                    const double edot_wrong = std::abs (edot_signed);
                    double f_mag = initial_viscosity
                                   * (target_edot + edot_wrong)
                                   / fault_half_width;
                    f_mag = std::min (f_mag, max_body_force);

                    for (unsigned int d = 0; d < dim; ++d)
                      props[base + 1 + d] = f_mag * slip_dir[d];
                  }

                ++p;
              }
          }


          UpdateTimeFlags need_update () const override
          {
            return update_time_step;
          }


          UpdateFlags get_update_flags (const unsigned int component) const override
          {
            // Velocity: need gradients to build the strain rate tensor.
            if (this->introspection().component_masks.velocities[component])
              return update_gradients;

            // Composition: need values (fault-zone check) and gradients
            // (local normal direction from composition gradient).
            if (this->introspection().component_masks.compositions[component])
              return update_values | update_gradients;

            return update_default;
          }


          std::vector<std::pair<std::string, unsigned int>>
          get_property_information () const override
          {
            std::vector<std::pair<std::string, unsigned int>> info;
            info.emplace_back ("fault_viscosity", 1);
            const std::array<std::string,3> suffixes = {{"x","y","z"}};
            for (unsigned int d = 0; d < dim; ++d)
              info.emplace_back ("fault_force_" + suffixes[d], 1);
            return info;
          }


          static void declare_parameters (ParameterHandler &prm)
          {
            prm.enter_subsection ("Fault viscosity");
            {
              prm.declare_entry ("Fault composition name", "wkzs",
                                 Patterns::Anything (),
                                 "Name of the compositional field that identifies the fault zone.");

              prm.declare_entry ("Fault composition threshold", "0.5",
                                 Patterns::Double (0., 1.),
                                 "Minimum composition value for a particle to be inside the fault.");

              prm.declare_entry ("Hanging wall side", "positive",
                                 Patterns::Selection ("positive|negative"),
                                 "Slip sense for the dip-slip fault. "
                                 "'positive': hanging wall (normal-pointing side) goes down = normal fault. "
                                 "'negative': hanging wall goes up = thrust fault. "
                                 "If you see large fault_force_* values in the output flip this.");

              prm.declare_entry ("Target slip velocity", "3.17e-10",
                                 Patterns::Double (0.),
                                 "Target slip velocity across the full fault zone [m/s]. "
                                 "1 cm/yr = 3.17e-10 m/s. "
                                 "Converted to edot_target = V_slip / (2 * Fault half-width).");

              prm.declare_entry ("Fault half-width", "5000.0",
                                 Patterns::Double (0.),
                                 "Half-width of the fault zone [m]. Used to convert target "
                                 "slip velocity to target strain rate, and to scale the body "
                                 "force for wrong-sense slip.");

              prm.declare_entry ("Initial viscosity", "1e20",
                                 Patterns::Double (0.),
                                 "Starting viscosity and neutral viscosity restored on "
                                 "wrong-sense slip [Pa s].");

              prm.declare_entry ("Minimum viscosity", "1e17",
                                 Patterns::Double (0.),
                                 "Hard lower bound on adaptive viscosity [Pa s]. "
                                 "Should be >= the global minimum viscosity.");

              prm.declare_entry ("Maximum viscosity", "1e24",
                                 Patterns::Double (0.),
                                 "Hard upper bound on adaptive viscosity [Pa s].");

              prm.declare_entry ("Relaxation exponent", "0.5",
                                 Patterns::Double (0., 2.),
                                 "Exponent alpha in eta_new = eta_old * (edot/edot_target)^alpha. "
                                 "0.3-0.5 gives stable convergence over several steps.");

              prm.declare_entry ("Maximum correction factor", "2.0",
                                 Patterns::Double (1.),
                                 "Maximum factor by which viscosity may change per timestep. "
                                 "Prevents runaway feedback during the initial transient.");

              prm.declare_entry ("Maximum body force", "1e10",
                                 Patterns::Double (0.),
                                 "Cap on body force for wrong-sense slip [N/m^3]. "
                                 "Reduce if Stokes convergence degrades. "
                                 "A physically motivated value is yield_stress / half_width.");

              prm.declare_entry ("Minimum strain rate", "1e-20",
                                 Patterns::Double (0.),
                                 "Floor strain rate to avoid division by zero [1/s].");
            }
            prm.leave_subsection ();
          }


          void parse_parameters (ParameterHandler &prm) override
          {
            prm.enter_subsection ("Fault viscosity");
            {
              fault_composition_name      = prm.get ("Fault composition name");
              fault_composition_threshold = prm.get_double ("Fault composition threshold");

              AssertThrow (this->introspection().compositional_name_exists (fault_composition_name),
                           ExcMessage ("Fault viscosity: no compositional field named '"
                                       + fault_composition_name + "' found."));

              hanging_wall_sign = (prm.get ("Hanging wall side") == "positive") ? 1.0 : -1.0;

              const double v_slip = prm.get_double ("Target slip velocity");
              fault_half_width    = prm.get_double ("Fault half-width");
              AssertThrow (fault_half_width > 0,
                           ExcMessage ("Fault viscosity: Fault half-width must be > 0."));
              target_edot = v_slip / (2.0 * fault_half_width);

              initial_viscosity     = prm.get_double ("Initial viscosity");
              min_viscosity         = prm.get_double ("Minimum viscosity");
              max_viscosity         = prm.get_double ("Maximum viscosity");
              relaxation_exponent   = prm.get_double ("Relaxation exponent");
              max_correction_factor = prm.get_double ("Maximum correction factor");
              max_body_force        = prm.get_double ("Maximum body force");
              min_strain_rate       = prm.get_double ("Minimum strain rate");

              AssertThrow (min_viscosity <= initial_viscosity &&
                           initial_viscosity <= max_viscosity,
                           ExcMessage ("Fault viscosity: Initial viscosity must lie within "
                                       "[Minimum viscosity, Maximum viscosity]."));
            }
            prm.leave_subsection ();
          }


        private:
          std::string fault_composition_name;
          double fault_composition_threshold;
          double hanging_wall_sign;
          double target_edot;
          double fault_half_width;
          double initial_viscosity;
          double min_viscosity;
          double max_viscosity;
          double relaxation_exponent;
          double max_correction_factor;
          double max_body_force;
          double min_strain_rate;
      };

    }
  }
}


namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      ASPECT_REGISTER_PARTICLE_PROPERTY (
        FaultViscosity,
        "fault viscosity",
        "Imposes a target slip rate on a curved dip-slip fault. "
        "For correct-sense slip the stored viscosity is nudged toward "
        "the value that produces edot_target = V_slip/(2W). "
        "For wrong-sense slip a body force is applied in the correct "
        "slip direction; this requires enable_additional_stokes_rhs = true "
        "and the visco_plastic.cc patch described in the usage notes.")
    }
  }
}
