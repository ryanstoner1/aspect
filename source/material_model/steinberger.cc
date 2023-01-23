/*
  Copyright (C) 2011 - 2022 by the authors of the ASPECT code.

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


#include <aspect/material_model/steinberger.h>
#include <aspect/material_model/equation_of_state/interface.h>
#include <aspect/adiabatic_conditions/interface.h>
#include <aspect/utilities.h>
#include <aspect/lateral_averaging.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/table.h>
#include <fstream>
#include <iostream>
#include <memory>


namespace aspect
{
  namespace MaterialModel
  {
    namespace internal
    {
      LateralViscosityLookup::LateralViscosityLookup(const std::string &filename,
                                                     const MPI_Comm &comm)
      {
        std::string temp;
        // Read data from disk and distribute among processes
        std::istringstream in(Utilities::read_and_distribute_file_content(filename, comm));

        std::getline(in, temp); // eat first line

        min_depth=1e20;
        max_depth=-1;

        while (!in.eof())
          {
            double visc, depth;
            in >> visc;;
            if (in.eof())
              break;
            in >> depth;
            depth *=1000.0;
            std::getline(in, temp);

            min_depth = std::min(depth, min_depth);
            max_depth = std::max(depth, max_depth);

            values.push_back(visc);
          }
        delta_depth = (max_depth-min_depth)/(values.size()-1);
      }

      double LateralViscosityLookup::lateral_viscosity(double depth) const
      {
        depth=std::max(min_depth, depth);
        depth=std::min(depth, max_depth);

        Assert(depth>=min_depth, ExcMessage("ASPECT found a depth less than min_depth."));
        Assert(depth<=max_depth, ExcMessage("ASPECT found a depth greater than max_depth."));
        const unsigned int idx = static_cast<unsigned int>((depth-min_depth)/delta_depth);
        Assert(idx<values.size(), ExcMessage("Attempting to look up a depth with an index that would be out of range. (depth-min_depth)/delta_depth too large."));
        return values[idx];
      }

      int LateralViscosityLookup::get_nslices() const
      {
        return values.size();
      }

      RadialViscosityLookup::RadialViscosityLookup(const std::string &filename,
                                                   const MPI_Comm &comm)
      {
        std::string temp;
        // Read data from disk and distribute among processes
        std::istringstream in(Utilities::read_and_distribute_file_content(filename, comm));

        min_depth=1e20;
        max_depth=-1;

        while (!in.eof())
          {
            double visc, depth;
            in >> visc;;
            if (in.eof())
              break;
            in >> depth;
            depth *=1000.0;
            std::getline(in, temp);

            min_depth = std::min(depth, min_depth);
            max_depth = std::max(depth, max_depth);

            values.push_back(visc);
          }
        delta_depth = (max_depth-min_depth)/(values.size()-1);
      }

      double RadialViscosityLookup::radial_viscosity(double depth) const
      {
        depth=std::max(min_depth, depth);
        depth=std::min(depth, max_depth);

        Assert(depth>=min_depth, ExcMessage("ASPECT found a depth less than min_depth."));
        Assert(depth<=max_depth, ExcMessage("ASPECT found a depth greater than max_depth."));
        const unsigned int idx = static_cast<unsigned int>((depth-min_depth)/delta_depth);
        Assert(idx<values.size(), ExcMessage("Attempting to look up a depth with an index that would be out of range. (depth-min_depth)/delta_depth too large."));
        return values[idx];
      }
    }

    // TODO: causes compilation to fail
    // template <int dim>
    // bool
    // Steinberger<dim>::
    // is_yielding (const double pressure,
    //              const double temperature,
    //              const std::vector<double> &composition,
    //              const SymmetricTensor<2,dim> &strain_rate) const
    // {
    //   /* The following returns whether or not the material is plastically yielding
    //    * as documented in evaluate.
    //    */
    //   bool plastic_yielding = false;

    //   MaterialModel::MaterialModelInputs <dim> in (/*n_evaluation_points=*/1,
    //                                                                        this->n_compositional_fields());
    //   unsigned int i = 0;

    //   in.pressure[i] = pressure;
    //   in.temperature[i] = temperature;
    //   in.composition[i] = composition;
    //   in.strain_rate[i] = strain_rate;

    //   const std::vector<double> volume_fractions
    //     = MaterialUtilities::compute_composition_fractions(composition,
    //                                                        rheology->get_volumetric_composition_mask());

    //   const IsostrainViscosities isostrain_viscosities
    //     = rheology->calculate_isostrain_viscosities(in, i, volume_fractions);

    //   std::vector<double>::const_iterator max_composition
    //     = std::max_element(volume_fractions.begin(),volume_fractions.end());

    //   plastic_yielding = isostrain_viscosities.composition_yielding[std::distance(volume_fractions.begin(),
    //                                                                               max_composition)];

    //   return plastic_yielding;
    // }


    // TODO: equation_of_state -> thermodynamic_table_lookup -> 3 args 
    //
    // template <int dim>
    // bool
    // Steinberger<dim>::
    // is_yielding(const MaterialModelInputs<dim> &in) const
    // {
    //   Assert(in.n_evaluation_points() == 1, ExcInternalError());

    //   const std::vector<double> volume_fractions = MaterialUtilities::compute_composition_fractions(in.composition[0], rheology->get_volumetric_composition_mask());

    //   /* The following handles phases in a similar way as in the 'evaluate' function.
    //    * Results then enter the calculation of plastic yielding.
    //    */
    //   std::vector<double> phase_function_values(phase_function.n_phase_transitions(), 0.0);

    //   if (phase_function.n_phase_transitions() > 0)
    //     {
    //       const double gravity_norm = this->get_gravity_model().gravity_vector(in.position[0]).norm();

    //       double reference_density;
    //       if (this->get_adiabatic_conditions().is_initialized())
    //         {
    //           reference_density = this->get_adiabatic_conditions().density(in.position[0]);
    //         }
    //       else
    //         {
    //           EquationOfStateOutputs<dim> eos_outputs_all_phases (this->n_compositional_fields()+1+phase_function.n_phase_transitions());
    //           equation_of_state.evaluate(in, 0, eos_outputs_all_phases);
    //           reference_density = eos_outputs_all_phases.densities[0];
    //         }

    //       MaterialUtilities::PhaseFunctionInputs<dim> phase_inputs(in.temperature[0],
    //                                                                in.pressure[0],
    //                                                                this->get_geometry_model().depth(in.position[0]),
    //                                                                gravity_norm*reference_density,
    //                                                                numbers::invalid_unsigned_int);

    //       for (unsigned int j=0; j < phase_function.n_phase_transitions(); j++)
    //         {
    //           phase_inputs.phase_index = j;
    //           phase_function_values[j] = phase_function.compute_value(phase_inputs);
    //         }
    //     }

    //   /* The following returns whether or not the material is plastically yielding
    //    * as documented in evaluate.
    //    */
    //   const IsostrainViscosities isostrain_viscosities = rheology->calculate_isostrain_viscosities(in, 0, volume_fractions, phase_function_values, phase_function.n_phase_transitions_for_each_composition());

    //   std::vector<double>::const_iterator max_composition = std::max_element(volume_fractions.begin(), volume_fractions.end());
    //   const bool plastic_yielding = isostrain_viscosities.composition_yielding[std::distance(volume_fractions.begin(), max_composition)];

    //   return plastic_yielding;
    // }


    template <int dim>
    void
    Steinberger<dim>::initialize()
    {
      equation_of_state.initialize();

      lateral_viscosity_lookup
        = std::make_unique<internal::LateralViscosityLookup>(data_directory+lateral_viscosity_file_name,
                                                             this->get_mpi_communicator());
      radial_viscosity_lookup
        = std::make_unique<internal::RadialViscosityLookup>(data_directory+radial_viscosity_file_name,
                                                            this->get_mpi_communicator());
      average_temperature.resize(n_lateral_slices);
    }



    template <int dim>
    void
    Steinberger<dim>::
    update()
    {
      if (use_lateral_average_temperature)
        {
          this->get_lateral_averaging().get_temperature_averages(average_temperature);
          for (double temperature : average_temperature)
            AssertThrow(numbers::is_finite(temperature),
                        ExcMessage("In computing depth averages, there is at"
                                   " least one depth band that does not have"
                                   " any quadrature points in it."
                                   " Consider reducing number of depth layers"
                                   " for averaging specified in the parameter"
                                   " file.(Number lateral average bands)"));
        }
    }



    template <int dim>
    double
    Steinberger<dim>::
    viscosity (const double temperature,
               const double /*pressure*/,
               const std::vector<double> &,
               const SymmetricTensor<2,dim> &,
               const Point<dim> &position) const
    {
      const double depth = this->get_geometry_model().depth(position);
      const double adiabatic_temperature = this->get_adiabatic_conditions().temperature(position);

      double delta_temperature;
      if (use_lateral_average_temperature)
        {
          const unsigned int idx = static_cast<unsigned int>((average_temperature.size()-1) * depth / this->get_geometry_model().maximal_depth());
          delta_temperature = temperature-average_temperature[idx];
        }
      else
        delta_temperature = temperature-adiabatic_temperature;

      // For an explanation on this formula see the Steinberger & Calderwood 2006 paper
      const double vis_lateral_exp = -1.0*lateral_viscosity_lookup->lateral_viscosity(depth)*delta_temperature/(temperature*adiabatic_temperature);
      // Limit the lateral viscosity variation to a reasonable interval
      const double vis_lateral = std::max(std::min(std::exp(vis_lateral_exp),max_lateral_eta_variation),1/max_lateral_eta_variation);

      const double vis_radial = radial_viscosity_lookup->radial_viscosity(depth);

      return std::max(std::min(vis_lateral * vis_radial,max_eta),min_eta);
    }



    template <int dim>
    bool
    Steinberger<dim>::
    is_compressible () const
    {
      return equation_of_state.is_compressible();
    }



    template <int dim>
    double
    Steinberger<dim>::
    thermal_conductivity (const double temperature,
                          const double pressure,
                          const Point<dim> &position) const
    {
      if (conductivity_formulation == constant)
        return thermal_conductivity_value;

      else if (conductivity_formulation == p_T_dependent)
        {
          // Find the conductivity layer that corresponds to the depth of the evaluation point.
          const double depth = this->get_geometry_model().depth(position);
          unsigned int layer_index = std::distance(conductivity_transition_depths.begin(),
                                                   std::lower_bound(conductivity_transition_depths.begin(),conductivity_transition_depths.end(), depth));

          const double p_dependence = reference_thermal_conductivities[layer_index] + conductivity_pressure_dependencies[layer_index] * pressure;

          // Make reasonably sure we will not compute any invalid values due to the temperature-dependence.
          // Since both the temperature-dependence and the saturation term scale with (Tref/T), we have to
          // make sure we can compute the square of this number. If the temperature is small enough to
          // be close to yielding NaN values, the conductivity will be set to the maximum value anyway.
          const double T = std::max(temperature, std::sqrt(std::numeric_limits<double>::min()) * conductivity_reference_temperatures[layer_index]);
          const double T_dependence = std::pow(conductivity_reference_temperatures[layer_index] / T, conductivity_exponents[layer_index]);

          // Function based on the theory of Roufosse and Klemens (1974) that accounts for saturation.
          // For the Tosi formulation, the scaling should be zero so that this term is 1.
          double saturation_function = 1.0;
          if (1./T_dependence > 1.)
            saturation_function = (1. - saturation_scaling[layer_index])
                                  + saturation_scaling[layer_index] * (2./3. * std::sqrt(T_dependence) + 1./3. * 1./T_dependence);

          return std::min(p_dependence * saturation_function * T_dependence, maximum_conductivity);
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
          return numbers::signaling_nan<double>();
        }
    }



    template <int dim>
    void
    Steinberger<dim>::evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                               MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      // added visco_plastic
      const ComponentMask volumetric_compositions = rheology->get_volumetric_composition_mask();
            EquationOfStateOutputs<dim> eos_outputs_all_phases (this->n_compositional_fields()+1+phase_function.n_phase_transitions());
      std::vector<double> average_elastic_shear_moduli (in.n_evaluation_points());

      // Store value of phase function for each phase and composition
      // While the number of phases is fixed, the value of the phase function is updated for every point
      std::vector<double> phase_function_values(phase_function.n_phase_transitions(), 0.0);
      // end added visco_plastic
      
      std::vector<EquationOfStateOutputs<dim>> eos_outputs (in.n_evaluation_points(), equation_of_state.number_of_lookups());
      std::vector<std::vector<double>> volume_fractions (in.n_evaluation_points(), std::vector<double> (equation_of_state.number_of_lookups()));

      // We need to make a copy of the material model inputs because we want to use the adiabatic pressure
      // rather than the real pressure for the equations of state (to avoid numerical instabilities).
      MaterialModel::MaterialModelInputs<dim> eos_in(in);
      for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
        eos_in.pressure[i] = this->get_adiabatic_conditions().pressure(in.position[i]);

      // Evaluate the equation of state properties over all evaluation points
      equation_of_state.evaluate(eos_in, eos_outputs);

      for (unsigned int i=0; i < in.n_evaluation_points(); ++i)
        {
          // First compute the equation of state variables and thermodynamic properties
          equation_of_state.evaluate_phases(in, i, eos_outputs_all_phases, eos_outputs);

          const double gravity_norm = this->get_gravity_model().gravity_vector(in.position[i]).norm();
          const double reference_density = (this->get_adiabatic_conditions().is_initialized())
                                           ?
                                           this->get_adiabatic_conditions().density(in.position[i])
                                           :
                                           eos_outputs_all_phases.densities[0];

          // The phase index is set to invalid_unsigned_int, because it is only used internally
          // in phase_average_equation_of_state_outputs to loop over all existing phases
          MaterialUtilities::PhaseFunctionInputs<dim> phase_inputs(in.temperature[i],
                                                                   in.pressure[i],
                                                                   this->get_geometry_model().depth(in.position[i]),
                                                                   gravity_norm*reference_density,
                                                                   numbers::invalid_unsigned_int);

          // Compute value of phase functions
          for (unsigned int j=0; j < phase_function.n_phase_transitions(); j++)
            {
              phase_inputs.phase_index = j;
              phase_function_values[j] = phase_function.compute_value(phase_inputs);
            }

          // Average by value of gamma function to get value of compositions
          phase_average_equation_of_state_outputs(eos_outputs_all_phases,
                                                  phase_function_values,
                                                  phase_function.n_phase_transitions_for_each_composition(),
                                                  eos_outputs[i]);


          out.thermal_conductivities[i] = thermal_conductivity(in.temperature[i], in.pressure[i], in.position[i]);
          for (unsigned int c=0; c<in.composition[i].size(); ++c)
            out.reaction_terms[i][c] = 0;

          // Calculate volume fractions from mass fractions
          // If there is only one lookup table, set the mass and volume fractions to 1
          std::vector<double> mass_fractions;
          if (equation_of_state.number_of_lookups() == 1)
            mass_fractions.push_back(1.0);
          else
            {
              // We only want to compute mass/volume fractions for fields that are chemical compositions.
              std::vector<double> chemical_compositions;
              const std::vector<typename Parameters<dim>::CompositionalFieldDescription> composition_descriptions = this->introspection().get_composition_descriptions();

              for (unsigned int c=0; c<in.composition[i].size(); ++c)
                if (composition_descriptions[c].type == Parameters<dim>::CompositionalFieldDescription::chemical_composition
                    || composition_descriptions[c].type == Parameters<dim>::CompositionalFieldDescription::unspecified)
                  chemical_compositions.push_back(in.composition[i][c]);

              mass_fractions = MaterialUtilities::compute_composition_fractions(chemical_compositions, *composition_mask);

              // The function compute_volumes_from_masses expects as many mass_fractions as densities.
              // But the function compute_composition_fractions always adds another element at the start
              // of the vector that represents the background field. If there is no lookup table for
              // the background field, the mass_fractions vector is too long and we remove this element.
              if (!has_background_field)
                mass_fractions.erase(mass_fractions.begin());
            }

          volume_fractions[i] = MaterialUtilities::compute_volumes_from_masses(mass_fractions,
                                                                               eos_outputs[i].densities,
                                                                               true);
          const std::vector<double> const_volume_fractions = MaterialUtilities::compute_volumes_from_masses(mass_fractions,
                                                                               eos_outputs[i].densities,
                                                                               true);

	  bool plastic_yielding = false;
          if (in.requests_property(MaterialProperties::viscosity)) {
            // Currently, the viscosities for each of the compositional fields are calculated assuming
            // isostrain amongst all compositions, allowing calculation of the viscosity ratio.
            // TODO: This is only consistent with viscosity averaging if the arithmetic averaging
            // scheme is chosen. It would be useful to have a function to calculate isostress viscosities.
            const IsostrainViscosities isostrain_viscosities = rheology->calculate_isostrain_viscosities(in, i, volume_fractions[i], phase_function_values, phase_function.n_phase_transitions_for_each_composition());
            //out.viscosities[i] = viscosity(in.temperature[i], in.pressure[i], in.composition[i], in.strain_rate[i], in.position[i]);
             out.viscosities[i] = MaterialUtilities::average_value(volume_fractions[i], isostrain_viscosities.composition_viscosities, rheology->viscosity_averaging);

             // Decide based on the maximum composition if material is yielding.
             // This avoids for example division by zero for harmonic averaging (as plastic_yielding
             // holds values that are either 0 or 1), but might not be consistent with the viscosity
             // averaging chosen.
              std::vector<double>::const_iterator max_composition = std::max_element(const_volume_fractions.begin(),const_volume_fractions.end());
              plastic_yielding = isostrain_viscosities.composition_yielding[std::distance(const_volume_fractions.begin(),max_composition)];                                                                       
          }
	  rheology->strain_rheology.fill_reaction_outputs(in, i, rheology->min_strain_rate, plastic_yielding, out);

          // Fill plastic outputs if they exist.
          rheology->fill_plastic_outputs(i,volume_fractions[i],plastic_yielding,in,out, phase_function_values, phase_function.n_phase_transitions_for_each_composition());
          MaterialUtilities::fill_averaged_equation_of_state_outputs(eos_outputs[i], mass_fractions, volume_fractions[i], i, out);
          fill_prescribed_outputs(i, volume_fractions[i], in, out);
        }

      rheology->strain_rheology.compute_finite_strain_reaction_terms(in, out);

      equation_of_state.fill_additional_outputs(in, volume_fractions, out);
    }



    template <int dim>
    void
    Steinberger<dim>::
    fill_prescribed_outputs(const unsigned int q,
                            const std::vector<double> &,
                            const MaterialModel::MaterialModelInputs<dim> &,
                            MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      // set up variable to interpolate prescribed field outputs onto compositional field
      PrescribedFieldOutputs<dim> *prescribed_field_out = out.template get_additional_output<PrescribedFieldOutputs<dim>>();

      if (this->introspection().composition_type_exists(Parameters<dim>::CompositionalFieldDescription::density)
          && prescribed_field_out != nullptr)
        {
          const unsigned int projected_density_index = this->introspection().find_composition_type(Parameters<dim>::CompositionalFieldDescription::density);
          prescribed_field_out->prescribed_field_outputs[q][projected_density_index] = out.densities[q];
        }
    }



    template <int dim>
    void
    Steinberger<dim>::declare_parameters (ParameterHandler &prm)
    {
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Steinberger model");
        {
        Rheology::ViscoPlastic<dim>::declare_parameters(prm);  
	      MaterialUtilities::PhaseFunction<dim>::declare_parameters(prm);
        // Equation of state parameters
        prm.declare_entry ("Thermal diffusivities", "0.8e-6",
                            Patterns::List(Patterns::Double (0.)),
                            "List of thermal diffusivities, for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of compositional fields. "
                            "If only one value is given, then all use the same value.  "
                            "Units: \\si{\\meter\\squared\\per\\second}.");
        prm.declare_entry ("Define thermal conductivities","false",
                            Patterns::Bool (),
                            "Whether to directly define thermal conductivities for each compositional field "
                            "instead of calculating the values through the specified thermal diffusivities, "
                            "densities, and heat capacities. ");
        prm.declare_entry ("Thermal conductivities", "3.0",
                            Patterns::List(Patterns::Double(0)),
                            "List of thermal conductivities, for background material and compositional fields, "
                            "for a total of N+1 values, where N is the number of compositional fields. "
                            "If only one value is given, then all use the same value. "
                            "Units: \\si{\\watt\\per\\meter\\per\\kelvin}.");
	      prm.declare_entry ("Data directory", "$ASPECT_SOURCE_DIR/data/material-model/steinberger/",
                             Patterns::DirectoryName (),
                             "The path to the model data. The path may also include the special "
                             "text '$ASPECT_SOURCE_DIR' which will be interpreted as the path "
                             "in which the ASPECT source files were located when ASPECT was "
                             "compiled. This interpretation allows, for example, to reference "
                             "files located in the `data/' subdirectory of ASPECT. ");
          prm.declare_entry ("Radial viscosity file name", "radial-visc.txt",
                             Patterns::Anything (),
                             "The file name of the radial viscosity data. ");
          prm.declare_entry ("Lateral viscosity file name", "temp-viscosity-prefactor.txt",
                             Patterns::Anything (),
                             "The file name of the lateral viscosity data. ");
          prm.declare_entry ("Use lateral average temperature for viscosity", "true",
                             Patterns::Bool (),
                             "Whether to use to use the laterally averaged temperature "
                             "instead of the adiabatic temperature as reference for the "
                             "viscosity calculation. This ensures that the laterally averaged "
                             "viscosities remain more or less constant over the model "
                             "runtime. This behaviour might or might not be desired.");
          prm.declare_entry ("Number lateral average bands", "10",
                             Patterns::Integer (1),
                             "Number of bands to compute laterally averaged temperature within.");
          prm.declare_entry ("Maximum lateral viscosity variation", "1e2",
                             Patterns::Double (0.),
                             "The relative cutoff value for lateral viscosity variations "
                             "caused by temperature deviations. The viscosity may vary "
                             "laterally by this factor squared.");
          prm.declare_entry ("Thermal conductivity", "4.7",
                             Patterns::Double (0.),
                             "The value of the thermal conductivity $k$. Only used in case "
                             "the 'constant' Thermal conductivity formulation is selected. "
                             "Units: \\si{\\watt\\per\\meter\\per\\kelvin}.");
          prm.declare_entry ("Thermal conductivity formulation", "constant",
                             Patterns::Selection("constant|p-T-dependent"),
                             "Which law should be used to compute the thermal conductivity. "
                             "The 'constant' law uses a constant value for the thermal "
                             "conductivity. The 'p-T-dependent' formulation uses equations "
                             "from Stackhouse et al. (2015): First-principles calculations "
                             "of the lattice thermal conductivity of the lower mantle "
                             "(https://doi.org/10.1016/j.epsl.2015.06.050), and Tosi et al. "
                             "(2013): Mantle dynamics with pressure- and temperature-dependent "
                             "thermal expansivity and conductivity "
                             "(https://doi.org/10.1016/j.pepi.2013.02.004) to compute the "
                             "thermal conductivity in dependence of temperature and pressure. "
                             "The thermal conductivity parameter sets can be chosen in such a "
                             "way that either the Stackhouse or the Tosi relations are used. "
                             "The conductivity description can consist of several layers with "
                             "different sets of parameters. Note that the Stackhouse "
                             "parametrization is only valid for the lower mantle (bridgmanite).");
          prm.declare_entry ("Thermal conductivity transition depths", "410000, 520000, 660000",
                             Patterns::List(Patterns::Double (0.)),
                             "A list of depth values that indicate where the transitions between "
                             "the different conductivity parameter sets should occur in the "
                             "'p-T-dependent' Thermal conductivity formulation (in most cases, "
                             "this will be the depths of major mantle phase transitions). "
                             "Units: \\si{\\meter}.");
          prm.declare_entry ("Reference thermal conductivities", "2.47, 3.81, 3.52, 4.9",
                             Patterns::List(Patterns::Double (0.)),
                             "A list of base values of the thermal conductivity for each of the "
                             "horizontal layers in the 'p-T-dependent' Thermal conductivity "
                             "formulation. Pressure- and temperature-dependence will be applied"
                             "on top of this base value, according to the parameters 'Pressure "
                             "dependencies of thermal conductivity' and 'Reference temperatures "
                             "for thermal conductivity'. "
                             "Units: \\si{\\watt\\per\\meter\\per\\kelvin}");
          prm.declare_entry ("Pressure dependencies of thermal conductivity", "3.3e-10, 3.4e-10, 3.6e-10, 1.05e-10",
                             Patterns::List(Patterns::Double ()),
                             "A list of values that determine the linear scaling of the "
                             "thermal conductivity with the pressure in the 'p-T-dependent' "
                             "Thermal conductivity formulation. "
                             "Units: \\si{\\watt\\per\\meter\\per\\kelvin\\per\\pascal}.");
          prm.declare_entry ("Reference temperatures for thermal conductivity", "300, 300, 300, 1200",
                             Patterns::List(Patterns::Double (0.)),
                             "A list of values of reference temperatures used to determine "
                             "the temperature-dependence of the thermal conductivity in the "
                             "'p-T-dependent' Thermal conductivity formulation. "
                             "Units: \\si{\\kelvin}.");
          prm.declare_entry ("Thermal conductivity exponents", "0.48, 0.56, 0.61, 1.0",
                             Patterns::List(Patterns::Double (0.)),
                             "A list of exponents in the temperature-dependent term of the "
                             "'p-T-dependent' Thermal conductivity formulation. Note that this "
                             "exponent is not used (and should have a value of 1) in the "
                             "formulation of Stackhouse et al. (2015). "
                             "Units: none.");
          prm.declare_entry ("Saturation prefactors", "0, 0, 0, 1",
                             Patterns::List(Patterns::Double (0., 1.)),
                             "A list of values that indicate how a given layer in the "
                             "conductivity formulation should take into account the effects "
                             "of saturation on the temperature-dependence of the thermal "
                             "conducitivity. This factor is multiplied with a saturation function "
                             "based on the theory of Roufosse and Klemens, 1974. A value of 1 "
                             "reproduces the formulation of Stackhouse et al. (2015), a value of "
                             "0 reproduces the formulation of Tosi et al., (2013). "
                             "Units: none.");
          prm.declare_entry ("Maximum thermal conductivity", "1000",
                             Patterns::Double (0.),
                             "The maximum thermal conductivity that is allowed in the "
                             "model. Larger values will be cut off.");

          // Table lookup parameters
          EquationOfState::ThermodynamicTableLookup<dim>::declare_parameters(prm);

          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
    }



    template <int dim>
    void
    Steinberger<dim>::parse_parameters (ParameterHandler &prm)
    {
      // increment by one for background:
      const unsigned int n_fields = this->n_compositional_fields() + 1;
      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Steinberger model");
        {
          phase_function.initialize_simulator (this->get_simulator());
	  phase_function.parse_parameters (prm);
	  std::vector<unsigned int> n_phase_transitions_for_each_composition
          (phase_function.n_phase_transitions_for_each_composition());
          // We require one more entry for density, etc as there are phase transitions
          // (for the low-pressure phase before any transition).
          for (unsigned int &n : n_phase_transitions_for_each_composition)
          	n += 1;

	  data_directory = Utilities::expand_ASPECT_SOURCE_DIR(prm.get ("Data directory"));
          radial_viscosity_file_name   = prm.get ("Radial viscosity file name");
          lateral_viscosity_file_name  = prm.get ("Lateral viscosity file name");
          use_lateral_average_temperature = prm.get_bool ("Use lateral average temperature for viscosity");
          n_lateral_slices = prm.get_integer("Number lateral average bands");
          min_eta              = 1e19; // prm.get_double ("Minimum viscosity");
          max_eta              = 1e23; // prm.get_double ("Maximum viscosity");
          max_lateral_eta_variation = 1e3; // prm.get_double ("Maximum lateral viscosity variation");
          thermal_conductivity_value = prm.get_double ("Thermal conductivity");

          // Rheological parameters
          if (prm.get ("Thermal conductivity formulation") == "constant")
            conductivity_formulation = constant;
          else if (prm.get ("Thermal conductivity formulation") == "p-T-dependent")
            conductivity_formulation = p_T_dependent;
          else
            AssertThrow(false, ExcMessage("Not a valid thermal conductivity formulation"));

          conductivity_transition_depths = Utilities::string_to_double
                                           (Utilities::split_string_list(prm.get ("Thermal conductivity transition depths")));
          const unsigned int n_conductivity_layers = conductivity_transition_depths.size() + 1;
          AssertThrow (std::is_sorted(conductivity_transition_depths.begin(), conductivity_transition_depths.end()),
                       ExcMessage("The list of 'Thermal conductivity transition depths' must "
                                  "be sorted such that the values increase monotonically."));

          reference_thermal_conductivities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Reference thermal conductivities"))),
                                                                                     n_conductivity_layers,
                                                                                     "Reference thermal conductivities");
          conductivity_pressure_dependencies = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Pressure dependencies of thermal conductivity"))),
                                                                                       n_conductivity_layers,
                                                                                       "Pressure dependencies of thermal conductivity");
          conductivity_reference_temperatures = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Reference temperatures for thermal conductivity"))),
                                                                                        n_conductivity_layers,
                                                                                        "Reference temperatures for thermal conductivity");
          conductivity_exponents = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Thermal conductivity exponents"))),
                                                                           n_conductivity_layers,
                                                                           "Thermal conductivity exponents");
          saturation_scaling = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Saturation prefactors"))),
                                                                       n_conductivity_layers,
                                                                       "Saturation prefactors");
          maximum_conductivity = prm.get_double ("Maximum thermal conductivity");

          // Parse the table lookup parameters
          equation_of_state.initialize_simulator (this->get_simulator());
          equation_of_state.parse_parameters(prm,
                                              std::make_unique<std::vector<unsigned int>>(n_phase_transitions_for_each_composition));

          equation_of_state_viscoplas.initialize_simulator (this->get_simulator());
          equation_of_state_viscoplas.parse_parameters(prm,
                                              std::make_unique<std::vector<unsigned int>>(n_phase_transitions_for_each_composition));

          thermal_diffusivities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Thermal diffusivities"))),
                                                                          n_fields,
                                                                          "Thermal diffusivities");

          define_conductivities = prm.get_bool ("Define thermal conductivities");

          thermal_conductivities = Utilities::possibly_extend_from_1_to_N (Utilities::string_to_double(Utilities::split_string_list(prm.get("Thermal conductivities"))),
                                                                           n_fields,
                                                                           "Thermal conductivities");

          rheology = std::make_unique<Rheology::ViscoPlastic<dim>>();
          rheology->initialize_simulator (this->get_simulator());
          rheology->parse_parameters(prm, std::make_unique<std::vector<unsigned int>>(n_phase_transitions_for_each_composition));
   
	// Check if compositional fields represent a composition
          const std::vector<typename Parameters<dim>::CompositionalFieldDescription> composition_descriptions = this->introspection().get_composition_descriptions();

          // All chemical compositional fields are assumed to represent mass fractions.
          // If the field type is unspecified (has not been set in the input file),
          // we have to assume it also represents a chemical composition for reasons of
          // backwards compatibility.
          composition_mask = std::make_unique<ComponentMask> (this->n_compositional_fields(), false);
          for (unsigned int c=0; c<this->n_compositional_fields(); ++c)
            if (composition_descriptions[c].type == Parameters<dim>::CompositionalFieldDescription::chemical_composition
                || composition_descriptions[c].type == Parameters<dim>::CompositionalFieldDescription::unspecified)
              composition_mask->set(c, true);

          const unsigned int n_chemical_fields = composition_mask->n_selected_components();

          // Assign background field and do some error checking
          AssertThrow ((equation_of_state.number_of_lookups() == 1) ||
                       (equation_of_state.number_of_lookups() == n_chemical_fields) ||
                       (equation_of_state.number_of_lookups() == n_chemical_fields + 1),
                       ExcMessage("The Steinberger material model assumes that all compositional "
                                  "fields of the type chemical composition correspond to mass fractions of "
                                  "materials. There must either be one material lookup file, the same "
                                  "number of material lookup files as compositional fields of type chemical "
                                  "composition, or one additional file (if a background field is used). You "
                                  "have "
                                  + Utilities::int_to_string(equation_of_state.number_of_lookups())
                                  + " material data files, but there are "
                                  + Utilities::int_to_string(n_chemical_fields)
                                  + " fields of type chemical composition."));

          has_background_field = (equation_of_state.number_of_lookups() == n_chemical_fields + 1);

          prm.leave_subsection();
        }
        prm.leave_subsection();

        // Declare dependencies on solution variables
        this->model_dependence.viscosity = NonlinearDependence::temperature;
        this->model_dependence.density = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
        this->model_dependence.compressibility = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
        this->model_dependence.specific_heat = NonlinearDependence::temperature | NonlinearDependence::pressure | NonlinearDependence::compositional_fields;
        this->model_dependence.thermal_conductivity = NonlinearDependence::none;
      }
    }



    template <int dim>
    void
    Steinberger<dim>::create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const
    {
      equation_of_state.create_additional_named_outputs(out);

      if (this->introspection().composition_type_exists(Parameters<dim>::CompositionalFieldDescription::density)
          && out.template get_additional_output<PrescribedFieldOutputs<dim>>() == nullptr)
        {
          const unsigned int n_points = out.n_evaluation_points();
          out.additional_outputs.push_back(
            std::make_unique<MaterialModel::PrescribedFieldOutputs<dim>> (n_points, this->n_compositional_fields()));
        }
    }

  }
}


// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(Steinberger,
                                   "Steinberger",
                                   "This material model looks up the viscosity from the tables that "
                                   "correspond to the paper of Steinberger and Calderwood "
                                   "2006 (``Models of large-scale viscous flow in the Earth's "
                                   "mantle with constraints from mineral physics and surface observations'', "
                                   "Geophys. J. Int., 167, 1461-1481, "
                                   "\\url{http://dx.doi.org/10.1111/j.1365-246X.2006.03131.x}) and material "
                                   "data from a database generated by the thermodynamics code \\texttt{Perplex}, "
                                   "see \\url{http://www.perplex.ethz.ch/}. "
                                   "The default example data builds upon the thermodynamic "
                                   "database by Stixrude 2011 and assumes a pyrolitic composition by "
                                   "Ringwood 1988 but is easily replaceable by other data files. ")
  }
}
