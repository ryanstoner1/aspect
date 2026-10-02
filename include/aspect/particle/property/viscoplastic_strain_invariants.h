/*
 Copyright (C) 2015 - 2024 by the authors of the ASPECT code.

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

#ifndef _aspect_particle_property_viscoplastic_strain_invariant_h
#define _aspect_particle_property_viscoplastic_strain_invariant_h

#include <aspect/particle/property/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/material_model/visco_plastic.h>

namespace aspect
{
  namespace Particle
  {
    namespace Property
    {
      /**
       * A class that calculates the finite strain invariant a particle has
       * experienced and assigns it to either the plastic and/or viscous strain field based
       * on whether the material is plastically yielding, or the total strain field
       * used in the visco plastic material model. The implementation of this property
       * is equivalent to the implementation for compositional fields that is located in
       * the plugin <code>benchmarks/buiter_et_al_2008_jgr/plugin/finite_strain_invariant.cc</code>,
       * and is effectively the same as what the visco plastic material model uses for compositional fields.
       * @ingroup ParticleProperties
       */
      template <int dim>
      class ViscoPlasticStrainInvariant : public Interface<dim>, public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * Constructor.
           */
          ViscoPlasticStrainInvariant ();

          void initialize () override;


          /**
           * @copydoc aspect::Particle::Property::Interface::initialize_one_particle_property()
           */
          void
          initialize_one_particle_property (const Point<dim> &position,
                                            std::vector<double> &particle_properties) const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::update_particle_properties()
           */
          void
          update_particle_properties (const ParticleUpdateInputs<dim> &inputs,
                                      typename ParticleHandler<dim>::particle_iterator_range &particles) const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::need_update()
           */
          UpdateTimeFlags
          need_update () const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::get_update_flags()
           */
          UpdateFlags
          get_update_flags (const unsigned int component) const override;

          /**
           * @copydoc aspect::Particle::Property::Interface::get_property_information()
           */
          std::vector<std::pair<std::string, unsigned int>>
          get_property_information() const override;

          /**
           * Declare the parameters this class takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm);
 
          /**
           * Read the parameters this class declares from the parameter file.
           */
          void
          parse_parameters (ParameterHandler &prm) override;

        private:
          unsigned int n_components;

          /**
           * An object that is used to compute the particle property. Since the
           * object is expensive to create and is needed often it is kept as a
           * member variable. Because it is changed inside a const member function
           * (update_particle_property) it has to be mutable, but since it is
           * only used inside that function and always set before being used
           * that is not a problem. This implementation is not thread safe,
           * but it is currently not used in a threaded context.
           */
          mutable MaterialModel::MaterialModelInputs<dim> material_inputs;

          /**
           * Whether newly-deposited particles above the y-threshold receive
           * a per-particle random plastic strain seed (true), or the standard
           * IC-based seed (false, default — recovers stock behavior).
           */
          bool use_random_sediment_seed;
 
          /**
           * Lower bound of the uniform distribution from which the
           * per-particle random plastic strain seed is drawn.
           */
          double random_seed_min;
 
          /**
           * Upper bound of the uniform distribution from which the
           * per-particle random plastic strain seed is drawn.
           */
          double random_seed_max;
 
          /**
           * Particles created at t > 0 whose initial vertical coordinate
           * exceeds this threshold are treated as freshly-deposited sediment
           * and receive the random seed. Below the threshold, AMR/refinement
           * events fall back to the standard IC-based seed.
           */
          double random_seed_y_threshold;
 
          /**
           * Integer base seed combined with the particle position in the
           * deterministic hash. Change to obtain a different ensemble
           * realization while keeping per-rank reproducibility.
           */
          unsigned int random_seed_base;
 
          /**
           * Compute a deterministic per-particle random plastic strain value
           * from the particle's birth position. Implemented as a SplitMix64-
           * style hash of the quantized coordinates mixed with @ref
           * random_seed_base, drawn through std::mt19937_64 onto the interval
           * [@ref random_seed_min, @ref random_seed_max]. The output depends
           * only on @p position and the seeding parameters, not on MPI rank,
           * particle ordering, or load-balancing history.
           */
          double random_from_position (const Point<dim> &position) const;

      };
    }
  }
}

#endif
