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

#ifndef _aspect_material_model_equation_of_state_thermodynamic_table_lookup_h
#define _aspect_material_model_equation_of_state_thermodynamic_table_lookup_h

#include <aspect/material_model/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/material_model/equation_of_state/interface.h>

#include <aspect/material_model/rheology/peierls_creep.h>

namespace aspect
{
  namespace MaterialModel
  {
    namespace EquationOfState
    {
      using namespace dealii;

      /**
       * An equation of state class that reads thermodynamic properties
       * from pressure-temperature tables in input files. These input files
       * can be created using codes such as Perple_X or HeFESTo.
       */
      template <int dim>
      class ThermodynamicTableLookup : public ::aspect::SimulatorAccess<dim>
      {
        public:
          /**
           * Initialization function. Loads the material data and sets up
           * pointers.
           */
          void
          initialize ();

          /**
           * Returns the number of lookups
           */
          virtual unsigned int number_of_lookups() const;

          /**
           * Return whether the model is compressible or not.  Incompressibility
           * does not necessarily imply that the density is constant; rather, it
           * may still depend on temperature or pressure. In the current
           * context, compressibility means whether we should solve the continuity
           * equation as $\nabla \cdot (\rho \mathbf u)=0$ (compressible Stokes)
           * or as $\nabla \cdot \mathbf{u}=0$ (incompressible Stokes).
           */
          bool is_compressible () const;

          /**
           * Function to compute the thermodynamic properties in @p out given the
           * inputs in @p in over all evaluation points.
           * This function also fills the mass_fraction and volume_fraction vectors.
           */
          void
          evaluate(const MaterialModel::MaterialModelInputs<dim> &in,
                   std::vector<MaterialModel::EquationOfStateOutputs<dim>> &eos_outputs) const;

          /**
           * A function that computes the output of the equation of state @p out
           * for all compositions and phases, given the inputs in @p in, material lookup tables, and an
           * index input_index that determines which entry of the vector of inputs is used.
           */
          void
          evaluate_phases(const MaterialModel::MaterialModelInputs<dim> &in,
               const unsigned int input_index,
               MaterialModel::EquationOfStateOutputs<dim> &out,
               std::vector<MaterialModel::EquationOfStateOutputs<dim>> &eos_outputs) const;          

          void
          get_h2o(std::vector<double> &h2omax, std::vector<bool> &islookup,
            const double &temperature,
            const double &pressure,
            const unsigned int &i,const unsigned int &j,const unsigned int &base,
            const std::vector<unsigned int> &n_phases_per_composition,
            const std::vector<double> &phase_function_values,
            const std::vector<double> &volume_fractions) const;

          void
          get_h2o_serp(std::vector<double> &h2omax,
          const double &temperature,
          const double &pressure,
          const unsigned int &i,const unsigned int &j,const unsigned int &base,
          const std::vector<unsigned int> &n_phases_per_composition,
          const std::vector<double> &phase_function_values) const;

          /**
           * Function to fill the seismic velocity and phase volume additional outputs
           */
          void
          fill_additional_outputs(const MaterialModel::MaterialModelInputs<dim> &in,
                                  const std::vector<std::vector<double>> &volume_fractions,
                                  MaterialModel::MaterialModelOutputs<dim> &out) const;

          /**
           * Function to fill rheological parameters from material lookup tables for dislocation creep
           */
          unsigned int
          fill_lookup_rheology (Rheology::DislocationCreep<dim> &dislocation_creep, 
                                Rheology::DiffusionCreep<dim> &diffusion_creep,
                                const Rheology::DislocationCreep<dim> &initial_dislocation_creep,
                                const int base,
                                const unsigned int j,
                                const double temperature_for_viscosity,
                                const double pressure_for_creep,
                                const std::vector<double> &volume_fractions,
                                const std::vector<double> &phase_function_values,
                                const std::vector<unsigned int> &n_phases_per_composition) const;




          /**
           * Declare the parameters this class takes through input files.
           */
          static
          void
          declare_parameters (ParameterHandler &prm,const double default_thermal_expansion = 3.5e-5);

          /**
           * Read the parameters this class declares from the parameter file.
           */
          void
          parse_parameters (ParameterHandler &prm,
                            const std::unique_ptr<std::vector<unsigned int>> &expected_n_phases_per_composition = nullptr);

          void
          create_additional_named_outputs (MaterialModel::MaterialModelOutputs<dim> &out) const;

          const MaterialModel::MaterialUtilities::Lookup::MaterialLookup &
          get_material_lookup (unsigned int lookup_index) const;


          const std::vector<double>
          get_phases_using_material_files() const;

          /**
           * List of pointers to objects that read and process data we get from
           * material data files. There is one pointer/object per lookup file.
           */
          std::vector<std::unique_ptr<MaterialModel::MaterialUtilities::Lookup::MaterialLookup>> material_lookup;
        private:
          /**
          * Vector of lookup tables to be used
          */
          std::vector<double> phases_using_material_files;

          /**
          * Vector of where viscosities used from lookup tables
          */
          std::vector<double> phases_using_lookup_viscosities;

          /**
          * Vector of activation volumes used from lookup tables
          */
          std::vector<double> activation_volumes_dislocation_lookup;

          /**
          * Vector of activation energies used from lookup tables
          */
          std::vector<double> activation_energies_dislocation_lookup;

          /**
          * Vector of prefactors used from lookup tables
          */
          std::vector<double> prefactors_dislocation_lookup;

          /**
          * Vector of stress exponents used from lookup tables
          */
          std::vector<double> stress_exponents_dislocation_lookup;

          /**
          * Vector of flags of whether to use water fugacity
          */
          std::vector<double> use_water_fugacity;

          /**
          * Vector of flags of whether to use exponential melt weakening for small melt fractions
          */
          std::vector<double> use_melt_weakening;

          /**
          * Melt weakening factor for small melt fractions
          */
          double exponential_melt_weakening_factor;

          /**
          * Peierls parameters
          */
          std::vector<double> fitting_parameters_Peierls_lookup;
          std::vector<double> p_Peierls_lookup;
          std::vector<double> q_Peierls_lookup;
          std::vector<double> reference_stress_Peierls_lookup;
          std::vector<double> activation_energies_Peierls_lookup;
          std::vector<double> prefactors_Peierls_lookup;
          std::vector<double> stress_exponents_Peierls_lookup;
          std::vector<double> use_water_fugacity_Peierls;
          std::vector<double> use_melt_weakening_Peierls;

          /**
          * Phase names or identifiers for viscosity lookups
          */
          std::vector<std::string> viscosity_lookup_phase_names;          

          /**
          * Phase names or identifiers for viscosity lookups
          */
          std::vector<std::string> peierls_viscosity_lookup_phase_names;

          /**
          * Material model file names with duplicates removed
          */
          std::vector<std::string> unique_material_file_names;

          /**
          * Number of phases used for viscosity lookups
          */
          std::unique_ptr<std::vector<unsigned int>> expected_n_phases_per_viscosity_lookup;

          /**
          * Number of phases used for viscosity lookups
          */
          std::unique_ptr<std::vector<unsigned int>> peierls_expected_n_phases_per_viscosity_lookup;

          /**
 *            * Vector of reference densities $\rho_0$ with one entry per composition and phase plus one
 *                       * for the background field.
 *                                  */
          std::vector<double> densities;

          /**
 *            * The reference temperature $T_0$ used in the computation of the density.
 *                       * All components use the same reference temperature.
 *                                  */
          double reference_T;

          /**
 *            * Vector of thermal expansivities with one entry per composition and phase plus one
 *                       * for the background field.
 *                                  */
          std::vector<double> thermal_expansivities;

          /**
 *            * Vector of specific heat capacities with one entry per composition and phase plus one
 *                       * for the background field.
 *                                  */
          std::vector<double> specific_heats;

          unsigned int n_material_lookups;
          bool use_bilinear_interpolation;
          bool latent_heat;

          /**
           * Information about the location of data files.
           */
          std::string data_directory;
          std::vector<std::string> material_file_names;
          std::vector<std::string> derivatives_file_names;

          /**
           * The maximum number of substeps over the temperature pressure range
           * to calculate the averaged enthalpy gradient over a cell
           */
          unsigned int max_latent_heat_substeps;

          /**
           * The format of the provided material files. Currently we support
           * the Perple_X and HeFESTo data formats.
           */
          enum formats
          {
            perplex,
            hefesto
          } material_file_format;

          /**
           * Vector of strings containing the names of the unique phases in all the material lookups.
           */
          std::vector<std::string> unique_phase_names;

          /**
           * Vector of vector of unsigned ints which constitutes mappings
           * between lookup phase name vectors and unique_phase_names.
           * The element unique_phase_indices[i][j] contains the
           * index of phase name j from lookup i as it is found in unique_phase_names.
           */
          std::vector<std::vector<unsigned int>> unique_phase_indices;

          /**
           * Vector of strings containing the names of the dominant phases in all the material lookups.
           * In case the lookup table contains one string with the dominant phase rather than separate
           * columns with volume fraction for each phase, this vector will be used instead of the
           * unique_phase_names vector above.
           */
          std::vector<std::string> list_of_dominant_phases;

          /**
           * Each lookup table reads in their own dominant phases and assigns indices based
           * on all phases in that particular lookup. Since a model can have more than one
           * lookup, it might have more phases than present in each lookup. We want to output
           * unique/consistent indices for each phase, so we have to convert the indices of a phase
           * in the individual lookup to the index in the global list of phases. This vector
           * of vectors of unsigned int stores the global index for each lookup (so there are
           * as many inner vectors as lookups, and each one stores the indices for an individual
           * lookup, to be filled in the initialize function).
           *
           * In case the lookup table contains one string with the dominant phase rather than separate
           * columns with volume fraction for each phase, this vector will be used instead of the
           * unique_phase_indices vector above.
           */
          std::vector<std::vector<unsigned int>> global_index_of_lookup_phase;

          /**
           * Compute the specific heat and thermal expansivity using the pressure
           * and temperature derivatives of the specific enthalpy.
           * This evaluation incorporates the effects of latent heat production.
           */
          void evaluate_thermal_enthalpy_derivatives(const MaterialModel::MaterialModelInputs<dim> &in,
                                                     std::vector<MaterialModel::EquationOfStateOutputs<dim>> &eos_outputs) const;

          /**
           * Returns the cell-wise averaged enthalpy derivatives for the evaluate
           * function and postprocessors. The function returns two pairs, the
           * first one represents the temperature derivative, the second one the
           * pressure derivative. The first member of each pair is the derivative,
           * the second one the number of vertex combinations the function could
           * use to compute the derivative. The second member is useful to handle
           * the case no suitable combination of vertices could be found (e.g.
           * if the temperature and pressure on all vertices of the current
           * cell is identical.
           */
          std::array<std::pair<double, unsigned int>,2>
          enthalpy_derivatives (const typename Interface<dim>::MaterialModelInputs &in) const;

          void fill_seismic_velocities (const MaterialModel::MaterialModelInputs<dim> &in,
                                        const std::vector<double> &composite_densities,
                                        const std::vector<std::vector<double>> &volume_fractions,
                                        SeismicAdditionalOutputs<dim> *seismic_out) const;

          /**
           * This function uses the MaterialModelInputs &in to fill the output_values
           * of the phase_volume_fractions_out output object with the volume
           * fractions of each of the unique phases at each of the evaluation points.
           * These volume fractions are obtained from the Perple_X- or HeFESTo-derived
           * pressure-temperature lookup tables.
           * The filled output_values object is a vector of vector<double>;
           * the outer vector is expected to have a size that equals the number
           * of unique phases, the inner vector is expected to have a size that
           * equals the number of evaluation points.
           */
          void fill_phase_volume_fractions (const MaterialModel::MaterialModelInputs<dim> &in,
                                            const std::vector<std::vector<double>> &volume_fractions,
                                            NamedAdditionalMaterialOutputs<dim> *phase_volume_fractions_out) const;

          /**
           * This function uses the MaterialModelInputs &in to fill the output_values
           * of the dominant_phases_out output object with the index of the
           * dominant phase at each of the evaluation points.
           * The phases are obtained from the Perple_X- or HeFESTo-derived
           * pressure-temperature lookup tables.
           * The filled output_values object is a vector of vector<unsigned int>;
           * the outer vector is expected to have a size of 1, the inner vector is
           * expected to have a size that equals the number of evaluation points.
           */
          void fill_dominant_phases (const MaterialModel::MaterialModelInputs<dim> &in,
                                     const std::vector<std::vector<double>> &volume_fractions,
                                     PhaseOutputs<dim> &dominant_phases_out) const;
      };
    }
  }
}

#endif
