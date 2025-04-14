The scripts and steps used to generate the figures in the paper Stoner et al. are as follows:

# Preprocessing: 

## Data itself:
setup.py: Saves .vtu data in .npz format, which loads more quickly in numpy, allowing scripts that need to be re-run to be re-run faster. 
Change `folder` to desired folder with run output to process. Check file structure if errors. E.g. `python setup.py`

## Calculate plate speeds, make preliminary plots:

Extract initial data
1. extract_csv_pyvista.py: Run in format `extract_csv_pyvista.py <<folder>> <<number timesteps+1>> <<step>>`
E.g. `python extract_csv_pyvista.py rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_
sed_1km_smu0_02_cmu0_04_deserp_erase_res6_5_run36 8801 100`

Extract slab surface data
2. PT_conds_valeria_up.py: Run in format `PT_conds_valeria_up.py <<folder>> <<number timesteps>> <<step>>`
E.g. `python PT_conds_valeria_up.py rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_
sed_1km_smu0_02_cmu0_04_deserp_erase_res6_5_run36 8800 100`

Make preliminary plots (not in paper) and calculate kinematics 
3. plot_model_and_extract_properties_freesurf.py: Run in format `plot_model_and_extract_properties_freesurf.py <<folder>> <<number timesteps+1>> <<step>>`
E.g. `python plot_model_and_extract_properties_freesurf.py rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_
sed_1km_smu0_02_cmu0_04_deserp_erase_res6_5_run36 8801 100`

OPTIONAL: plot kinematic and decoupling depth data
4. kinematics_plot_freesurf.py: E.g. `python kinematics_plot_freesurf.py`

## General check

Filesystem:
I had the format 
>>main folder
  visualization scripts
  >>plotting folder (plots)
  >>results folder
    >>run_folder1
    >>run_folder2
    >>etc.

Fonts: 
I use Calibri, which is not included by default in Unix systems. Change font or follow online instructions to install Calibri.

## Plotting

Most of figure 1 is in Adobe Illustrator, but outline and viscosity plot. Change `folder` to appropriate run output directory. 
1. Figure 1: setup_plot1.py

2. Figure 2: flowchart just in illustrator

Reference model. Change `folder` to appropriate run output directory. Output from file `rc3_smu0_02_cmu0_04_sed1km_pel4_h2oAUG.prm`
3. Figure 3: main_model2.py (e.g., `python main_model2.py`)

Kinematics from reference model. Change `folder` to appropriate run output directory. Output from file `rc3_smu0_02_cmu0_04_sed1km_pel4_h2oAUG.prm`
4. Figure 4: main_model4.py (e.g. `python main_model4.py`)

Cases with varying incoming H2O content.
5. Figure 5: sed_comparison_fig5.py

Kinematics for cases with varying incoming H2O content.
6. Figure 6: h2o5_old.py

Cases with varying friction parameters.
5. Figure 7: h2o7.py

Kinematics for cases with varying friction parameters.
6. Figure 8: h2o7_new.py

9. Figure 9: Adobe Illustrator

Supplementary figures:

Rheology_output.py - supp. fig. 1 - in `pseudosections` folder
h2o_S2.py - supp. fig. 2
h2o_S3.py - supp. fig. 3
h2o_S4_hires.py - supp. fig. 4 high resolution case
S_0mat.py - no density lookup
S_0matserp.py - no lookups whatsoever
S_0serp.py - no serpentinization








