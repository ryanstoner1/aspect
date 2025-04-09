import matplotlib.collections as collections
import scipy.interpolate
import glob
import time
import matplotlib as mpl
import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt
mpl.use('agg')

# 0. load data
# Path to the .npz files
folders = ["rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18",
        "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel2_smu0_02_cmu0_04_deserp_erase_res5_5_run19",
        "rc3_part_lookup_serp_morb_ue_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_smu0_02_cmu0_04_deserp_erase_res5_5_run13"]

fig, ax = plt.subplots(figsize=(12, 6),nrows=1,ncols=2)

for idf,folder in enumerate(folders):
    file_pattern = folder+"/"+"solution/solution-0??00.npz"
    files = sorted(glob.glob(file_pattern))
    stats_file = ''.join([folder,'/statistics'])
    model_output_dt  = 100

    times = np.zeros(len(files))
    serp_sums = np.zeros(len(files))
    for idx,file in enumerate(files):
        # Load the mesh and field data
        data = np.load(files[idx])
        points = data["points"][:,0:2]/1000
        cells = data["cells"]
        viscosity = data["viscosity"]
        temperature = data["temperature"]
        grid_viscosity = data["grid_viscosity"]
        serp_total = data["serp_total"]   

        i = (idx)//2
        j = idx%2

        f=open(stats_file)
        lines=f.readlines()
        num_header_lines = len(list(filter(lambda line: line.startswith("#"),lines))) # num header lines in stats_files (for getting the dimensional time)
        stats_line_num = num_header_lines + (idx * model_output_dt)

        line=lines[stats_line_num]
        time_dim=float(line.split()[1])/1.e6

        times[idx] = time_dim
        serp_sums[idx] = serp_total
    ax[1].plot(times,serp_sums,label=folders[idf])

plt.savefig("vc_evolution_fig4.png",dpi=300)
    


