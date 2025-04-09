import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.collections as collections
import scipy.interpolate
import glob
import time
import matplotlib as mpl
import numpy as np
import matplotlib.tri as tri
import pandas as pd
import time as timemodule
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os
from main_model2 import plot_quads_with_tricontourf

params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

# Define a single-color colormap (e.g., "blue")
single_color_cmap = LinearSegmentedColormap.from_list("SingleColor", ["#D2D344", "#D2D344"])

is_plot_outline = False # or just viscosity 

# ASPECT output 
csvs_loc = os.getcwd()+'/csv_outputs/'

# setup plotting
mpl.use('agg')
plt.rcParams['font.family']="Times New Roman"
plt.rcParams['font.size']=12
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['axes.linewidth'] = 2.0

# 0. load data
# Path to the .npz files1
folder1 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel0_smu0_02_cmu0_04_deserp_erase_res5_5_run20"
file_pattern1 = os.getcwd()+folder1+"/"+"solution/solution-0??00.npz"
files1_full = sorted(glob.glob(file_pattern1))

ids1 = ["01000","03900"]

files1 = [os.getcwd()+folder1+"/solution/solution-"+id1+".npz" for id1 in ids1]

folder2 = "/rc3_part_lookup_serp_morb_bas_h2o_0denslookup_0visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_smu0_02_cmu0_04_deserp_erase_res5_5_run12_new"
ids2 = ["00700","01900"]
files2 = [os.getcwd()+folder2+"/solution/solution-"+id2+".npz" for id2 in ids2]

folder_baseline = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"

fig, ax = plt.subplots(figsize=(16, 10),nrows=3,ncols=2)
# Sub-grid for gs[0, 0] (top-left corner)
sub_gs = [GridSpecFromSubplotSpec(1, 2, subplot_spec=axi.get_subplotspec(), wspace=0.2) for axi in ax.flatten()]

def plot_col(fig,ax,sub_gs,files1,col,ids):

    for idx,file1 in enumerate(files1):
        if col==0:
            stats_file=''.join([os.getcwd(),folder1,'/statistics'])
        elif col==1:
            stats_file=''.join([os.getcwd(),folder2,'/statistics'])

        f=open(stats_file)
        lines=f.readlines()
        num_header_lines = len(list(filter(lambda line: line.startswith("#"),lines)))

        # num header lines in stats_files (for getting the dimensional time)
        idx_time = ''.join(c for c in file1.split('/')[-1] if c.isdigit())
        stats_line_num = num_header_lines + (int(idx_time[1:]) )

        line=lines[stats_line_num]
        time_dim=float(line.split()[1])/1.e6

        ymax = 1450e3
        step = 1000

        t1 = timemodule.time()



        time = idx*step
        if col==0:
            csv_filename=''.join([csvs_loc,folder1,'/full.',str(int(ids[idx])),'.gzip'])
        elif col==1:
            csv_filename=''.join([csvs_loc,folder2,'/full.',str(int(ids[idx])),'.gzip'])
        df = pd.read_parquet(csv_filename)
        model_data = df.values
        header_terms = df.columns.to_list()
        if "Points:0" in header_terms:
            x_col = header_terms.index("Points:0")
        if "Points:1" in header_terms:
            y_col = header_terms.index("Points:1")
        if "ocrust_init" in header_terms:
            ocrust_col = header_terms.index("ocrust_init")
        if "ocrust" in header_terms:
            ocrust_col2 = header_terms.index("ocrust")
        # extract mid-plate profile
        plate_prof_loc = ymax - 0.5e3                           # 20 km depth
        plate_prof = model_data[model_data[:,y_col] < (plate_prof_loc+4.e3)] 
        plate_prof = plate_prof[plate_prof[:,y_col] > (plate_prof_loc-4.e3)] 
        plate_prof = plate_prof[plate_prof[:,x_col].argsort()] # sort by x

        # get trench location
        trench_x = 0;
        for k in range(len(plate_prof)):
            if (((plate_prof[k,ocrust_col] + plate_prof[k,ocrust_col2]) > 0.4)  and plate_prof[k,x_col] > trench_x):
                trench_x = plate_prof[k,x_col]
        if trench_x == 0:
            trench_x = 0#np.nan

        # topo_data = np.loadtxt(files_topo[idx],skiprows=1)
        # x_max = np.max(topo_data[:,0]/1e3)
        # trench_ind = np.argmin(topo_data[:,1][(topo_data[:,0]>0.3*x_max) & (topo_data[:,0]<0.7*x_max)])
        # trench_x = topo_data[:,0][trench_ind]



        # Load the mesh and field data
        data = np.load(files1[idx])
        points = data["points"][:,0:2]/1000
        points[:,1] = (ymax/1e3)-points[:,1]
        cells = data["cells"]
        viscosity = data["viscosity"]
        temperature = data["temperature"]
        grid_viscosity = data["grid_viscosity"]
        vx = data["vx"]
        vy = data["vy"]

        ccrusts = data["ccrust"]
        serps = data["serp"]
        gabbros = data["gabbro"]+data["gabbro_init"]
        ocrusts = data["ocrust"]+data["ocrust_init"]
        sediment = data["sediment"]
        freefluid = data["freefluid"]

        comp_matrix = np.stack([serps, gabbros, ocrusts, sediment], axis=1)
        i_comp = np.argmax(comp_matrix, axis=1)
        mask_sum = np.sum(comp_matrix, axis=1) < 0.2
        mask_gabbros = i_comp == 1
        mask_ocrusts = i_comp == 2
        mask_sediment = i_comp == 3
        # gabbros[mask_gabbros] = 1.0    
        ocrusts[mask_ocrusts] = 1.0    
        # sediment[mask_sediment] = 1.0
        gabbros[(mask_sediment | mask_ocrusts) | mask_sum] = 0.0
        ocrusts[(mask_sediment | mask_gabbros) | mask_sum] = 0.0
        sediment[(mask_ocrusts | mask_gabbros) | mask_sum] = 0.0



        triangulation, triangle_values = plot_quads_with_tricontourf(points, cells, viscosity)

        #ax[idx,0].set_axis_off()
        ax[idx,col].set_axis_off()
        ax_comp = fig.add_subplot(sub_gs[2*idx+col][1])



        triangulation.set_mask(np.mean(gabbros[triangulation.triangles]+
                                        ccrusts[triangulation.triangles]+
                                        ocrusts[triangulation.triangles]+
                                        sediment[triangulation.triangles],
                                        axis=1)<0.01)
        
        ax_comp.fill_between(x=[trench_x/1e3,trench_x/1e3+250], y1=-8, y2=8, color='white',  interpolate=True, alpha=1,zorder=1)

        ax_comp.tricontourf(triangulation,gabbros,levels=[0.5,1.1],cmap="Blues",zorder=2)
        ax_comp.tricontourf(triangulation,ocrusts,levels=[0.4,1.1],cmap="Oranges",zorder=2)
        ax_comp.tricontourf(triangulation,ccrusts,levels=[0.1,1.1],cmap="Greys",zorder=2)
        ax_comp.tricontourf(triangulation,sediment,levels=[0.4,1.1],cmap=single_color_cmap,zorder=2)
        ax_comp.tricontourf(triangulation, serps,levels=[0.01,1.1],colors=["black",],zorder=2)
        ax_comp.tricontour(triangulation, freefluid,levels=[1e-4,1.1],colors=["blue",],alpha=0.6,zorder=2)

        triangulation.set_mask(None)
        CB = ax_comp.tricontour(triangulation, temperature-273,levels=[400,800,1200],colors=[[1,1,1,0.8],],zorder=2)


        elapsed1 = timemodule.time() - t1
        print("first elapsed is: "+ str(elapsed1))

        mask = temperature>373
        n_sample = 400

        vel_vects = ax_comp.quiver(points[:,0][mask][::n_sample], points[:,1][mask][::n_sample], 
                        vx[mask][::n_sample]*100, vy[mask][::n_sample]*100,
                        color='black',scale=75,width=0.003,zorder=2) # scale=150, width=0.0015
        ax_comp.quiverkey(vel_vects, 0.23, 0.05, 5, '5 cm/yr', labelpos='W',fontproperties={'size': '10'},color='black',labelcolor='black')

        ax_comp.set_facecolor((154/255,185/255,115/255,0.5))
        font = {'fontname': 'Times New Roman',
        'weight': 'normal',
        'size': 13,
        }  

        ax_comp.set_ylim([200,-2])
        ax_comp.set_yticks([0,50,100,150,200])
        ax_comp.set_xlim([trench_x/1e3,trench_x/1e3+250])
        ax_comp.spines[['top']].set_visible(False)

        ax_s = fig.add_subplot(sub_gs[2*idx+col][0])
        ax_s.set_xlim([2800,3650])
        ax_s.set_ylim([750,-10]) 

        triangulation.set_mask((((np.max(data["points"][:,0][triangulation.triangles],
                                        axis=1)/1e3>3650) & (np.mean(data["points"][:,1][triangulation.triangles],
                                        axis=1)/1e3>1250))) | (np.min(data["points"][:,0][triangulation.triangles],
                                        axis=1)/1e3<2800))
        
        contour2 = ax_s.tricontourf(triangulation, temperature-273,levels=[0,1300],cmap="Greys",alpha=0.5)
        contour2.set_clip_on(False)
        triangulation.set_mask(None)
        contour = ax_s.tricontourf(triangulation, np.log10(viscosity),levels=100,cmap='viridis_r',antialiased=True)
        #contour2 = ax_s.tricontour(triangulation, np.log10(viscosity),levels=100,cmap='viridis_r',linewidths=1)


        ax_s.set_aspect('equal', adjustable='box')
        #cb = plt.colorbar(contour, ax=ax, label="Viscosity",ticks=[19, 20, 21,22,23]) 
        font = {'fontname': 'Times New Roman',
        'weight': 'normal',
        'size': 13,
        }  

        i = (idx)//2
        j = idx%2
        if j==1:
            ax_s.set_xlabel("x [km]",fontdict=font)
        #ax_s.set_ylabel("Depth [km]",fontdict=font)
        if j==1:
            ax_comp.set_xlabel("x [km]",fontdict=font)
        if col==0:
            ax_s.set_ylabel("Depth [km]",fontdict=font)
        ax_s.annotate(''.join(['t = ',str("%.1f" % (time_dim)),' Myr']), xy=(0.025,0.14), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=13,color='white')
        ax_s.spines[['top']].set_visible(False)
        # Add a rectangle (manual bbox)
        box_width = 250  # Desired width of the bbox
        box_height = 200  # Desired height of the bbox
        trench_y = -8.0
        box_height = 200-trench_y  # Desired height of the bbox
        rect = patches.Rectangle(
            (trench_x/1e3, trench_y ),  # Bottom-left corner
            box_width,                         # Width
            box_height,                        # Height
            linewidth=1.75,
            edgecolor=[1,1,1,0.75],
            facecolor="none",
        )
        ax_s.add_patch(rect)
        rect2 = patches.Rectangle(
            (trench_x/1e3, trench_y ),  # Bottom-left corner
            box_width,                         # Width
            box_height,                        # Height
            linewidth=1.1,
            edgecolor=[0,0,0,1],
            facecolor="none",
        )
        ax_s.add_patch(rect2)

plot_col(fig,ax,sub_gs,files1,0,ids1)
plot_col(fig,ax,sub_gs,files2,1,ids2)

def get_serp_vc(files,row):
    serps = np.zeros(len(files))
    times = np.zeros_like(serps)
    for idx,file in enumerate(files):
        if row==0:
            stats_file=''.join([os.getcwd(),folder1,'/statistics'])
        elif row==1:
            stats_file=''.join([os.getcwd(),folder2,'/statistics'])
        elif row==-1:
            stats_file=''.join([os.getcwd(),folder_baseline,'/statistics'])

        f=open(stats_file)
        lines=f.readlines()
        num_header_lines = len(list(filter(lambda line: line.startswith("#"),lines)))

        # num header lines in stats_files (for getting the dimensional time)
        idx_time = ''.join(c for c in file.split('/')[-1] if c.isdigit())
        stats_line_num = num_header_lines + (int(idx_time[1:]) )

        line=lines[stats_line_num]
        time_dim=float(line.split()[1])/1.e6

        time = idx*100
        if row==0:
            csv_filename=''.join([csvs_loc,folder1,'/full.',str(time),'.gzip'])
        elif row==1:
            csv_filename=''.join([csvs_loc,folder2,'/full.',str(time),'.gzip'])
        elif row==-1:
            csv_filename=''.join([csvs_loc,folder_baseline,'/full.',str(time),'.gzip'])
        data = np.load(file)
        serps[idx] = data["serp_total"]
        times[idx] = time_dim
    return serps,times


file_pattern2 = os.getcwd()+folder2+"/"+"solution/solution-0??00.npz"
files2_full = sorted(glob.glob(file_pattern2))


file_pattern_baseline = os.getcwd()+folder_baseline+"/"+"solution/solution-0??00.npz"
files_baseline_full = sorted(glob.glob(file_pattern_baseline))
    
serp1,times1 = get_serp_vc(files1_full,0)
serp2,times2 = get_serp_vc(files2_full,1)
serp_full,times_full = get_serp_vc(files_baseline_full,-1)

run_folder1 = "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
kin_files = [folder1[1:]+".txt",folder_baseline[1:]+".txt",folder2[1:]+".txt"]

legends = ["pelite 6 wt%","pelite 4 wt%", "pelite 2 wt%","pelite 0 wt%"]
# kin_files = ["dd200_orig.txt","dd200.txt"]
# legends = ["no lookup\nMDD=200 km","MDD=200 km"]
colors = ["navy","skyblue","green","coral"]#["green","grey","navy"]#


gs = ax[2, 0].get_gridspec()
# remove the underlying Axes
for axi in ax[-1, :]:
    axi.remove()
ax_serp1_base = fig.add_subplot(gs[-1, :])
ax_serp1_base.set_axis_off()
gs_new = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax_serp1_base.get_subplotspec(), wspace=0.0)
ax_serp1 = fig.add_subplot(gs_new[1])
plt.subplots_adjust(hspace=0.3)

#ax_serp1.set_aspect('equal', adjustable='box')

#ax_serp1 = fig.add_subplot(sub_gs[2][0])
ax[2,0].set_axis_off()

ax_serp1_vc = ax_serp1.twinx()
#ax_serp1_vc.set_aspect('equal', adjustable='box')
color = "tab:grey"

ax_serp1.plot(times1,serp1,color=colors[0],linewidth=2.0)
ax_serp1.plot(times_full,serp_full,color=colors[1],linewidth=2.0)
ax_serp1.plot(times2,serp2,color="green",linewidth=2.0)

font = {'fontname': 'Times New Roman',
'weight': 'normal',
'size': 15,
} 

ax_serp1_vc.set_ylabel("vc [cm/yr]",color=color,fontdict=font)
ax_serp1.set_ylabel("serpentinite [km$^{3}$/km]",fontdict=font)
ax_serp1.set_xlabel("time [Myr]",fontdict=font)
#ax_serp2 = fig.add_subplot(sub_gs[5][0])
#ax_serp1.set_aspect('equal', adjustable='datalim')
ax[2,1].set_axis_off()

# ax_serp2_vc = ax_serp2.twinx()
for kin_file,color,legend in zip(kin_files,colors,legends):
    f = np.loadtxt(os.getcwd()+"/kinematics/"+kin_file)
    t = f[:,1]
    vsp = f[:,2]
    vt = f[:,3]
    vc = f[:,4]
    ax_serp1_vc.plot(t,vc,color=color,linestyle="dashed",linewidth=2.0)
    #ax_serp2_vc.plot(t,vc,color=color,linestyle="dashed")
#ax_serp1_vc.tick_params(axis="y",labelcolor=color)

ax_serp1_vc.set_xlim([0,50])

# ax_serp2.plot(times2,serp2,color="navy",linewidth=1.75)
# ax_serp2.plot(times_full,serp_full,color="grey")
# ax_serp2.plot(times1,serp1,color="green")

output_file = files2[0].replace(".npz", "_fig2.png")
plt.savefig(output_file,dpi=500)
print("saved: "+output_file)