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
from scipy.interpolate import griddata
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

# Define a single-color colormap (e.g., "blue")
single_color_cmap = LinearSegmentedColormap.from_list("SingleColor", ["#D2D344", "#D2D344"])

is_plot_outline = False # or just viscosity 

# ASPECT output 
csvs_loc = os.getcwd()+'/csv_outputs/'

# setup plotting
mpl.use('agg')
plt.rcParams['font.family']="Calibri"
plt.rcParams['font.size']=12
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['axes.linewidth'] = 2.0

def main():
    # 0. load data
    # Path to the .npz files1
    folder1 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_smu0_02_cmu0_04_deserp_erase_res5_5_run35"
    ids1 = ["03100"]
    files1 = [os.getcwd()+folder1+"/solution/solution-"+id1+".npz" for id1 in ids1]

    folder2 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
    ids2 = ["03100"]
    files2 = [os.getcwd()+folder2+"/solution/solution-"+id2+".npz" for id2 in ids2]

    folder3 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel0_smu0_02_cmu0_04_deserp_erase_res5_5_run20"
    ids3 = ["03100"]
    files3 = [os.getcwd()+folder3+"/solution/solution-"+id3+".npz" for id3 in ids3]

    folder4 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_0km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run37"
    ids4 = ["03100"]
    files4 = [os.getcwd()+folder4+"/solution/solution-"+id4+".npz" for id4 in ids4]


    fig, ax = plt.subplots(figsize=(18, 7),nrows=2,ncols=2)

    plot_loc = ''.join(
        [os.getcwd()+'/plots', str(folder1)])
    if not os.path.exists(plot_loc):
        os.mkdir(plot_loc)
    # Sub-grid for gs[0, 0] (top-left corner)
    sub_gs = [GridSpecFromSubplotSpec(1, 2, subplot_spec=axi.get_subplotspec(), wspace=0.1) for axi in ax.flatten()]

    plot_col(fig,ax,sub_gs,files1,0,0,ids1,folder1)
    plot_col(fig,ax,sub_gs,files2,0,1,ids2,folder2)
    plot_col(fig,ax,sub_gs,files3,1,0,ids3,folder3)
    plot_col(fig,ax,sub_gs,files4,1,1,ids4,folder4)



    plt.subplots_adjust(bottom=0.12,top=0.95,left=0.05,right=0.95,wspace=0.05,hspace = 0.25) # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    output_file = files1[0].replace(".npz", "_fig5_outline.png")
    fout_name = plot_loc+"/"+output_file.split('/')[-1]    
    plt.savefig(fout_name,dpi=500)
    print("saved at: "+fout_name)

def plot_col(fig,ax,sub_gs,files,row,col,ids,folder):

    for idx,file in enumerate(files):

        stats_file=''.join([os.getcwd(),folder,'/statistics'])

        f=open(stats_file)
        lines=f.readlines()
        num_header_lines = len(list(filter(lambda line: line.startswith("#"),lines)))

        # num header lines in stats_files (for getting the dimensional time)
        idx_time = ''.join(c for c in file.split('/')[-1] if c.isdigit())
        stats_line_num = num_header_lines + (int(idx_time[1:]) )

        line=lines[stats_line_num]
        time_dim=float(line.split()[1])/1.e6

        ymax = 1450e3
        step = 1000

        t1 = timemodule.time()

        time = idx*step

        csv_filename=''.join([csvs_loc,folder,'/full.',str(int(ids[idx])),'.gzip'])

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
        if "p" in header_terms:
            p_col = header_terms.index("p")
        if "T" in header_terms:
            T_col = header_terms.index("T")
                
    
        # extract mid-plate profile
        plate_prof_loc = ymax - 0.5e3                           # 20 km depth
        plate_prof = model_data[model_data[:,y_col] < (plate_prof_loc+4.e3)] 
        plate_prof = plate_prof[plate_prof[:,y_col] > (plate_prof_loc-4.e3)] 
        plate_prof = plate_prof[plate_prof[:,x_col].argsort()] # sort by x
        p_prof_sort = plate_prof[:,x_col].argsort()

        p_prof = model_data[:,p_col][p_prof_sort]
        T_prof = model_data[:,T_col][p_prof_sort]
        # if col==0:
        #     ax[2,0].scatter(p_prof,T_prof-273,linewidth=2,color="navy")
        # else:
        #     ax[2,0].scatter(p_prof,T_prof-273,linewidth=2,color="green")

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
        data = np.load(files[idx])
        points = data["points"][:,0:2]/1000
        points[:,1] = (ymax/1e3)-points[:,1]
        cells = data["cells"]
        viscosity = data["viscosity"]
        temperature = data["temperature"]
        pressure = data["pressure"]
        #grid_viscosity = data["grid_viscosity"]
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
        ax[row,col].set_axis_off()
        ax_comp = fig.add_subplot(sub_gs[2*row+col][1])

        bins_serp = data["bins_serp"]
        bins_y = data["bins_y"]
        divider = make_axes_locatable(ax_comp)
        ax_serp_bin = divider.append_axes('right',size="20%",pad=0.0)
        ax_serp_bin.plot(bins_serp/1e6,1450-(bins_y/1e3),color="black")
        ax_serp_bin.set_xlim([0.0,4.0])
        ax_serp_bin.set_ylim([200,-2])
        ax_serp_bin.set_xticks([0.0,4.0])
        ax_serp_bin.spines[['top','right']].set_visible(False)

        ax_serp_bin.set_yticklabels([])
        # GREEN
        ax_serp_bin.set_yticks([])
        font = {'fontname': 'Calibri',
            'weight': 'normal',
            'size': 14} 
        fontserp = {'fontname': 'Calibri',
            'weight': 'normal',
            'size': 13} 
        if row==1:
            ax_serp_bin.set_xlabel("Serpentinite\n $\mathregular{10^{6}[kg\hspace{0.15} m^{3}/m^{2}]}$",fontdict=fontserp)

        triangulation.set_mask(np.mean(gabbros[triangulation.triangles]+
                                        ccrusts[triangulation.triangles]+
                                        ocrusts[triangulation.triangles]+
                                        sediment[triangulation.triangles],
                                        axis=1)<0.01)
        
        ax_comp.fill_between(x=[trench_x/1e3-100,trench_x/1e3+330], y1=-8, y2=8, color='white',  interpolate=True, alpha=1,zorder=1)

        ax_comp.tricontourf(triangulation,gabbros,levels=[0.5,1.1],cmap="Blues",zorder=2)
        ax_comp.tricontourf(triangulation,ocrusts,levels=[0.4,1.1],cmap="Oranges",zorder=2)
        ax_comp.tricontourf(triangulation,ccrusts,levels=[0.1,1.1],cmap="Greys",zorder=2)
        ax_comp.tricontourf(triangulation,sediment,levels=[0.4,1.1],cmap=single_color_cmap,zorder=2)
        ax_comp.tricontourf(triangulation, serps,levels=[0.121*0.1,1.1],colors=["black",],zorder=2)
        triangulation.set_mask(None)
        ax_comp.tricontourf(triangulation, freefluid,levels=[5e-5,1.1],colors=["blue",],alpha=0.35,zorder=2)
        

        
        CB = ax_comp.tricontour(triangulation, temperature-273,levels=[400,800,1200],colors=[[1,1,1,0.8],],zorder=2)

        # Katz, Spiegelman, 2003 solidus
        A1 = 1085.7
        A2 = 132.9
        A3 = -5.1
        Tsolidus = A1+A2*np.abs(pressure/1e9)+A3*np.abs(pressure/1e9)**2
        Tsolidus_in = Tsolidus<(temperature-273)
        ax_comp.tricontourf(triangulation, Tsolidus_in,levels=[0.5,1.1],colors=[[0.8,0.1,0.1,0.18],],zorder=1)

        elapsed1 = timemodule.time() - t1
        print("first elapsed is: "+ str(elapsed1))

        mask = points[:,1]>10
        n_sample = 400
        nx_quiv = 80
        ny_quiv = 60

        x_mask = points[:,0][mask]
        y_mask = points[:,1][mask]
        xpoints = np.linspace(np.min(x_mask),np.max(x_mask),num=nx_quiv)
        ypoints = np.linspace(np.min(y_mask),np.max(y_mask),num=ny_quiv)
        x_grid,y_grid = np.meshgrid(xpoints,ypoints)
        vx_grid = griddata((x_mask,y_mask),vx[mask],(x_grid,y_grid),method="linear")
        vy_grid = griddata((x_mask,y_mask),vy[mask],(x_grid,y_grid),method="linear")

        vel_vects = ax_comp.quiver(x_grid, y_grid, 
                        vx_grid*100, vy_grid*100,
                        color='black',scale=75,width=0.003,zorder=2) # scale=150, width=0.0015
        # vel_vects = ax_comp.quiver(points[:,0][mask][::n_sample], points[:,1][mask][::n_sample], 
        #                 vx[mask][::n_sample]*100, vy[mask][::n_sample]*100,
        #                 color='black',scale=75,width=0.003,zorder=2) # scale=150, width=0.0015
        if row==0:
            ax_comp.quiverkey(vel_vects, 0.20, 0.05, 5, '5 cm/yr', labelpos='W',fontproperties={'size': '10'},color='black',labelcolor='black')

        ax_comp.set_facecolor((154/255,185/255,115/255,0.5))
        font = {'fontname': 'Calibri',
        'weight': 'normal',
        'size': 13,
        }  

        ax_comp.set_ylim([200,-2])
        ax_comp.set_yticks([0,50,100,150,200])
        ax_comp.set_xlim([trench_x/1e3,trench_x/1e3+330])
        ax_comp.spines[['top']].set_visible(False)
        ax_comp.set_xticks(ax_comp.get_xticks()[:-2])

        ax_s = fig.add_subplot(sub_gs[2*row+col][0])
        ax_s.set_xlim([2800,3650])
        ax_s.set_ylim([700,-10]) 

        triangulation.set_mask((((np.max(data["points"][:,0][triangulation.triangles],
                                        axis=1)/1e3>3650) & (np.mean(data["points"][:,1][triangulation.triangles],
                                        axis=1)/1e3>1250))) | (np.min(data["points"][:,0][triangulation.triangles],
                                        axis=1)/1e3<2800))

        if is_plot_outline:
            ax_s.set_ylim([750,-10]) 
            ax_s.axhspan(900,660,color="grey")
            contour = ax_s.tricontourf(triangulation, viscosity,levels=[1e22,3e23],cmap="Blues",antialiased=False)
            #contour = ax_s.tricontourf(triangulation, viscosity*((serps+ocrusts+gabbros+sediment)>0.75),levels=[2e18,1e21],colors="black",antialiased=False)
            ax_s.spines[['left']].set_visible(False)
            ax_s.spines[['right']].set_visible(False)
            ax_s.spines[['bottom']].set_visible(False)
            ax_s.spines[['top']].set_visible(False)
        else:        
            contour2 = ax_s.tricontourf(triangulation, temperature-273,levels=[0,1300],cmap="Greys",alpha=0.5)
            contour2.set_clip_on(False)
            triangulation.set_mask(None)
            viscosity_kwarg = np.linspace(18.39,23.4,num=100)
            contour = ax_s.tricontourf(triangulation, np.log10(viscosity),levels=viscosity_kwarg,cmap='viridis_r',antialiased=True)
            #contour2 = ax_s.tricontour(triangulation, np.log10(viscosity),levels=100,cmap='viridis_r',linewidths=1)


            ax_s.set_aspect('equal', adjustable='box')
            #cb = plt.colorbar(contour, ax=ax, label="Viscosity",ticks=[19, 20, 21,22,23]) 
            font = {'fontname': 'Calibri',
            'weight': 'normal',
            'size': 14,
            }  

            i = (idx)//2
            j = idx%2
            if row==1:
                ax_s.set_xlabel("x [km]",fontdict=font)
            #ax_s.set_ylabel("Depth [km]",fontdict=font)
            if row==1:
                ax_comp.set_xlabel("x [km]",fontdict=font)
            if col==0:
                ax_s.set_ylabel("Depth [km]",fontdict=font)
            ax_s.annotate(''.join(['t = ',str("%.1f" % (time_dim)),' Myr']), xy=(0.025,0.09), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=13,color='white')
            ax_s.spines[['top']].set_visible(False)
            # Add a rectangle (manual bbox)
            box_width = 330  # Desired width of the bbox
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

            if (2*row+col)==0:
                title_name = "6.89 wt% $\mathregular{H_{2}O}$"
            elif (2*row+col)==1:
                title_name = "4 wt% $\mathregular{H_{2}O}$"
            elif (2*row+col)==2:
                title_name = "0 wt% $\mathregular{H_{2}O}$"
            elif (2*row+col)==3:
                title_name = "No sediment"
            ax_s.set_title(title_name,fontweight="bold",fontsize=16)

if __name__=="__main__":
    main()
