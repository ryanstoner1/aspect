import os
import glob
import pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec
from main_model4 import get_serp_vc
from main_model2 import plot_quads_with_tricontourf

plt.rcParams['font.family']="Times New Roman"
plt.rcParams['font.size']=12
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['axes.linewidth'] = 2.0

# 0. load data
# Path to the .npz files1

# folder1 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_smu0_02_cmu0_04_deserp_erase_res5_5_run35"
# ids1 = ["01500"]
# files1 = [os.getcwd()+folder1+"/solution/solution-"+id1+".npz" for id1 in ids1]

# folder2 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
# ids2 = ["01500"]
# files2 = [os.getcwd()+folder2+"/solution/solution-"+id2+".npz" for id2 in ids2]

# folder3 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel0_smu0_02_cmu0_04_deserp_erase_res5_5_run20"
# ids3 = ["01500"]
# files3 = [os.getcwd()+folder3+"/solution/solution-"+id3+".npz" for id3 in ids3]

# folder4 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_0km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run37"
# ids4 = ["01500"]
# files4 = [os.getcwd()+folder4+"/solution/solution-"+id4+".npz" for id4 in ids4]


folder1 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_01_cmu0_04_deserp_erase_res5_5_run22"
ids1 = ["01500"]
files1 = [os.getcwd()+folder1+"/solution/solution-"+id1+".npz" for id1 in ids1]

folder2 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
ids2 = ["01500"]
files2 = [os.getcwd()+folder2+"/solution/solution-"+id2+".npz" for id2 in ids2]

folder3 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_03_cmu0_04_deserp_erase_res5_5_run23"
ids3 = ["01500"]
files3 = [os.getcwd()+folder3+"/solution/solution-"+id3+".npz" for id3 in ids3]

folder4 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_04_cmu0_04_deserp_erase_res5_5_run24"
ids4 = ["01500"]
files4 = [os.getcwd()+folder4+"/solution/solution-"+id4+".npz" for id4 in ids4]

# 0. load data
# Path to the .npz files
folders = [folder1,folder2,folder3,folder4]
#file_pattern = folder+"/"+"solution/solution-0[1,2,3,4,5,6]000.npz"
files = files1+files2+files3+files4
#files[0]= files[0][:-7]+"1"+files[0][-6:]
timings = []
fig, ax = plt.subplots(figsize=(15, 6),nrows=1,ncols=3)

# Sub-grid for gs[0, 0] (top-left corner)
sub_gs = [GridSpecFromSubplotSpec(1, 2, subplot_spec=axi.get_subplotspec(), wspace=0.2) for axi in ax.flatten()]


ymax = 1450e3
step = 1000

time = 1
strnum = str(1000)
input_file_serp=str(pathlib.Path().resolve()) + "/pseudosections/Serpentinite/serp_sat_niu_aspect_H2O.txt"

# Read the input file
data_serp = np.genfromtxt(input_file_serp, skip_header=0, usecols=(0, 1, 2))

# Separate the columns
pressure_h2o_serp = (data_serp[:, 1]).reshape((250,250))
temperature_h2o_serp = (data_serp[:, 0]).reshape((250,250))
xh2o_serp = (data_serp[:, 2]).reshape((250,250))

input_file_basalt=str(pathlib.Path().resolve()) + "/pseudosections/MORB/morb_green_aspect_H2O.txt"

# Read the input file
data_basalt = np.genfromtxt(input_file_basalt, skip_header=0, usecols=(0, 1, 2))
xh2o_basalt = (data_basalt[:, 2]).reshape((250,250))
pressure_h2o_basalt = (data_basalt[:, 1]).reshape((250,250))
temperature_h2o_basalt = (data_basalt[:, 0]).reshape((250,250))

# Read the input file
input_file_sed=str(pathlib.Path().resolve()) + "/pseudosections/Pelagic_sediment/pelagic_bound_h2o.tab"
data_sed = np.genfromtxt(input_file_sed, skip_header=0, usecols=(0, 1, 2))
xh2o_sed = (data_sed[:, 2]).reshape((250,250))
pressure_h2o_sed = (data_sed[:, 1]).reshape((250,250))
temperature_h2o_sed = (data_sed[:, 0]).reshape((250,250))


temperature_h2o_serp = np.vstack((temperature_h2o_serp[0,:],temperature_h2o_serp))
pressure_h2o_serp = np.vstack((0.0*pressure_h2o_serp[0,:],pressure_h2o_serp))
xh2o_serp = np.vstack((xh2o_serp[0,:],xh2o_serp))
xh2o_serp[0,:] = np.hstack((xh2o_serp[0,3:],0.0*xh2o_serp[0,:3])) # fudge to make the plot not weird

pressure_h2o_basalt = np.vstack((0.5*pressure_h2o_basalt[0,:],pressure_h2o_basalt))
temperature_h2o_basalt = np.vstack((temperature_h2o_basalt[0,:],temperature_h2o_basalt))
pressure_h2o_basalt = np.vstack((0.0*pressure_h2o_basalt[0,:],pressure_h2o_basalt))
temperature_h2o_basalt = np.vstack((temperature_h2o_basalt[0,:],temperature_h2o_basalt))

xh2o_basalt = np.vstack((xh2o_basalt[0,:],xh2o_basalt))
xh2o_basalt = np.vstack((xh2o_basalt[0,:],xh2o_basalt))

pressure_h2o_sed = np.vstack((0.0*pressure_h2o_sed[0,:],pressure_h2o_sed))
temperature_h2o_sed = np.vstack((temperature_h2o_sed[0,:],temperature_h2o_sed))
xh2o_sed = np.vstack((xh2o_sed[0,:],xh2o_sed))

fig4 = plt.figure(figsize=(15, 4))
if time==0:
    gs4 = GridSpec(1, 3,width_ratios=[0.67,1,1])
else:
    gs4 = GridSpec(1, 3)

axL1_boundwater = fig4.add_subplot(gs4[0])
# axR2 = fig4.add_subplot(gs4[0, 0])
axM1 = fig4.add_subplot(gs4[1])
axR1 = fig4.add_subplot(gs4[2])      

#colors = ["#FA2338","skyblue","#FAF323","#7A4E53"]
colors = ["navy","skyblue","green","coral"]
for fidx,folder in enumerate(folders):
    stats_file = ''.join([os.getcwd(),'/',folder,'/statistics'])
    model_output_dt  = 1000

    f=open(stats_file)
    lines=f.readlines()
    num_header_lines = len(list(filter(lambda line: line.startswith("#"),lines)))

    plot_loc = ''.join(
        [os.getcwd()+'/plots', str(folder)])
    if not os.path.exists(plot_loc):
        os.mkdir(plot_loc)

    fnums = [3100]

    Psurfload = len(fnums)*[np.nan]
    Tsurfload = len(fnums)*[np.nan]

    P_1km = len(fnums)*[np.nan]
    T_1km = len(fnums)*[np.nan]

    for idx_fnum,fnum in enumerate(fnums):
        strnum = str(fnum)
        strnum_1km = strnum
        if len(strnum)<4:
            strnum = "0"+strnum
        filename = "/pt_"+strnum+".txt"
        print("loading: "+filename+"\n")
        PTdata = np.loadtxt(plot_loc+filename,skiprows=1)
        
        Psurfload[idx_fnum] = PTdata[:,1]/1e9
        Tsurfload[idx_fnum] = PTdata[:,2]

        filename_basPT = "/"+strnum_1km+".1km.txt"
        print("loading: "+filename_basPT+"\n")
        PTdata_bas = np.loadtxt(plot_loc+filename_basPT,skiprows=1)
        
        T_1km[idx_fnum] = PTdata_bas[:,4]
        P_1km[idx_fnum] = PTdata_bas[:,3]


    if time ==0:

        # axR3 = fig4.add_subplot(gs4[2, 1])
        # axR4 = fig4.add_subplot(gs4[3, 1])

        filename_track = "/pt_"+strnum+".txt"
        print("loading: "+filename_track+"\n")




        axL1_boundwater.contourf(temperature_h2o_basalt-273, pressure_h2o_basalt*100/1e6, np.abs(xh2o_basalt),
                                                    cmap="magma_r", levels=np.linspace(0, 6.0, 101), extend="both")
        axL1_boundwater.contour(temperature_h2o_serp-273,pressure_h2o_serp*100/1e6,xh2o_serp,levels=np.array([5]),colors="black",extend="both",linestyles='dotted')
        axM1.contour(temperature_h2o_serp-273,pressure_h2o_serp*100/1e6,xh2o_serp,levels=np.array([5]),colors="black",extend="both",linestyles='dotted')
    #else:
    #    if is_load_PT:
        for idx_rgba,(Tsurf,Psurf,T1,P1) in enumerate(zip(Tsurfload,Psurfload,T_1km,P_1km)):
            cmap = mpl.cm.get_cmap("BuPu_r")
            rgba = cmap(idx_rgba/len(Tsurfload))
            axL1_boundwater.plot(Tsurf, Psurf,color=colors[fidx])
            axL1_boundwater.plot(T1, P1,color=colors[fidx],linestyle='--')
            axM1.plot(Tsurf, Psurf,color=colors[fidx],linestyle="dashed",alpha=0.6)
            #axM1.plot(T1, P1,color=rgba)
        axL1_boundwater.set_xlim([0,np.max(temperature_h2o_basalt-273)])
        axL1_boundwater.set_ylim([np.min(pressure_h2o_basalt*100/1e6),np.max(pressure_h2o_basalt*100/1e6)])


        font = {'fontname': 'Times New Roman',
        'weight': 'normal',
        'size': 14} 
        axL1_boundwater.set_ylabel("pressure [GPa]",fontdict=font)
        axL1_boundwater.set_xlabel("temperature [$^{\circ}$C]",fontdict=font)
        axL1_boundwater.set_yticks([0,2,4,6,8])

        h2o_plot_boundwater = axM1.contourf(temperature_h2o_sed-273, pressure_h2o_sed*100/1e6, np.abs(xh2o_sed),
                                                    cmap="magma_r", levels=np.linspace(0, 6.0, 101), extend="both")
        axM1.set_xlabel("temperature [$^{\circ}$C]",fontdict=font)
        
        if fidx==0:
            cbar4 = plt.colorbar(h2o_plot_boundwater,ax=axM1, orientation='vertical',
                                ticks=[0, 1, 2, 3, 4, 5, 6],pad=0.18)
            cbar4.ax.tick_params(labelsize=14)
            cbar4.set_label("Bound $\mathregular{H_{2}O}$ [%]", size=15)

            axM1.set_xlim([0,np.max(temperature_h2o_basalt-273)])
            axM1.set_ylim([np.min(pressure_h2o_basalt*100/1e6),np.max(pressure_h2o_basalt*100/1e6)])
            axM1.set_yticks([0,2,4,6,8])
            axL1_y = axM1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'

            axL1_y.set_ylim((0,1e9*np.max(pressure_h2o_basalt)*100/1e6/(3300*9.8*1e3)))

            axM1_y = axL1_boundwater.twinx()  # instantiate a second axes that shares the same x-axis

            axL1_y.set_ylabel('depth [km]')  # we already handled the x-label with ax1

            axM1_y.tick_params(axis='y')
            axM1_y.set_ylim((0,1e9*np.max(pressure_h2o_basalt)*100/1e6/(3300*9.8*1e3)))
    elif (fidx==0.0):
        stats_file=''.join([os.getcwd(),folder,'/statistics'])

        f=open(stats_file)
        lines=f.readlines()
        num_header_lines = len(list(filter(lambda line: line.startswith("#"),lines)))

        # num header lines in stats_files (for getting the dimensional time)
        idx_time = ''.join(c for c in files[fidx].split('/')[-1] if c.isdigit())
        stats_line_num = num_header_lines + (int(idx_time[1:]) )

        line=lines[stats_line_num]
        time_dim=float(line.split()[1])/1.e6

        # Load the mesh and field data
        data = np.load(files[fidx])
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
        serps_stable = data["serp_total_stable"]
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
        #ax[row,col].set_axis_off()
        ax_comp = axL1_boundwater#fig.add_subplot(sub_gs[2*row+col][1])

        bins_serp = data["bins_serp"]
        bins_y = data["bins_y"]

        
        #ax_comp.fill_between(x=[trench_x/1e3-100,trench_x/1e3+330], y1=-8, y2=8, color='white',  interpolate=True, alpha=1,zorder=1)
        contour = ax_comp.tricontourf(triangulation, np.log10(viscosity),levels=100,cmap='viridis_r')
        # ax_comp.tricontourf(triangulation,gabbros,levels=[0.5,1.1],cmap="Blues",zorder=2)
        # ax_comp.tricontourf(triangulation,ocrusts,levels=[0.4,1.1],cmap="Oranges",zorder=2)
        # ax_comp.tricontourf(triangulation,ccrusts,levels=[0.1,1.1],cmap="Greys",zorder=2)
        # #ax_comp.tricontourf(triangulation,sediment,levels=[0.4,1.1],cmap=single_color_cmap,zorder=2)
        #ax_comp.tricontour(triangulation, gabbros+ocrusts+sediment,levels=[0.35,1.1],colors=["green",],zorder=2)
        ax_comp.tricontour(triangulation, serps,levels=[0.01,1.1],colors=["white",],zorder=2)

        #ax_comp.tricontourf(triangulation, freefluid,levels=[1e-4,1.1],colors=["skyblue",],alpha=0.75,zorder=2)
        CB = ax_comp.tricontour(triangulation, temperature-273,levels=[400,800,1200],colors=[[1,1,1,0.8],],zorder=2)
        ax_comp.set_xlim([3000,3250])
        ax_comp.set_ylim([80,-10])
        ax_comp.annotate(''.join(['t = ',str("%.1f" % (time_dim)),' Myr']), xy=(0.02,0.05), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=14,color='white')
        font = {'fontname': 'Times New Roman',
            'weight': 'normal',
            'size': 14} 
        ax_comp.set_ylabel("Depth [km]",fontdict=font)
        ax_comp.set_xlabel("x [km]",fontdict=font)
    elif (fidx==3.0):
        stats_file=''.join([os.getcwd(),folder,'/statistics'])

        f=open(stats_file)
        lines=f.readlines()
        num_header_lines = len(list(filter(lambda line: line.startswith("#"),lines)))

        # num header lines in stats_files (for getting the dimensional time)
        idx_time = ''.join(c for c in files[fidx].split('/')[-1] if c.isdigit())
        stats_line_num = num_header_lines + (int(idx_time[1:]) )

        line=lines[stats_line_num]
        time_dim=float(line.split()[1])/1.e6        
        # Load the mesh and field data
        data = np.load(files[fidx])
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
        #ax[row,col].set_axis_off()
        ax_comp = axM1#fig.add_subplot(sub_gs[2*row+col][1])

        bins_serp = data["bins_serp"]
        bins_y = data["bins_y"]
        # divider = make_axes_locatable(ax_comp)
        # ax_serp_bin = divider.append_axes('right',size="20%",pad=0.0)
        # ax_serp_bin.plot(bins_serp/1e6,1450-(bins_y/1e3),color="black")
        # ax_serp_bin.set_xlim([0.0,4.0])
        # ax_serp_bin.set_ylim([200,-2])
        # ax_serp_bin.set_xticks([0.0,4.0])
        # ax_serp_bin.spines[['top','right']].set_visible(False)

        # ax_serp_bin.set_yticklabels([])
        # # GREEN
        # ax_serp_bin.set_yticks([])
        # font = {'fontname': 'Times New Roman',
        #     'weight': 'normal',
        #     'size': 14} 
        # fontserp = {'fontname': 'Times New Roman',
        #     'weight': 'normal',
        #     'size': 13} 
        # if row==1:
        #     ax_serp_bin.set_xlabel("serpentinite\n $\mathregular{10^{6}[kg\hspace{0.15} m^{3}/m^{2}]}$",fontdict=fontserp)

        # triangulation.set_mask(np.mean(gabbros[triangulation.triangles]+
        #                                 ccrusts[triangulation.triangles]+
        #                                 ocrusts[triangulation.triangles]+
        #                                 sediment[triangulation.triangles],
        #                                 axis=1)<0.01)
        
        #ax_comp.fill_between(x=[trench_x/1e3-100,trench_x/1e3+330], y1=-8, y2=8, color='white',  interpolate=True, alpha=1,zorder=1)
        contour = ax_comp.tricontourf(triangulation, np.log10(viscosity),levels=100,cmap='viridis_r',antialiased=True)
        #ax_comp.tricontour(triangulation, gabbros+ocrusts+sediment,levels=[0.5,1.1],colors=["green",],zorder=2)
        # ax_comp.tricontourf(triangulation,gabbros,levels=[0.5,1.1],cmap="Blues",zorder=2)
        # ax_comp.tricontourf(triangulation,ocrusts,levels=[0.4,1.1],cmap="Oranges",zorder=2)
        # ax_comp.tricontourf(triangulation,ccrusts,levels=[0.1,1.1],cmap="Greys",zorder=2)
        # #ax_comp.tricontourf(triangulation,sediment,levels=[0.4,1.1],cmap=single_color_cmap,zorder=2)
        ax_comp.tricontour(triangulation, serps,levels=[0.01,1.1],colors=["white",],zorder=2)
        CB = ax_comp.tricontour(triangulation, temperature-273,levels=[400,800,1200],colors=[[1,1,1,0.8],],zorder=2)
        # ax_comp.tricontourf(triangulation, freefluid,levels=[5e-5,1.1],colors=["blue",],alpha=0.45,zorder=2)
        ax_comp.set_xlim([3030,3250])
        ax_comp.set_ylim([80,-5])
        ax_comp.annotate(''.join(['t = ',str("%.1f" % (time_dim)),' Myr']), xy=(0.02,0.05), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=14,color='white')

        ax_comp.set_xlabel("x [km]",fontdict=font)

    # retrieve H2O and kinematic variables
    file_pattern_hires = folder+"/"+"solution/solution-0??00.npz"
    files_hires = sorted(glob.glob(os.getcwd()+file_pattern_hires))
    serp_full,serps_stable,times_full, boundwaters = get_serp_vc(files_hires,-1,"/"+folder)
    kin_files = [folder+".txt"]

    legends = ["pelite 4 wt%"]
    #["green","grey","navy"]#
    ax_serp1 = axR1#fig.add_subplot(gs_new[1])
    
    if fidx==0:
        plt.subplots_adjust(hspace=0.3)
        ax_serp1_vc = ax_serp1.twinx()
        #ax_serp1_vc.set_aspect('equal', adjustable='box')
        color = "tab:grey"

        
        
        


        font = {'fontname': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
        } 

        ax_serp1_vc.set_ylabel("vc [cm/yr]",color="black",fontdict=font)
        ax_serp1.set_ylabel("serpentinite [km$\mathregular{^{3}}$/km]",fontdict=font)
        ax_serp1.set_xlabel("time [Myr]",fontdict=font)
        #ax_serp2 = fig.add_subplot(sub_gs[5][0])
        #ax_serp1.set_aspect('equal', adjustable='datalim')

    ax_serp1.plot(times_full[1:],serp_full[1:],color=colors[fidx],linewidth=2.0)
    ax_serp1.plot(times_full[1:],serps_stable[1:],color=colors[fidx],linewidth=2.0,linestyle="--")

    # ax_serp2_vc = ax_serp2.twinx()
    for kin_file in kin_files:
        f = np.loadtxt(os.getcwd()+"/kinematics/"+kin_file)
        t = f[:,1]
        vsp = f[:,2]
        vt = f[:,3]
        vc = f[:,4]
        ax_serp1_vc.plot(t[1:],vc[1:],color=colors[fidx],linewidth=2.0,linestyle="--")
        #ax_serp2_vc.plot(t,vc,color=color,linestyle="dashed")
    #ax_serp1_vc.tick_params(axis="y",labelcolor=color)

    ax_serp1_vc.set_xlim([0,50])

plt.tight_layout()

fig4.savefig(plot_loc+"/boundwater7_visc.jpg",dpi=250)
print(plot_loc+"/boundwater7_.svg")

