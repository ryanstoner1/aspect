import os
import glob
import pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec
from main_model4 import get_serp_vc

plt.rcParams['font.family']="Calibri"

plot_time_xaxis = True # for convergence and serp plots; otherwise distance
plot_high_friction = False
plot_high_res = False
# 0. load data
# Path to the .npz files1

if plot_high_friction:
    folder2 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
    ids2 = ["03500"]
    files2 = [os.getcwd()+folder2+"/solution/solution-"+id2+".npz" for id2 in ids2]

    folder5 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_06_cmu0_04_deserp_erase_res5_5_run38"
    ids5 = ["03500"]
    files5 = [os.getcwd()+folder5+"/solution/solution-"+id5+".npz" for id5 in ids5]
    folders = [folder2,folder5]
    #file_pattern = folder+"/"+"solution/solution-0[1,2,3,4,5,6]000.npz"
    files = files2+files5
    colors = ["skyblue","teal"]
elif plot_high_res:
    folder2 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
    ids2 = ["03500"]
    files2 = [os.getcwd()+folder2+"/solution/solution-"+id2+".npz" for id2 in ids2]

    folder5 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_smu0_02_cmu0_04_deserp_erase_res6_5_run36_10"
    ids5 = ["03500"]
    files5 = [os.getcwd()+folder5+"/solution/solution-"+id5+".npz" for id5 in ids5]
    folders = [folder2,folder5]
    #file_pattern = folder+"/"+"solution/solution-0[1,2,3,4,5,6]000.npz"
    files = files2+files5
    colors = ["skyblue","navy"]    
else:
    folder1 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_01_cmu0_04_deserp_erase_res5_5_run22"
    ids1 = ["03500"]
    files1 = [os.getcwd()+folder1+"/solution/solution-"+id1+".npz" for id1 in ids1]

    folder2 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
    ids2 = ["03500"]
    files2 = [os.getcwd()+folder2+"/solution/solution-"+id2+".npz" for id2 in ids2]

    folder3 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_03_cmu0_04_deserp_erase_res5_5_run23"
    ids3 = ["03500"]
    files3 = [os.getcwd()+folder3+"/solution/solution-"+id3+".npz" for id3 in ids3]

    folder4 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_04_cmu0_04_deserp_erase_res5_5_run24"
    ids4 = ["03500"]
    files4 = [os.getcwd()+folder4+"/solution/solution-"+id4+".npz" for id4 in ids4]
    folders = [folder1,folder2,folder3,folder4]
    files = files1+files2+files3+files4
    colors = ["navy","skyblue","green","coral"]#,"teal"]

# 0. load data
# Path to the .npz files

#files[0]= files[0][:-7]+"1"+files[0][-6:]
timings = []
fig, ax = plt.subplots(figsize=(15, 6),nrows=1,ncols=3)

# Sub-grid for gs[0, 0] (top-left corner)
sub_gs = [GridSpecFromSubplotSpec(1, 2, subplot_spec=axi.get_subplotspec(), wspace=0.2) for axi in ax.flatten()]


ymax = 1450e3
step = 1000

time = 0
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
gs4 = GridSpec(1, 1,width_ratios=[1])

gs4_sub = GridSpec(1, 2, width_ratios=[0.75,0.75],wspace=0.2)

axR1_boundwater = fig4.add_subplot(gs4[0])
# axR2 = fig4.add_subplot(gs4[0, 0])
axM1 = fig4.add_subplot(gs4_sub[1])
axL1 = fig4.add_subplot(gs4_sub[0])
gs4_sub.update(left=0.05,right=0.58,bottom=0.15,top=0.95)
gs4.update(left=0.65,right=0.99,bottom=0.15,top=0.95)

# axR1_boundwater = fig4.add_subplot(gs4[0])
# # axR2 = fig4.add_subplot(gs4[0, 0])
# axM1 = fig4.add_subplot(gs4[1])
# axL1 = fig4.add_subplot(gs4[2])      

#colors = ["#FA2338","skyblue","#FAF323","#7A4E53"]

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

    if plot_high_friction:
        if fidx==0:
            fnums = [2000,6000]
        elif fidx==1:
            fnums = [1700,4900]
    elif plot_high_res:
        if fidx==0:
            fnums = [2000,6000]
        elif fidx==1:
            fnums = [3400,9800]        
    else:
        if fidx==0:
            fnums = [2000,6000]
        elif fidx==1:
            fnums = []
        elif fidx==2:
            fnums = []
        elif fidx==3:
            fnums = [1800,5100]
        elif fidx==4:
            fnums = []

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





        axR1_boundwater.contour(temperature_h2o_serp-273,pressure_h2o_serp*100/1e6,xh2o_serp,levels=np.array([5]),colors="black",extend="both",linestyles='dotted')

    #else:
    #    if is_load_PT:

        axR1_boundwater.set_xlim([0,np.max(temperature_h2o_basalt-273)])
        axR1_boundwater.set_ylim([np.min(pressure_h2o_basalt*100/1e6),np.max(pressure_h2o_basalt*100/1e6)])


        font = {'fontname': 'Calibri',
        'weight': 'normal',
        'size': 14} 
        axR1_boundwater.set_ylabel("Pressure [GPa]",fontdict=font)
        axR1_boundwater.set_xlabel("Temperature [$^{\circ}$C]",fontdict=font)
        axR1_boundwater.set_yticks([0,2,4,6,8])

        h2o_plot_boundwater = axR1_boundwater.contourf(temperature_h2o_sed-273, pressure_h2o_sed*100/1e6, np.abs(xh2o_sed),
                                                    cmap="magma_r", levels=np.linspace(0, 6.0, 101), extend="both")
        #axM1.set_xlabel("Temperature [$^{\circ}$C]",fontdict=font)
        
        if fidx==0:
            cbar4 = plt.colorbar(h2o_plot_boundwater,ax=axR1_boundwater, orientation='vertical',
                                ticks=[0, 1, 2, 3, 4, 5, 6],pad=0.13)
            cbar4.ax.tick_params(labelsize=14)
            cbar4.set_label("Bound $\mathregular{H_{2}O}$ [%]", size=15)

            
            #axM1.set_ylim([np.min(pressure_h2o_basalt*100/1e6),np.max(pressure_h2o_basalt*100/1e6)])
            #axM1.set_yticks([0,2,4,6,8])
            # axL1_y = axM1.twinx()  # instantiate a second axes that shares the same x-axis

            # color = 'tab:blue'

            # axL1_y.set_ylim((0,1e9*np.max(pressure_h2o_basalt)*100/1e6/(3300*9.8*1e3)))

            axR1_y = axR1_boundwater.twinx()  # instantiate a second axes that shares the same x-axis

            axR1_y.set_ylabel('Depth [km]')  # we already handled the x-label with ax1

            axR1_y.tick_params(axis='y')
            axR1_y.set_ylim((0,1e9*np.max(pressure_h2o_basalt)*100/1e6/(3300*9.8*1e3)))

        # retrieve H2O and kinematic variables
        file_pattern_hires = folder+"/"+"solution/solution-0??00.npz"
        files_hires = sorted(glob.glob(os.getcwd()+file_pattern_hires))
        print("processing folder: "+folder)
        serp_full,_,serps_stable,times_full,boundwaters = get_serp_vc(files_hires,-1,"/"+folder)
        kin_files = [folder+".txt"]

        legends = ["pelite 4 wt%"]
        #colors = ["black","green","coral"]#["green","grey","navy"]#

        
        if fidx==0:
            plt.subplots_adjust(hspace=0.3)
            #ax_serp1_vc = axL1.twinx()
            #ax_serp1_vc.set_aspect('equal', adjustable='box')
            color = "tab:grey"
            font = {'fontname': 'Calibri',
            'weight': 'normal',
            'size': 15,
            } 

            #ax_serp1_vc.set_ylabel("vc [cm/yr]",color=color,fontdict=font)
            axL1.set_ylabel("Serpentinite [km$\mathregular{^{3}}$/km]",fontdict=font)
            axL1.set_xlabel("Net convergence [km]",fontdict=font)
            #ax_serp2 = fig.add_subplot(sub_gs[5][0])
            #axL1.set_aspect('equal', adjustable='datalim')

        #axL1.plot(np.arange(len(times_full[1:])),serp_full[1:],color=colors[fidx],linewidth=2.0)
        
        #axL1.plot(times_full[1:],boundwaters[1:],color=colors[fidx],linewidth=1.0,linestyle='-.')


        # ax_serp2_vc = ax_serp2.twinx()
        for kin_file in kin_files:
            f = np.loadtxt(os.getcwd()+"/kinematics/"+kin_file)
            t = f[:,1]
            vsp = f[:,2]
            vt = f[:,3]
            vc = f[:,4]
            axM1.set_ylabel("vc [cm/yr]",color="black",fontdict=font)
            if plot_time_xaxis:
                x_axis = t[1:]
                axM1.set_xlim([0,50])
            elif fidx<=3:
                x_axis = np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)
                axM1.set_xlim([0,1750])
            else:
                x_axis = np.cumsum(1e6*np.diff(t[:-1])*vc[1:-1]*0.01/1e3)
                axM1.set_xlim([0,1750])

            axM1.plot(x_axis,vc[1:],color=colors[fidx],linewidth=2.0,zorder=2)

            if (fidx==0) | (fidx==3):
                axM1.plot(sum(1e6*np.diff(t[:32])*vc[1:32]*0.01/1e3),vc[31],marker="*",markerfacecolor=colors[fidx],markeredgecolor="black",markersize=16)
            
            axM1.set_xlabel("Net convergence [km]",color="black",fontdict=font)
            
            #ax_serp2_vc.plot(t,vc,color=color,linestyle="dashed")
        
        if plot_time_xaxis:
            x_axis_serp = t[1:]
            axL1.set_xlim([0,50]) 
        elif fidx<=3:
            x_axis_serp = np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)
            axL1.set_xlim([0,1750]) 
        else:
            x_axis_serp = np.cumsum(1e6*np.diff(t[:-15])*vc[1:-15]*0.01/1e3)    
            axL1.set_xlim([0,1750]) 
        
        if (fidx<=3):
            axL1.plot(x_axis_serp,serps_stable[(-len(vc[1:])):]/1e6,color=colors[fidx],linewidth=1.5,linestyle=":",zorder=10)
            axL1.plot(x_axis_serp,serp_full[(-len(vc[1:])):]/1e6,color=colors[fidx],linewidth=1.5,zorder=10)
        else:
            axL1.plot(x_axis_serp,serps_stable[(-len(vc[1:])):-15]/1e6,color=colors[fidx],linewidth=1.5,linestyle=":",zorder=3)
            axL1.plot(x_axis_serp,serp_full[(-len(vc[1:])):-15]/1e6,color=colors[fidx],linewidth=1.5,zorder=3)

        axL1.set_ylim([0,85])
        axM1.set_ylim([0,10.25])
        print(np.max(vc))
        for idx_rgba,(Tsurf,Psurf,T1,P1) in enumerate(zip(Tsurfload,Psurfload,T_1km,P_1km)):
            cmap = mpl.cm.get_cmap("BuPu_r")
            rgba = cmap(idx_rgba/len(Tsurfload))
            #axR1_boundwater.plot(Tsurf, Psurf,color=rgba)
            #axR1_boundwater.plot(T1, P1,color=colors[fidx])
            #axR1_boundwater.plot(T1, P1,color=colors[fidx],linestyle='--')
            axR1_boundwater.plot(Tsurf, Psurf,color=colors[fidx])
            axM1.scatter(np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)[fnums[idx_rgba]//100-1],vc[fnums[idx_rgba]//100],75,facecolor=colors[fidx],edgecolors="black",zorder=10)



plt.tight_layout()

if plot_time_xaxis:
    boundwater_name = "/boundwater8_time"
if plot_high_res:
    boundwater_name = "/boundwater8_hires"
else:
    boundwater_name = "/boundwater8_convergence"

fig4.savefig(plot_loc+boundwater_name+".svg",dpi=200)
print(plot_loc+boundwater_name+".svg")

#file_pattern2 = os.getcwd()+folder2+"/"+"solution/solution-0??00.npz"
# files2_full = sorted(glob.glob(file_pattern2))


# file_pattern_baseline = os.getcwd()+folder_baseline+"/"+"solution/solution-0??00.npz"
# files_baseline_full = sorted(glob.glob(file_pattern_baseline))
    
# serp1,times1 = get_serp_vc(files1_full,0)
# serp2,times2 = get_serp_vc(files2_full,1)
# serp_full,times_full = get_serp_vc(files_baseline_full,-1)

# run_folder1 = "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
# kin_files = [folder1[1:]+".txt",folder_baseline[1:]+".txt",folder2[1:]+".txt"]

# legends = ["pelite 6 wt%","pelite 4 wt%", "pelite 2 wt%","pelite 0 wt%"]
# # kin_files = ["dd200_orig.txt","dd200.txt"]
# # legends = ["no lookup\nMDD=200 km","MDD=200 km"]
# colors = ["navy","skyblue","green","coral"]#["green","grey","navy"]#


# # gs = ax[2, 0].get_gridspec()
# # # remove the underlying Axes
# # for axi in ax[-1, :]:
# #     axi.remove()
# #ax_serp1_base = fig.add_subplot(gs[-1, :])
# #ax_serp1_base.set_axis_off()
# #gs_new = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_serp1_base.get_subplotspec(), wspace=0.2)
# axL1 = ax[2,1]#fig.add_subplot(gs_new[1])
# plt.subplots_adjust(hspace=0.3)

# #axL1.set_aspect('equal', adjustable='box')

# #axL1 = fig.add_subplot(sub_gs[2][0])
# ax[2,0].set_axis_off()

# ax_serp1_vc = axL1.twinx()
# #ax_serp1_vc.set_aspect('equal', adjustable='box')
# color = "tab:grey"

# axL1.plot(times1,serp1,color=colors[0],linewidth=2.0)
# axL1.plot(times_full,serp_full,color=colors[1],linewidth=2.0)
# axL1.plot(times2,serp2,color="green",linewidth=2.0)

# font = {'fontname': 'Calibri',
# 'weight': 'normal',
# 'size': 15,
# } 

# ax_serp1_vc.set_ylabel("vc [cm/yr]",color=color,fontdict=font)
# axL1.set_ylabel("Serpentinite [km$^{3}$/km]",fontdict=font)
# axL1.set_xlabel("time [Myr]",fontdict=font)
# #ax_serp2 = fig.add_subplot(sub_gs[5][0])
# #axL1.set_aspect('equal', adjustable='datalim')
# ax[2,1].set_axis_off()

# # ax_serp2_vc = ax_serp2.twinx()
# for kin_file,color,legend in zip(kin_files,colors,legends):
#     f = np.loadtxt(os.getcwd()+"/kinematics/"+kin_file)
#     t = f[:,1]
#     vsp = f[:,2]
#     vt = f[:,3]
#     vc = f[:,4]
#     ax_serp1_vc.plot(t,vc,color=color,linestyle="dashed",linewidth=2.0)
#     #ax_serp2_vc.plot(t,vc,color=color,linestyle="dashed")
# #ax_serp1_vc.tick_params(axis="y",labelcolor=color)

# ax_serp1_vc.set_xlim([0,50])

# # ax_serp2.plot(times2,serp2,color="navy",linewidth=1.75)
# # ax_serp2.plot(times_full,serp_full,color="grey")
# # ax_serp2.plot(times1,serp1,color="green")

# output_file = files1[0].replace(".npz", "_fig2.png")

