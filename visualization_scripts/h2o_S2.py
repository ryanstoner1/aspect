import os
import glob
import pathlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec, GridSpec
from main_model4 import get_serp_vc

plot_time_xaxis = False # for convergence and serp plots; otherwise distance

# 0. load data
# Path to the .npz files1

folder3 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel0_smu0_02_cmu0_04_deserp_erase_res5_5_run20"
ids3 = ["03100"]
files3 = [os.getcwd()+folder3+"/solution/solution-"+id3+".npz" for id3 in ids3]

folder4 = "/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_0visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_smu0_02_cmu0_04_deserp_erase_res5_5_run38_new"
ids4 = ["03100"]
files4 = [os.getcwd()+folder4+"/solution/solution-"+id4+".npz" for id4 in ids4]

# 0. load data
# Path to the .npz files
folders = [folder3,folder4]
#file_pattern = folder+"/"+"solution/solution-0[1,2,3,4,5,6]000.npz"
files = files3+files4
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

fig4 = plt.figure(figsize=(12, 4))
gs4 = GridSpec(1, 1,width_ratios=[0.67])

gs4_sub = GridSpec(1, 2, width_ratios=[0.75,0.75],wspace=0.19)

axR1_boundwater = fig4.add_subplot(gs4[0])
# axR2 = fig4.add_subplot(gs4[0, 0])
axM1 = fig4.add_subplot(gs4_sub[1])
axL1 = fig4.add_subplot(gs4_sub[0])
gs4_sub.update(left=0.06,right=0.62,bottom=0.15,top=0.95)
gs4.update(left=0.70,right=0.95,bottom=0.15,top=0.95)     

colors = ["#FAF323","#7A4E53"]
#colors = ["navy","skyblue","green","coral"]
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

    if fidx==0:
        fnums = [1700,4200]
    elif fidx==1:
        fnums = [1500,3000]


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

            
            #axM1.plot(Tsurf, Psurf,color=colors[fidx])
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
            # cbar4 = plt.colorbar(h2o_plot_boundwater,ax=axR1_boundwater, orientation='vertical',
            #                     ticks=[0, 1, 2, 3, 4, 5, 6],pad=0.15)
            # cbar4.ax.tick_params(labelsize=14)
            # cbar4.set_label("Bound $\mathregular{H_{2}O}$ [%]", size=15)

            
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
        serp_full,serps_stable,times_full,boundwaters = get_serp_vc(files_hires,-1,"/"+folder)
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

        # ax_serp2_vc = ax_serp2.twinx()
        for kin_file in kin_files:
            f = np.loadtxt(os.getcwd()+"/kinematics/"+kin_file)
            t = f[:,1]
            vsp = f[:,2]
            vt = f[:,3]
            vc = f[:,4]
            if fidx<2:
                if plot_time_xaxis:
                    distance_vc = t[1:]
                    axM1.set_xlim([0,50])
                else:
                    distance_vc = np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)
                    axM1.set_xlim([0,1100])
                    axM1.set_ylim([0,6.3])
                axM1.set_ylabel("vc [cm/yr]",color="black",fontdict=font) # np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)

                axM1.plot(distance_vc,vc[1:],color=colors[fidx],linewidth=2.0)

                axM1.set_xlabel("Net convergence [km]",fontdict=font)
                axM1.plot(sum(1e6*np.diff(t[:31])*vc[1:31]*0.01/1e3),vc[30],marker="*",markerfacecolor=colors[fidx],markeredgecolor="black",markersize=16,zorder=10)
                #ax_serp2_vc.plot(t,vc,color=color,linestyle="dashed")
        if fidx<2:
            if plot_time_xaxis:
                distance = t[1:]
                axL1.set_xlim([0,50])
            else:
                distance = np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)
                axL1.set_xlim([0,1100])
                axL1.set_ylim([0,24])
            axL1.plot(distance,serp_full[1:len(distance)+1]/1e6,color=colors[fidx],linewidth=2.0)
            axL1.plot(distance,serps_stable[1:len(distance)+1]/1e6,color=colors[fidx],linewidth=2.0,linestyle="--")
        

        for idx_rgba,(Tsurf,Psurf,T1,P1) in enumerate(zip(Tsurfload,Psurfload,T_1km,P_1km)):
            cmap = mpl.cm.get_cmap("BuPu_r")
            rgba = 0.0#cmap(times_full[np.array(fnums)//100-1]/50)
            #axR1_boundwater.plot(Tsurf, Psurf,color=rgba)
            if fidx==0:
                axR1_boundwater.plot(T1, P1,color=colors[fidx])
                axM1.scatter(np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)[fnums[idx_rgba]//100-1],vc[fnums[idx_rgba]//100],75,facecolor=colors[fidx],edgecolors="black",zorder=10)
            elif fidx==1: 
                axR1_boundwater.plot(Tsurf, Psurf,color=colors[fidx],linestyle='--')
                #axM1.scatter(np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)[fnums[idx_rgba]//100-1],vc[fnums[idx_rgba]//100],color=colors[fidx])
                axM1.scatter(np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)[fnums[idx_rgba]//100-1],vc[fnums[idx_rgba]//100],75,facecolor=colors[fidx],edgecolors="black",zorder=8)


plt.tight_layout()

if plot_time_xaxis:
    boundwater_name = "/S2_time"
else:
    boundwater_name = "/S2_convergence"

fig4.savefig(plot_loc+boundwater_name+".svg",dpi=200)
print(plot_loc+boundwater_name+".svg")
