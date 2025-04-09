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
import pathlib
import os
#import gridspec
# Define a single-color colormap (e.g., "blue")
single_color_cmap = LinearSegmentedColormap.from_list("SingleColor", ["#D2D344", "#D2D344"])

is_plot_outline = True # or just viscosity 
plot_time_xaxis = False # in first subplot

# ASPECT output 
csvs_loc = os.getcwd()+'/csv_outputs/'

# setup plotting
mpl.use('agg')
plt.rcParams['font.family']="Calibri"
plt.rcParams['font.size']=13
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.linewidth'] = 1.5

def get_serp_vc(files,row,folder_baseline):
    serps = np.zeros(len(files))
    melts = np.zeros_like(serps)
    serps_stable = np.zeros_like(serps)
    times = np.zeros_like(serps)
    boundwaters = np.zeros_like(serps)
    for idx,file in enumerate(files):

        if row==-1:
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

        if row==-1:
            csv_filename=''.join([csvs_loc,folder_baseline,'/full.',str(time),'.gzip'])
        data = np.load(file)
        serps[idx] = data["serp_total"]
        if 'melt_total' in np.fromiter(data.keys(),dtype='<U16'):
            melts[idx] = data["melt_total"]
        else:
            melts[idx] = np.nan
        serps_stable[idx] = data["serp_total_stable"]
        times[idx] = time_dim
        boundwaters[idx] = data["boundwater_total"]
    return serps,melts,serps_stable,times,boundwaters


# Function to plot quadrilateral data as triangles
def plot_quads_with_tricontourf(points, cells, values):
    triangles = []
    triangle_values = []
    x = []
    y = []

    for i, quad in enumerate(cells):
        # Split each quadrilateral into two triangles
        triangles.append([quad[0], quad[1], quad[2]])  # Triangle 1
        triangle_values.append([values[i]]) # , values[i], values[i]
        # x.append(np.mean(points[:, 0][quad[[0,1,2]]]))
        # y.append(np.mean(points[:, 1][quad[[0,1,2]]])) 

        triangles.append([quad[0], quad[2], quad[3]])  # Triangle 2
        # x.append(np.mean(points[:, 0][quad[[0,2,3]]]))
        # y.append(np.mean(points[:, 1][quad[[0,2,3]]]))
        triangle_values.append([values[i]])

    triangles = np.array(triangles)
    triangle_values = np.array(triangle_values).flatten()
    triangulation = tri.Triangulation(points[:, 0], points[:, 1], triangles)

    return triangulation, triangle_values

def main():
    # 0. load data
    # Path to the .npz files
    folder = "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
    
    #"rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"

    # 1. load topo for trench
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
    plot_loc = ''.join(
        [os.getcwd()+'/plots/', str(folder)])
    if not os.path.exists(plot_loc):
        os.mkdir(plot_loc)
    
    fnums = [100,1000,3000,6000]

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

    # Read the input file
    input_file_sed=str(pathlib.Path().resolve()) + "/pseudosections/Pelagic_sediment/pelagic_bound_h2o.tab"
    data_sed = np.genfromtxt(input_file_sed, skip_header=0, usecols=(0, 1, 2))
    xh2o_sed = (data_sed[:, 2]).reshape((250,250))
    pressure_h2o_sed = (data_sed[:, 1]).reshape((250,250))
    temperature_h2o_sed = (data_sed[:, 0]).reshape((250,250))

    if time ==0:
        fig4 = plt.figure(figsize=(12, 4))
        gs4 = GridSpec(1, 2,width_ratios=[0.67,0.67],wspace=0.1)

        gs4_sub = GridSpec(1, 1, width_ratios=[0.75])

        axL1_boundwater = fig4.add_subplot(gs4[0])
        # axR2 = fig4.add_subplot(gs4[0, 0])
        axM1 = fig4.add_subplot(gs4[1])
        axR1 = fig4.add_subplot(gs4_sub[0])
        gs4_sub.update(left=0.055,right=0.32,bottom=0.15,top=0.95)
        gs4.update(left=0.43,right=0.95,bottom=0.15,top=0.95)
        
        # axR3 = fig4.add_subplot(gs4[2, 1])
        # axR4 = fig4.add_subplot(gs4[3, 1])

        filename_track = "/pt_"+strnum+".txt"
        print("loading: "+filename_track+"\n")

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
        
        axM1.contourf(temperature_h2o_basalt-273, pressure_h2o_basalt*100/1e6, np.abs(xh2o_basalt),
                                                    cmap="magma_r", levels=np.linspace(0, 6.0, 101), extend="both")
        axL1_boundwater.contour(temperature_h2o_serp-273,pressure_h2o_serp*100/1e6,xh2o_serp,levels=np.array([5]),colors="black",extend="both",linestyles='dotted')
        axM1.contour(temperature_h2o_serp-273,pressure_h2o_serp*100/1e6,xh2o_serp,levels=np.array([5]),colors="black",extend="both",linestyles='dotted')
    #else:
    #    if is_load_PT:
        for idx_rgba,(Tsurf,Psurf,T1,P1) in enumerate(zip(Tsurfload,Psurfload,T_1km,P_1km)):
            cmap = mpl.cm.get_cmap("PuBu_r")
            rgba = cmap(idx_rgba/len(Tsurfload))
            axL1_boundwater.plot(Tsurf, Psurf,color=rgba)
            axM1.plot(Tsurf, Psurf,color=rgba,alpha=0.35)
            axM1.plot(T1, P1,color=rgba)
        axL1_boundwater.set_xlim([0,np.max(temperature_h2o_basalt-273)])
        axL1_boundwater.set_ylim([np.min(pressure_h2o_basalt*100/1e6),np.max(pressure_h2o_basalt*100/1e6)])


        font = {'fontname': 'Calibri',
        'weight': 'normal',
        'size': 14} 
        axL1_boundwater.set_ylabel("Pressure [GPa]",fontdict=font)
        axL1_boundwater.set_xlabel("Temperature [$^{\circ}$C]",fontdict=font)
        axL1_boundwater.set_yticks([0,2,4,6,8])

        h2o_plot_boundwater = axL1_boundwater.contourf(temperature_h2o_sed-273, pressure_h2o_sed*100/1e6, np.abs(xh2o_sed),
                                                    cmap="magma_r", levels=np.linspace(0, 6.0, 101), extend="both")
        axM1.set_xlabel("Temperature [$^{\circ}$C]",fontdict=font)
        # cbar4 = plt.colorbar(h2o_plot_boundwater,ax=axM1, orientation='vertical',
        #                     ticks=[0, 1, 2, 3, 4, 5, 6],pad=0.18)
        # cbar5 = plt.colorbar(h2o_plot_boundwater,ax=axL1_boundwater, orientation='vertical',
        #                     ticks=[0, 1, 2, 3, 4, 5, 6],pad=0.15)
        # cbar5.ax.set_visible(False)
        # cbar4.ax.tick_params(labelsize=14)
        # cbar4.set_label("Bound $\mathregular{H_{2}O}$ [%]", size=15)

        axM1.set_xlim([0,np.max(temperature_h2o_basalt-273)])
        axM1.set_ylim([np.min(pressure_h2o_basalt*100/1e6),np.max(pressure_h2o_basalt*100/1e6)])
        axM1.set_yticks([0,2,4,6,8])
        axM1_y = axM1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'

        axM1_y.set_ylim((0,1e9*np.max(pressure_h2o_basalt)*100/1e6/(3300*9.8*1e3)))

        # axL1_y = axL1_boundwater.twinx()  # instantiate a second axes that shares the same x-axis

        axM1_y.set_ylabel('Depth [km]')  # we already handled the x-label with ax1

        # axL1_y.tick_params(axis='y')
        # axL1_y.set_ylim((0,1e9*np.max(pressure_h2o_basalt)*100/1e6/(3300*9.8*1e3)))

        # retrieve H2O and kinematic variables
        file_pattern_hires = folder+"/"+"solution/solution-0[0,1,2,3,4,5,6]?00.npz"
        files_hires = sorted(glob.glob(file_pattern_hires))
        serp_full,melt_full,serp_stable,times_full,boundwaters = get_serp_vc(files_hires,-1,"/"+folder)
        kin_files = [folder+".txt"]

        legends = ["pelite 4 wt%"]
        colors = ["black","green","coral"]#["green","grey","navy"]#
        ax_serp1 = axR1#fig.add_subplot(gs_new[1])
        plt.subplots_adjust(hspace=0.3)
        ax_serp1_vc = ax_serp1.twinx()
        #ax_serp1_vc.set_aspect('equal', adjustable='box')
        color = "orangered"


        #ax_serp2 = fig.add_subplot(sub_gs[5][0])
        #ax_serp1.set_aspect('equal', adjustable='datalim')
        ax_serp1_vc.tick_params(axis="y",labelcolor=color)
        ax_serp1.set_ylabel("$\mathregular{v_{c}}$ [cm/yr]",fontdict=font)
        # ax_serp2_vc = ax_serp2.twinx()
        for kin_file,color,legend in zip(kin_files,colors,legends):
            f = np.loadtxt(os.getcwd()+"/kinematics/"+kin_file)
            t = f[:,1]
            vsp = f[:,2]
            vt = f[:,3]
            vc = f[:,4]
            if plot_time_xaxis:
                distance_vc = t[1:]
            else:
                distance_vc = np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)
            ax_serp1.plot(distance_vc,vc[1:],color="black",linewidth=2.0,zorder=2)
            ax_serp1.set_ylim([0,1.05*np.max(vc)])
            #ax_serp2_vc.plot(t,vc,color=color,linestyle="dashed")
        #ax_serp1_vc.set_xlim([0,np.max(np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3))])
        if plot_time_xaxis:
            distance = t[1:]
        else:
            distance = np.cumsum(1e6*np.diff(t)*vc[1:]*0.01/1e3)
        ax_serp1_vc.plot(distance,serp_full[1:]/1e6,color="orangered",linewidth=2.0) # times_full[1:]
        ax_serp1_vc.plot(distance,melt_full[1:]/1e6,color="fuchsia",linewidth=2.0)
        ax_serp1_vc.plot(distance,serp_stable[1:]/1e6,color="orangered",linestyle="dashed",linewidth=2.0)
        
        ax_serp1_vc.set_xlim([0,1640])
        ax_serp1_vc.set_ylim([0,79])


        font = {'fontname': 'Calibri',
        'weight': 'normal',
        'size': 15,
        } 

        
        ax_serp1_vc.set_ylabel("Serpentinite [km$\mathregular{^{3}}$/km]",fontdict=font,color="orangered")
        ax_serp1.set_xlabel("Net convergence [km]",fontdict=font)
        

    plt.tight_layout()
    ax_serp1.set_zorder(ax_serp1_vc.get_zorder()+1)
    ax_serp1.set_frame_on(False)
    if plot_time_xaxis:
        boundwater_name = "/boundwater3_time"
    else:
        boundwater_name = "/boundwater3_convergence"
    fig4.savefig(plot_loc+boundwater_name+".svg",dpi=200)
    print(plot_loc+"/boundwater3.svg")
if __name__=="__main__":
    main()