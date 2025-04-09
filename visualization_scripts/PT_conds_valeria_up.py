#! /usr/bin/python3
from posix import times_result
#from matplotlib.cm import get_cmap
import pandas as pd
import numpy as np
import json
#from tqdm import tqdm
# import matplotlib.cm as cm
# import matplotlib as mpl
import sys, os, subprocess
# from matplotlib.pyplot import show, xlabel, ylabel
import matplotlib.pyplot as plt
# import matplotlib.tri as tri
import argparse
import math as math
from scipy.signal import savgol_filter 
from scipy.interpolate import griddata, Rbf, RegularGridInterpolator
import timeit

mod_name = str(sys.argv[1])
max_file = int(sys.argv[2])
step = int(sys.argv[3])

########## FUNCTIONS ################

def create_grid_velocities_crust (xmin_plot:float, xmax_plot:float, ymin_plot:float, ymax_plot:float, grid_res:float, grid_low_res:float, grid_high_res: float)  :
    # create grid to interpolate stuff onto (for plotting)
    x_low = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_res))
    y_low =  np.linspace(ymin_plot,ymax_plot,int((ymax_plot-ymin_plot)/grid_res))
    X_low, Y_low = np.meshgrid(x_low,y_low)
    # lower res grid for velocities
    x_vels = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_low_res))
    y_vels =  np.linspace(ymin_plot,ymax_plot,int((ymax_plot-ymin_plot)/grid_low_res))
    X_vels, Y_vels = np.meshgrid(x_vels,y_vels)
    # higher res grid for crust
    x_crust = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_high_res))
    y_crust =  np.linspace(ymin_plot,ymax_plot,int((ymax_plot-ymin_plot)/grid_high_res))
    X_crust, Y_crust = np.meshgrid(x_crust,y_crust)
    return (X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust)


def grab_dimTime_fields (dir: str, stat:str, time, step):
    time[:,0] = np.arange(0,time.shape[0]*step,step)
    filt = stat.loc[:,15].notnull()
    time[:,1] = stat[filt].loc[:,1]
    return time


def interp_compCrust (x, y, C, X_crust, Y_crust):
    Comp = griddata((x, y), C,   (X_crust, Y_crust), method='nearest')
    return (Comp)


def slab_surf_moho(contour, thresh: float):    
    conts = len(contour.get_paths())
    j = 0
    for i in range(conts):
        if len(contour.get_paths()[j]) < len(contour.get_paths()[i]):
            j = i
    pts = contour.get_paths()[j].vertices
    threshold_x = (pts[pts[:,1] > thresh]).min(0)[0]
    slab = pts[pts[:,0]> threshold_x]
    itip = slab[:,1].argmax()
    itop = slab[:,1].argmin()
    iset = 0
    if np.abs(len(slab)//2 - itop)<np.abs(len(slab)//2 - itip):
        iset = itop
    else:
        iset = itip


    moho = slab[:iset, :]
    slab_surf = slab[iset:, :][::-1, :]
    return slab_surf, moho

# Use Green's theorem to compute the area
# enclosed by the given contour.
def area(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return a

############## EXTRACT AND PLOT P-T-T ###################

def main():
    #step = 100
    
    # read model
    json_file_name = "input.json"

    cwd = os.getcwd()
    csvs_loc =  cwd+'/csv_outputs/'
    models_loc =  cwd+'/'
    json_loc = cwd+'/pyInput/'

    with open(f"{json_loc}{json_file_name}") as json_file:  # args.json_file
            configs = json.load(json_file)

    # 2d equivalent grid variables
    xmin_plot = 0.e3; xmax_plot = 5800.e3
    ymin_plot=0.e3; ymax_plot = 1450.e3
    grid_res=7.e3; grid_low_res = 75.e3; grid_high_res = 1.0e3

    # create grids
    X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust = create_grid_velocities_crust (xmin_plot, xmax_plot, ymin_plot, ymax_plot, grid_res, grid_low_res, grid_high_res)
    interp_method='nearest'

    
    for ind_m, m in (enumerate([mod_name])): 
        plot_loc = f"{cwd}/plots/{mod_name}"
    
        # read statistics file from ASPECT
        stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=configs['head_lines']+7,sep='\s+',header=None)

        # grab times
        csv_file_names = os.listdir(f"{csvs_loc}{m}")
        # if csv_file_names[0][-4:] == 'gzip':
        #     max_file = max([int(csv_file_name[5:-5]) for csv_file_name in csv_file_names])
        # elif csv_file_names[0][-3:] == 'csv':
        #     max_file = max([int(csv_file_name[5:-4])
        #                     for csv_file_name in csv_file_names])
        # max_file = 7100
        time_array = np.zeros((max_file//step,2)) 
        time_array = grab_dimTime_fields(f"{csvs_loc}{m}", stat.loc[0:max_file-1:step,:], time_array, step)

        #colmap = plt.get_cmap('copper_r',len(time_array))

        for t in (range(0, max_file+1,step)):
            strnum = str(t)
            if len(strnum)<4:
                strnum = '0'*(4-len(strnum))+strnum
            
            if not os.path.exists(plot_loc):
                os.mkdir(plot_loc)

            # read ASPECT output and isolate the crust
            data = pd.read_parquet(f"{csvs_loc}{m}/full.{t}.gzip")
            
            comp = interp_compCrust(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'ocrust']+data.loc[:,'ocrust_init']+data.loc[:,'gabbro_init']+data.loc[:,'gabbro']+data.loc[:,'sediment'],  X_crust, Y_crust) 
            
            crust_cont = plt.contour(X_crust/1.e3, (ymax_plot-Y_crust)/1.e3, comp, levels=[0.25], linewidths=0.5, colors='blue', zorder=2, alpha = 0)
            # thresh - threshold depth
            slab_surf, moho = slab_surf_moho(crust_cont, thresh=10.)
            
            slab_properties_1km = np.zeros((len(slab_surf),5))
            slab_properties_5km = np.zeros((len(slab_surf),5))
            slab_properties_10km = np.zeros((len(slab_surf),5))

            for i in range(0,len(slab_surf)):
                if i == 0:
                    slope = (slab_surf[i,1]-slab_surf[i+1,1])/(slab_surf[i,0]-slab_surf[i+1,0])
                    dip = np.arctan(slope) # radians
                elif i == len(slab_surf)-1:
                    slope = (slab_surf[i-1,1]-slab_surf[i,1])/(slab_surf[i-1,0]-slab_surf[i,0])
                    dip = np.arctan(slope) # radians
                else:
                    
                    if (slab_surf[i-1,0]-slab_surf[i,0])!=0:
                        slope = (slab_surf[i-1,1]-slab_surf[i,1])/(slab_surf[i-1,0]-slab_surf[i,0])
                    elif (slab_surf[i,0]-slab_surf[i+1,0])!=0:
                        slope = (slab_surf[i,1]-slab_surf[i+1,1])/(slab_surf[i,0]-slab_surf[i+1,0])
                    elif (slab_surf[i-1,0]-slab_surf[i+1,0])!=0: 
                        slope = (slab_surf[i-1,1]-slab_surf[i+1,1])/(slab_surf[i-1,0]-slab_surf[i+1,0])


                    dip = np.arctan(slope) # radians

                dx = 1.0 * np.sin(dip)
                dz = 1.0 * np.cos(dip)
                xmid =  slab_surf[i,0] - dx
                zmid =  slab_surf[i,1] + dz
                slab_properties_1km[i,0] = xmid
                slab_properties_1km[i,1] = zmid

                dx = 5.0 * np.sin(dip)
                dz = 5.0 * np.cos(dip)
                xmid =  slab_surf[i,0] - dx
                zmid =  slab_surf[i,1] + dz
                slab_properties_5km[i,0] = xmid
                slab_properties_5km[i,1] = zmid

                dx = 10.0 * np.sin(dip)
                dz = 10.0 * np.cos(dip)
                xmid =  slab_surf[i,0] - dx
                zmid =  slab_surf[i,1] + dz
                slab_properties_10km[i,0] = xmid
                slab_properties_10km[i,1] = zmid    

            start_time = timeit.default_timer()
            # interpolate the P-T conditions along the slab top and moho for every time
            T_surf = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'T']-273, (slab_surf[:,0], slab_surf[:,1]), method=interp_method)
            P_surf = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'p'], (slab_surf[:,0], slab_surf[:,1]), method=interp_method)

            # velocity conditions for surface tracking
            vx_surf = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'velocity:0'], (slab_surf[:,0], slab_surf[:,1]), method=interp_method)
            vz_surf = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'velocity:1'], (slab_surf[:,0], slab_surf[:,1]), method=interp_method)

            # SET UP PSEUDO-PARTICLE TRACKING
            # Access the column names
            header_terms = data.columns.tolist()
            if 'velocity:0' in header_terms:
                vx_col = header_terms.index("velocity:0")
            if "Points:0" in header_terms:
                x_col = header_terms.index("Points:0")
            if "Points:1" in header_terms:
                y_col = header_terms.index("Points:1")
            if "ocrust" in header_terms:
                ocrust_col = header_terms.index("ocrust")
            if "ocrust_init" in header_terms:
                ocrust_col2 = header_terms.index("ocrust_init")

            # extract mid-plate profile
            plate_prof_loc = ymax_plot - 20.e3                           # 20 km depth
            plate_prof = data.values[data.values[:,y_col] < (plate_prof_loc+4.e3)] 
            plate_prof = plate_prof[plate_prof[:,y_col] > (plate_prof_loc-4.e3)] 
            plate_prof = plate_prof[plate_prof[:,x_col].argsort()] # sort by x

            # get trench location
            trench_x = 0;
            for i in range(len(plate_prof)):
                if (((plate_prof[i,ocrust_col] + plate_prof[i,ocrust_col2]) > 0.4)  and plate_prof[i,x_col] > trench_x):
                    trench_x = plate_prof[i,x_col]
            if trench_x == 0:
                trench_x = 0#np.nan

            # get plate velocities either side of trench
            vsp_tot = 0; nsp = 0
            vop_tot = 0; nop = 0
            plt.plot()
            for i in range(len(plate_prof)):
                if plate_prof[i,x_col] > (trench_x - 200e3) and plate_prof[i,x_col] < (trench_x - 100e3):
                    vsp_tot = vsp_tot + plate_prof[i,vx_col] 
                    nsp = nsp + 1
                if plate_prof[i,x_col] > (trench_x + 50e3) and plate_prof[i,x_col] < (trench_x + 150e3):
                    vop_tot = vop_tot + plate_prof[i,vx_col] 
                    nop = nop + 1
            if nsp>0:
                vsp = 100.*(vsp_tot/nsp) # to get cm/yr
            else:
                vsp = 0
            
            if nop>0:
                vop = 100.*(vop_tot/nop)
            else:
                vop = 0

            vc  = vsp - vop            


            if (t>0):
                T_surf,P_surf = savgol_filter((T_surf,P_surf), 59, 3) 
                z_surf,x_surf = savgol_filter((slab_surf[:,1],slab_surf[:,0]), 59, 3) 
                vz_surf,vx_surf = savgol_filter((vz_surf,vx_surf), 59, 3) 

            if (t//step==1):
                n_to_track = 7
                p_bins_to_track = np.linspace(0,np.max(P_surf),n_to_track+1)
                inds = [0]*n_to_track
                for (ind,p_bin) in enumerate(p_bins_to_track[1:]):
                    ind_p, = np.where((P_surf<p_bin) & (P_surf>p_bins_to_track[ind]))
                    argind = np.argmax(np.abs(p_bin-P_surf[ind_p]))
                    inds[ind] = ind_p[argind]

                x_slab_track = x_surf[inds]
                z_slab_track = z_surf[inds]
                P_slab_track = P_surf[inds]
                vz_slab_track = vz_surf[inds]
                T_slab_track = T_surf[inds]
            elif (t==0):
                x_slab_track = np.empty([0,1])
                z_slab_track = np.empty([0,1])
                vz_slab_track = np.empty_like(x_slab_track)
                P_slab_track = np.empty_like(x_slab_track)
                T_slab_track = np.empty_like(x_slab_track)
            
            if (t//step>1):
                dt =  time_array[(t//step)-1,1] - time_array[(t//step)-2,1]
                surf_sorted = np.sort(list(zip(z_surf,x_surf)),axis=0)
                z_surf = surf_sorted[:,0]
                x_surf = surf_sorted[:,1]
                interp = RegularGridInterpolator((z_surf,),x_surf)
                interpT = RegularGridInterpolator((z_surf,),T_surf)
                interpP = RegularGridInterpolator((z_surf,),P_surf)
                interpvz = RegularGridInterpolator((z_surf,),vz_surf)
                z_slab_track_offset = (max(z_surf)-max(z_surf_prev))
                x_slab_track_offset = vsp*dt/1e2/1e3
                z_slab_track_new = np.zeros_like(z_slab_track)
                for i in range(0,len(x_slab_track)):
                    if i == 0:
                        slope = (z_slab_track[i]-z_slab_track[i+1])/(x_slab_track[i]-x_slab_track[i+1])
                        dip = np.arctan(slope) # radians
                    elif i == len(slab_surf)-1:
                        slope = (z_slab_track[i-1]-z_slab_track[i])/(x_slab_track[i-1]-x_slab_track[i])
                        dip = np.arctan(slope) # radians
                    else:
                        
                        if (x_slab_track[i-1]-x_slab_track[i])!=0:
                            slope = (z_slab_track[i-1]-z_slab_track[i])/(x_slab_track[i-1]-x_slab_track[i])
                        elif (x_slab_track[i]-x_slab_track[i+1])!=0:
                            slope = (z_slab_track[i]-z_slab_track[i+1])/(x_slab_track[i]-x_slab_track[i+1])
                        elif (x_slab_track[i-1]-x_slab_track[i+1])!=0: 
                            slope = (z_slab_track[i-1]-z_slab_track[i+1])/(x_slab_track[i-1]-x_slab_track[i+1])


                        dip = np.arctan(slope) # radians  
                    if ((z_slab_track[i] + z_slab_track_offset*np.sin(dip))<np.max(z_surf)) & ((z_slab_track[i] + z_slab_track_offset*np.sin(dip))>np.min(z_surf)):
                        z_slab_track_new[i] = z_slab_track[i] +   z_slab_track_offset*np.sin(dip) # x_slab_track_offset*np.cos(dip) #
                    elif ((z_slab_track[i] + z_slab_track_offset*np.sin(dip))<np.max(z_surf)): # undershoot case
                        z_slab_track_new[i] = np.min(z_surf)
                    elif ((z_slab_track[i] + z_slab_track_offset*np.sin(dip))>np.min(z_surf)): # overshoot case
                        z_slab_track_new[i] = np.max(z_surf)                    
                
                z_slab_track = z_slab_track_new
                #z_slab_track += vz_slab_track*dt/1e3 
                # 
                x_slab_track = interp(z_slab_track)
                P_slab_track = interpP(z_slab_track)
                T_slab_track = interpT(z_slab_track)
                vz_slab_track = interpvz(z_slab_track)

            if len(P_slab_track)>0:
                pt_slab_track = open(f"{plot_loc}/alt_track_{strnum}.txt", "w")
                # write X, Y, Plith [GPa], Ptotal [GPa], T_cels
                for i in range(len(P_slab_track)):
                    pt_slab_track.write("%.0f %.3f %.3f\n" % (t, P_slab_track[i], T_slab_track[i]))
                pt_slab_track.close()

            T_crust_points_1km  = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'T']-273, (slab_properties_1km[:,0], slab_properties_1km[:,1]), method=interp_method)
            P_crust_points_1km  = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'p'], (slab_properties_1km[:,0], slab_properties_1km[:,1]), method=interp_method)
            T_crust_points_5km  = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'T']-273, (slab_properties_5km[:,0], slab_properties_5km[:,1]), method=interp_method)
            P_crust_points_5km  = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'p'], (slab_properties_5km[:,0], slab_properties_5km[:,1]), method=interp_method)
            T_crust_points_10km = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'T']-273, (slab_properties_10km[:,0], slab_properties_10km[:,1]), method=interp_method)
            P_crust_points_10km = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'p'], (slab_properties_10km[:,0], slab_properties_10km[:,1]), method=interp_method)

            if (t>0):
                T_crust_points_1km,P_crust_points_1km = savgol_filter((T_crust_points_1km,P_crust_points_1km), 59, 3)
                T_crust_points_5km,P_crust_points_5km = savgol_filter((T_crust_points_5km,P_crust_points_5km), 59, 3)
                T_crust_points_10km,P_crust_points_10km = savgol_filter((T_crust_points_10km,P_crust_points_10km), 59, 3)

            for i in range(len(slab_properties_1km)):
                slab_properties_1km[i,2] = P_crust_points_1km[i]/1.e9
                slab_properties_1km[i,3] = (3300. * 9.81 * slab_properties_1km[i,1]*1.e3)/1.e9
                slab_properties_1km[i,4] = T_crust_points_1km[i] + slab_properties_1km[i,1]*0.3*1e-3

            for i in range(len(slab_properties_5km)):
                slab_properties_5km[i,2] = P_crust_points_5km[i]/1.e9
                slab_properties_5km[i,3] = (3300. * 9.81 * slab_properties_5km[i,1]*1.e3)/1.e9
                slab_properties_5km[i,4] = T_crust_points_5km[i] + slab_properties_5km[i,1]*0.3*1e-3

            for i in range(len(slab_properties_10km)):
                slab_properties_10km[i,2] = P_crust_points_10km[i]/1.e9
                slab_properties_10km[i,3] = (3300. * 9.81 * slab_properties_10km[i,1]*1.e3)/1.e9
                slab_properties_10km[i,4] = T_crust_points_10km[i] + slab_properties_10km[i,1]*0.3*1e-3

            T_moho = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'T']-273, (moho[:,0], moho[:,1]), method=interp_method)
            P_moho = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'p'], (moho[:,0], moho[:,1]), method=interp_method)
            
            # if (t>0):
            #     T_moho,P_moho = savgol_filter((T_moho,P_moho), 59, 3)

            # original style PT data
            # write P-T-t conditions to file
            pt = open(f"{plot_loc}/pt_{strnum}.txt", "w")

            # write X, Y, Plith [GPa], Ptotal [GPa], T_cels
            for i in range(len(P_surf)):
                pt.write("%.0f %.3f %.3f\n" % (t, P_surf[i], T_surf[i]))
            pt.close()

            # Gabe-style data
            # write P-T-t conditions to file
            ptslabtop = open(f"{plot_loc}/{t}.slabtop.txt", "w")

            # write X, Y, Plith [GPa], Ptotal [GPa], T_cels
            for i in range(len(P_surf)):
                ptslabtop.write("%.3f %.3f %.3f %.3f %.3f\n" % (slab_surf[i,0], slab_surf[i,1], (3300.0 * 9.81 * slab_surf[i,1]*1.e3)/1.e9, P_surf[i]/1e9, T_surf[i]))
            ptslabtop.close()

            # write P-T-t conditions to file
            pt1 = open(f"{plot_loc}/{t}.1km.txt", "w")
            #pt5.write("time Psurf Tsurf\n")
            # write X, Y, Plith [GPa], Ptotal [GPa], T_cels to file
            for i in range(len(P_crust_points_1km)):
                pt1.write("%.3f %.3f %.3f %.3f %.3f\n" % (slab_properties_1km[i,0], slab_properties_1km[i,1] ,(3300.0 * 9.81 * slab_properties_1km[i,1]*1.e3)/1.e9, P_crust_points_1km[i]/1e9, T_crust_points_1km[i]))
            pt1.close()

            # write P-T-t conditions to file
            pt5 = open(f"{plot_loc}/{t}.5km.txt", "w")
            #pt5.write("time Psurf Tsurf\n")
            # write X, Y, Plith [GPa], Ptotal [GPa], T_cels to file
            for i in range(len(P_crust_points_5km)):
                pt5.write("%.3f %.3f %.3f %.3f %.3f\n" % (slab_properties_5km[i,0], slab_properties_5km[i,1] ,(3300.0 * 9.81 * slab_properties_5km[i,1]*1.e3)/1.e9, P_crust_points_5km[i]/1e9, T_crust_points_5km[i]))
            pt5.close()

            pt10 = open(f"{plot_loc}/{t}.10km.txt", "w")
            #pt10.write("time Psurf Tsurf\n")
            # write X, Y, Plith [GPa], Ptotal [GPa], T_cels to file
            for i in range(len(P_crust_points_10km)):
                pt10.write("%.3f %.3f %.3f %.3f %.3f\n" % (slab_properties_10km[i,0], slab_properties_10km[i,1] ,(3300.0 * 9.81 * slab_properties_10km[i,1]*1.e3)/1.e9, P_crust_points_10km[i]/1e9, T_crust_points_10km[i]))
            pt10.close()

            ptmoho = open(f"{plot_loc}/{t}.Moho.txt", "w")
            #ptmoho.write("time Psurf Tsurf\n")
            # write them to file
            for i in range(len(P_moho)):
                ptmoho.write("%.3f %.3f %.3f %.3f %.3f\n" % (moho[i,0], moho[i,1], (3300.0 * 9.81 * moho[i,1]*1.e3)/1.e9, P_moho[i]/1e9, T_moho[i]))
            ptmoho.close()

            elapsed_time = timeit.default_timer() - start_time
            print("time elapsed:", elapsed_time)
            # norm = mpl.colors.Normalize(vmin=0, vmax=len(time_array))
            # fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colmap), ax=axs,orientation='horizontal', cax = fig.add_axes([0.75, -0.15, 0.125, 0.0125]),ticks=[0,len(time_array)], ticklocation = 'top')    
            
            # fig.subplots_adjust(hspace = 2)
            # strnum = str(t)
            # if len(strnum)<4:
            #     strnum = '0'*(4-len(strnum))+strnum

            # plt.savefig(plotname+strnum, bbox_inches='tight', format='eps', dpi=500)
            # axs[0].clear()
            # axs[1].clear()

            plt.close()
            if (t>0):
                z_surf_prev = z_surf


if __name__ == "__main__":
    main()


