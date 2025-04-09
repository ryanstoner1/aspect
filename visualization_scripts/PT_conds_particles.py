#! /usr/bin/python3
from posix import times_result
from matplotlib.cm import get_cmap
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.cm as cm
import matplotlib.tri as tri
import matplotlib as mpl
import sys, os, subprocess
from matplotlib.pyplot import show, xlabel, ylabel
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import argparse
import math as math
from scipy.signal import savgol_filter 
from scipy.interpolate import griddata, Rbf, RegularGridInterpolator
import timeit
from scipy.spatial import Delaunay
from particle_funcs import *
from pathlib import Path

mpl.rcParams['axes.linewidth'] = 2
is_plot_triangulation = False
is_plot_temperature_contour = False

############## EXTRACT AND PLOT P-T-T ###################
def main(mod_name: str, time_full: int, step: int):
    """
    Main function for processing model data and initializing interpolators.
    
    Parameters:
        mod_name (str): Model name.
        time_full (int): Total simulation time.
        step (int): Time step interval.
    """
    t_start = 100
    n_x, n_y = 250, 250
    
    # File paths
    pseudosection_files = {
        "gabbro": "pseudosections/MORB/gabbro_aspect_H2O.txt",
        "basalt": "pseudosections/MORB/morb_green_aspect_H2O.txt",
        "sediment": "pseudosections/Pelagic_sediment/pelagic_bound_h2o.tab",
        "serpentine": "pseudosections/Serpentinite/serp_sat_niu_aspect_H2O.txt"
    }

    interpolators = initialize_interpolators(pseudosection_files, n_x, n_y)

    # Directory paths
    cwd = Path.cwd()
    csvs_loc = cwd / 'csv_outputs'
    models_loc = cwd

    # 2d equivalent grid variables
    xmin_plot = 0.e3; xmax_plot = 5800.e3
    ymin_plot=0.e3; ymax_plot = 1450.e3
    grid_res=7.e3; grid_low_res = 75.e3; grid_high_res = 1.0e3

    # create grids
    X_low, Y_low, X_vels, Y_vels, X_crust, Y_crust = create_grid_velocities_crust (xmin_plot, xmax_plot, ymin_plot, ymax_plot, grid_res, grid_low_res, grid_high_res)
    interp_method='nearest'

    
    for ind_m, m in tqdm(enumerate([mod_name])): 
        plot_loc = f"{cwd}/plots/{m}"
    
        # read statistics file from ASPECT
        stat = pd.read_csv(f"{models_loc}/{m}/statistics",skiprows=37,sep='\s+',header=None)
        if stat.loc[0,1]>0.0:
            raise Exception("First timestep is greater than zero in statistics file. skiprows likely too large.")
        # grab times
        csv_file_names = os.listdir(f"{csvs_loc}/{m}")
        if (csv_file_names[0][-4:] == 'gzip') & (csv_file_names[0][0:4] != 'serp'):
            max_file = max([int(csv_file_name[5:-5]) for csv_file_name in csv_file_names if (csv_file_name[0:4] != 'serp')])
        elif csv_file_names[0][-3:] == 'csv':
            max_file = max([int(csv_file_name[5:-4])
                            for csv_file_name in csv_file_names])

        time_array = np.zeros((max_file//step,2)) 
        time_array[:,0] = np.arange(0,time_array.shape[0]*step,step)
        time_array[:,1] = stat.loc[t_start:max_file:step,1]

        fluid_release_start = 0
        for t in tqdm(range(0, max_file+1,step)):
            strnum = str(t)
            if len(strnum)<4:
                strnum = '0'*(4-len(strnum))+strnum
            
            if not os.path.exists(plot_loc):
                os.mkdir(plot_loc)

            # read ASPECT output and isolate the crust
            data = pd.read_parquet(f"{csvs_loc}/{m}/part.{t}.gzip")



            # Set up the figure
            nbins = 90

            if t==t_start:
                ids_orig = data["id"].to_numpy()
                ids_mask = np.ones(len(ids_orig),dtype=bool)
                # get last timestep to avoid lost particles
                for t_beg in range(t_start+step, max_file+1,step):
                    data_end = pd.read_parquet(f"{csvs_loc}/{m}/part.{t_beg}.gzip")
                    ids_cand = data_end["id"].to_numpy()
                    ids_mask_new = np.isin(ids_orig,ids_cand)
                    ids_mask *= ids_mask_new
                ids_orig = ids_orig[ids_mask]    
                ids_orig = ids_orig.astype(int)

                # extract particle props
                
                p = data["p"][ids_mask]
                T = data["T"][ids_mask]

                xx = data["Points:0"][ids_mask]
                yy = data["Points:1"][ids_mask]
                serp = data["serp"][ids_mask]
                h2o_bound = data["boundwater"][ids_mask]

                # track selected properties
                # 1 bound properties and fractions
                n_ids_orig = len(ids_orig)
                h2o_bound_slab_track = np.zeros((max_file//step,n_ids_orig))
                h2o_bound_slab_track[0,:] = h2o_bound
                h2o_bound_basalt_track_interp = np.zeros((max_file//step,n_ids_orig))
                h2o_bound_gabbro_track_interp = np.zeros((max_file//step,n_ids_orig))
                h2o_bound_sediment_track_interp = np.zeros((max_file//step,n_ids_orig))
                for idx, (pval,Tval) in enumerate(zip(p,T)):
                    if pval<150e6:
                        pval = 150e6
                    if pval>5.5e9:
                        pval = 5.5e9
                    if Tval<273:
                        Tval = 273
                    if Tval>1700:
                        Tval = 1700
                    h2o_bound_basalt_track_interp[0,idx] = interpolators["basalt"]((Tval,pval))/100
                    h2o_bound_gabbro_track_interp[0,idx] = interpolators["gabbro"]((Tval,pval))/100
                    h2o_bound_sediment_track_interp[0,idx] = interpolators["sediment"]((Tval,pval))/100
                
                sediment_track = np.zeros_like(h2o_bound_sediment_track_interp)
                sediment_track[0,:] = data["sediment"][ids_mask]
                basalt_track = np.zeros_like(h2o_bound_basalt_track_interp)
                basalt_track[0,:] = data["ocrust_init"][ids_mask]+data["ocrust"][ids_mask]
                gabbro_track = np.zeros_like(h2o_bound_gabbro_track_interp)
                gabbro_track[0,:] = data["gabbro_init"][ids_mask]+data["gabbro"][ids_mask]

                # 2 release
                h2o_release_basalt_track = np.zeros((max_file//step,n_ids_orig))
                h2o_release_gabbro_track = np.zeros((max_file//step,n_ids_orig))
                h2o_release_sediment_track = np.zeros((max_file//step,n_ids_orig))  
                h2o_release_boundwater_track = np.zeros((max_file//step,n_ids_orig))  

                p_track = np.zeros((max_file//step,n_ids_orig))  
                T_track = np.zeros((max_file//step,n_ids_orig))  

                P_slab_track = np.copy(p)
                T_slab_track = np.copy(T) 

                h2o_bins_time = np.zeros(nbins)
                h2o_release_boundwater_min = np.zeros_like(p)

                h2o_release_gabbro_track_cur = np.zeros_like(p)
                h2o_release_basalt_track_cur = np.zeros_like(p)
                h2o_release_sediment_track_cur = np.zeros_like(p)
            elif t>t_start:
                ids = [int(id) for id in data["id"]]
                ids = np.array(ids).astype(int)
                ids_mask = np.isin(ids,ids_orig)

                ids = ids[ids_mask]
                ids_ordered = np.zeros_like(ids_orig,dtype=int)
                for i, id_orig in enumerate(ids_orig):   
                    is_original_id = id_orig==np.array(ids)
                    ids_ordered[i] = np.where(is_original_id)[0]
                
                ids = ids[ids_ordered]
                
                # extract new particle properties
                p = np.array(data["p"])[ids_mask][ids_ordered]
                T = np.array(data["T"])[ids_mask][ids_ordered]
                h2o_bound = np.array(data["boundwater"])[ids_mask][ids_ordered]
                serp = np.array(data["serp"])[ids_mask][ids_ordered]
                xx = np.array(data["Points:0"])[ids_mask][ids_ordered]
                yy = np.array(data["Points:1"])[ids_mask][ids_ordered]

                # calculate fluid release
                # correct for crust composition
                h2o_bound_slab_track[t//step-1,:] = h2o_bound
                T_slab_track = np.vstack((T_slab_track, T))

                # basalt check
                frac_basalt_orig = np.array(data["initial ocrust"] + data["initial ocrust_init"])[ids_mask][ids_ordered]
                frac_basalt = np.array(data["ocrust"] + data["ocrust_init"])[ids_mask][ids_ordered]
                h2o_bound_basalt_raw_direct = np.zeros_like(ids_orig,dtype=float)

                # gabbro
                frac_gabbro_orig = np.array(data["initial gabbro"] + data["initial gabbro_init"])[ids_mask][ids_ordered]
                frac_gabbro = np.array(data["gabbro"] + data["gabbro_init"])[ids_mask][ids_ordered]
                h2o_bound_gabbro_raw_direct = np.zeros_like(ids_orig,dtype=float)

                # sediment
                frac_sediment = np.array(data["sediment"])[ids_mask][ids_ordered]
                h2o_bound_sediment_raw_direct = np.zeros_like(ids_orig,dtype=float)
                
                pval = np.copy(p)
                pval[pval<150e6] = 150e6
                pval[pval>5.5e9] = 5.5e9
                Tval = np.copy(T)
                Tval[Tval<273] = 273
                p_raw_direct = np.copy(pval)
                T_raw_direct = np.copy(Tval)

                h2o_bound_cutoff = 2e-4
                basalt_cutoff = 0.01
                gabbro_cutoff = 0.01
                sediment_cutoff = 0.01
                h2o_bound_basalt_raw_direct[(h2o_bound>h2o_bound_cutoff) & (frac_basalt>basalt_cutoff)] = interpolators["basalt"]((Tval[(h2o_bound>h2o_bound_cutoff) & (frac_basalt>basalt_cutoff)],
                                                             pval[(h2o_bound>h2o_bound_cutoff) & (frac_basalt>basalt_cutoff)]))/100  
                h2o_bound_gabbro_raw_direct[(h2o_bound>h2o_bound_cutoff) & (frac_gabbro>gabbro_cutoff)] = interpolators["gabbro"]((Tval[(h2o_bound>h2o_bound_cutoff) & (frac_gabbro>gabbro_cutoff)],
                                                             pval[(h2o_bound>h2o_bound_cutoff) & (frac_gabbro>gabbro_cutoff)]))/100  
                h2o_bound_sediment_raw_direct[(h2o_bound>h2o_bound_cutoff) & (frac_sediment>sediment_cutoff)] = interpolators["sediment"]((Tval[(h2o_bound>h2o_bound_cutoff) & (frac_sediment>sediment_cutoff)],
                                                             pval[(h2o_bound>h2o_bound_cutoff) & (frac_sediment>sediment_cutoff)]))/100                  
                h2o_bound_basalt_track_interp[(t//step)-1,:] = h2o_bound_basalt_raw_direct
                h2o_bound_gabbro_track_interp[(t//step)-1,:] = h2o_bound_gabbro_raw_direct
                h2o_bound_sediment_track_interp[(t//step)-1,:] = h2o_bound_sediment_raw_direct
                
                if t==(t_start+2*step):
                    h2o_release_basalt_track_cur = h2o_bound_basalt_track_interp[(t//step)-2,:]*frac_basalt
                    h2o_release_gabbro_track_cur = h2o_bound_gabbro_track_interp[(t//step)-2,:]*frac_gabbro
                    h2o_release_sediment_track_cur = h2o_bound_sediment_track_interp[(t//step)-2,:]*frac_sediment
                    t_set = 1
                 
                for i, id_orig in enumerate(ids_orig):                      
                    new_h2o_basalt = h2o_bound_basalt_raw_direct[i]*frac_basalt[i]
                    new_h2o_gabbro = h2o_bound_gabbro_raw_direct[i]*frac_gabbro[i]
                    new_h2o_sediment = h2o_bound_sediment_raw_direct[i]*frac_sediment[i]
                    if (new_h2o_basalt<h2o_release_basalt_track_cur[i]) & (frac_basalt[i]>0.1):
                        h2o_release_basalt_track[t//step-2,i] = h2o_bound_basalt_track_interp[(t//step)-2,i]*frac_basalt[i] - new_h2o_basalt
                        h2o_release_basalt_track_cur[i] = h2o_bound_basalt_track_interp[(t//step)-2,i]*frac_basalt[i]
                    if (new_h2o_gabbro<h2o_release_gabbro_track_cur[i]) & (frac_gabbro[i]>0.1):
                        h2o_release_gabbro_track[t//step-2,i] = h2o_bound_gabbro_track_interp[(t//step)-2,i]*frac_gabbro[i] - new_h2o_gabbro
                        h2o_release_gabbro_track_cur[i] = h2o_bound_gabbro_track_interp[(t//step)-2,i]*frac_gabbro[i]
                    if (new_h2o_sediment<h2o_release_sediment_track_cur[i]) & (frac_sediment[i]>0.1):
                        h2o_release_sediment_track[t//step-2,i] = h2o_bound_sediment_track_interp[(t//step)-2,i]*frac_sediment[i] - new_h2o_sediment 
                        h2o_release_sediment_track_cur[i] = h2o_bound_sediment_track_interp[(t//step)-2,i]*frac_sediment[i]  

                    if (t//step-2)>=t_start//step:
                        if ((frac_sediment[i]>0.1) | (frac_basalt[i]>0.1) | (frac_gabbro[i]>0.1)) & (p_raw_direct[i]>=1.50e8):
                            #h2o_release_boundwater_track[t//step-2,i] = h2o_bound_slab_track[t//step-2,i]-h2o_bound_slab_track[t//step-1,i]
                            h2o_release_boundwater_track[t//step-2,i] = h2o_bound_slab_track[t//step-1,i]
                            if (frac_sediment[i]>0.0):
                                h2o_release_boundwater_track[t//step-2,i] -= frac_sediment[i]*(h2o_bound_sediment_raw_direct[i])
                            if (frac_basalt[i]>0.0):
                                h2o_release_boundwater_track[t//step-2,i] -= frac_basalt[i]*(h2o_bound_basalt_raw_direct[i])
                            if (frac_gabbro[i]>0.0):
                                h2o_release_boundwater_track[t//step-2,i] -= frac_gabbro[i]*(h2o_bound_gabbro_raw_direct[i])
                            if (h2o_release_boundwater_track[t//step-2,i]<0):
                                h2o_release_boundwater_track[t//step-2,i] = 0 
                    else:                        
                        if ((frac_sediment[i]>0.1) | (frac_basalt[i]>0.1) | (frac_gabbro[i]>0.1)) & (p_raw_direct[i]>=1.50e8):
                            h2o_release_boundwater_track[t//step-2,i] = h2o_bound_slab_track[(t//step)-1,i]
                            if (frac_sediment[i]>0.0):
                                h2o_release_boundwater_track[t//step-2,i] -= frac_sediment[i]*(h2o_bound_sediment_raw_direct[i])
                            if (frac_basalt[i]>0.0):
                                h2o_release_boundwater_track[t//step-2,i] -= frac_basalt[i]*(h2o_bound_basalt_raw_direct[i])
                            if (frac_gabbro[i]>0.0):
                                h2o_release_boundwater_track[t//step-2,i] -= frac_gabbro[i]*(h2o_bound_gabbro_raw_direct[i])

                            if (h2o_release_boundwater_track[t//step-2,i]<0):
                                h2o_release_boundwater_track[t//step-2,i] = 0 


                p_track[t//step-1,:] = p_raw_direct
                T_track[t//step-1,:] = T_raw_direct

                basalt_track[t//step-1,:] = frac_basalt
                gabbro_track[t//step-1,:] = frac_gabbro
                sediment_track[t//step-1,:] = frac_sediment

                if (time_array[(t//step)-1][1]/(1e6))>0.0:
                    
                    nbins = 40
                    if fluid_release_start==0:
                        fluid_release_start = t
                        range_released_full = range((fluid_release_start//step)-2,((max_file//step)))
                        h2o_release_grid = np.zeros((nbins,len(range_released_full)))
                        h2o_release_grid_sediment = np.zeros((nbins,len(range_released_full)))
                        h2o_release_grid_basalt = np.zeros((nbins,len(range_released_full)))
                        h2o_release_grid_gabbro = np.zeros((nbins,len(range_released_full)))
                        h2o_release_grid_tracked = np.zeros((nbins,len(range_released_full)))                       
                        h2o_release_gridx = np.zeros((nbins,len(range_released_full)))
                        h2o_release_gridx_tracked = np.zeros((nbins,len(range_released_full)))  
                        h2o_release_serp = np.zeros(len(range_released_full)) 
                        h2o_release_aspect_mass = np.zeros(len(range_released_full)) 
                        h2o_release_serp_mass = np.zeros(len(range_released_full)) 
                        h2o_release_serp_aspect_mass = np.zeros(len(range_released_full)) 
                        h2o_release_serp_aspect_contourmass = np.zeros(len(range_released_full))  
                        h2o_release_serp_tracked = np.zeros(len(range_released_full)) 
                        h2o_release_serp_tracked_mass = np.zeros(len(range_released_full)) 
           
                    
                    ival = ((t//step)-1) - ((fluid_release_start//step)-1)
                    depth_bins = np.linspace(0,160e3,nbins)
                    dy_bin = np.diff(depth_bins)[0]
                    for i,y_bin in enumerate(depth_bins):
                        is_in_bin = ((ymax_plot-yy)>y_bin) & ((ymax_plot-yy)<(y_bin+dy_bin)) 
                        if sum(is_in_bin)>0:
                            h2o_release_grid[i,ival] = np.sum(h2o_release_basalt_track[(fluid_release_start//step)-2+ival,is_in_bin]+
                                                              h2o_release_gabbro_track[(fluid_release_start//step)-2+ival,is_in_bin]+
                                                              h2o_release_sediment_track[(fluid_release_start//step)-2+ival,is_in_bin])   
                            h2o_release_grid_sediment[i,ival] = np.sum(h2o_release_sediment_track[(fluid_release_start//step)-2+ival,is_in_bin])  
                            h2o_release_grid_basalt[i,ival] = np.sum(h2o_release_basalt_track[(fluid_release_start//step)-2+ival,is_in_bin])   
                            h2o_release_grid_gabbro[i,ival] = np.sum(h2o_release_gabbro_track[(fluid_release_start//step)-2+ival,is_in_bin])    
                            h2o_release_grid_tracked[i,ival] = np.sum(h2o_release_boundwater_track[(fluid_release_start//step)-2+ival,is_in_bin])

                    x_bins = np.linspace(0,np.max(xx),nbins)
                    dx_bin = np.diff(x_bins)[0]
                    for i,x_bin in enumerate(x_bins):
                        is_in_bin = (xx>x_bin) & (xx<(x_bin+dx_bin)) 
                        if sum(is_in_bin)>0:
                            h2o_release_gridx[i,ival] = np.sum(h2o_release_basalt_track[(fluid_release_start//step)-2+ival,is_in_bin]+
                                                              h2o_release_gabbro_track[(fluid_release_start//step)-2+ival,is_in_bin]+
                                                              h2o_release_sediment_track[(fluid_release_start//step)-2+ival,is_in_bin])    
                            h2o_release_gridx_tracked[i,ival] = np.sum(h2o_release_boundwater_track[(fluid_release_start//step)-2+ival,is_in_bin])

                    """ 
                    Plotting 
                    """  
                    nth_step = 1

                    # plot once enough processed
                    range_released_plot = range((fluid_release_start//step)-2,((t//step)-1))

                    if (h2o_release_grid[:,np.array(range_released_plot)-((fluid_release_start//step)-2)].shape[1]>1) & (ind_m%nth_step==0):  
                        fig_all, axs = plt.subplots(figsize=(16, 9),nrows=2, ncols=2)
                        y_top_plot = 10e3
                        fluid_val = axs[0,0].contourf(time_array[range_released_plot,1]/1e6,depth_bins[depth_bins>y_top_plot]/1e3,
                                                      h2o_release_grid_tracked[:,np.array(range_released_plot)-((fluid_release_start//step)-2)][depth_bins>y_top_plot,:],100)
                        axs[0,0].set_xlabel("time (Myr)")
                        axs[0,0].set_ylabel("depth (km)")
                        axs[0,0].set_title("released H2O (aspect)")
                        axs[0,0].invert_yaxis()
                        plt.colorbar(fluid_val)

                        fluid_test = axs[0,1].contourf(time_array[range_released_plot,1]/1e6,depth_bins[depth_bins>y_top_plot]/1e3,
                                                       h2o_release_grid[:,np.array(range_released_plot)-((fluid_release_start//step)-2)][depth_bins>y_top_plot,:],100)
                        axs[0,1].set_xlabel("time (Myr)")
                        axs[0,1].set_ylabel("depth (km)")
                        axs[0,1].set_title("released H2O (predicted)")
                        axs[0,1].invert_yaxis()
                        plt.colorbar(fluid_test)

                        scatter_release = axs[1,0].scatter(xx,(ymax_plot-yy)/1e3,s=1,c=(h2o_bound))
                        axs[1,0].set_xlabel("distance (km)")
                        axs[1,0].set_ylabel("depth (km)")
                        axs[1,0].set_title("locations of h2o release")
                        axs[1,0].invert_yaxis()
                        plt.colorbar(scatter_release)

                        scatter_bound = axs[1,1].scatter(xx,(ymax_plot-yy)/1e3,s=1,c=((h2o_bound_sediment_raw_direct*frac_sediment+
                                                                                       h2o_bound_basalt_raw_direct*frac_basalt+
                                                                                       h2o_bound_gabbro_raw_direct*frac_gabbro)))
                        axs[1,1].set_xlabel("distance (km)")
                        axs[1,1].set_ylabel("depth (km)")
                        axs[1,1].set_title("bound H2O difference")
                        axs[1,1].invert_yaxis()
                        plt.colorbar(scatter_bound)
                        fig_all.tight_layout()
                        fig_all.savefig(plot_loc+"/d05_fluid_release_2_"+str(t)+".jpg",dpi=250)
                        

                        # sum fluid release plots over depth and time
                        depth_cutoff = 2e3
                        fig_sums, axs_sums = plt.subplots(figsize=(16, 8),nrows=3, ncols=2)
                        h2o_loss_aspect_depth = np.sum(h2o_release_grid_tracked[:,np.array(range_released_plot)-((fluid_release_start//step)-2)][depth_bins>depth_cutoff,:],axis=1)
                        h2o_loss_predicted_depth = np.sum(h2o_release_grid[:,np.array(range_released_plot)-((fluid_release_start//step)-2)][depth_bins>depth_cutoff,:],axis=1)
                        h2o_loss_predicted_depth_sediment = np.sum(h2o_release_grid_sediment[:,np.array(range_released_plot)-((fluid_release_start//step)-2)][depth_bins>depth_cutoff,:],axis=1)
                        h2o_loss_predicted_depth_basalt = np.sum(h2o_release_grid_basalt[:,np.array(range_released_plot)-((fluid_release_start//step)-2)][depth_bins>depth_cutoff,:],axis=1)

                        cumulative_h2o_loss_aspect_depth = np.cumsum(h2o_loss_aspect_depth[::-1])[::-1]
                        cumulative_h2o_loss_predicted_depth = np.cumsum(h2o_loss_predicted_depth[::-1])[::-1]

                        axs_sums[0,0].plot(h2o_loss_aspect_depth,depth_bins[depth_bins>depth_cutoff]/1e3)
                        axs_sums[0,0].set_xlabel("time (Myr)")
                        axs_sums[0,0].set_ylabel("depth (km)")
                        axs_sums[0,0].set_title("depths of cumulative released H2O (aspect)")
                        axs_sums[0,0].set_xlim(0,np.max(h2o_loss_aspect_depth))

                        axs_sums[0,1].plot(h2o_loss_predicted_depth,depth_bins[depth_bins>depth_cutoff]/1e3,label="total",color="navy")
                        axs_sums[0,1].plot(h2o_loss_predicted_depth_sediment,depth_bins[depth_bins>depth_cutoff]/1e3,label="sediment",color="lime")
                        axs_sums[0,1].plot(h2o_loss_predicted_depth_basalt,depth_bins[depth_bins>depth_cutoff]/1e3,label="basalt",color="brown")
                        axs_sums[0,1].set_xlabel("time (Myr)")
                        axs_sums[0,1].set_ylabel("depth (km)")
                        axs_sums[0,1].set_title("depths of cumulative released H2O (predicted)")
                        axs_sums[0,1].set_xlim(0,np.max(h2o_loss_aspect_depth))
                        axs_sums[0,1].legend(labelcolor=["navy","lime","brown"]) # 

                        h2o_loss_aspect = np.sum(h2o_release_grid_tracked[:,np.array(range_released_plot)-((fluid_release_start//step)-2)][depth_bins>depth_cutoff,:],axis=0)
                        cumulative_h2o_loss_aspect = np.cumsum(h2o_loss_aspect)
                        axs_sums[1,0].plot(time_array[range_released_plot,1]/1e6,cumulative_h2o_loss_aspect)
                        axs_sums[1,0].set_xlabel("time (Myr)")
                        axs_sums[1,0].set_ylabel("H2O released")
                        axs_sums[1,0].set_title("time-evolving cumulative released H2O (aspect)")
                        axs_sums[1,0].set_ylim(0,np.max(cumulative_h2o_loss_aspect))



                        h2o_loss_predicted = np.sum(h2o_release_grid[:,np.array(range_released_plot)-((fluid_release_start//step)-2)][depth_bins>depth_cutoff,:],axis=0)
                        cumulative_h2o_loss_predicted = np.cumsum(h2o_loss_predicted)
                        axs_sums[1,1].plot(time_array[range_released_plot,1]/1e6,
                                                      np.cumsum(np.sum(h2o_release_grid[:,np.array(range_released_plot)-((fluid_release_start//step)-2)][depth_bins>depth_cutoff,:],axis=0)))
                        axs_sums[1,1].set_xlabel("time (Myr)")
                        axs_sums[1,1].set_ylabel("H2O released")
                        axs_sums[1,1].set_title("time-evolving cumulative released H2O (predicted)")
                        axs_sums[1,1].set_ylim(0,np.max(cumulative_h2o_loss_predicted))

                        h2o_lossx_aspect = np.sum(h2o_release_gridx_tracked[:,np.array(range_released_plot)-((fluid_release_start//step)-2)],axis=1)
                        
                        axs_sums[2,0].plot(x_bins/1e3,h2o_lossx_aspect)
                        axs_sums[2,0].set_xlabel("distance (km)")
                        axs_sums[2,0].set_ylabel("H2O released")
                        axs_sums[2,0].set_title("location of released H2O (aspect)")
                        axs_sums[2,0].set_ylim(0,np.max(h2o_lossx_aspect))


                        h2o_lossx_predicted = np.sum(h2o_release_gridx[:,np.array(range_released_plot)-((fluid_release_start//step)-2)],axis=1)
                        axs_sums[2,1].plot(x_bins/1e3,h2o_lossx_predicted)
                        axs_sums[2,1].plot(x_bins/1e3,h2o_lossx_predicted)
                        axs_sums[2,1].set_xlabel("distance (km)")
                        axs_sums[2,1].set_ylabel("H2O released")
                        axs_sums[2,1].set_title("location of released H2O (predicted)")
                        axs_sums[2,1].set_ylim(0,np.max(h2o_lossx_aspect))

                        fig_sums.tight_layout()
                        fig_sums.savefig(plot_loc+"/d05_thresh_2km_cumulative_fluid_release_"+str(t)+"_cutoff150MPa.jpg",dpi=250)
                        plt.close()

                if ind_m%nth_step==0:
                    fig, axs = plt.subplots(figsize=(12, 10),nrows=2, ncols=4)

                    data = pd.read_parquet(f"{csvs_loc}/{m}/full.{t}.gzip")
                    stringt = str(t)
                    if len(stringt)<5:
                        stringt = (5-len(stringt))*"0"+stringt
                    data_serp_boundwater = np.load(f"{models_loc}/{m}/solution/solution-" +stringt +".npz") #pd.read_parquet(f"{csvs_loc}{m}/serparea.{t}.gzip")
                    data_serp_mult = data_serp_boundwater["serp_total"]
                    data_boundwater_bin = data_serp_boundwater["boundwater_total"] #pd.read_parquet(f"{csvs_loc}{m}/serp_boundwater_bin.{t}.gzip")
                    
                    #h2o_release_serp_aspect_mass[ival] = calculate_total_mass(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'serp'])
                    h2o_release_serp_aspect_contourmass[ival] = data_serp_mult#np.sum(data_serp_mult['Cell_Area']*data_serp_mult['Serp_Value'])
                    h2o_release_serp_aspect_mass[ival] = data_serp_mult#np.sum(data_serp_mult['Cell_Area']*data_serp_mult['Serp_Value_Thresh'])



                    comp = interp_compCrust(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'ocrust']+data.loc[:,'ocrust_init']+data.loc[:,'gabbro_init']+data.loc[:,'gabbro'],  X_crust, Y_crust) 

                    #serp_region = interp_compCrust(data.loc[:,'Points:0'], data.loc[:,'Points:1'], data.loc[:,'serp'],  X_crust, Y_crust) 
                    
                    crust_cont = plt.contour(X_crust/1.e3, (ymax_plot-Y_crust)/1.e3, comp, levels=[0.5], linewidths=0.5, colors='blue', zorder=2, alpha = 0)
                    #serp_cont = plt.contour(X_crust/1.e3, (ymax_plot-Y_crust)/1.e3, serp_region, levels=[0.01], linewidths=0.5, colors='blue', zorder=2, alpha = 0)
                    # thresh - threshold depth
                    slab_surf, moho = slab_surf_moho(crust_cont, thresh=10.)
                    slab_surf[:,0] = np.flip(np.maximum.accumulate(np.flip(slab_surf[:,0])))
                    T_surf = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'T']-273, (slab_surf[:,0], slab_surf[:,1]), method=interp_method)
                    P_surf = griddata((data.loc[:,'Points:0']/1.e3, (ymax_plot - data.loc[:,'Points:1'])/1.e3), data.loc[:,'p'], (slab_surf[:,0], slab_surf[:,1]), method=interp_method)
                    T_surf,P_surf = savgol_filter((T_surf,P_surf), 59, 3) 

                    dx_slab = np.diff(slab_surf[:,0],prepend=slab_surf[0,0])
                    
                    # check for rollover
                    is_not_rolled_over = dx_slab<0.0
                    xs_slab = slab_surf[:,0]
                    ys_slab = slab_surf[:,1]

                    # remove rollover bit
                    dx_slab = dx_slab[is_not_rolled_over]
                    x_surf = xs_slab[is_not_rolled_over] #np.hstack((xs_slab[0], xs_slab[:-1][is_not_rolled_over]))
                    y_surf = ys_slab[is_not_rolled_over] #np.hstack((ys_slab[0], ys_slab[:-1][is_not_rolled_over]))
                    T_surf = T_surf[is_not_rolled_over] #np.hstack((T_surf[0], T_surf[:-1][is_not_rolled_over]))
                    P_surf = P_surf[is_not_rolled_over] #np.hstack((P_surf[0], P_surf[:-1][is_not_rolled_over]))
                    h2o_release_serp_x = np.zeros(len(slab_surf))
                    h2o_release_serp_x_tracked = np.zeros(len(slab_surf))
                    
                    y_thresh = 10e3
                    is_in_slab = (xx/1e3>np.min(x_surf)) & (xx/1e3<np.max(x_surf)) & ((ymax_plot-yy)>y_thresh) & (h2o_release_boundwater_track[(fluid_release_start//step)-2+ival,:]>0)
                    is_in_slab_track = (xx/1e3>np.min(x_surf)) & (xx/1e3<np.max(x_surf)) & ((ymax_plot-yy)>y_thresh)
                    print(x_surf)
                    is_x_bin = np.digitize(xx/1e3,x_surf)

                    T_serp = T_surf[is_x_bin[is_in_slab]]+273
                    P_serp = P_surf[is_x_bin[is_in_slab]]+273
                    P_serp[P_serp<1.5e8] = 1.5e8

                    serp_frac_particles = interpolators["serpentine"]((T_serp,P_serp))/100
                    #serp_frac_particles = interpolators["serpentine"]((T_track[t//step-1,is_in_slab],p_track[t//step-1,is_in_slab]))/100
                    if len(serp_frac_particles)>2:
                        h2o_release_serp_tracked_mass[ival] = calculate_total_mass(xx[is_in_slab], yy[is_in_slab], h2o_release_boundwater_track[(fluid_release_start//step)-2+ival,is_in_slab]*(serp_frac_particles>0.01))
                        if np.sum(((h2o_release_basalt_track[(fluid_release_start//step)-2+ival,is_in_slab]+
                                                                            h2o_release_gabbro_track[(fluid_release_start//step)-2+ival,is_in_slab]+
                                                                            h2o_release_sediment_track[(fluid_release_start//step)-2+ival,is_in_slab])*(
                                                                            serp_frac_particles>0.01))>0.0)>2:
                            h2o_release_serp_mass[ival] = calculate_total_mass(xx[is_in_slab], 
                                                                                yy[is_in_slab], 
                                                                            (h2o_release_basalt_track[(fluid_release_start//step)-2+ival,is_in_slab]+
                                                                            h2o_release_gabbro_track[(fluid_release_start//step)-2+ival,is_in_slab]+
                                                                            h2o_release_sediment_track[(fluid_release_start//step)-2+ival,is_in_slab])*(
                                                                            serp_frac_particles>0.01))
                        h2o_release_aspect_serp = (h2o_release_boundwater_track[(fluid_release_start//step)-2+ival,is_in_slab]*(serp_frac_particles>0.01))
                        T_aspect_serp = (T_track[(fluid_release_start//step)-2+ival,is_in_slab]*(serp_frac_particles>0.01))
                        p_aspect_serp = (p_track[(fluid_release_start//step)-2+ival,is_in_slab]*(serp_frac_particles>0.01))
                        is_release_aspect = h2o_release_aspect_serp>1e-5
                        h2o_release_pt_postprocess_serp = ((h2o_release_basalt_track[(fluid_release_start//step)-2+ival,is_in_slab]+
                                                                        h2o_release_gabbro_track[(fluid_release_start//step)-2+ival,is_in_slab]+
                                                                        h2o_release_sediment_track[(fluid_release_start//step)-2+ival,is_in_slab])*(
                                                                        serp_frac_particles>0.01))
                        

                        
                        is_release_pt_postprocess = h2o_release_pt_postprocess_serp>1e-5
                        ids_bins_aspect = np.digitize(
                            ymax_plot-yy[is_in_slab][is_release_aspect],depth_bins)
                        ids_bins_pt = np.digitize(
                            ymax_plot-yy[is_in_slab][is_release_pt_postprocess],depth_bins)  

                        bins_h2o_serp_aspect = np.zeros(len(depth_bins))
                        bins_h2o_serp_pt_postprocess = np.zeros(len(depth_bins))

                        for id, ybin in enumerate(depth_bins): 
                            if (sum(id==ids_bins_aspect)>0):
                                bins_h2o_serp_aspect[id] += np.sum(h2o_release_aspect_serp[is_release_aspect][id==ids_bins_aspect])
                            if (sum(id==ids_bins_pt)>0):
                                bins_h2o_serp_pt_postprocess[id] += np.sum(h2o_release_pt_postprocess_serp[is_release_pt_postprocess][id==ids_bins_pt])


                        fig_depth_serp, axs_depth_serp = plt.subplots(figsize=(8, 6),nrows=3, ncols=2)
                        axs_depth_serp[0,0].plot(bins_h2o_serp_pt_postprocess,depth_bins/1e3,label="pt postprocess")
                        axs_depth_serp[0,0].plot(bins_h2o_serp_aspect,depth_bins/1e3,label="aspect")      
                        axs_depth_serp[0,0].set_xlabel("serp (m^3/m*wt%)")

                        axs_depth_serp[0,0].set_ylabel("depth (km)")  


                        axs_depth_serp[1,0].scatter(T_aspect_serp[is_release_aspect],p_aspect_serp[is_release_aspect],c=h2o_release_aspect_serp[is_release_aspect])

                        axs_depth_serp[2,0].scatter(T_aspect_serp[is_release_pt_postprocess],p_aspect_serp[is_release_pt_postprocess],c=h2o_release_pt_postprocess_serp[is_release_pt_postprocess])

                        axs_depth_serp[0,1].scatter(T_aspect_serp[is_release_pt_postprocess],p_aspect_serp[is_release_pt_postprocess],c=h2o_release_pt_postprocess_serp[is_release_pt_postprocess])
                        axs_depth_serp[0,1].plot(T_surf,P_surf,label="pt")
                        axs_depth_serp[0,1].grid(True)
                        axs_depth_serp[1,1].plot(x_surf,y_surf,label="xy")
                        yticks = np.arange(0.0, 100, 10)

                        axs_depth_serp[1,1].set_yticks(yticks)
                        axs_depth_serp[1,1].grid(True)

                        axs_depth_serp[2,1].contour(X_crust/1.e3, (ymax_plot-Y_crust)/1.e3, comp, levels=[0.25,0.5], linewidths=0.5, colors=['blue','green'], zorder=2, alpha = 0)
                        #axs_depth_serp[2,1].plot(x_surf,y_surf,label="xy")

                        axs_depth_serp[2,1].grid(True)


                        fig_depth_serp.tight_layout()                      
                        fig_depth_serp.savefig(plot_loc+"/d05_release_serp_"+str(t)+".jpg",dpi=250)
                        plt.close()                                             

                            

                    for (i, (x_slab,y_slab,T_slab,P_slab)) in enumerate(zip(x_surf,y_surf,T_surf,P_surf)):
                        if i>0:
                            is_in_x_slice_slab = (((xx/1e3)<(x_slab+dx_slab[i-1])) & ((xx/1e3)>x_slab))
                            h2o_release_serp_x[i] = np.sum(h2o_release_basalt_track[(fluid_release_start//step)-2+ival,is_in_x_slice_slab]+
                                                              h2o_release_gabbro_track[(fluid_release_start//step)-2+ival,is_in_x_slice_slab]+
                                                              h2o_release_sediment_track[(fluid_release_start//step)-2+ival,is_in_x_slice_slab])
                            h2o_release_serp_x_tracked[i] = np.sum(h2o_release_boundwater_track[(fluid_release_start//step)-2+ival,is_in_x_slice_slab])
                            if P_slab<150e6:
                                P_slab = 150e6
                            if P_slab>5.5e9:
                                P_slab = 5.5e9
                            if T_slab<273:
                                T_slab = 273
                            serp_frac = interpolators["serpentine"]((T_slab+273,P_slab))/100
                            if (serp_frac>0) & (h2o_release_serp_x[i]>0.0) & (y_slab>y_thresh/1e3):
                                h2o_release_serp[ival] += h2o_release_serp_x[i]
                            if (serp_frac>0) & (h2o_release_serp_x_tracked[i]>0.0) & (y_slab>y_thresh/1e3):
                                h2o_release_serp_tracked[ival] += h2o_release_serp_x_tracked[i]


                    fig_full, axs_full = plt.subplots(figsize=(18, 8),nrows=1, ncols=3)
                    depth_cutoff = 2e3

                    if h2o_release_grid[:,np.array(range_released_plot)-((fluid_release_start//step)-2)].shape[1]>2:   
                        fig_serp, axs_serp = plt.subplots(figsize=(14, 24),nrows=3, ncols=1)
                        range_released_arr = np.array(range_released_full)
                        time_array_plot = time_array[((fluid_release_start//step)-2):,:]
                        axs_serp[0].set_title("Semi-quantitative instantaneous serpentinite \n"+"postprocessing particles", size=30)
                        axs_serp[0].plot(time_array_plot[h2o_release_serp_tracked>0,1]/1e6,np.cumsum(h2o_release_serp_tracked[h2o_release_serp_tracked>0]),linewidth=4,label="ASPECT dehydration")   
                        axs_serp[0].set_xlabel("time (Ma)", size=24)
                        axs_serp[0].tick_params(labelsize=20)
                        axs_serp[0].set_ylabel("fluid released \n(n_particles*wt%/(20 timesteps)", size=24)
                        

                        axs_serp[0].plot(time_array_plot[h2o_release_serp>0,1]/1e6,np.cumsum(h2o_release_serp[h2o_release_serp>0]),linewidth=4,label="particle PT-predicted dehydration")
                        axs_serp[0].plot((time_array_plot[h2o_release_aspect_mass>0,1]/1e6)[1:],np.cumsum(-np.diff(h2o_release_aspect_mass[h2o_release_aspect_mass>0])),linewidth=4,label="mesh ASPECT dehydration")  

                        axs_serp[0].legend(prop={'size': 20})
                        
                        axs_serp[1].set_title("Quantitative instantaneous serpentinite \n"+"postprocessing particles", size=30)
                        axs_serp[1].plot(time_array_plot[h2o_release_serp_tracked_mass>0,1]/1e6,np.cumsum(h2o_release_serp_tracked_mass[h2o_release_serp_tracked_mass>0]),linewidth=4,label="ASPECT dehydration")  
                        axs_serp[1].plot(time_array_plot[h2o_release_serp_mass>0,1]/1e6,np.cumsum(h2o_release_serp_mass[h2o_release_serp_mass>0]),linewidth=4,label="particle PT-predicted dehydration")  

                        axs_serp[1].set_xlabel("time (Ma)", size=24)
                        axs_serp[1].tick_params(labelsize=20)
                        axs_serp[1].set_ylabel("serpentinite \n(m^3/m*wt%)", size=24)                        
                        axs_serp[1].legend(prop={'size': 20})

                        axs_serp[2].set_title("Quantitative ASPECT serpentinite", size=30)
                        axs_serp[2].plot(time_array_plot[h2o_release_serp_aspect_contourmass>0,1]/1e6, h2o_release_serp_aspect_contourmass[h2o_release_serp_aspect_contourmass>0],linewidth=4, label="original ASPECT\n mesh")  
                        axs_serp[2].plot(time_array_plot[h2o_release_serp_aspect_mass>0,1]/1e6, h2o_release_serp_aspect_mass[h2o_release_serp_aspect_mass>0],linewidth=4, label="original ASPECT\n mesh (>0.001 serpbound H2O)")  

                        fig_serp.tight_layout()                      
                        fig_serp.savefig(plot_loc+"/d05_serpcompare_originalmesh_"+str(t)+".jpg",dpi=250)
                        plt.close()  

                        fig_val, axs_val = plt.subplots(figsize=(14, 14),nrows=1, ncols=1)
                        #axs_val.plot(data_boundwater_bin['Boundwater_Depth_Binned'],(ymax_plot-data_boundwater_bin['Boundwater_Bins'])/1e3)      

                        axs_val.set_xlabel("boundwater (m^3/m*wt%)", size=24)
                        axs_val.tick_params(labelsize=20)
                        axs_val.set_ylabel("depth (km)", size=24)                           
                        fig_val.tight_layout()                      
                        fig_val.savefig(plot_loc+"/d05_boundwater_originalmesh_"+str(t)+".jpg",dpi=250)
                        plt.close() 
                                        
                    fig_full.savefig(plot_loc+"/d05_fluid_release_"+str(t)+"_w_cutoff150MPa.jpg",dpi=250)
                    plt.close()

if __name__ == "__main__":
    mod_name = str(sys.argv[1])
    time_full = int(sys.argv[2])
    step = int(sys.argv[3])
    main(mod_name,time_full,step)
