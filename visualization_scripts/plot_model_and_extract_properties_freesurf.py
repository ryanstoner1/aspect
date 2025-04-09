#!/bin/python
import matplotlib.style as mplstyle
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import Delaunay
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import sys, os
import time as timemodule
#import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap

# set options to control processing/plotting
# for performance purposes
is_plotting_main = False
is_plotting_fluid = False
is_plotting_boundwater = True
is_load_PT = True
is_plotting_heat_flux = True
is_plotting_PT_depth_profiles = False


# get colormap
ncolors = 256
color_array = plt.get_cmap('gray')(range(ncolors))

# change alpha values
color_array[:,-1] = np.linspace(1.0,0.0,ncolors)

# create a colormap object
#map_object = mpl.colors.LinearSegmentedColormap.from_list(name='viridis',colors=color_array)
map_object = ListedColormap(color_array)
# register this new colormap with matplotlib
#plt.register_cmap(cmap=map_object) 

# # construct cmap
# my_cmap = sns.color_palette("flare", as_cmap=True)
# my_cmapPT = sns.color_palette("coolwarm", as_cmap=True)



def calculate_total_mass(x, y, rho):
    # # Filter points where rho (serp usually) is above zero
    # is_non_zero = rho > 0
    # x_non_zero = x[is_non_zero]
    # y_non_zero = y[is_non_zero]
    # rho_non_zero = rho[is_non_zero]

    # Calculate the total mass where rho is above zero
    total_mass = np.sum(rho)#np.sum(rho_non_zero)

    # Perform triangulation
    points = np.column_stack((x,y))#np.column_stack((x_non_zero, y_non_zero))
    tri = Delaunay(points)

    # Calculate the area of each triangle
    triangle_areas = 0.5 * np.abs(np.dot(tri.points[tri.simplices[:, 0]],
                                         np.cross(tri.points[tri.simplices[:, 1]] - tri.points[tri.simplices[:, 0]],
                                                  tri.points[tri.simplices[:, 2]] - tri.points[tri.simplices[:, 0]])))

    # Multiply triangle areas by corresponding rho values and sum them up
    weighted_areas = triangle_areas * rho#rho_non_zero
    total_weighted_mass = np.sum(weighted_areas)

    return total_weighted_mass, total_mass

model_name=str(sys.argv[1])             
max_time=int(sys.argv[2])   # largest number in csv_outputs/filenames
step = int(sys.argv[3])
MDD = 100e3
MinDD = 50e3
z_occ_sample = 100

# model domain
xmax=5800.e3 # m
ymax=1450.e3
# domain to plot
xmin_plot=2000.e3 # 
xmax_plot=xmax-2000e3 # 1850.e3 # 
ymin_plot=1160e3 # 675.e3; #  #
grid_res=5.0e3; grid_low_res = 70.e3; 

# ASPECT output 
csvs_loc = os.getcwd()+'/csv_outputs/'
models_loc = os.getcwd()+'/'
stats_file = ''.join([models_loc,str(model_name),'/statistics'])
model_output_dt  = 1 # output dt as set in ASPECT .prm file (for getting the dimensional time)

# Agard et al., 2018 dataset
#file_path = '/home/rstoner/Documents/agard18_data/Agard_2018_chopped.txt'

# Load data from file
# agard18_data = np.array([0])#;np.loadtxt(file_path)

# # Extract columns into separate arrays
# P_agard18 = agard18_data[:, 1]
# T_agard18 = agard18_data[:, 0]
# t_prime_agard18 = agard18_data[:, 2]

# where to put the plots
plot_loc = ''.join(
    [os.getcwd(),'/plots/', str(model_name)])
if not os.path.exists(plot_loc):
    os.mkdir(plot_loc)

# where to put kinematics file
kinematics_loc = ''.join(
    [os.getcwd()+'/kinematics/'])
if not os.path.exists(kinematics_loc):
    os.mkdir(kinematics_loc)
kinematics_file  = ''.join([kinematics_loc,model_name,'.txt'])
kinematics_serp_file  = ''.join([kinematics_loc,model_name,'_serp.txt'])
kinematics_freefluid_file  = ''.join([kinematics_loc,model_name,'_freefluid.txt'])
kinematics_epstein_file  = ''.join([kinematics_loc,model_name,'_epstein.txt'])
kinematics = np.zeros((max_time,16)) # t', time [Ma], vsp [cm/yr], vt [cm/yr], vc [cm/yr], dip [degs], dip @ 400 km [degs], slab depth [km], decoupling x-location [km], decoupling depth [km], T_surf (C), T_moho (C), sample isotherm (km), serpentinite, serpentinite (<120 km)
kinematics_serp = np.zeros((max_time,int((ymax-ymin_plot)/grid_res)))
kinematics_freefluid = np.zeros((max_time,int((ymax-ymin_plot)/grid_res)))
kinematics_epstein = np.zeros((max_time,4))

is_first_iter = True

# initial figure creation
fig_opt, axs_opt = plt.subplots(nrows=3, ncols=2, figsize=(10, 5))

for time in range(0,max_time,step):
    
    csv_filename=''.join([csvs_loc,model_name,'/full.',str(time),'.gzip'])

    timestr = str(time)
    if len(timestr)<4:
        timestr = "0"*(4-len(timestr))+timestr
    plotname=''.join([plot_loc,'/',timestr,'_centerzoom.png'])
    plotname_pdf=''.join([plot_loc,'/',timestr,'.pdf'])

    # get dimensional time
    f=open(stats_file)
    lines=f.readlines()
    num_header_lines = len(list(filter(lambda line: line.startswith("#"),lines))) # num header lines in stats_files (for getting the dimensional time)
    stats_line_num = num_header_lines + (time * model_output_dt)

    line=lines[stats_line_num]
    time_dim=float(line.split()[1])/1.e6 # Myrs
    print("%.0f: t = %.1f Ma" % (time,time_dim))
    if is_first_iter:
        # Load the Parquet file into a pandas DataFrame
        df = pd.read_parquet(csv_filename)

        # Access the column names
        header_terms = df.columns.tolist()
        if 'velocity:0' in header_terms:
            vx_col = header_terms.index("velocity:0")
        if "velocity:1" in header_terms:
            vz_col = header_terms.index("velocity:1")
        if "p" in header_terms:
            p_col = header_terms.index("p")
        if "T" in header_terms:
            T_col = header_terms.index("T")
        if "Points:0" in header_terms:
            x_col = header_terms.index("Points:0")
        if "Points:1" in header_terms:
            y_col = header_terms.index("Points:1")
        if "ocrust_init" in header_terms:
            ocrust_col = header_terms.index("ocrust_init")
        if "ocrust" in header_terms:
            ocrust_col2 = header_terms.index("ocrust")
        if "ccrust" in header_terms:
            ccrust_col = header_terms.index("ccrust")
        if "serp" in header_terms:
            serp_col = header_terms.index("serp")
        if "viscosity" in header_terms:
            visc_col = header_terms.index("viscosity")
        if "density" in header_terms:
            rho_col = header_terms.index("density")
        if "strain_rate" in header_terms:
            strainrate_col = header_terms.index("strain_rate")
        if "dislocation_viscosity" in header_terms:
            disl_visc_col = header_terms.index("dislocation_viscosity")
        if "boundwater" in header_terms:
            boundwater_col = header_terms.index("boundwater")
        if "freefluid" in header_terms:
            freefluid_col = header_terms.index("freefluid")
        if "melth2o" in header_terms:
            melth2o_col = header_terms.index("melth2o")
        if "stress_second_invariant" in header_terms:
            stressII_col = header_terms.index("stress_second_invariant")
        if "gabbro" in header_terms:
            gabbro_col = header_terms.index("gabbro")
        if "gabbro_init" in header_terms:
            gabbro_init_col = header_terms.index("gabbro_init")        
        if "vertical_heat_flux" in header_terms:
            vertical_heat_flux_col = header_terms.index("vertical_heat_flux")    


    # load relevant model file
    #model_data  = np.loadtxt(csv_filename, delimiter=',', skiprows=1)
    model_data = df.values

    # extract surface profile
    surf_prof = model_data[model_data[:,y_col] == ymax]     # profile at y = ymax (surface)
    surf_prof = surf_prof[surf_prof[:,x_col].argsort()]     # sort by x

    # extract mid-plate profile
    plate_prof_loc = ymax - 10e3                           # 20 km depth
    plate_prof = model_data[model_data[:,y_col] < (plate_prof_loc+4.e3)] 
    plate_prof = plate_prof[plate_prof[:,y_col] > (plate_prof_loc-4.e3)] 
    plate_prof = plate_prof[plate_prof[:,x_col].argsort()] # sort by x

    # create grid to interpolate stuff onto (for plotting)
    
    x_low = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_res))
    y_low =  np.linspace(ymin_plot,ymax,int((ymax-ymin_plot)/grid_res))
    X_low, Y_low = np.meshgrid(x_low,y_low) 
    # low res grid for plotting velocities
    x_vels = np.linspace(xmin_plot,xmax_plot,int((xmax_plot-xmin_plot)/grid_low_res))
    y_vels =  np.linspace(ymin_plot,ymax,int((ymax-ymin_plot)/grid_low_res))
    X_vels, Y_vels = np.meshgrid(x_vels,y_vels) 

    print("interpolating model outputs to regular grid...")

    t = timemodule.time()
    P      =  griddata((model_data[:,x_col], model_data[:,y_col]), 
                    model_data[:,p_col],    (X_low, Y_low), method='nearest')
    T      =  griddata((model_data[:,x_col], model_data[:,y_col]), 
                    model_data[:,T_col]-273,    (X_low, Y_low), method='nearest') # fast nearest; best cubic
    visc = griddata((model_data[:, x_col], model_data[:, y_col]),
                    model_data[:, visc_col], (X_low, Y_low), method='nearest') # best linear
    density = griddata((model_data[:, x_col], model_data[:, y_col]),
                       model_data[:, rho_col], (X_low, Y_low), method='nearest')
    vx = griddata((model_data[:, x_col], model_data[:, y_col]),
                  model_data[:, vx_col],   (X_vels, Y_vels), method='nearest')
    vz = griddata((model_data[:, x_col], model_data[:, y_col]),
                  model_data[:, vz_col],   (X_vels, Y_vels), method='nearest')
    ocrust_init = griddata((model_data[:, x_col], model_data[:, y_col]),
                     model_data[:, ocrust_col],   (X_low, Y_low), method='nearest') # best cubic
    ocrust = griddata((model_data[:, x_col], model_data[:, y_col]),
                     model_data[:, ocrust_col2],   (X_low, Y_low), method='nearest') # best cubic 
    gabbro = griddata((model_data[:, x_col], model_data[:, y_col]),
                     model_data[:, gabbro_col],   (X_low, Y_low), method='nearest') # best cubic
    gabbro_init = griddata((model_data[:, x_col], model_data[:, y_col]),
                     model_data[:, gabbro_init_col],   (X_low, Y_low), method='nearest') # best cubic 
    ccrust = griddata((model_data[:, x_col], model_data[:, y_col]),
                     model_data[:, ccrust_col],   (X_low, Y_low), method='nearest') # best cubic 
    serp = griddata((model_data[:, x_col], model_data[:, y_col]),
                     model_data[:, serp_col],   (X_low, Y_low), method='nearest') # best cubic 
    boundwater = griddata((model_data[:, x_col], model_data[:, y_col]),
                    model_data[:, boundwater_col],   (X_low, Y_low), method='nearest')  # best cubic
    freefluid = griddata((model_data[:, x_col], model_data[:, y_col]),
                    model_data[:, freefluid_col],   (X_low, Y_low), method='nearest')
    # melth2o = griddata((model_data[:, x_col], model_data[:, y_col]),
    #             model_data[:, melth2o_col],   (X_low, Y_low), method='nearest')
    stressII = griddata((model_data[:, x_col], model_data[:, y_col]),
                    model_data[:, stressII_col],   (X_low, Y_low), method='nearest')
    vertical_heat_flux = griddata((model_data[:, x_col], model_data[:, y_col]),
                    model_data[:, vertical_heat_flux_col],   (X_low, Y_low), method='nearest')


    ocrust_all = ocrust_init + ocrust
    strainrate = griddata((model_data[:, x_col], model_data[:, y_col]),
                     model_data[:, strainrate_col],   (X_low, Y_low), method='nearest') # best cubic
    disl_visc = griddata((model_data[:, x_col], model_data[:, y_col]),
                     model_data[:, visc_col],   (X_low, Y_low), method='nearest') # best linear
    elapsed = timemodule.time() - t
    print("interp duration is:"+str(elapsed)+"\n")

    # get trench location
    # trench_x = 0
    # for i in range(len(plate_prof)):
    #     if (((plate_prof[i,ocrust_col] + plate_prof[i,ocrust_col2]) > 0.25)  and plate_prof[i,x_col] > trench_x):
    #         trench_x = plate_prof[i,x_col]

    ind_x = (plate_prof[:,x_col]>1500e3) & (plate_prof[:,x_col]<(xmax-1500e3))
    ind_trench_x_alt = np.argmax((plate_prof[:,strainrate_col])[ind_x])
    trench_x = plate_prof[:,x_col][ind_x][ind_trench_x_alt]
    #print("trench_x alt:" + str(trench_x_alt/1e3) + "\n")
    print("trench_x:" + str(trench_x/1e3))
    # if trench_x == 0:
    #     trench_x = 0#np.nan

    tcur = timemodule.time()
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

    # get slab depths from temperature contour
    tc = plt.contour(X_low/1.e3, (ymax-Y_low)/1.e3, T, levels=[1000])
    plt.xlim(3320,3460)
    plt.ylim(50,120)
    # plt.savefig("temp_slab.png")
    # plt.savefig(plot_loc+"/Tslabcompare_"+timestr+".png",bbox_inches="tight", format="png",dpi=400)
    slab_depth = 0
    for d in range(len(tc.get_paths())):
        p = tc.get_paths()[d]
        x = p.vertices[:,0]; z = p.vertices[:,1]
        for j in range(len(z)):
            if z[j] > slab_depth:
                slab_depth = z[j]

    # get deep slab dip (@ 250 km) from temperature contour
    x_shall = 0; x_deep = 0;
    x_shall250 = 0; x_deep250 = 0;
    z_shall250 = 0; z_deep250 = 0;
    for d in range(len(tc.get_paths())): 
        p = tc.get_paths()[d]
        x = p.vertices[:,0]; z = p.vertices[:,1]
        for j in range(len(x)):
            if x[j] > x_shall and z[j] < (110.+(grid_res/1.e3)) and z[j] > (110.-(grid_res/1.e3)):
                x_shall = x[j]; z_shall = z[j]
            if x[j] > x_deep and  z[j] < (140.+(grid_res/1.e3)) and z[j] > (140.-(grid_res/1.e3)):
                x_deep = x[j];  z_deep = z[j]
            if x[j] > x_shall250 and z[j] < (235.+(grid_res/1.e3)) and z[j] > (235.-(grid_res/1.e3)):
                x_shall250 = x[j]; z_shall250 = z[j]
            if x[j] > x_deep250 and  z[j] < (265.+(grid_res/1.e3)) and z[j] > (265.-(grid_res/1.e3)):
                x_deep250 = x[j];  z_deep250 = z[j]                
    if x_deep > 0:
        dip_deep = np.rad2deg(np.arctan((z_deep-z_shall)/(x_deep-x_shall)))
    else:
        dip_deep = 0

    if x_deep250 > 0:
        dip_deep250 = np.rad2deg(np.arctan((z_deep250-z_shall250)/(x_deep250-x_shall250)))
    else:
        dip_deep250 = 0

    slab_surf_past_trench = np.zeros((0,2))
    for d in range(len(tc.get_paths())): 
            p = tc.get_paths()[d]
            x = p.vertices[:,0]; z = p.vertices[:,1]
            xz = np.vstack([x,z]).T 
            slab_surf_past_trench = np.append(slab_surf_past_trench,xz,axis = 0)
            # slab_surf_past_trench[:,0] = x
            # slab_surf_past_trench[:,1] = z
    slab_surf_past_trench = slab_surf_past_trench[slab_surf_past_trench[:,0]>(trench_x/1e3)]


    tf = plt.contour(X_low/1.e3, (ymax-Y_low)/1.e3, visc, levels=[5e20])    
    plt.xlim(3220,3460)
    plt.ylim(0,120)

    plt.xlim(3220,(9.9*xmax_plot/1e3+trench_x/1e3)/10)
    plt.ylim(0,120)

    occ = plt.contour(X_low/1e3,(ymax-Y_low)/1e3,ocrust_all,levels=[0.1])

    plt.xlim(3220,3460)
    plt.ylim(0,120)

    oc_surf_past_trench = np.zeros((0,2))

    x_occ_temp = []
    x_occ_surf = np.nan
    z_occ_surf = np.nan

    x_occ_moho = np.nan
    z_occ_moho = np.nan

    for d in range(len(occ.get_paths())): 
            p = occ.get_paths()[d]
            x = p.vertices[:,0]; z = p.vertices[:,1]
            xz = np.vstack([x,z]).T            
            oc_surf_past_trench = np.append(oc_surf_past_trench,xz,axis = 0)
            
            for j in range(len(x)):
                if j>0:
                    is_z_in = ((z_occ_sample>=z[j-1]) & (z_occ_sample<=z[j])) | ((z_occ_sample<=z[j-1]) & (z_occ_sample>=z[j]))
                    if (is_z_in):
                        t = ((z_occ_sample)-z[j])/(z[j-1]-z[j])
                        x_occ_temp.append(t*x[j-1] + (1-t)*x[j])

    if len(x_occ_temp)<2:
        x_occ_temp = np.array([np.nan,np.nan])
    ind_z_occ_sample = np.argmin(abs((ymax-Y_low[:,0])/1000-z_occ_sample))
    x_occ_moho = min(x_occ_temp)
    x_occ_surf = max(x_occ_temp)
    ind_x_occ_moho = np.argmin(abs((X_low[ind_z_occ_sample,:])/1000 - x_occ_moho))
    ind_x_occ_surf = np.argmin(abs((X_low[ind_z_occ_sample,:])/1000 - x_occ_surf))
    T_moho = T[ind_z_occ_sample,ind_x_occ_moho]
    T_surf = T[ind_z_occ_sample,ind_x_occ_surf]


    oc_surf_past_trench = oc_surf_past_trench[oc_surf_past_trench[:,0]>(trench_x/1e3)]

    Z_OP = 1e9
    for d in range(len(tf.get_paths())):
        p = tf.get_paths()[d]
        x = p.vertices[:,0]; z = p.vertices[:,1]
        plt.plot(x,z)
        #plt.savefig(plot_loc+"/dval_"+timestr+"dn"+str(d)+".png",bbox_inches="tight", format="png",dpi=300)
        for k in range(len(z)):
            if x[k] > (trench_x + 195.e3)/1.e3 and x[k] < (trench_x + 205.e3)/1.e3 and z[k] < Z_OP:
                Z_OP = z[k]
    print("OP thickness = %.0f km" % Z_OP)

    # find decoupling depth
    x_decoupling = 0
    z_decoupling = 0
    z_slabsurf_total = 0
    z_slabsurf_cand = 1e99
    z_slabsurf_decoupling = 1e98
    for d in range(len(tc.get_paths())):
        
        p = tc.get_paths()[d]
        x = p.vertices[:,0]; z = p.vertices[:,1]

        xop_edge = x[x>(9*xmax_plot/1e3+trench_x/1e3)/10]
        zop_edge = z[x>(9*xmax_plot/1e3+trench_x/1e3)/10]
        if len(xop_edge>=1) and (np.min(zop_edge)<MDD/1e3): # if contour is not in overriding plate don't bother
            z_decoupling_alt = np.max(z)
            # z_decoupling_alt = (z_decoupling_alt+np.partition(z.flatten(), -2)[-2])/2
            iz_decoupling_alt = np.argmax(z)
            x_decoupling_alt = x[iz_decoupling_alt]           
            for h in range(len(z)):  ## loop through viscosity contour points
                dist_slabsurf_cur = 1e99
                xcont = x[h]; zcont = z[h]
                z_slabsurf_max = 0

                x_slabsurf = oc_surf_past_trench[:,0]
                z_slabsurf = oc_surf_past_trench[:,1]
                # x_slabsurf = slab_surf_past_trench[:,0]
                # z_slabsurf = slab_surf_past_trench[:,1]
                dist =  np.sqrt((xcont-x_slabsurf)**2+(zcont-z_slabsurf)**2)  ## distance between each slab surf pt and 5e22 Pas viscosity contour
                distmin_new = min(dist)
                distminind = np.argmin(dist)
                horiz_dist = xcont-x_slabsurf
                vert_dist = np.abs(zcont-z_slabsurf)

                #if vert_dist < 0.75 and horiz_dist > 3:# and horiz_dist < 8 and z_slabsurf > z_slabsurf_max and z_slabsurf > 0.75*Z_OP and z_slabsurf < 1.6*Z_OP and z_slabsurf <= (MDD - 1.) and zcont <= (MDD - 1.):
                #is_closest = (vert_dist<3) & (horiz_dist>3.5) & (horiz_dist<8) & (z_slabsurf <= (1.5*MDD/1e3 - 1.)) & (zcont <= (1.5*MDD/1e3 - 1.))
                is_closest = (dist<9) & (z_slabsurf <= (2*MDD/1e3 - 1.)) 
                
                x_slabsurf_max_raw = x_slabsurf[is_closest]
                z_slabsurf_max_raw = z_slabsurf[is_closest]
                if len(z_slabsurf_max_raw)>0:
                    dist_close = dist[is_closest]
                    z_decoupling_cand = z_slabsurf_max_raw[-1]
                    z_slabsurf_cand = zcont
                    x_decoupling_cand = x_slabsurf_max_raw[-1] 
                    x_slabsurf_cand = xcont
                    dist_slabsurf_cnad = min(dist_close)

                if z_slabsurf_cand < z_slabsurf_decoupling: # assuming monotonic slab
                    z_slabsurf_decoupling = z_slabsurf_cand
                    z_decoupling = z_decoupling_cand
                    x_decoupling = x_decoupling_cand



    # find isotherm depth
    isotherm_val = 600
    x_isotherm = 0
    z_isotherm = 0
    z_slabsurf_total = 0
    z_slabsurf_cand = 1e99
    z_slabsurf_isotherm = 1e98
    tc_sample = plt.contour(X_low/1.e3, (ymax-Y_low)/1.e3, T, levels=[isotherm_val])
    for d in range(len(tc_sample.get_paths())):
        
        p = tc_sample.get_paths()[d]
        x = p.vertices[:,0]; z = p.vertices[:,1]

        xop_edge = x[x>(9*xmax_plot/1e3+trench_x/1e3)/10]
        zop_edge = z[x>(9*xmax_plot/1e3+trench_x/1e3)/10]
        if len(xop_edge>=1) and (np.min(zop_edge)<MDD/1e3): # if contour is not in overriding plate don't bother
            z_isotherm_alt = np.max(z)
            # z_isotherm_alt = (z_isotherm_alt+np.partition(z.flatten(), -2)[-2])/2
            iz_isotherm_alt = np.argmax(z)
            x_isotherm_alt = x[iz_isotherm_alt]           
            for h in range(len(z)):  ## loop through viscosity contour points
                dist_slabsurf_cur = 1e99
                xcont = x[h]; zcont = z[h]
                z_slabsurf_max = 0

                x_slabsurf = oc_surf_past_trench[:,0]
                z_slabsurf = oc_surf_past_trench[:,1]
                dist =  np.sqrt((xcont-x_slabsurf)**2+(zcont-z_slabsurf)**2)  ## distance between each slab surf pt and 5e22 Pas viscosity contour
                distmin_new = min(dist)
                distminind = np.argmin(dist)
                horiz_dist = xcont-x_slabsurf
                vert_dist = np.abs(zcont-z_slabsurf)

                #if vert_dist < 0.75 and horiz_dist > 3:# and horiz_dist < 8 and z_slabsurf > z_slabsurf_max and z_slabsurf > 0.75*Z_OP and z_slabsurf < 1.6*Z_OP and z_slabsurf <= (MDD - 1.) and zcont <= (MDD - 1.):
                #is_closest = (vert_dist<3) & (horiz_dist>3.5) & (horiz_dist<8) & (z_slabsurf <= (1.5*MDD/1e3 - 1.)) & (zcont <= (1.5*MDD/1e3 - 1.))
                is_closest = (dist<6) & (z_slabsurf <= (1.5*MDD/1e3 - 1.)) 
                
                x_slabsurf_max_raw = x_slabsurf[is_closest]
                z_slabsurf_max_raw = z_slabsurf[is_closest]
                if len(z_slabsurf_max_raw)>0:
                    dist_close = dist[is_closest]
                    z_isotherm_cand = z_slabsurf_max_raw[-1]
                    z_slabsurf_cand = zcont
                    x_isotherm_cand = x_slabsurf_max_raw[-1] 
                    x_slabsurf_cand = xcont
                    dist_slabsurf_cnad = min(dist_close)

                if z_slabsurf_cand < z_slabsurf_isotherm: # assuming monotonic slab
                    z_slabsurf_isotherm = z_slabsurf_cand
                    z_isotherm = z_isotherm_cand
                    x_isotherm = x_isotherm_cand

    assert "x_decoupling" in locals(), "x_decoupling not found successfully!"
    print("DD = %.0f km" % z_decoupling)
    print("x DD = %.0f km" % x_decoupling)
    print("altDD = %.1f km" % z_decoupling_alt)
    print("altx DD = %.1f km" % x_decoupling_alt)
    print("vc = %.1f cm/yr" % vc)

    serptotal = np.diff(x_low[0:2])[0]*np.diff(y_low[0:2])[0]*sum(sum(serp))
    freefluidtotal = np.diff(x_low[0:2])[0]*np.diff(y_low[0:2])[0]*sum(sum(freefluid))
    boundwatertotal = np.diff(x_low[0:2])[0]*np.diff(y_low[0:2])[0]*sum(sum(boundwater))
    melth2ototal = 0 #np.diff(x_low[0:2])[0]*np.diff(y_low[0:2])[0]*sum(sum(melth2o))
    # serptotal = np.diff(x_low[0:2])[0]*np.diff(y_low[0:2])[0]*sum(sum(serp))


    serptotal120 = np.diff(x_low[0:2])[
        0]*np.diff(y_low[0:2])[0]*sum(serp[((ymax-Y_low) < 120e3) & (T < 600)])

    y_above_oc = np.max((ymax-Y_low)*((ocrust+ocrust_init)>0.05),axis=0)
    is_above_oc = (ymax-Y_low)<y_above_oc
    is_serpentinizable = np.diff(x_low[0:2])[
        0]*np.diff(y_low[0:2])[0]*np.ones_like(T)*(T<600)*((ocrust_init+ccrust+ocrust+serp+gabbro+gabbro_init)<0.1)*(X_low>trench_x)*(X_low<(trench_x+150e3))*is_above_oc

    kinematics[time,:] = time, time_dim, vsp, vop, vc, dip_deep, dip_deep250, slab_depth, x_decoupling, z_decoupling, sum(sum(is_serpentinizable)), T_surf, T_moho, z_isotherm, serptotal, serptotal120
    kinematics_serp[time,:] = np.sum(serp*(T<600)*((ymax-Y_low)<200e3),axis=1)
    kinematics_freefluid[time,:] = np.sum(freefluid*((ymax-Y_low)<200e3),axis=1)
    kinematics_epstein[time,:] = time, time_dim, vc, dip_deep250 # time, time_dim, serptotal, freefluidtotal, boundwatertotal, melth2ototal   #

    

    

    # ################# plotting visc, density, temperature fields #######################
    if is_plotting_main:
        t = timemodule.time()
        fig=plt.figure(figsize=(6, 3))
        gs=GridSpec(2,1) 

        print("plotting temp field...")
        ax1=fig.add_subplot(gs[0,0], aspect=1)
        T_plot = ax1.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, T,cmap=cm.get_cmap('RdBu_r'),levels=np.linspace(0,1575,201),extend='neither')
        ax1.contour( X_low/1.e3, (ymax-Y_low)/1.e3, T, levels=[500,750,1000,1250],    linewidths=0.3, colors='white',zorder=2)   
        ax1.set_ylim([(ymax-ymin_plot)/1.e3,0])   
        # ax1.set_xlim([xmin_plot/1e3,xmax_plot/1e3])   
        ax1.tick_params(direction='out',length=2, labelsize=7)
        #ax1.tick_params(labelbottom=False)    
        ax1.annotate(''.join(['t = ',str("%.1f" % (time_dim)),' Myr']), xy=(0.025,0.15), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=9.5,color='white')       
       
        cbar1 = plt.colorbar(T_plot, orientation='vertical',ticks=[0,350,700,1050,1400])
        cbar1.ax.tick_params(labelsize=6.5)
        cbar1.set_label("T  [$^\circ$C]",size=7.5)

        # print("plotting density field...")
        # ax2=fig.add_subplot(gs[1,0], aspect=1)
        # density_plot = ax2.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, density, cmap=cm.get_cmap(my_cmap),levels=np.linspace(2800,3600,101),extend='neither')
        # occ_plot = ax2.contour(X_low/1e3,(ymax-Y_low)/1e3,ocrust_init,levels=[0.5],colors=["crimson"],linewidths=0.75)
        # ax2.set_ylim([(ymax-ymin_plot)/1.e3,0])   
        # ax2.set_xlim([xmin_plot/1e3,xmax_plot/1e3])   
        # ax2.tick_params(direction='out',length=2, labelsize=6)
        # cbar2 = plt.colorbar(density_plot, orientation='vertical',ticks=[2800,3000,3200,3400])
        # cbar2.ax.tick_params(labelsize=5.5)
        # cbar2.set_label("density  [kg/m$^3$]",size=7.5)

        print("plotting viscosity field...")
        ax3=fig.add_subplot(gs[1,0], aspect=1)
        visc_plot = ax3.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, np.log10(visc), cmap=cm.get_cmap('plasma_r'),levels=np.linspace(18,24,501),extend='max')
        contour_hi = 0.08
        contour_low = 0.04

        if np.max(serp)>=contour_low:
            serp_plot0 = ax3.contour(X_low/1e3,(ymax-Y_low)/1e3,serp,levels=[contour_hi],colors=["springgreen"],linewidths=0.75)
        if np.max(serp) >= contour_hi:
            serp_plot1 = ax3.contour(X_low/1e3,(ymax-Y_low)/1e3,serp,levels=[contour_low],colors=["green"],alpha=0.7,linewidths=0.75)
        ax3.contour( X_low/1.e3, (ymax-Y_low)/1.e3, T, levels=[600],    linewidths=0.3, colors='white',zorder=2)   
        ax3.set_ylim([(ymax-ymin_plot)/1.e3,0])   
        # ax3.set_xlim([xmin_plot/1e3,xmax_plot/1e3])   
        ax3.tick_params(direction='out',length=2, labelsize=6)
        cbar3 = plt.colorbar(visc_plot, orientation='vertical',ticks=[19,20,21,22,23,24])
        cbar3.ax.tick_params(labelsize=5.5)
        cbar3.set_label("log(${\eta}$)  [Pa.s]",size=7.5)
        # vector plotting
        vel_plot_thresh = 0.05 # don't plot velocity vectors smaller than this (cm/yr)
        vx[100.*np.sqrt(vx**2 + vz**2)<vel_plot_thresh]=float('nan')
        vz[100.*np.sqrt(vx**2 + vz**2)<vel_plot_thresh]=float('nan')
        vel_vects = ax3.quiver(X_vels/1.e3,(ymax-Y_vels)/1.e3,vx*100,vz*100,color='black',scale=100,width=0.0025) # scale=150, width=0.0015
        ax3.quiverkey(vel_vects, 0.22, 0.1, 5, '5 cm/yr', labelpos='W',fontproperties={'size': '7'},color='white',labelcolor='white')

        print("saved figure to %s..." % plotname)
        fig.tight_layout()
        plt.savefig(plotname, bbox_inches='tight', format='png', dpi=600)
    

    ##############################################################################################
    ocfrac = 0.9
    isocrust = (ocrust_init>=ocfrac) | (ocrust>=ocfrac)
    if (len(density[isocrust])<1):
        ocfrac = 0.1
        isocrust = (ocrust_init>=ocfrac) | (ocrust>=ocfrac)
    mindens_oc = np.min(density[isocrust])
    maxdens_oc = np.max(density[isocrust])

    DD_press = np.max(P[(ymax-Y_low)<MDD])

    ##############################################################################################
    #input_file="upper_extrusives_AKA_hydrated_basalt/032323_ue90_bound_H2O.tab"  # "Pelagic_sediment/pelagic_bound_h2o.tab" # 
    #input_file="pseudosections/MORB/morb_green_finev3_bound_H2O.tab"
    input_file_gabbro=os.getcwd()+"/pseudosections/MORB/gabbro_aspect_H2O.txt"
    input_file_basalt=os.getcwd()+"/pseudosections/MORB/morb_green_aspect_H2O.txt"
    #input_file="pseudosections/Pelagic_sediment/pelagic_bound_H2O_volume.txt"
    # Read the input file
    data_gabbro = np.genfromtxt(input_file_gabbro, skip_header=0, usecols=(0, 1, 2))
    data_basalt = np.genfromtxt(input_file_basalt, skip_header=0, usecols=(0, 1, 2))

    # Separate the columns
    pressure_h2o_gabbro = (data_gabbro[:, 1]).reshape((250,250))
    temperature_h2o_gabbro = (data_gabbro[:, 0]).reshape((250,250))
    xh2o_gabbro = (data_gabbro[:, 2]).reshape((250,250))

   

    pressure_h2o_basalt = (data_basalt[:, 1]).reshape((250,250))
    temperature_h2o_basalt = (data_basalt[:, 0]).reshape((250,250))
    xh2o_basalt = (data_basalt[:, 2]).reshape((250,250))
    interp_basalt = RegularGridInterpolator((temperature_h2o_basalt[0,:],pressure_h2o_basalt[:,0]*100*1e3), xh2o_basalt,method='nearest')


    oc_fluid_release = np.zeros_like(model_data[:, ocrust_col])
    ids_oc = np.where(((model_data[:, ocrust_col] + model_data[:, ocrust_col2])>1e-5) & (model_data[:, p_col]<np.max(pressure_h2o_basalt[:,0]*100*1e3)) & (model_data[:,y_col]<(ymax-10e3)) & (model_data[:,x_col]>(trench_x)))
    
    elapsed = timemodule.time() - tcur
    print("first plotting duration:"+str(elapsed)+"\n")

    # for (id0_oc) in ids_oc[0]:
    #     pval = model_data[id0_oc, p_col]
    #     Tval = model_data[id0_oc, T_col]
    #     if Tval>max(data_basalt[:, 0]):
    #         Tval = max(data_basalt[:, 0])
    #     if pval<min(data_basalt[:, 1]*1e2*1e3):
    #         pval = min(data_basalt[:, 1]*1e2*1e3)

    #     fluid_out = model_data[id0_oc,boundwater_col]/(model_data[id0_oc,ocrust_col] + model_data[id0_oc,ocrust_col2]) - interp_basalt((Tval,pval))/1e2
    #     if fluid_out>0:
    #         oc_fluid_release[id0_oc] += fluid_out

    elapsed = timemodule.time() - tcur
    print("second plotting duration:"+str(elapsed)+"\n")

    oc_fluid = griddata((model_data[:, x_col], model_data[:, y_col]),
                    oc_fluid_release,   (X_low, Y_low), method='nearest')



    if time==0:
        h2o_release = np.zeros([len(y_low),1+max_time//step])
        h2o_release_y = np.sum(oc_fluid,axis=1)
        h2o_release[:,time//step] = h2o_release_y
    else: 

        h2o_release_y = np.sum(oc_fluid,axis=1)
        h2o_release[:,time//step] = h2o_release_y

    input_file_serp=os.getcwd()+"/pseudosections/Serpentinite/serp_sat_niu_aspect_H2O.txt"

    # Read the input file
    data_serp = np.genfromtxt(input_file_serp, skip_header=0, usecols=(0, 1, 2))

    # Separate the columns
    pressure_h2o_serp = (data_serp[:, 1]).reshape((250,250))
    temperature_h2o_serp = (data_serp[:, 0]).reshape((250,250))
    xh2o_serp = (data_serp[:, 2]).reshape((250,250))



    strnum = str(time)
    if len(strnum)<4:
        strnum = '0'*(4-len(strnum))+strnum
    
    if is_load_PT:
        filename = "/pt_"+strnum+".txt"
        print("loading: "+filename+"\n")
        PTdata = np.loadtxt(plot_loc+filename,skiprows=1)
        Psurfload = PTdata[:,1]
        Tsurfload = PTdata[:,2]
    

        if time != (max_time-1):
            strnum2 = str(time)
            if len(strnum2) < 4:
                strnum2 = '0'*(4-len(strnum2))+strnum2
            filename2 = "/pt_"+strnum2+".txt"
            PTdata2 = np.loadtxt(plot_loc+filename2, skiprows=1)
            Psurfload2 = PTdata2[:, 1]
            Tsurfload2 = PTdata2[:, 2]

        ocfrac = 0.99
        fig3 = plt.figure(figsize=(10,5))
        gs = gridspec.GridSpec(3,2)

        axL1 = fig3.add_subplot(gs[:,0])
        axR1 = fig3.add_subplot(gs[0,1])
        axR2 = fig3.add_subplot(gs[1,1])
        axR3 = fig3.add_subplot(gs[2,1])

        h2o_plot = axL1.contourf(temperature_h2o_basalt-273,pressure_h2o_basalt*100/1e6,xh2o_basalt,levels=np.linspace(0,8,101),extend="both")
        axL1.contour(temperature_h2o_basalt-273,pressure_h2o_basalt*100/1e6,xh2o_serp,levels=np.linspace(0,12,3),extend="both")
        axL1.plot(Tsurfload,Psurfload/1e9)
        axL1.set_xlabel("temperature [$^{\circ}$C]")
        axL1.tick_params(direction='out', length=2, labelsize=12)
        axL1.set_ylabel("pressure [GPa]")
        axL1.set_title('pseudosection and PT paths', fontstyle='italic')
        axL1.set_xlim([0,np.max(temperature_h2o_basalt-273)])
        axL1.set_ylim([0,8])

        cbar3 = plt.colorbar(h2o_plot,orientation='vertical',ticks=[0,2,4])
        cbar3.ax.tick_params(labelsize=11.5)
        cbar3.set_label("$X_{H_{2}O}$",size=12.5)

        visc_plotR1 = axR1.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, np.log10(visc), cmap=cm.get_cmap('plasma_r'),levels=np.linspace(18,24,501),extend='max')
        serp_plot0 = axR1.contour(X_low/1e3,(ymax-Y_low)/1e3,serp,levels=[0.08],colors=["springgreen"],linewidths=0.75)
        serp_plot1 = axR1.contour(X_low/1e3,(ymax-Y_low)/1e3,serp,levels=[0.05],colors=["green"],alpha=0.7,linewidths=0.75)
        serp_plot2 = axR1.contour(X_low/1e3,(ymax-Y_low)/1e3,serp,levels=[0.02],colors=["blue"],alpha=0.7,linewidths=0.75)
        axR1.contour( X_low/1.e3, (ymax-Y_low)/1.e3, T, levels=[600],    linewidths=0.3, colors='white',zorder=2)   
        axR1.set_ylim([(ymax-ymin_plot)/1.e3,0])   
        axR1.set_xlim([xmin_plot/1e3,xmax_plot/1e3])   
        axR1.tick_params(direction='out',length=2, labelsize=6)
        cbar3 = plt.colorbar(visc_plotR1, orientation='vertical',ticks=[19,20,21,22,23,24])
        cbar3.ax.tick_params(labelsize=5.5)
        cbar3.set_label("log(${\eta}$)  [Pa.s]",size=7.5)

        vel_vectsR1 = axR1.quiver(X_vels/1.e3,(ymax-Y_vels)/1.e3,vx*100,vz*100,color='black',scale=100,width=0.0025) # scale=150, width=0.0015
        axR1.quiverkey(vel_vectsR1, 0.22, 0.1, 5, '5 cm/yr', labelpos='W',fontproperties={'size': '7'},color='white',labelcolor='white')

        # R2
        T_plotR2 = axR2.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, T,cmap=cm.get_cmap('RdBu_r'),levels=np.linspace(0,1575,201),extend='neither')
        axR2.contour( X_low/1.e3, (ymax-Y_low)/1.e3, T, levels=[500,750,1000,1250],    linewidths=0.3, colors='white',zorder=2)   
        axR2.set_ylim([(ymax-ymin_plot)/1.e3,0])   
        axR2.set_xlim([xmin_plot/1e3,xmax_plot/1e3])   
        axR2.tick_params(direction='out',length=2, labelsize=6)    
        axR2.annotate(''.join(['t = ',str("%.1f" % (time_dim)),' Myr']), xy=(0.025,0.15), xycoords='axes fraction',verticalalignment='center',horizontalalignment='left',fontsize=9.5,color='white')       
                
        cbar2 = plt.colorbar(T_plotR2, orientation='vertical',ticks=[0,350,700,1050,1400])
        cbar2.ax.tick_params(labelsize=5.5)
        cbar2.set_label("T  [$^\circ$C]",size=7.5)

        # R3
        print("plotting density field...")
        density_plot = axR3.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, density,levels=np.linspace(2800,3600,101),extend='both')
        occ_plot = axR3.contour(X_low/1e3,(ymax-Y_low)/1e3,ocrust_init,levels=[0.5],colors=["crimson"],linewidths=0.75)
        axR3.set_ylim([(ymax-ymin_plot)/1.e3,0])   
        axR3.set_xlim([xmin_plot/1e3,xmax_plot/1e3])   
        axR3.tick_params(direction='out',length=2, labelsize=6)
        cbar2 = plt.colorbar(density_plot, orientation='vertical',ticks=[2700,3000,3300,3700])
        cbar2.ax.tick_params(labelsize=5.5)
        cbar2.set_label("density  [kg/m$^3$]",size=7.5)

        plt.tight_layout()

        plt.savefig(plot_loc+"/PT_"+timestr+"_extrusives.png",bbox_inches="tight", format="png",dpi=400)

    ##############################################################################################

    t = timemodule.time()
    ocfrac = 0.99

    if is_plotting_boundwater:
        # #fig3,axs3 = plt.subplots(ncols=1,figsize=(4, 3))
        fig4 = plt.figure(figsize=(12, 5))
        gs4 = gridspec.GridSpec(2, 2)

        axL1_boundwater = fig4.add_subplot(gs4[:, 1])
        axR2 = fig4.add_subplot(gs4[0, 0])
        axR1 = fig4.add_subplot(gs4[1, 0])
        # axR3 = fig4.add_subplot(gs4[2, 1])
        # axR4 = fig4.add_subplot(gs4[3, 1])

        filename_track = "/pt_"+strnum+".txt"
        print("loading: "+filename_track+"\n")
        # if (time//step)==5:
        #     if os.path.isfile(plot_loc+filename_track):
        #         PTdata_track2 = np.loadtxt(plot_loc+filename_track,skiprows=0)
        #         Psurfload_track2 = PTdata_track2[:,1][np.newaxis,:]
        #         Tsurfload_track2 = PTdata_track2[:,2][np.newaxis,:]
        # elif (time//step)>5:
        #     if os.path.isfile(plot_loc+filename_track):
        #         PTdata_track2 = np.loadtxt(plot_loc+filename_track,skiprows=0)
        #         Psurfload_track2 = np.concatenate((Psurfload_track2,PTdata_track2[:,1][np.newaxis,:]),axis=0)
        #         Tsurfload_track2 = np.concatenate((Tsurfload_track2,PTdata_track2[:,2][np.newaxis,:]),axis=0)


        h2o_plot_boundwater = axL1_boundwater.contourf(temperature_h2o_basalt-273, pressure_h2o_basalt*100/1e6, np.abs(xh2o_basalt),
                                                       cmap="Reds", levels=np.linspace(0, 3.5, 101), extend="both")
        axL1_boundwater.contour(temperature_h2o_basalt-273,pressure_h2o_basalt*100/1e6,xh2o_serp,levels=np.array([5]),colors="forestgreen",extend="both")
        cbar4 = plt.colorbar(h2o_plot_boundwater,ax=axL1_boundwater, orientation='vertical',
                            ticks=[0, 1, 2, 3],pad=0.15)
        cbar4.ax.tick_params(labelsize=11.5)
        cbar4.set_label("$X_{H_{2}O}$ [%]", size=13)

        if is_load_PT:
            axL1_boundwater.plot(Tsurfload, Psurfload/1e9)
        axL1_boundwater.set_xlabel("temperature [$^{\circ}$C]")
        axL1_boundwater.set_ylabel("pressure [GPa]")
        axL1_boundwater.tick_params(direction='out', length=2, labelsize=6)
        axL1_boundwater.set_title('pseudosection and P-T paths', fontstyle='italic')
        axL1_boundwater.set_xlim([0, np.max(temperature_h2o_basalt-273)])
        axL1_boundwater.set_ylim([0, 8])
        axL1_boundwater.locator_params(axis='y', nbins=6)
        axL1_boundwater.locator_params(axis='x', nbins=6)


        axL1_y = axL1_boundwater.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        axL1_y.set_ylabel('depth [km]', color=color)  # we already handled the x-label with ax1
        # axL1_y.plot(np.array([0,1]), np.array([0,8e9/(3300*9.8*1e3)]), color=color,linewidth=0)
        axL1_y.tick_params(axis='y', labelcolor=color)
        axL1_y.set_ylim((0,1e9*np.max(pressure_h2o_basalt)*100/1e6/(3300*9.8*1e3)))

        # R2
        T_plotR2 = axR2.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, T, cmap=cm.get_cmap(
            'RdBu_r'), levels=np.linspace(0, 1575, 201), extend='neither')
        stressIIoc = (stressII>4e6) & ((ocrust+ocrust_init)>0.5)
        T_plotR2_serp = axR2.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, (1-is_serpentinizable), cmap='viridis',extend='neither',zorder=1)
        axR2.contour(X_low/1.e3, (ymax-Y_low)/1.e3, T,
                    levels=[500, 750, 1000, 1250],    linewidths=0.3, colors='white', zorder=2)
        
        axR2.set_ylabel("depth [km]")
        axR2.set_ylim([(ymax-ymin_plot)/1.e3, 0])
        axR2.set_xlim([xmin_plot/1e3, xmax_plot/1e3])
        axR2.tick_params(direction='out', length=2, size=8)
        # axR2.tick_params(labelbottom=False)
        axR2.annotate(''.join(['t = ', str("%.1f" % (time_dim)), ' Myr']), xy=(
            0.025, 0.15), xycoords='axes fraction', verticalalignment='center', horizontalalignment='left', fontsize=9.5, color='white')
        axR2.locator_params(axis='y', nbins=3)
        axR2.locator_params(axis='x', nbins=6)

        visc_plotR1 = axR1.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, np.log10(
            visc), cmap=cm.get_cmap('plasma_r'), levels=np.linspace(18, 24, 501), extend='max')
        serp_plot0 = axR1.contour(X_low/1e3, (ymax-Y_low)/1e3, serp,
                                levels=[0.05], colors=["springgreen"], linewidths=0.75)
        serp_plot1 = axR1.contour(X_low/1e3, (ymax-Y_low)/1e3, serp,
                                levels=[0.01], colors=["green"], alpha=0.7, linewidths=0.75)
        axR1.contour(X_low/1.e3, (ymax-Y_low)/1.e3, T,
                    levels=[600],    linewidths=0.3, colors='white', zorder=2)
        axR1.set_ylabel("depth [km]")
        axR1.set_ylim([(ymax-ymin_plot)/1.e3, 0])
        # axR1.set_xlim([xmin_plot/1e3, xmax_plot/1e3])
        

        axR1.tick_params(direction='out', length=2, size=8)
        cbar3 = plt.colorbar(visc_plotR1,ax=axR1, orientation='vertical',
                            ticks=[19, 21, 23])
        cbar3.ax.tick_params(labelsize=7)
        cbar3.set_label("log(${\eta}$)  [Pa.s]", size=9)

        vel_vectsR1 = axR1.quiver(X_vels/1.e3, (ymax-Y_vels)/1.e3, vx*100, vz*100, 
                                color='black', scale=100, width=0.003)  # scale=150, width=0.0015
        axR1.quiverkey(vel_vectsR1, 0.19, 0.1, 5, '5 cm/yr', labelpos='W',
                    fontproperties={'size': '8'}, color='white', labelcolor='white')
        axR1.locator_params(axis='y', nbins=3)
        axR1.locator_params(axis='x', nbins=6)

        cbar2 = plt.colorbar(T_plotR2, ax=axR2, orientation='vertical',
                            ticks=[0, 350, 700, 1050, 1400])
        cbar2.ax.tick_params(labelsize=7)
        cbar2.set_label("T  [$^\circ$C]", size=9)
        axR2.set_xlabel("distance [km]")

        if (time<500):
            axR1.set_xlim([3100, 3500])
            axR2.set_xlim([3100, 3500])

        if ((time<2000) & (time>=500)):
            axR1.set_xlim([3000, 3350])
            axR2.set_xlim([3000, 3350])

        if ((time<4000) & (time>=2000)):
            axR1.set_xlim([2900, 3250])
            axR2.set_xlim([2900, 3250])

        if ((time<6000) & (time>=4000)):
            axR1.set_xlim([2800, 3100])
            axR2.set_xlim([2800, 3100])

        if ((time>=6000)):
            axR1.set_xlim([2750, 3150])
            axR2.set_xlim([2750, 3150])

        # R3
        print("plotting serp field...")
        # boundwater_plot = axR3.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, oc_fluid, cmap=cm.get_cmap(
        #     my_cmap), levels=np.linspace(-0.1, 0.4, 101), extend='both')
        # #boundwater_plot = axR3.contourf(h2o_release, levels=50)        
        
        # axR3.set_ylabel("depth [km]")
        # axR3.set_ylim([50, 0])
        # 
        # axR3.set_xlim([xmin_plot/1e3, xmax_plot/1e3])
        # axR3.tick_params(direction='out', length=2, size=8)
        # cbar2 = plt.colorbar(boundwater_plot, ax=axR3, orientation='vertical', ticks=[
        #                     0, 0.2, 0.4])
        # cbar2.ax.tick_params(labelsize=7)
        # cbar2.set_label("$X_{H_{2}O}$ [%]", size=9)
        # axR3.contour(X_low/1.e3, (ymax-Y_low)/1.e3, T,
        #             levels=[600],    linewidths=0.3, colors='white', zorder=2)
        # axR3.locator_params(axis='y', nbins=3)
        # axR3.locator_params(axis='x', nbins=6)
        plt.tight_layout()

        fig4.savefig(plot_loc+"/boundwater2_"+timestr+".jpg",dpi=250) # ,
                   # bbox_inches="tight", format="jpg", dpi=250

    elapsed = timemodule.time() - t
    print("optimizing section is:"+str(elapsed)+"\n")

    ##############################################################################################

    ocfrac = 0.99
    if is_plotting_fluid:
        fig4 = plt.figure(figsize=(8, 5))#plt.figure(figsize=(10, 5))
        gs4 = gridspec.GridSpec(2, 2)

        axL1_freefluid = fig4.add_subplot(gs4[:, 0])
        axR1 = fig4.add_subplot(gs4[0, 1])
        axR2 = fig4.add_subplot(gs4[1, 1])

        h2o_plot_boundwater = axL1_freefluid.contourf(temperature_h2o_basalt-273, pressure_h2o_basalt*100/1e6, xh2o_basalt,
                                                    levels=np.linspace(0, 5, 101), extend="both")

        cbar4 = plt.colorbar(h2o_plot_boundwater,ax=axL1_freefluid, orientation='vertical',
                            ticks=[0, 2, 4],pad=0.2)
        cbar4.ax.tick_params(labelsize=11.5)
        cbar4.set_label("$X_{H_{2}O}$ [%]", size=13)

        if is_load_PT:
            axL1_freefluid.plot(Tsurfload, Psurfload/1e9)
        axL1_freefluid.set_xlabel("temperature [$^{\circ}$C]")
        axL1_freefluid.set_ylabel("pressure [GPa]")
        axL1_freefluid.tick_params(direction='out', length=2, labelsize=12)
        axL1_freefluid.set_title('pseudosection and PT paths', fontstyle='italic')
        axL1_freefluid.set_xlim([0, np.max(temperature_h2o_basalt-273)])
        axL1_freefluid.set_ylim([0, 8])
        axL1_freefluid.locator_params(axis='y', nbins=6)
        axL1_freefluid.locator_params(axis='x', nbins=6)

        axL1_fy = axL1_freefluid.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        axL1_fy.set_ylabel('depth [km]', color=color)  # we already handled the x-label with ax1
        # axL1_y.plot(np.array([0,1]), np.array([0,8e9/(3300*9.8*1e3)]), color=color,linewidth=0)
        axL1_fy.tick_params(axis='y', labelcolor=color)
        axL1_fy.set_ylim((0,1e9*np.max(pressure_h2o_basalt)*100/1e6/(3300*9.8*1e3)))

        visc_plotR1 = axR1.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, np.log10(
            visc), cmap=cm.get_cmap('plasma_r'), levels=np.linspace(18, 24, 501), extend='max')
        serp_plot0 = axR1.contour(X_low/1e3, (ymax-Y_low)/1e3, serp,
                                levels=[0.08], colors=["springgreen"], linewidths=0.75)
        serp_plot1 = axR1.contour(X_low/1e3, (ymax-Y_low)/1e3, serp,
                                levels=[0.04], colors=["green"], alpha=0.7, linewidths=0.75)
        axR1.contour(X_low/1.e3, (ymax-Y_low)/1.e3, T,
                    levels=[600],    linewidths=0.3, colors='white', zorder=2)
        axR1.set_ylabel("depth [km]")
        axR1.set_ylim([(ymax-ymin_plot)/1.e3, 0])
        axR1.set_xlim([xmin_plot/1e3, xmax_plot/1e3])
        axR1.tick_params(direction='out', length=2, size=8)
        cbar3 = plt.colorbar(visc_plotR1,ax=axR1, orientation='vertical',
                            ticks=[19, 21, 23])
        cbar3.ax.tick_params(labelsize=7)
        cbar3.set_label("log(${\eta}$)  [Pa.s]", size=9)

        vel_vectsR1 = axR1.quiver(X_vels/1.e3, (ymax-Y_vels)/1.e3, vx*100, vz *
                                100, color='black', scale=100, width=0.0025)  # scale=150, width=0.0015
        axR1.quiverkey(vel_vectsR1, 0.20, 0.06, 5, '5 cm/yr', labelpos='W',
                    fontproperties={'size': '9.5'}, color='white', labelcolor='white')
        axR1.locator_params(axis='y', nbins=3)
        axR1.locator_params(axis='x', nbins=6)

        # R2
        T_plotR2 = axR2.contourf(X_low/1.e3, (ymax-Y_low)/1.e3, T, cmap=cm.get_cmap(
            'RdBu_r'), levels=np.linspace(0, 1575, 201), extend='neither')
        axR2.contour(X_low/1.e3, (ymax-Y_low)/1.e3, T,
                     levels=[500, 750, 1000, 1450],    linewidths=0.3, colors='white', zorder=2)
        axR2.set_ylabel("depth [km]")
        axR2.set_ylim([(ymax-ymin_plot)/1.e3, 0])
        axR2.set_xlim([xmin_plot/1e3, xmax_plot/1e3])
        axR2.tick_params(direction='out', length=2, size=8)
        # axR2.tick_params(labelbottom=False)
        axR2.annotate(''.join(['t = ', str("%.1f" % (time_dim)), ' Myr']), xy=(
            0.025, 0.06), xycoords='axes fraction', verticalalignment='center', horizontalalignment='left', fontsize=9.5, color='white')
        axR2.locator_params(axis='y', nbins=3)
        axR2.locator_params(axis='x', nbins=6)

        cbar2 = plt.colorbar(T_plotR2, ax=axR2, orientation='vertical',
                             ticks=[0, 350, 700, 1050, 1400])
        cbar2.ax.tick_params(labelsize=7)
        cbar2.set_label("T  [$^\circ$C]", size=9)
        plt.tight_layout()

        plt.savefig(plot_loc+"/bigserp_"+timestr+".png",
                    bbox_inches="tight", format="png", dpi=400)

        ##############################################################################################

    if is_plotting_heat_flux:
        figHF = plt.figure(figsize=(8, 5))#plt.figure(figsize=(10, 5))
        plt.plot(X_low[-1,:]/1.e3, vertical_heat_flux[-1,:])
        plt.tight_layout()

        plt.savefig(plot_loc+"/heat_flux_"+timestr+".png",
                    bbox_inches="tight", format="png", dpi=400)        

        ##############################################################################################

    if is_plotting_PT_depth_profiles:
        #filename_track = "/particle_track_"+strnum+".txt"
        filename_track = "/pt_"+strnum+".txt"
        print("loading: "+filename_track+"\n")
        if (time//step)==0:
            if os.path.isfile(plot_loc+filename_track):
                PTdata_track = np.loadtxt(plot_loc+filename_track,skiprows=0)
                Psurfload_track = PTdata_track[:,1][np.newaxis,:]
                Tsurfload_track = PTdata_track[:,2][np.newaxis,:]
        else:
            if os.path.isfile(plot_loc+filename_track):
                PTdata_track = np.loadtxt(plot_loc+filename_track,skiprows=0)
                Psurfload_track = np.concatenate((Psurfload_track,PTdata_track[:,1][np.newaxis,:]),axis=0)
                Tsurfload_track = np.concatenate((Tsurfload_track,PTdata_track[:,2][np.newaxis,:]),axis=0)

        if time==0:
            fig5 = plt.figure(figsize=(15, 5))
            gs5 = gridspec.GridSpec(2, 2)

            axL1_PT = fig5.add_subplot(gs5[:, 0])
            # axR1_PT = fig5.add_subplot(gs5[0, 1])
            # axR2_PT = fig5.add_subplot(gs5[1, 1])

            h2o_plot_boundwater = axL1_PT.contourf(temperature_h2o_basalt-273, pressure_h2o_basalt*100/1e6, xh2o_basalt,
                                                levels=np.linspace(0, 5, 101), extend="both")

            cbar5 = plt.colorbar(h2o_plot_boundwater,ax=axL1_PT, orientation='vertical',
                                ticks=[0, 2, 4],pad=0.2)
            cbar5.ax.tick_params(labelsize=11.5)
            cbar5.set_label("$X_{H_{2}O}$ [%]", size=13)

            axL1_yPT = axL1_PT.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            axL1_yPT.set_ylabel('depth [km]', color=color)  # we already handled the x-label with ax1
            axL1_yPT.tick_params(axis='y', labelcolor=color)
            axL1_yPT.set_ylim((0,1e9*4/(3300*9.8*1e3)))

        else:

            if is_load_PT:
                axL1_PT.plot(Tsurfload, Psurfload/1e9,linewidth=0.85,color="black",zorder=1)
                axL1_PT.plot(Tsurfload, Psurfload/1e9,linewidth=0.5,zorder=2)
                axL1_PT.set_xlabel("temperature [$^{\circ}$C]")
                axL1_PT.set_ylabel("pressure [GPa]")
                axL1_PT.tick_params(direction='out', length=2, labelsize=12)
                axL1_PT.set_title('pseudosection and PT paths', fontstyle='italic')
                axL1_PT.set_xlim([0, 1100])
                axL1_PT.set_ylim([0, 4])
                axL1_PT.locator_params(axis='y', nbins=6)
                axL1_PT.locator_params(axis='x', nbins=6)
            if time+step>=max_time:
                for (P_path,T_path) in zip(Psurfload_track.T,Tsurfload_track.T):
                    axL1_PT.plot(T_path,P_path/1e9,color="black",linewidth=0.5)
                #axL1_PT.scatter(T_agard18,P_agard18,s=15*(t_prime_agard18)+4,facecolors="teal",alpha=0.9,edgecolors="white",linewidths=0.4,zorder=3)

        fig5.savefig(plot_loc+"/alt2_PT_depth_profile_"+timestr+"_black.png",
                            bbox_inches="tight", format="png", dpi=500)



    elapsed = timemodule.time() - t
    print("plotting duration is:"+str(elapsed)+"\n")


# save kinematics file
print("saved kinematics to %s..." % kinematics_file)
# remove rows having all zeroes
kinematics = kinematics[~np.all(kinematics == 0, axis=1)]
np.savetxt(kinematics_file, kinematics, fmt='%.5f')  

# save kinematics 2 file
print("saved kinematics serp to %s..." % kinematics_serp_file)
# remove rows having all zeroes
kinematics_serp = kinematics_serp[~np.all(kinematics_serp == 0, axis=1)]
np.savetxt(kinematics_serp_file, kinematics_serp, fmt='%.5f') 

# save kinematics epstein file
print("saved kinematics epstein to %s..." % kinematics_epstein_file)
# remove rows having all zeroes
kinematics_epstein = kinematics_epstein[~np.all(kinematics_epstein == 0, axis=1)]
np.savetxt(kinematics_epstein_file, kinematics_epstein, fmt='%.5f') 

# save kinematics 2 file
print("saved kinematics freefluid to %s..." % kinematics_freefluid_file)
# remove rows having all zeroes
kinematics_freefluid = kinematics_freefluid[~np.all(kinematics_freefluid == 0, axis=1)]
np.savetxt(kinematics_freefluid_file, kinematics_freefluid, fmt='%.5f') 
