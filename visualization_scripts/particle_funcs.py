import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib as mpl
import math as math
from scipy.signal import savgol_filter 
from scipy.interpolate import griddata, Rbf, RegularGridInterpolator
import timeit
from scipy.spatial import Delaunay
from typing import Dict, Tuple

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

def calculate_total_mass(x, y, rho):
    # # Filter points where rho (serp usually) is above zero
    is_non_zero = rho > 0
    x_non_zero = x[is_non_zero]
    y_non_zero = y[is_non_zero]
    rho_non_zero = rho[is_non_zero]

    # Perform triangulation
    points = np.column_stack((x_non_zero, y_non_zero))
    tri = Delaunay(points)

    # Calculate the area of each triangle
    triangle_areas = 0.5 * np.abs(np.cross(tri.points[tri.simplices[:, 1]] - tri.points[tri.simplices[:, 0]],
                                                  tri.points[tri.simplices[:, 2]] - tri.points[tri.simplices[:, 0]])) # np.dot(tri.points[tri.simplices[:, 0]],
                                         

    # Multiply triangle areas by corresponding rho values and sum them up
    rho_non_zero = np.array(rho_non_zero)
    average_rho_triangle = np.sum(rho_non_zero[tri.simplices],axis=1)/3
    weighted_areas = triangle_areas * (average_rho_triangle)

    return np.sum(weighted_areas)

def calculate_total_mass_contour(x, y, rho, t):
    # # Filter points where rho (serp usually) is above zero
    is_non_zero = rho > 0
    x_non_zero = x[is_non_zero]
    y_non_zero = y[is_non_zero]
    rho_non_zero = rho[is_non_zero]

    # Perform triangulation
    points = np.column_stack((x_non_zero, y_non_zero))
    
    tri = Delaunay(points)

    # Calculate the area of each triangle
    triangle_areas = 0.5 * np.abs(np.cross(tri.points[tri.simplices[:, 1]] - tri.points[tri.simplices[:, 0]],
                                                  tri.points[tri.simplices[:, 2]] - tri.points[tri.simplices[:, 0]])) # np.dot(tri.points[tri.simplices[:, 0]],
                                         

    # Multiply triangle areas by corresponding rho values and sum them up
    rho_non_zero = np.array(rho_non_zero)
    average_rho_triangle = np.sum(rho_non_zero[tri.simplices],axis=1)/3

    mpltri = mpl.tri.Triangulation(x_non_zero,y_non_zero,triangles=tri.simplices)
    
    fig=plt.figure(figsize=(6, 3))
    tri_out = plt.tricontourf(mpltri, rho_non_zero,levels=np.linspace(0.001,np.max(rho_non_zero),10))
    plt.colorbar(tri_out)
    fig.savefig("outcontour_"+str(t)+".jpg")
    area_out = 0
    level_used = np.zeros(len(tri_out.levels))
    for i, (level,collection) in enumerate(zip(tri_out.levels,tri_out.collections)):
        if len(collection.get_paths())>0:
            level_used[i] = level            
            for path in collection.get_paths():
                if i==0: # base contour
                    area_out += np.abs(area(path.vertices))*level_used[i]
                else: # replace lower level with higher level
                    area_out += np.abs(area(path.vertices))*(level_used[i]-level_used[i-1])
        


    return area_out

#### LOADING DATA
def read_pseudosection_data(file_path: str, n_x: int, n_y: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, RegularGridInterpolator]:
    """
    Reads pseudosection data and sets up a RegularGridInterpolator.
    
    Parameters:
        file_path (str): Path to the pseudosection file.
        n_x (int): Number of points along the x-axis.
        n_y (int): Number of points along the y-axis.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, RegularGridInterpolator]: 
        Pressure, temperature, xH2O arrays, and the interpolator.
    """
    data = np.genfromtxt(file_path, skip_header=0, usecols=(0, 1, 2))
    pressure = data[:, 1].reshape((n_x, n_y))
    temperature = data[:, 0].reshape((n_x, n_y))
    xh2o = data[:, 2].reshape((n_x, n_y))
    interpolator = RegularGridInterpolator(
        (temperature[0, :], pressure[:, 0] * 100 * 1e3),
        xh2o.T,
        method='nearest'
    )
    return pressure, temperature, xh2o, interpolator

def initialize_interpolators(config: Dict[str, str], n_x: int, n_y: int) -> Dict[str, RegularGridInterpolator]:
    """
    Initializes interpolators for various materials based on configuration.
    
    Parameters:
        config (Dict[str, str]): Dictionary containing file paths for materials.
        n_x (int): Number of points along the x-axis.
        n_y (int): Number of points along the y-axis.

    Returns:
        Dict[str, RegularGridInterpolator]: Dictionary of interpolators.
    """
    interpolators = {}
    for material, file_path in config.items():
        _, _, _, interpolator = read_pseudosection_data(file_path, n_x, n_y)
        interpolators[material] = interpolator
    return interpolators

                    # axs[0,0].plot(h2o_bound_slab_track[0:(t//step-1),1516],h2o_bound_basalt_track_interp[0:(t//step-1),1516]*basalt_track[0:(t//step-1),1516]+
                    #             h2o_bound_gabbro_track_interp[0:(t//step-1),1516]*gabbro_track[0:(t//step-1),1516]+
                    #             h2o_bound_sediment_track_interp[0:(t//step-1),1516]*sediment_track[0:(t//step-1),1516])
                    # axs[0,0].set_xlabel("tracked H2O (aspect)")
                    # axs[0,0].set_ylabel("predicted H2O")
                    # axs[0,0].set_title("predicted vs. tracked H2O")

                    # axs[0,1].plot(T_slab_track[0:(t//step-1),1516],h2o_bound_slab_track[0:(t//step-1),1516])
                    # axs[0,1].set_xlabel("T particle (K)")
                    # axs[0,1].set_ylabel("tracked H2O")
                    # axs[0,1].set_title("tracked H2O of particle")

                    # axs[0,2].plot(T_slab_track[0:(t//step-1),1516],h2o_bound_basalt_track_interp[0:(t//step-1),1516]*basalt_track[0:(t//step-1),1516]+
                    #             h2o_bound_gabbro_track_interp[0:(t//step-1),1516]*gabbro_track[0:(t//step-1),1516]+
                    #             h2o_bound_sediment_track_interp[0:(t//step-1),1516]*sediment_track[0:(t//step-1),1516])
                    # axs[0,2].set_xlabel("T particle (K)")
                    # axs[0,2].set_ylabel("predicted sediment H2O")
                    # axs[0,2].set_title("predicted H2O from sediment")

                    # axs[1,0].plot(T_slab_track[0:(t//step-1),1516],basalt_track[0:(t//step-1),1516])
                    # axs[1,0].set_xlabel("T particle (K)")
                    # axs[1,0].set_ylabel("tracked basalt fraction")
                    # axs[1,0].set_title("tracked basalt diffusion")

                    # axs[1,1].plot(T_slab_track[0:(t//step-1),1516],gabbro_track[0:(t//step-1),1516])
                    # axs[1,1].set_xlabel("T particle (K)")
                    # axs[1,1].set_ylabel("tracked gabbro fraction")
                    # axs[1,1].set_title("tracked gabbro diffusion")

                    # axs[1,2].plot(T_slab_track[0:(t//step-1),1516],h2o_release_basalt_track[0:(t//step-1),1516])
                    # axs[1,2].set_xlabel("T particle (K)")
                    # axs[1,2].set_ylabel("tracked sediment fraction")
                    # axs[1,2].set_title("tracked sediment diffusion")

                    # axs[1,3].plot(T_slab_track[0:(t//step-1),1516],np.cumsum(h2o_release_basalt_track[0:(t//step-1),1516]))
                    # plt.tight_layout()
                    # fig.savefig(plot_loc+"/particle_compare_"+str(t)+".jpg",dpi=250)
                    # plt.close()



                    # fig_full, axs_full = plt.subplots(figsize=(18, 8),nrows=1, ncols=3)
                    # depth_cutoff = 2e3
                    # scatter_full = axs_full[0].scatter(xx[(xx>3e6) & ((ymax_plot-yy)>depth_cutoff)],yy[(xx>3e6) & ((ymax_plot-yy)>depth_cutoff)],s=1.5,c=(h2o_release_basalt_track[(fluid_release_start//step)-2+ival,((xx>3e6) & ((ymax_plot-yy)>depth_cutoff))]+
                    #                                                                       h2o_release_sediment_track[(fluid_release_start//step)-2+ival,((xx>3e6) & ((ymax_plot-yy)>depth_cutoff))]))

                    # scatter_full = axs_full[0].scatter(xx[(xx>3e6) & ((ymax_plot-yy)>20e3)]/1e3,(ymax_plot-yy[(xx>3e6) & ((ymax_plot-yy)>20e3)])/1e3,s=1,c=((h2o_bound_sediment_raw_direct[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)]*frac_sediment[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)]+
                    #                                                                 h2o_bound_basalt_raw_direct[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)]*frac_basalt[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)]+
                    #                                                                 h2o_bound_gabbro_raw_direct[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)]*frac_gabbro[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)])))
                    # axs_full[0].invert_yaxis()
                    # axs_full[0].set_xlabel("distance (km)")
                    # axs_full[0].set_ylabel("depth")
                    # axs_full[0].set_title("predicted (postprocessing)")
                    # plt.colorbar(scatter_full)

                    # scatter_predicted = axs_full[1].scatter(xx[(xx>3e6) & ((ymax_plot-yy)>20e3)]/1e3,(ymax_plot-yy[(xx>3e6) & ((ymax_plot-yy)>20e3)])/1e3,s=1,c=(h2o_bound_slab_track_new[(xx>3e6) & ((ymax_plot-yy)>20e3)]))
                    # axs_full[1].invert_yaxis()
                    # axs_full[1].set_xlabel("distance (km)")
                    # axs_full[1].set_ylabel("depth")
                    # axs_full[1].set_title("aspect (current formulation)")


                    # scatter_predicted = axs_full[1].scatter(xx[((xx>3e6) & ((ymax_plot-yy)>depth_cutoff))],yy[((xx>3e6) & ((ymax_plot-yy)>depth_cutoff))],s=1.5,c=(
                    #                     h2o_release_boundwater_track[(fluid_release_start//step)-2+ival,((xx>3e6) & ((ymax_plot-yy)>depth_cutoff))]-
                    #                     (h2o_release_basalt_track[(fluid_release_start//step)-2+ival,((xx>3e6) & ((ymax_plot-yy)>depth_cutoff))]+
                    #                                                                       h2o_release_sediment_track[(fluid_release_start//step)-2+ival,((xx>3e6) & ((ymax_plot-yy)>depth_cutoff))])))
                    # scatter_predicted = axs_full[1].scatter(xx[((xx>3e6) & ((ymax_plot-yy)>depth_cutoff))],yy[((xx>3e6) & ((ymax_plot-yy)>depth_cutoff))],s=1.5,c=(
                    #                     h2o_release_boundwater_track[(fluid_release_start//step)-2+ival,((xx>3e6) & ((ymax_plot-yy)>depth_cutoff))]))
                    # plt.colorbar(scatter_predicted)

                    # scatter_diff = axs_full[2].scatter(xx[(xx>3e6) & ((ymax_plot-yy)>20e3)]/1e3,(ymax_plot-yy[(xx>3e6) & ((ymax_plot-yy)>20e3)])/1e3,s=1,c=(h2o_bound_slab_track_new[(xx>3e6) & ((ymax_plot-yy)>20e3)]-
                    #                                                                                     (h2o_bound_sediment_raw_direct[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)]*frac_sediment[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)]+
                    #                                                                 h2o_bound_basalt_raw_direct[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)]*frac_basalt[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)]+
                    #                                                                 h2o_bound_gabbro_raw_direct[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)]*frac_gabbro[np.isin(ids,ids_orig)][(xx>3e6) & ((ymax_plot-yy)>20e3)])))
                    # axs_full[2].invert_yaxis()
                    # axs_full[2].set_xlabel("distance (km)")
                    # axs_full[2].set_ylabel("depth")
                    # axs_full[2].set_title("low (predicted loss>postprocess loss)")
                    # plt.colorbar(scatter_diff)

                    # read ASPECT output and isolate the crust
