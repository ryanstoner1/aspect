import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.collections as collections
import scipy.interpolate
import os
import glob
import time
import matplotlib as mpl
import numpy as np
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager

# Set the path to Calibri Regular

# Define a single-color colormap (e.g., "blue")
single_color_cmap = LinearSegmentedColormap.from_list("SingleColor", ["#D2D344", "#D2D344"])
mpl.use('agg')

# import matplotlib.font_manager
# print(matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='otf'))

plt.rcParams['font.family']="Calibri"
plt.rcParams['font.size']=12
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.linewidth'] = 1.5

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

# 0. load data
# Path to the .npz files
folder = os.getcwd()+"/rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_smu0_02_cmu0_04_deserp_erase_res6_5_run36"
file_pattern = folder+"/"+"solution/solution-000?0.npz"
print("loading from: "+file_pattern)
files = sorted(glob.glob(file_pattern))

timings = []
#for file in files:
    
# Load the mesh and field data
data = np.load(files[1])
points = data["points"][:,0:2]/1000
points[:,1] = 1450 - points[:,1]
cells = data["cells"]
viscosity = data["viscosity"]
temperature = data["temperature"]
grid_viscosity = data["grid_viscosity"]

ccrusts = data["ccrust"]
serps = data["serp"]
gabbros = data["gabbro"]+data["gabbro_init"]
ocrusts = data["ocrust"]+data["ocrust_init"]
sediment = data["sediment"]

for idx,(serp,gabbro,ocrust,i_sediment) in enumerate(zip(serps,gabbros,ocrusts,sediment)):
    # if serps[idx]<0.01:
    #     serps[idx] = 0.0  
    #     serp = 0.0

    comp_array = np.array([serp,gabbro,ocrust,i_sediment])
    i_comp = np.argmax(comp_array)

    # if i_comp==0:
    #     serps[idx] = 1.0
    #     gabbros[idx] = 0.0
    #     ocrusts[idx] = 0.0
    if i_comp==1:
        gabbros[idx] = 1.0
        serps[idx] = 0.0
        ocrusts[idx] = 0.0
        sediment[idx] = 0.0
    elif i_comp==2:
        ocrusts[idx] = 1.0
        serps[idx] = 0.0
        gabbros[idx] = 0.0
        sediment[idx] = 0.0
    elif i_comp==3:
        ocrusts[idx] = 0.0
        serps[idx] = 0.0
        gabbros[idx] = 0.0
        sediment[idx] = 1.0
    if np.sum(comp_array)<0.2:
        ocrusts[idx] = 0.0
        gabbros[idx] = 0.0
        sediment[idx] = 0.0
      
grid_x = data["grid_x"]/1e3
grid_y = data["grid_y"]/1e3
x0 = np.min(grid_x)
x1 = np.max(grid_x)
y0 = np.min(grid_y)
y1 = np.max(grid_y)

# 1a. viscosity plot
is_fullres = True
is_Tcontour = True

if is_fullres:


    print("plotting viscosity!")

    triangulation, triangle_values = plot_quads_with_tricontourf(points, cells, viscosity)
    start_time = time.time()
    fig, ax = plt.subplots(figsize=(12, 6),nrows=2)
    ax[0].set_aspect('equal', adjustable='box')
    contour = ax[0].tricontourf(triangulation, np.log10(viscosity),levels=900,cmap='viridis_r',antialiased=False)
    #cb = plt.colorbar(contour, ax=ax, label="Viscosity",ticks=[19, 20, 21,22,23]) 
    font = {'fontname': 'Calibri',
    'weight': 'normal',
    'size': 14,
    }  
    ax[0].set_xlabel("x [km]",fontdict=font)
    ax[0].set_ylabel("Depth [km]",fontdict=font)
    ax[0].set_ylim([1450,-2])
    #ax[0].set_xlim([2000,3500])
    #ax[0].set_yticks([0,500,1000,1450])
    ax[0].spines[['top']].set_visible(False)
    ax[1].spines[['top']].set_visible(False)
    # 1b. add temperature contours
    # if is_Tcontour:
    #     ax[0].tricontour(triangulation, temperature-273,levels=[800,1200],colors=[[1,1,1,0.8],],width=0.75)

    ax[1].tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False)
    ax[1].set_axis_off()
    # Save the plot
    output_file = files[0].replace(".npz", "_interpolated_viscosity_plot_raster_fullres.png")
    plt.savefig(output_file,dpi=300)
    #cb.remove() 

    # 1c.
    output_file_T = files[0].replace(".npz", "_interpolated_temperature_plot_raster_fullres.png")
    contour_T = ax[0].tricontourf(triangulation, temperature-273,levels=900,cmap='coolwarm',antialiased=False)
    cb_T = plt.colorbar(contour_T, ax=ax, label="Temperature ($^{\circ}$C)",ticks=[1400,1100,800,500,200,0])
    plt.savefig(output_file_T,dpi=300)
    cb_T.remove()
    ax[0].clear()

# 2a. composition plot
is_zoom_comp = False
if not is_zoom_comp:
    
    ax[0].tricontourf(triangulation,gabbros,levels=[0.5,1.1],cmap="Blues",antialiased=False)
    ax[0].tricontourf(triangulation,ocrusts,levels=[0.5,1.1],cmap="Oranges",antialiased=False)
    ax[0].tricontourf(triangulation,ccrusts,levels=[0.1,1.1],cmap="Greys")
    ax[0].tricontourf(triangulation,sediment,levels=[0.3,1.1],cmap=single_color_cmap)
    
    
    ax[0].tricontour(triangulation,serps,levels=[0.01,1.1],linewidths=0.25,colors='k')
    if is_Tcontour:
        ax[0].tricontour(triangulation, temperature-273,levels=[400,800,1200],colors=[[1,1,1,0.8],])
    ax[0].spines[['top']].set_visible(False)
    ax[0].set_facecolor((154/255,185/255,115/255,0.5))
    ax[0].set_xlabel("x [km]",fontdict=font)
    ax[0].set_ylabel("Depth [km]",fontdict=font)
    ax[0].set_ylim([150,-2])
    ax[0].set_yticks([0,50,100,150])
    ax[0].set_xlim([3250,3500])
    ax[0].set_xticks([3200,3300,3400,3500])
    ax[1].set_axis_off()
    #ax[0].legend(fontsize="small")

    #ax.tricontourf(triangulation,gabbros,"red",antialiased=False)
    output_file_comps = files[0].replace(".npz", "_interpolated_composition_plot_raster_fullres.svg")
    plt.savefig(output_file_comps,dpi=500)

# # 3. topography
# fig_topo, ax_topo = plt.subplots(figsize=(12, 1))
# topo_file_pattern = folder+"/"+"topography.000?0.txt"
# files_topo = sorted(glob.glob(topo_file_pattern))
# topo_data = np.loadtxt(files_topo[1],skiprows=1)
# print("done")
# ax_topo.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
# ax_topo.plot(topo_data[:,0]/1e3,topo_data[:,2]/1e3,linewidth=2,color="k")
# ax_topo.set_yticks([-2,0,2])
# ax_topo.set_xlim([0,np.max(topo_data[:,0]/1e3)])
# ax_topo.spines[['top']].set_visible(False)
# ax_topo.set_ylabel(" ",fontdict=font)
# output_file_topo = files_topo[0].replace(".txt", "_topo_plot_raster_fullres.png")
# plt.savefig(output_file_topo,dpi=500)
#print("saved: "+output_file_topo)