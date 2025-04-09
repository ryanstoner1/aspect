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


# Define a single-color colormap (e.g., "blue")
single_color_cmap = LinearSegmentedColormap.from_list("SingleColor", ["#D2D344", "#D2D344"])

is_plot_outline = False # or just viscosity 

# ASPECT output 
csvs_loc = '/Users/ryanstoner/Documents/results_miami/results/csv_outputs/'

# setup plotting
mpl.use('agg')
plt.rcParams['font.family']="Times New Roman"
plt.rcParams['font.size']=10
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
folder1 = "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel4_smu0_02_cmu0_04_deserp_erase_res5_5_run18"
folder2 = "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel0_smu0_02_cmu0_04_deserp_erase_res5_5_run33"
file_pattern1 = folder1+"/"+"solution/solution-0?000.npz"
files1 = sorted(glob.glob(file_pattern1))
file_pattern2 = folder2+"/"+"solution/solution-0?000.npz"
files2 = sorted(glob.glob(file_pattern2))
files2[0]= files2[0][:-7]+"1"+files2[0][-6:]

for idx,file_name in enumerate(np.hstack((files1[::2],files2[::2]))):

    