import pyvista as pv
import numpy as np
import scipy as scipy
import glob
import matplotlib.pyplot as plt
import pathlib
from scipy.interpolate import RegularGridInterpolator
import time as timemodule

fig, ax = plt.subplots()
input_file_serp=str(pathlib.Path().resolve()) + "/pseudosections/Serpentinite/serp_sat_niu_aspect_H2O.txt"

# Read the input file
data_serp = np.genfromtxt(input_file_serp, skip_header=0, usecols=(0, 1, 2))

# Separate the columns
pressure_h2o_serp = (data_serp[:, 1]).reshape((250,250))
temperature_h2o_serp = (data_serp[:, 0]).reshape((250,250))
xh2o_serp = (data_serp[:, 2]).reshape((250,250))
contour = ax.contour(temperature_h2o_serp-273,pressure_h2o_serp*100/1e6,xh2o_serp,levels=np.array([5]),colors="black",extend="both",linestyles='dotted')

# contour_level = contour  # Index depends on the level order
# paths = contour_level.get_paths()

# Define the grid points (assuming regular spacing)
pressure_grid = np.linspace(pressure_h2o_serp.min()*100*1e3, pressure_h2o_serp.max()*100*1e3, 250)
temperature_grid = np.linspace(temperature_h2o_serp.min(), temperature_h2o_serp.max(), 250)

# Create the interpolator
interpolator = RegularGridInterpolator(
    (temperature_grid, pressure_grid), xh2o_serp, method='nearest', bounds_error=False, fill_value=None
)


def points_inside_contour(paths, points):
    results = np.zeros(len(points), dtype=bool)
    for path in paths:
        # Check which points are inside the current path
        results |= path.contains_points(points)
    return results

low_res_val = 400

# Function to interpolate data to a regular grid
def interpolate_to_regular_grid(points, values, grid_x, grid_y):
    # Perform interpolation to a regular grid using griddata
    grid_z = scipy.interpolate.griddata(points, values, (grid_x, grid_y), method='linear')
    return grid_z

# File pattern for .pvtu files
folder = "rc3_part_lookup_serp_morb_bas_h2o_1denslookup_1visclookup_cohes_1-MPa_freesurf_y_extent_1450km_sed_1km_pel0_smu0_02_cmu0_04_deserp_erase_res5_5_run20"
file_pattern = folder+"/"+"solution/solution-0??00.pvtu"
files = sorted(glob.glob(file_pattern))

for file in files:
    t1 = timemodule.time()
    # Load the dataset with PyVista
    mesh = pv.read(file)

    # Extract points, connectivity, and fields
    points = mesh.points  # Nx3 array of node coordinates
    cells = mesh.cells_dict[9]  # Assumes quadrilateral cells, type 9 in VTK
    viscosity = mesh["viscosity"]
    temperature = mesh["T"]
    pressure = mesh["p"]
    serp = mesh["serp"]
    sediment = mesh["sediment"]
    ccrust = mesh["ccrust"]
    ocrust = mesh["ocrust"]
    ocrust_init = mesh["ocrust_init"]
    gabbro = mesh["gabbro"]
    gabbro_init = mesh["gabbro_init"]
    freefluid = mesh["freefluid"]
    boundwater = mesh["boundwater"]
    velocity = mesh["velocity"]
    vx = velocity[:,0]
    vy = velocity[:,1]


    points = points[:,0:2]
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, low_res_val), np.linspace(y_min, y_max, low_res_val))

    # calculate total_serp
    mesh = mesh.compute_cell_sizes()

    # Access the cell areas
    cell_areas = mesh.cell_data['Area'] 

    # Initialize an array to store the mean values for each cell
    serp_values = np.zeros(mesh.n_cells)
    serp_values_stable = np.zeros(mesh.n_cells)
    serp_values_stable_test = np.zeros(mesh.n_cells)
    boundwater_values = np.zeros(mesh.n_cells)

    step_bin = 10e3
    bins_y = np.max(points[:,1])-np.arange(0,220e3,step=step_bin)
    bins_serp = np.zeros_like(bins_y)
    bins_serp1 = np.zeros_like(bins_y)
    bins_freefluid = np.zeros_like(bins_y)
    
    areas = mesh.compute_cell_sizes()['Area']
    # Loop over all cells



    # Pre-compute properties for all points
    test_points = np.vstack((mesh.point_data['T'], mesh.point_data['p'])).T
    interpolated_values = interpolator(test_points)
    is_inside_all = interpolated_values > 1.25
    depth_indices_pre = np.digitize(mesh.points[:, 1], bins_y)

    # Prepare cell properties
    cell_connectivity = mesh.cells
    cell_offsets = mesh.offset
    n_cells = mesh.n_cells
    cell_indices = np.arange(n_cells)

    # Map cell indices to their point indices
    cell_point_ranges = [np.arange(cell_offsets[i], cell_offsets[i + 1]) for i in cell_indices]
    cell_point_indices = np.hstack([cell_connectivity[r] for r in cell_point_ranges])
    cell_map = np.hstack([[i] * len(r) for i, r in enumerate(cell_point_ranges)])

    # Extract point-specific properties
    serp_all = mesh.point_data['serp']
    boundwater_all = mesh.point_data['boundwater']
    depth_indices_all = depth_indices_pre[cell_point_indices]
    is_inside_all_cells = is_inside_all[cell_point_indices]

    # Group operations by cell
    serp_values = np.bincount(cell_map, weights=serp_all[cell_point_indices]) / np.bincount(cell_map)
    serp_values_stable = (
        np.bincount(cell_map, weights=serp_all[cell_point_indices] * is_inside_all_cells) / np.bincount(cell_map)
    )
    boundwater_values = np.bincount(cell_map, weights=boundwater_all[cell_point_indices]) / np.bincount(cell_map)

    # Bin updates (sparse or batch updates for bins_serp and bins_serp1)
    bin_updates_serp = (serp_values / 0.121) > 0.1
    bin_updates_serp1 = (serp_values / 0.121) > 0.01

    # Distribute contributions to bins based on depths
    depth_bin_contributions = areas[cell_indices] * serp_values[cell_map]
    bins_serp += np.bincount(depth_indices_all, weights=depth_bin_contributions * bin_updates_serp[cell_map])
    bins_serp1 += np.bincount(depth_indices_all, weights=depth_bin_contributions * bin_updates_serp1[cell_map])

    elapsed2 = timemodule.time() - t1
    print("first elapsed is: "+ str(elapsed2))

    # Interpolate the viscosity field to the regular grid
    #grid_viscosity = interpolate_to_regular_grid(points, viscosity, grid_x, grid_y)

    # Save data to a compressed .npz file
    npz_file = file.replace(".pvtu", ".npz")
    np.savez_compressed(
        npz_file,
        points=points,
        cells=cells,
        viscosity=viscosity,
        temperature=temperature,
        pressure=pressure,
        sediment=sediment,
        serp=serp,
        ccrust=ccrust,
        ocrust=ocrust,
        ocrust_init=ocrust_init,
        gabbro=gabbro,
        gabbro_init=gabbro_init,
        grid_x=grid_x, 
        grid_y=grid_y,
        vx=vx, 
        vy=vy,
        freefluid=freefluid,
        serp_total=np.sum(serp_values),
        serp_total_stable=np.sum(serp_values_stable),
        boundwater_total=np.sum(boundwater_values),
        bins_serp=bins_serp,
        bins_serp1=bins_serp1,
        bins_y=bins_y,
    )

    print(f"Saved quadrilateral mesh and field data to {npz_file}")