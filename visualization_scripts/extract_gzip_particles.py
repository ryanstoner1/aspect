import os
import sys
import numpy as np
import timeit
import pyvista as pv
import pandas as pd


mod_name = str(sys.argv[1])
max_time = int(sys.argv[2])
step = int(sys.argv[3])
models_loc = '/home/ryan-stoner/Documents/results/'
output_loc = '/home/ryan-stoner/Documents/results/csv_outputs/'
solution = ''.join([models_loc, str(mod_name), '/particles.pvd'])
print(solution)

# Load the VTK file
reader = pv.get_reader(solution)

# directory for csv files
if not os.path.exists(''.join([output_loc, str(mod_name)])):
    os.mkdir(''.join([output_loc, str(mod_name)]))

times = reader.time_values



for i in range(0, max_time, step):
    
    start_time = timeit.default_timer()
    time = times[i]
    ofile = 'part.%d.gzip' % i
    ofiled = ''.join([output_loc, str(mod_name), '/', ofile])
    print('extracting', i, time, ofiled)

    # Skip extraction if the file already exists
    if os.path.exists(ofiled):
        print("File already exists. Skipping extraction.")
        continue

    reader.set_active_time_point(i)
    mesh = reader.read()[0]

    # Assuming you have a PyVista dataset called 'mesh'
    # You can access the point data and coordinates as follows:
    points = mesh.points
    point_data = mesh.point_data

    # Create an empty dictionary to hold the data
    data_dict = {}

    # Iterate over each array in the point data
    for data_name in point_data.keys():
        # Get the array associated with the name
        ivector = point_data[data_name]
        ivector_nparray = ivector

        # Check if the array is not 1D
        if ivector_nparray.ndim > 1:
            # Iterate over the dimensions and create separate 1D arrays
            for j in range(ivector_nparray.shape[1]):
                # Create the new array name with the iteration number
                new_array_name = f"{data_name}:{j}"

                # Extract the 1D array
                iarray = ivector_nparray[:, j]

                # Add the 1D array to the dictionary with the new array name
                data_dict[new_array_name] = iarray
        else:
            # Array is already 1D, add it to the dictionary as is
            data_dict[data_name] = ivector_nparray

        # # Add the NumPy array to the dictionary
        # data_dict[data_name] = ivector_nparray


    # Create the DataFrame using the data dictionary
    df = pd.DataFrame(data_dict)

    # You can also add the coordinates as separate columns if needed
    df['Points:0'] = points[:, 0]
    df['Points:1'] = points[:, 1]
    df['Points:2'] = points[:, 2]

    # elapsed_time = timeit.default_timer() - start_time
    # print("time elapsed:", elapsed_time)

    # mesh = reader.read()[0]

    # point_coords = pv.Table(mesh.points)
    # point_data = pv.Table(mesh.point_data)
    # # # Extract the data at the given time
    # # point_data = mesh.point_data_at_time(time)

    # # Convert point data to a pandas DataFrame
    # df = point_coords.to_pandas()
    # df_data = point_data.to_pandas()

    df.to_parquet(ofiled,
           compression='gzip')
    # # Save the DataFrame to a CSV file
    #df.to_csv(ofiled, index=False)

    elapsed_time = timeit.default_timer() - start_time
    print("time elapsed:", elapsed_time)


