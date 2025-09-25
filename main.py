import os
import numpy as np
from dataclasses import dataclass

import spiceypy as spice
METAKERNEL = './data/kernels/meta-kernel.tm'

#import src.getEllipse as old 
import src.ellipse as ellipse 
import src.tube as tube 


AU = 149597870700

# TODO: Rename to be more descriptive
@dataclass
class Data:
    variants_coordinates: list
    variants_velocities: list
    time_data: list
    num_variants: int
    num_time_steps: int  


def loadData(data_directory):
    """
    Load the data from the given data directory. The directory must exist and contain the
    appropiate data files.

    Input:
        data_directory: The full directory path to the data, should end with '/'
    Output:
        Data.variants_coordinates: The coordinates of the variants ordered per timestep
        Data.variants_velocities: The velocities of the variants ordered per timestep
        Data.time_data: The timesteps
        Data.num_variants: The total number of samples
        Data.num_time_steps: The total number of time steps
    """

    # Extract the variants coordinates and velocities files from the given data 
    # directory. The positions and velocities are given with respect to the Sun.
    # TODO: The Sun or SSB?
    coordinates_filename_start = "variants_coordinates_"
    velocities_filename_start = "variants_velocity_"
    times_filename = "times_isot.npy"

    # Find the variants coordinates file. There should only be one in the directory, if
    # there are more than one, then only the first to be found is the one used
    variants_coordinates_file = ""
    for filename in os.listdir(data_directory):
        if filename.startswith(coordinates_filename_start):
            variants_coordinates_file = filename
            break
    print("Loading file", variants_coordinates_file)
    
    # Find the variants velocities file
    variants_velocities_file = ""
    for filename in os.listdir(data_directory):
        if filename.startswith(velocities_filename_start):
            variants_velocities_file = filename
            break
    print("Loading file", variants_velocities_file)
    
    # Get the time file from the directory. All variants use the same timesteps
    time_file = os.path.join(data_directory, times_filename)
    time_data = np.load(time_file)

    # Load the coordinate data
    variants_coordinates = np.load(
        os.path.join(data_directory, variants_coordinates_file)
    )
    print("Coordinate numpy shape", variants_coordinates.shape)

    # Scale the coordinate data to be in meters (from AU)
    for v in range(variants_coordinates.shape[0]):
        variants_coordinates[v] = AU * variants_coordinates[v]

    # Load the velocity data
    variants_velocities = np.load(
        os.path.join(data_directory, variants_velocities_file)
    )
    print("Velocity numpy shape", variants_velocities.shape)
    
    # Scale the velocity data to be in meters per second (from AU per second)
    for v in range(variants_velocities.shape[0]):
        variants_velocities[v] = AU * variants_velocities[v]

    # Get meta data from the filename
    # Examplefilename: variants_coordinates_10000.npy -> 10000.npy -> 10 000 samples
    num_variants = int(variants_coordinates_file.split("_")[-1].split(".")[0])
    print("Number of variants", num_variants)
    num_time_steps = len(time_data)
    print("Number of time steps", num_time_steps)
    
    # The coordinates and velocities are orderd per orbit, but we want to find
    # all varaint cooridnates and velocities per timestep to create time-slices
    ordered_variants_coordinates = []
    ordered_variants_velocities = []
    for t in range(num_time_steps):
        # We only take the coordinate or velocity cooresponding to the t:th timestamp
        # for each orbit
        ordered_variants_coordinates.append(variants_coordinates[t::num_time_steps])
        ordered_variants_velocities.append(variants_velocities[t::num_time_steps])

    print("Size of ordered_variants_coordinates", len(ordered_variants_coordinates))
    print("Size of first item", len(ordered_variants_coordinates[0]))
    print("Shape of first item", ordered_variants_coordinates[0].shape)
 
    return Data(
        ordered_variants_coordinates,
        ordered_variants_velocities,
        time_data,
        num_variants,
        num_time_steps
    )


if __name__ == "__main__":
    # Initialize SPICE
    spice.furnsh(METAKERNEL)

    # Parse any input arguments
    do_plotting = True

    # Select the input data
    input_directory = "./src/orbit_propagation/generated_data/historical/2023 CX1/2023-02-13T02.38.19.001/"
    print("Loading data from", input_directory)

    # Create the directory for output
    out_directory = "./src/generated_data/historical/2023 CX1/2023-02-13T02.38.19.001/"
    os.makedirs(out_directory, exist_ok = True)
    print("Results will be stored in", out_directory)

    # Read the data
    data = loadData(input_directory)

    # Get the polygons of the tube
    #old.main(input_dir, out_dir)
    ellipses = ellipse.createEllipses(data, out_directory, do_plotting)

    # Perform analysis (if needed)

    # Create the tube file
