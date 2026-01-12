import os
import json
import rootutils
import numpy as np
import spiceypy as spice

from dataclasses import dataclass

import variant_generation.historical_uncertainty as historical_uncertainty
import variant_generation.sectioned_uncertainty as sectioned_uncertainty


import tube_generation.ellipse as ellipse 
import tube_generation.tube as tube
import transform_generation.toTransforms as toTransforms
import transform_generation.toEllipsoids as toEllipsoids


# Path to the SPICE kernel files, to initialize SPICE
LSK_KERNEL = "../data/kernels/lsk/naif0012.tls.pc"
SPK_KERNEL = "../data/kernels/spk/de432s.bsp"
PCK_KERNEL = "../data/kernels/pck/pck00011.tpc"

# The number of meters in one Astronomical Unit (AU)
AU = 149597870700 

# Current version of the configuration files
CONFIGURATION_VERSION = "0.1"

# Data object to hold variants data
@dataclass
class VariantsData:
    variants_coordinates: list
    variants_velocities: list
    times: list
    covariances: list
    num_variants: int
    num_time_steps: int 


def loadVariantsData(data_directory, num_variants = -1):
    """
    Load the data from the given data directory. The directory must exist and contain the
    appropiate data files.

    Input:
        data_directory: The full directory path to the data, should end with '/'
        num_variants: The number of desired variants, this is used to identify the files
        to load. If -1 is given, then load the first file that matches the filename
        format. 
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
    times_filename_start = "times_isot_"

    # If a specific number of variants is given, then add that to the filename to find
    # that particular file
    if num_variants != -1:
        coordinates_filename_start += str(num_variants)
        velocities_filename_start += str(num_variants)
        times_filename_start += str(num_variants)

    # Find the variants data files. There should only be one file per type in the
    # directory, if there are more than one, then only the first to be found is the one
    # used. If num_variants is given, then the file with that specific name is used.
    variants_coordinates_file = ""
    variants_velocities_file = ""
    time_file = ""
    if num_variants != -1:
        variants_coordinates_file = coordinates_filename_start + ".npy"
        variants_velocities_file = velocities_filename_start + ".npy"
        time_file = times_filename_start + ".npy"
    else:
        for filename in os.listdir(data_directory):
            # Variants coordinates file
            if not variants_coordinates_file and \
               filename.startswith(coordinates_filename_start):
                variants_coordinates_file = filename
                break
            # Find the variants velocities file
            if not variants_velocities_file and \
               filename.startswith(velocities_filename_start):
                variants_velocities_file = filename
                break
            # Get the time file from the directory. All variants use the same timesteps
            if not time_file and filename.startswith(times_filename_start):
                time_file = filename
                break

    # Check if all files were found and if they exist
    variants_coordinates_filepath = os.path.join(
        data_directory,
        variants_coordinates_file
    )
    if not os.path.exists(variants_coordinates_filepath):
        print("Could not find variants coordinates file ", variants_coordinates_filepath)
        assert False, "Missing variants coordinates file"

    variants_velocities_filepath = os.path.join(data_directory, variants_velocities_file)
    if not os.path.exists(variants_velocities_filepath):
        print("Could not find variants velocities file ", variants_velocities_filepath)
        assert False, "Missing variants velocities file"

    time_filepath = os.path.join(data_directory, time_file)
    if not os.path.exists(time_filepath):
        print("Could not find time file ", time_filepath)
        assert False, "Missing time file"

    # Load the time data
    print("Loading file", time_file)
    time_data = np.load(time_filepath)

    # Get the number of time steps
    num_time_steps = len(time_data)
    print("Number of time steps", num_time_steps)

    # Load the coordinate data
    print("Loading file", variants_coordinates_file)
    variants_coordinates = np.load(variants_coordinates_filepath)
    print("Coordinate numpy shape", variants_coordinates.shape)

    # Scale the coordinate data to be in meters (from AU)
    print("Scaling coordinates from AU to meters")
    for v in range(variants_coordinates.shape[0]):
        variants_coordinates[v] = AU * variants_coordinates[v]

    # Load the velocity data
    print("Loading file", variants_velocities_file)
    variants_velocities = np.load(variants_velocities_filepath)
    print("Velocity numpy shape", variants_velocities.shape)
    
    # Scale the velocity data to be in meters per second (from AU per second)
    print("Scaling velocities from AU/s to m/s")
    for v in range(variants_velocities.shape[0]):
        variants_velocities[v] = AU * variants_velocities[v]

    if num_variants == -1:
        # Get the number of variants from the filename of the loaded file(s)
        # Examplefilename: variants_coordinates_10000.npy -> 10000.npy -> 10 000 samples
        num_variants = int(variants_coordinates_file.split("_")[-1].split(".")[0])
    print("Number of variants", num_variants)
    
    # The coordinates and velocities in the files are orderd per orbit, but we want to
    # find all varaint cooridnates and velocities per timestep to create time-slices
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
    
    variants_data = []
    variants_data.append(VariantsData(
        ordered_variants_coordinates,
        ordered_variants_velocities,
        time_data,
        num_variants,
        num_time_steps
    ))
    return variants_data


if __name__ == "__main__":
    # Set the working directory
    root = rootutils.setup_root(
        __file__,
        indicator = ".project-root",
        dotenv = False,
        pythonpath = True,
        cwd = False
    )

    # Initialize SPICE
    spice.furnsh(LSK_KERNEL)
    spice.furnsh(SPK_KERNEL)
    spice.furnsh(PCK_KERNEL)

    # Get the input configuration file that contain all settings and parameters
    configuration_file = "../config/historical_2004_MN4_test.json"
    configuration_data = None

    # Try to read the configuration json file
    try:
        with open(configuration_file, 'r') as file:
            configuration_data = json.load(file)
    except FileNotFoundError:
        print("Error: The file", configuration_file, "was not found.")
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from file", configuration_file)
    
    # Check version number of the configuration file
    version = configuration_data["version"]
    if version != CONFIGURATION_VERSION:
        print("Warning: Configuration file version", version, 
              "does not match expected version", CONFIGURATION_VERSION)
        assert False, "Configuration file version mismatch"

    # Extract the necessary configuration settings
    object_id = configuration_data["object_id"]
    override_existing_results = configuration_data["override_existing_results"]

    # Create paths to all input data
    tube_type = configuration_data["tube_type"]
    input_directory = configuration_data["input_directory"]
    mpc_directory = os.path.join(input_directory, "mpc_data/")
    orbit_fits_directory = os.path.join(input_directory, "orbit_fits/")
    output_directory = configuration_data["output_directory"]
    
    # Create all the output directories
    generated_variants_directory = os.path.join(
        output_directory,
        "variants/" + tube_type + "/"
    ) 
    os.makedirs(generated_variants_directory, exist_ok = True)

    tube_directory = os.path.join(
        output_directory,
        "tubes/" + tube_type + "/"
    ) 
    os.makedirs(tube_directory, exist_ok = True)

    transforms_directory = os.path.join(
        output_directory,
        "transforms/" + tube_type + "/"
    ) 
    os.makedirs(transforms_directory, exist_ok = True)

    ellipsoids_directory = os.path.join(
        output_directory,
        "ellipsoids/" + tube_type + "/"
    ) 
    os.makedirs(ellipsoids_directory, exist_ok = True)
    
    # Select the asteroid that was specified in the configuration
    mpc_directory = os.path.join(mpc_directory, object_id)
    orbit_fits_directory = os.path.join(orbit_fits_directory, object_id)
    generated_variants_directory = os.path.join(generated_variants_directory, object_id)
    tube_directory = os.path.join(tube_directory, object_id)

    # Check if there are existing variants in the output directory
    has_existing_variants = True
    if len(os.listdir(generated_variants_directory)) == 0:
        # TODO: Make this check more robust by checking for specific files
        has_existing_variants = False

    # Run the variant generation
    variants_data = None
    if override_existing_results or has_existing_variants == False:
        if tube_type == "historical":
            print("Generating historical uncertainty variants for", object_id)
            variants_data = historical_uncertainty.generateVariants(
                mpc_directory,
                orbit_fits_directory,
                generated_variants_directory,
                configuration_data
            )
        elif tube_type == "sectioned":
            print("Generating sectioned uncertainty variants for", object_id)
            variants_data = sectioned_uncertainty.generateVariants(
                mpc_directory,
                orbit_fits_directory,
                generated_variants_directory,
                configuration_data
            )
        else:
            assert False, "Unknown tube type: " + tube_type
    # Or load existing variants data from file
    else:
        print("Loading existing variants from", generated_variants_directory)
        variants_data = loadVariantsData(generated_variants_directory)

    # Run the polygon generation
    # TODO: Only do this if there already isnt a tube file in the output directory
    print("Generating tube ellipses for", object_id)
    time_ellipses = ellipse.createEllipses(
        variants_data,
        tube_directory,
        configuration_data
    )

    # Create the final tube file
    tube_filename = "tube_" + object_id.replace(" ", "_") + ".json"
    print("Writing tube file", tube_filename)
    tube.writeTube(
        tube_filename,
        tube_directory,
        time_ellipses,
        configuration_data
    )

    # Generate transforms for visualization in OpenSpace
    transforms_filename = "transforms_" + object_id.replace(" ", "_") + ".asset"
    print("Writing transforms file", transforms_filename)
    toTransforms.generateTransforms(
        transforms_filename,
        transforms_directory,
        time_ellipses,
        configuration_data
    )

    # Generate ellipsoids for visualization in OpenSpace
    ellipsoids_filename = "ellipsoids_" + object_id.replace(" ", "_") + ".asset"
    print("Writing ellipsoids file", ellipsoids_filename)
    toEllipsoids.generateEllipsoids(
        ellipsoids_filename,
        ellipsoids_directory,
        time_ellipses,
        transforms_filename,
        configuration_data
    )


"""
    # Select the input data
    input_directory = "./src/orbit_propagation/generated_data/historical/2023 CX1/2023-02-13T02.38.19.001/"
    #input_directory = "./src/data/2004 MN4 prev/2004-12-27T21.28.37.000"
    print("Loading data from", input_directory)

    # Create the directory for output
    out_directory = "./src/generated_data/historical/2023 CX1/2023-02-13T02.38.19.001/"
    #out_directory = "./src/generated_data/prev/2004 MN4 prev/2004-12-27T21.28.37.000"
    os.makedirs(out_directory, exist_ok = True)
    print("Results will be stored in", out_directory)

    # Read the data
    variants_data = loadVariantsData(input_directory)

    # Get the polygons of the tube
    #old.main(input_dir, out_dir)
    time_ellipses = ellipse.createEllipses(
        data,
        num_ellipse_samples,
        out_directory,
        save_textures,
        texture_resolution,
        do_plotting
    )

    # Perform analysis (if needed)

    # Write the tube file
    tube.writeTube(
        tube_filename,
        out_directory,
        time_ellipses,
        num_ellipse_samples
    )"""
