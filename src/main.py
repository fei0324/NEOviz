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
import tube_generation.util as util
import transform_generation.toTransforms as toTransforms
import transform_generation.toEllipsoids as toEllipsoids
import variant_generation.adam_util as adam_util


# Path to the SPICE kernel files, to initialize SPICE
LSK_KERNEL = "../data/kernels/lsk/naif0012.tls.pc"
SPK_KERNEL = "../data/kernels/spk/de432s.bsp"
PCK_KERNEL = "../data/kernels/pck/pck00011.tpc"

# The number of meters in one Astronomical Unit (AU)
AU = 149597870700
SECONDS_PER_DAY = 86400

# Current version of the configuration files
CONFIGURATION_VERSION = "0.1"


if __name__ == "__main__":
    """
    """

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
    #configuration_file = "../config/historical_2004_MN4_high_res.json"
    #configuration_file = "../config/historical_2004_MN4_test.json"
    #configuration_file = "../config/sectioned_2012_DA14_test.json"
    configuration_file = "../config/historical_2023_CX1_test.json"
    configuration_data = None

    # Try to read the configuration json file
    try:
        with open(configuration_file, 'r') as file:
            configuration_data = json.load(file)
    except FileNotFoundError:
        print("Error: The file", configuration_file, "was not found.")
        assert False, "Configuration file not found"
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from file", configuration_file)
        assert False, "Configuration file could not be decoded"
    
    # Check version number of the configuration file
    version = configuration_data["version"]
    if version != CONFIGURATION_VERSION:
        print("Warning: Configuration file version", version, 
              "does not match expected version", CONFIGURATION_VERSION)
        assert False, "Configuration file version mismatch"

    # Extract the necessary configuration settings
    object_id = configuration_data["object_id"]
    object_identifier = object_id.replace(" ", "_")
    override_existing_results = configuration_data["override_existing_results"]
    has_impact = configuration_data["has_impact"]

    # Create paths to all input data
    tube_type = configuration_data["tube_type"]
    input_directory = configuration_data["input_directory"]
    mpc_directory = os.path.join(input_directory, "mpc_data/")
    orbit_fits_directory = os.path.join(input_directory, "orbit_fits/")
    output_directory = configuration_data["output_directory"]

    # If there are impacts then make sure to read the impact file to exlude them from the
    # propagation at then moment of impact
    impact_data = None
    if has_impact:
        # Impact code will only work for historic tubes
        assert tube_type == "historical", "Cannot handle impacts for sectioned tubes"

        # Find the impact file
        impact_file = os.path.join(
            input_directory,
            "impact/" + object_identifier + "_impact.txt"
        )

        # Read the impact file to get the impact times and identifiers
        # It is very important that the variants used to generate the impact file is the
        # same that will be used to create this tube
        impact_data = adam_util.loadImpactData(impact_file, object_id)
    
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

    os.makedirs(generated_variants_directory, exist_ok = True)
    os.makedirs(tube_directory, exist_ok = True)

    # Run the variant generation
    variants_data = None
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
    
    # Run the polygon generation
    # TODO: Only do this if there already isnt a tube file in the output directory
    print("Generating tube ellipses for", object_id)
    time_ellipses = ellipse.createEllipses(
        variants_data,
        tube_directory,
        configuration_data,
        impact_data
    )

    # Create the final tube file
    tube_filename = "tube_" + object_id.replace(" ", "_") + ".json"
    tube.writeTube(
        tube_filename,
        tube_directory,
        time_ellipses,
        configuration_data
    )

    # Generate transforms for visualization in OpenSpace
    transforms_filename = "transforms_" + object_id.replace(" ", "_") + ".asset"
    toTransforms.generateTransforms(
        transforms_filename,
        transforms_directory,
        time_ellipses,
        configuration_data
    )

    # Generate ellipsoids for visualization in OpenSpace
    ellipsoids_filename = "ellipsoids_" + object_id.replace(" ", "_") + ".asset"
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
