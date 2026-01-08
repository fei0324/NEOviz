import os
import math
import ray
import numpy as np
import pandas as pd
import pyarrow as pa

from astropy.time import Time
from adam_core.time import Timestamp
from adam_core.orbits import Orbits
from adam_core.orbits import VariantOrbits
from adam_assist import ASSISTPropagator

import variant_generation.util as util
import variant_generation.kernels as kernels
#from main import VariantsData


# TODO: Set this higher when multithreading is supported
MAX_THREADS = 1


def generateVariants(mpc_directory, orbit_fits_directory, output_directory,
                     configuration):
    """
    

    Input:
        mpc_directory: The directory containing the MPC submission data
        orbit_fits_directory: The directory containing the best fit orbit for each
                              submission
        output_directory: The directory to save the generated variants to. This will only
                          be used if the setting 'save_intermediate_results' of the
                          configuration is True.
        configuration: A dictionary containing all configuration parameters. This
                       includes the time to start and end the propagation.   
    """
    # Initialize the propagator
    if not ray.is_initialized():
        ray.init(num_cpus = MAX_THREADS)
    propagator = ASSISTPropagator()

    # Store configuration parameters
    start_time = Time(configuration["start_time"], format = "isot")
    end_time = Time(configuration["end_time"], format = "isot")
    gap_percentage = configuration["gap_percentage"]
    num_variants = configuration["num_variants"]
    
    # Load the submission and best fit orbit data
    submissions, submission_orbits = util.loadSubmissionsAndOrbits(
        mpc_directory,
        orbit_fits_directory
    )

    # Get submission timestamps
    submission_times = Time(
        submissions["timestamp"].values,
        scale = "utc",
        format = "datetime64"
    )

    # Process each submission
    for i, submission_orbit in enumerate(submission_orbits):
        # Get the submission information (ID, timestamp)
        orbit_id = submission_orbit.orbit_id[0].as_py()
        submission_number, submission_id = orbit_id.split("::")
        submission_number = int(submission_number)

        submission_time = submission_times[submission_number]
        submission_time_str = str(submission_time)[:23]
        submission_time_str = submission_time_str.replace(":", ".")
        print("Processing submission", submission_number, "at time", submission_time_str)

        # Check the submission time
            # If the time is before the start time, continue to the next submission.
            # This is common since the orbit fitting needs a number of observations to
            # become stable, so many submissions in the beginning needs to be skipped.

            # If the time is past the end time then we stop as well
        
        # Create a new data folder in the output directory for this submission
        submission_output_directory = os.path.join(output_directory, submission_time_str)
        os.makedirs(submission_output_directory, exist_ok = True)

        # Calculate the time to the next submission
        if i < len(submission_orbit) - 1:
            next_submission_time = submission_times[submission_number + 1]
        else:
            # Or if this is the last submission, use the set end time
            print("This is the last submission, using end time")
            next_submission_time = Time(end_time, format = "isot")

        # Create a new data folder in the output directory for this submission,
        # if requested
        submission_output_directory = None
        if configuration["save_intermediate_results"]:
            submission_output_directory = os.path.join(
                output_directory,
                submission_time_str
            )
            os.makedirs(submission_output_directory, exist_ok = True)
        
        # Create sample times from now to the next submission time
        propagation_times, num_time_steps = util.getTimeSteps(
            submission_time,
            next_submission_time
        )

        # Add a small time gap after the start time and before the end time to ensure we
        # capture the changed uncertainty effect
        # TODO: This needs to be inside the getTimeSteps function
        if submission_number > 0: # No gap for the first submission
            time_steps[0] = time_steps[0] + gap_percentage * \
                (time_steps[1] - time_steps[0]) / 100.0
        if submission_number < len(submission_orbit) - 1: # No gap for the last submission
            time_steps[-1] = time_steps[-1] - gap_percentage * \
                (time_steps[-1] - time_steps[-2]) / 100.0

        # Propagate the best fit orbit for this submission forward in time to the end
        # time, using the propagation sample times from the previous step
        propagation_times = Timestamp.from_mjd(time_steps, scale = "utc")
        propagated_best_fit_orbit = util.propagateBestFitOrbit(
            submission_orbit,
            propagator,
            propagation_times,
            num_variants * 2, # Use more samples for a better covariance matrix estimation
            num_threads = MAX_THREADS
        )

        # Generate variants based on the propagated best fit orbit
        propagated_variants = util.propagateVariants(
            propagated_best_fit_orbit,
            propagator,
            propagation_times,
            num_variants,
            submission_number,
            submission_id,
            num_threads = MAX_THREADS
        )

        # Finally, save the variant coordinates and velocities to file
        # The reference frame used by adam is Ecliptic J2000
        # Positions in AU (.r is the position vector)
        print("Saving propagated variants data to files")
        variants_coordinates_filepath = os.path.join(
            submission_output_directory,
            "variants_coordinates_" + str(num_variants)
        )
        np.save(variants_coordinates_filepath, propagated_variants.coordinates.r)

        # Velocities in AU/day (.v is the velocity vector)
        variants_velocity_filepath = os.path.join(
            submission_output_directory,
            "variants_velocity_" + str(num_variants)
        )
        np.save(variants_velocity_filepath, propagated_variants.coordinates.v)

        # Save the time steps to file as well in isot format
        times_isot = propagated_best_fit_orbit.coordinates.time.to_astropy().isot
        times_filepath = os.path.join(submission_output_directory, "times_isot")
        np.save(times_filepath, times_isot)

        # Save the covariance matricies too. This is a 6x6 matrix for each timestep of
        # the best fitting orbit with the covariance for position and velocity
        covariances = propagated_best_fit_orbit.coordinates.covariance.to_matrix()
        covariances_filepath = os.path.join(submission_output_directory, "covariances")
        np.save(covariances_filepath, covariances)

        # Create SPICE kernels for each variant, if requested
        if save_kernels:
            print("Creating SPICE kernels for the propagated variants")
            kernels_output_directory = os.path.join(
                submission_output_directory,
                "variant_kernels"
            )
            kernels.create_kernels(propagated_variants, kernels_output_directory)

    return None


if __name__ == "__main__":
    """
    Main function to calculate the sectioned uncertainty for a chosen asteroid from the
    start of its first observation, to a set end time. The sectioned uncertainty
    is calculated by sampling variant orbits based on the best fitting orbit for each
    observation submission. The variants are then propagated forward in time to the next
    submission. The generated variants and some additional information is saved to file 
    in the given output directory. This data can then be used to create a sectioned 
    uncertainty tube. The sectioned uncertainty adds more data to the uncertainty tube at
    each submission, showing how the uncertainty changes over time as more observations
    are added.
    """
    # Parse any input arguments
    # Settings
    num_variants = 5000

    # Choose an asteroid object
    object_id = "2012 DA14"
    #object_id = "1998 SG172"
    print("Calculating historical uncertainty for object:", object_id)

    # Set the time to start propagation and when to end propagation
    end_time = Time("2023-02-16T00:00:00.000", format = "isot") # 2012 DA14
    #end_time = Time("2013-12-01T00:00:00.000", format = "isot")
    #end_time = Time("2010-01-01T00:00:00.000", format = "isot")
    #end_time = Time("2075-01-01T00:00:00.000", format = "isot") # 2000 SG344

    # Setup input directories
    orbit_fits_directory = os.path.join("./data/orbit_fits", object_id)
    submissions_directory = os.path.join("./data/mpc_data", object_id)

    # Setup output directories
    output_directory = os.path.join("./generated_data/sectioned", object_id)
    os.makedirs(output_directory, exist_ok = True)

    # Find the input files
    # Observation submission history
    submissions_file = os.path.join(submissions_directory, "submissions.parquet")
    if not os.path.isfile(submissions_file):
        print("Could not find submissions file ", submissions_file)
        assert False, "Missing submissions file"

    # Calculated best orbit fit for each submission
    orbits_file = os.path.join(orbit_fits_directory, "orbits.parquet")
    if not os.path.isfile(orbits_file):
        print("Could not find orbits file ", orbits_file)
        assert False, "Missing orbits file"

    # Read the data in the input files
    print("Loading submissions from", submissions_file)
    submissions = pd.read_parquet(submissions_file)
    print("Loaded", len(submissions), "submissions:")

    print("Loading fitted orbits from", orbits_file)
    submission_orbits = util.FittedOrbits.from_parquet(orbits_file)
    print("Loaded", len(submission_orbits), "orbits")
    submission_orbits_dataframe = submission_orbits.to_dataframe()
    print(submission_orbits_dataframe)

    # The number of orbits should be one less than the number of submissions
    if len(submission_orbits) != len(submissions) - 1:
        print("\033[41mWarning:\033[0m Missmatch between number of submissions and number of fitted orbits")

    # Calculate the sectioned uncertainty of the chosen asteroid over the set timeframe
    calcSectionedUncertainty(
        output_directory,
        submissions,
        submission_orbits,
        end_time,
        num_variants
    )

    # We then have several shorter sections for each submission. These need to be combined. The variant SPICE kernels is not combinable, they will stay as is in their respective folder
    # We could combine the covariance matrices, positions, velocities and times into single files. This would show the drastic changes in uncertainty over time as more observations are added. 
