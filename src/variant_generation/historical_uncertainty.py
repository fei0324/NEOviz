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

import variant_generation.adam_util as adam_util
import variant_generation.kernels as kernels
from main import VariantsData

# TODO: Set this higher when multithreading is supported
MAX_THREADS = 8
CHUNK_SIZE = 16

# The number of meters in one Astronomical Unit (AU)
AU = 149597870700

def generateVariants(mpc_directory, orbit_fits_directory, output_directory,
                     configuration):
    """
    Sample variants of the asteroid based on the best fitting orbit for each submission.
    When the time of historical interest has been reached, we stop adding more data as
    submissions come in. This effectevly simulates the knowledge we had of the asteroid
    at that point in time. We then propagate sampled variants to the end time as an
    proximation of the uncertainty of the orbit. The generated variants and some
    additional information is put intp a dataobject and returned to the caller.

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
    time_of_historical_interest = Time(
        configuration["time_of_historical_interest"],
        format = "isot"
    )
    num_variants = configuration["num_variants"]

    use_high_res_timeframe = configuration["use_high_res_timeframe"]
    high_res_start_time = Time(configuration["high_res_start_time"], format = "isot") 
    high_res_end_time = Time(configuration["high_res_end_time"])
    
    # Load the submission and best fit orbit data
    submissions, submission_orbits = adam_util.loadSubmissionsAndOrbits(
        mpc_directory,
        orbit_fits_directory
    )

    # Get submission timestamps
    submission_times = Time(
        submissions["timestamp"].values,
        scale = "utc",
        format = "datetime64"
    )

    # Process each submission. Usually the time of interest is very close to the start
    # time (Since the uncertainty is high when the object is newly discovered) so only a
    # very few number of submissions will be processed. 
    variants_data = []
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
        if submission_time < start_time:
            # If the time is before the start time, continue to the next submission.
            # This is common since the orbit fitting needs a number of observations to
            # become stable, so many submissions in the beginning needs to be skipped.
            continue
        if submission_time > time_of_historical_interest:
            # If the time is past the time of historical interest, stop processing
            # further submissions
            print("Reached time of historical interest, skipping any further submissions")
            break
        if submission_time > end_time:
            # If the time is past the end time then we stop as well
            print("Reached end time, stopping propagation")
            break

        # Create a new data folder in the output directory for this submission,
        # if requested
        submission_output_directory = None
        if configuration["save_intermediate_results"]:
            submission_output_directory = os.path.join(
                output_directory,
                submission_time_str
            )
            os.makedirs(submission_output_directory, exist_ok = True)

        # Create sample times from now to the end time
        propagation_times, num_time_steps = adam_util.getTimeSteps(
            submission_time,
            end_time
        )
        
        # Propagate the best fit orbit for this submission forward in time
        # TODO: Make it posible to save this data to file and load it again so we do not
        # have to re-run it so often.
        # TODO: Use more samples for a better covariance matrix estimation
        num_samples = num_variants 
        propagated_best_fit_orbit = adam_util.propagateBestFitOrbit(
            submission_orbit,
            propagator,
            propagation_times,
            num_samples,
            num_threads = MAX_THREADS,
            chunk_size = CHUNK_SIZE
        )

        # If high resolution time frame is used, create new propagation times for the
        # variants in that time frame
        start_index = 0
        num_variant_time_steps = num_time_steps
        variant_propagation_times = propagation_times
        if use_high_res_timeframe:
            # Find the timestep in the propagated best fit orbit that is closest to the
            # high resolution start time
            high_res_start_time_actual, start_index = adam_util.findVariantsStartTime(
                propagated_best_fit_orbit,
                high_res_start_time
            )

            # Create new high resolution time steps for the variants over the high
            # resolution timeframe
            variant_propagation_times, num_variant_time_steps = adam_util.getTimeSteps(
                high_res_start_time_actual.to_astropy(),
                high_res_end_time,
                configuration["high_res_sample_multiplier"]
            )

        # Generate variants based on the propagated best fit orbit and propagate them
        # over time
        propagated_variants = adam_util.propagateVariants(
            propagated_best_fit_orbit,
            propagator,
            variant_propagation_times,
            num_variants,
            submission_number,
            submission_id,
            start_index,
            MAX_THREADS,
            CHUNK_SIZE
        )

        # The coordinates and velocities are orderd per orbit, but we want to find
        # all varaint coordinates and velocities per timestep to create time-slices
        ordered_variants_coordinates = []
        ordered_variants_velocities = []

        # We only take the coordinate or velocity cooresponding to the t:th timestamp
        # for each orbit. Also rescale the coordinates from AU to meters and velocities
        # from AU/s to m/s.
        print("Sorting coordinates and velocities of variants")
        for t in range(num_variant_time_steps):
            # The reference frame used by adam is Ecliptic J2000
            # Positions in AU (.r is the position vector)
            ordered_variants_coordinates.append(
                propagated_variants.coordinates.r[t::num_variant_time_steps] * AU
            )
            # Velocities in AU/day (.v is the velocity vector)
            ordered_variants_velocities.append(
                propagated_variants.coordinates.v[t::num_variant_time_steps] * AU
            )
        print("Finished sorting coordinates and velocities of variants")

        # Create a numpy array from the list of timesteps for the variants
        times_isot = propagated_variants.coordinates.time.to_astropy().isot

        # Do the same with the covariance matricies. This is a 6x6 matrix for each
        # timestep of the best fitted orbit with the covariance for position and
        # velocity.
        # TODO: Remember that the timesteps for the propagated_best_fit_orbit (i.e the
        # covariance matricies) is not the same as the variants. Can we generate new
        # covariance matricies for the high resolution timeframe? Then we have a finer
        # resolution for the covariance matricies.
        covariances = propagated_best_fit_orbit.coordinates.covariance.to_matrix()

        # Store the generated data in a data object
        variant_data = VariantsData(
            ordered_variants_coordinates,
            ordered_variants_velocities,
            times_isot,
            covariances,
            num_variants,
            num_variant_time_steps
        )
        variants_data.append(variant_data)

        # Save variants data to file, if requested
        if configuration["save_intermediate_results"]:
            # NOTE: The VariantsData object expect the order of the variant coordinates
            # and velocities to already be sorted per timestep. However, to keep
            # backwards compatibility with previous variant files, we save the variants
            # to file in the original order
            adam_util.saveVariantsToFile(
                propagated_variants,
                times_isot,
                covariances,
                num_variants,
                submission_output_directory
            )

        # Create SPICE kernels for each variant, if requested
        if configuration["save_kernels"]:
            kernels_output_directory = os.path.join(
                submission_output_directory,
                "variant_kernels"
            )

            # Create kernels and save them to file
            kernels.saveKernels(
                propagated_variants,
                num_variants,
                kernels_output_directory,
                configuration
            )
             
    return variants_data

"""
if __name__ == "__main__":
    # TODO: Parse any input arguments
    # Settings
    num_variants = 10000
    save_kernels = False

    # Choose an asteroid object
    object_id = "2004 MN4"
    #object_id = "2023 CX1"
    print("Calculating historical uncertainty for object:", object_id)

    # Set the time to start propagation and when to end propagation
    start_time = Time("2004-12-27T21:00:00.000", format = "isot") # 2004 MN4 
    #start_time = Time("2023-02-13T02:38:00.000", format = "isot") # 2023 CX1
    print("Start time:", start_time.iso)

    #end_time = Time("2005-12-27T21:00:00.000", format = "isot") # 2004 MN4 (test)
    end_time = Time("2029-12-31T00:00:00.000", format = "isot") # 2004 MN4 
    #end_time = Time("2023-02-13T03:40:00.000", format = "isot") # 2023 CX1
    print("End time:", end_time.iso)

    # Set the time of historical interest, from this point onwards we stop adding
    # more observation submission data
    time_of_interest = Time("2004-12-28T00:00:00.000", format = "isot") # 2004 MN4 
    #time_of_interest = Time("2023-02-13T02:39:00.000", format = "isot") # 2023 CX1

    # Set a start and end time for the variants. This can be sharter than the overall
    # timeframe and sampled at a much higher resolution.
    use_high_res_timeframe = True # 2004 MN4
    #use_high_res_timeframe = False # 2023 CX1

    #high_res_start_time = Time("2005-01-01T21:00:00.000", format = "isot") # 2004 MN4 (test)
    high_res_start_time = Time("2029-03-13T00:00:00.000", format = "isot") # 2004 MN4
    #high_res_start_time = start_time # 2023 CX1

    #high_res_end_time = Time("2005-02-01T21:00:00.000", format = "isot") # 2004 MN4 (test)
    high_res_end_time = Time("2029-05-13T00:00:00.000", format = "isot") # 2004 MN4
    #high_res_end_time = end_time # 2023 CX1
"""