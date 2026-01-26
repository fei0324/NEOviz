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

# Settings for chunking the propagation
MAX_THREADS = 8
CHUNK_SIZE = 256 # 128

# The number of meters in one Astronomical Unit (AU)
AU = 149597870700.0
SECONDS_PER_DAY = 86400.0

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
    high_res_start_time = None
    high_res_end_time = None
    if use_high_res_timeframe:
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
        print("\nProcessing submission", submission_number, "at time",
            submission_time_str)
        
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
        
        # Check if there are existing results for this submission
        parquet_path = os.path.join(
            submission_output_directory,
            "propagated_best_fit_orbit_"+ str(num_variants) + ".parquet"
        )
        parquet_exists = os.path.exists(parquet_path)

        # If the file exist, and we want to use it according to the configuration, then
        # use it, even if the configuration override flag is true
        should_propagate = True
        if configuration["override_existing_results"] and \
           configuration["use_existing_best_fit_orbit"] and parquet_exists:
           should_propagate = False
        elif not configuration["override_existing_results"] and parquet_exists:
            should_propagate = False

        # Propagate the best fit orbit for this submission forward in time
        propagated_best_fit_orbit = None
        if should_propagate:
            # TODO: Use more samples for a better covariance matrix estimation
            propagated_best_fit_orbit = adam_util.propagateBestFitOrbit(
                submission_orbit,
                propagator,
                propagation_times,
                num_variants,
                num_threads = MAX_THREADS,
                chunk_size = CHUNK_SIZE
            )

            if configuration["save_intermediate_results"]:
                # Save the propagated best fit orbit to file
                print("Saving propagated best fit orbit to", parquet_path)
                propagated_best_fit_orbit.to_parquet(parquet_path)
        else:
            # Load the stored data
            print(
                "Loading existing data for propagated best fit orbit from",
                parquet_path
            )
            propagated_best_fit_orbit = Orbits.from_parquet(parquet_path)

        # Check if there are existing variants in the output directory
        has_existing_variants_data = adam_util.hasVariantsData(
            submission_output_directory,
            num_variants
        )

        # If the files exist, and we want to use them according to the configuration, then
        # use them, even if the configuration override flag is true
        should_generate_variants = True
        if configuration["override_existing_results"] and \
           configuration["use_existing_variants"] and has_existing_variants_data:
           should_generate_variants = False
        elif not configuration["override_existing_results"] and \
            has_existing_variants_data:
            should_generate_variants = False

        # Use existing data instead of generating a new if possible
        if not should_generate_variants:
            print("Loading existing variants from", submission_output_directory)
            variant_data = adam_util.loadVariantsData(
                submission_output_directory,
                propagated_best_fit_orbit,
                num_variants
            )
            variants_data.append(variant_data)
            continue

        # If high resolution time frame is used, create new propagation times for the
        # variants in that time frame.
        # NOTE: If previous variants exist, and are being used, then if the high res
        # timeframe is different from before (when the variants were generated and stored)
        # it will use the old timesteps. A rerun is only triggered when the
        # override_existing_results flag is set to True (while use_existing_variants is
        # False). Or if the number of variants is different than before.
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

        # Prosess the propagated variants to get ordered arrays of coordinates,
        # velocities, and times
        ordered_variants_coordinates, \
        ordered_variants_velocities, \
        times_isot = adam_util.processVariants(propagated_variants, num_time_steps)

        # Save variants data to file, if requested
        if configuration["save_intermediate_results"]:
            variants_filepath = os.path.join(
                submission_output_directory,
                "propagated_variants_" + str(num_variants) + ".parquet"
            )
            print("Saving propagated variants to", variants_filepath)
            propagated_variants.to_parquet(variants_filepath)

        # Create SPICE kernels for each variant, if requested
        if configuration["save_kernels"]:
            kernels_output_directory = os.path.join(
                submission_output_directory,
                "variant_kernels_" + str(num_variants)
            )
            os.makedirs(kernels_output_directory, exist_ok = True)

            # Create kernels and save them to file
            kernels.saveKernels(
                propagated_variants,
                num_variants,
                kernels_output_directory,
                configuration
            )

        # TODO: Fix this
        # Recompute covariances of propagated variants, collapse the variants into a
        # single orbit to get one covariance matrix per timestep. Do this last as it will
        # change the variants data structure
        #collapsed_variants = propagated_variants.collapse(propagated_best_fit_orbit)
        #covariances = collapsed_variants.coordinates.covariance.to_matrix()
        #covariances = propagated_best_fit_orbit.coordinates.covariance.to_matrix()
        covariances = propagated_variants.coordinates.covariance.to_matrix()

        # Store the generated data in a data object
        variant_data = adam_util.VariantsData(
            ordered_variants_coordinates,
            ordered_variants_velocities,
            times_isot,
            covariances,
            num_variants,
            num_variant_time_steps
        )
        variants_data.append(variant_data)
             
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