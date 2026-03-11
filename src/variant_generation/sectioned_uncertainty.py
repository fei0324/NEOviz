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
MAX_THREADS = 2
VARIANTS_CHUNK_SIZE = 128
SAMPLES_CHUNK_SIZE = 256


def generateVariants(mpc_directory, orbit_fits_directory, output_directory,
                     configuration):
    """
    Calculate the sectioned uncertainty for a chosen asteroid from the start of its first
    observation, to a set end time. The sectioned uncertainty is calculated by sampling
    variant orbits based on the best fitting orbit for each observation submission. The
    variants are then propagated forward in time to the next submission. This data can
    then be used to create a sectioned uncertainty tube. The sectioned uncertainty adds
    more data to the uncertainty tube at each submission, showing how the uncertainty
    changes over time as more observations are added.

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
    propagator = ASSISTPropagator()

    # Store configuration parameters
    start_time = Time(configuration["start_time"], format = "isot")
    end_time = Time(configuration["end_time"], format = "isot")
    gap_percentage = configuration["gap_percentage"]
    num_variants = configuration["num_variants"]

    use_high_res_timeframe = configuration["use_high_res_timeframe"]
    high_res_start_time = None
    high_res_end_time = None
    if use_high_res_timeframe:
        high_res_start_time = Time(configuration["high_res_start_time"], format = "isot") 
        high_res_end_time = Time(configuration["high_res_end_time"], format = "isot") 

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

    # Process each submission and add the generated variants to a list
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

        if submission_time > end_time:
            # If the time is past the end time then we stop
            print("Reached end time, stopping propagation")
            break

        # Check if we are within the limit of the high resolution timeframe (if used)
        if use_high_res_timeframe:
            if submission_time > high_res_end_time:
                print("Reached end of high resolution timeframe, stopping propagation")
                break

        # Create a new data folder in the output directory for this submission,
        # if requested
        submission_output_directory = os.path.join(
            output_directory,
            submission_time_str
        )
        os.makedirs(submission_output_directory, exist_ok = True)

        # Calculate the time to the next submission
        next_submission_time = None
        if submission_number < len(submission_orbits) - 1:
            next_submission_time = submission_times[submission_number + 1]
        else:
            if use_high_res_timeframe and high_res_end_time < end_time:
                # If this is the last submission but we are still within the high
                # resolution timeframe, use the end of the high res timeframe as the next
                # submission time
                print("This is the last submission, using high res end time")
                next_submission_time = high_res_end_time
            else:
                # Or use the set end time
                print("This is the last submission, using end time")
                next_submission_time = Time(end_time, format = "isot")
        
        # If we are too early for the high resolution timeframe, skip this submission
        if use_high_res_timeframe and next_submission_time < high_res_start_time:
            continue

        # Create sample times from now to the next submission time
        propagation_times = None
        num_time_steps = None
        if use_high_res_timeframe:
            # Check if we are in range to use the high resolution start time 
            use_high_res_start = False
            used_start_time = submission_time
            if high_res_start_time > submission_time and \
                high_res_start_time < next_submission_time:  
                use_high_res_start = True
                used_start_time = high_res_start_time
            
            # Check if we are in range to use the high resolution end time
            use_high_res_end = False
            used_end_time = next_submission_time
            if high_res_end_time > submission_time and \
                high_res_end_time < next_submission_time:
                use_high_res_end = True
                used_end_time = high_res_end_time
            
            # If either the high resolution start or end time is within range of the
            # current submission and the next submission, then we use the high resolution
            # sampling.
            if use_high_res_start or use_high_res_end:
                propagation_times, num_time_steps = adam_util.getTimeSteps(
                    submission_time,
                    next_submission_time,
                    configuration["sample_multiplier"],
                    used_start_time,
                    used_end_time,
                    configuration["high_res_sample_multiplier"]
                )
            else:
                # If not, then use the normal sampling rate for this tube section
                propagation_times, num_time_steps = adam_util.getTimeSteps(
                    submission_time,
                    next_submission_time,
                    configuration["sample_multiplier"]
                )
        else:
            propagation_times, num_time_steps = adam_util.getTimeSteps(
                submission_time,
                next_submission_time,
                configuration["sample_multiplier"]
            )
        
        print("Number of time steps", num_time_steps)
        print("From", submission_time, "to", next_submission_time)

        # Add a small time gap after the start time and before the end time to ensure we
        # capture the changed uncertainty effect
        if submission_number > 0: # No gap for the first submission
            propagation_times[0] = propagation_times[0] + gap_percentage * \
                (propagation_times[1] - propagation_times[0]) / 100.0
        if submission_number < len(submission_orbit) - 1: # No gap for the last submission
            propagation_times[-1] = propagation_times[-1] - gap_percentage * \
                (propagation_times[-1] - propagation_times[-2]) / 100.0

        # Convert to a Timestamp array
        propagation_times = Timestamp.from_mjd(
            propagation_times.reshape(-1),
            scale = "utc"
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
                chunk_size = SAMPLES_CHUNK_SIZE
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
        
        # Generate variants based on the propagated best fit orbit and propagate them
        # over time
        propagated_variants = adam_util.propagateVariants(
            propagated_best_fit_orbit,
            propagator,
            propagation_times,
            num_variants,
            submission_number,
            submission_id,
            configuration,
            num_threads = MAX_THREADS,
            chunk_size = VARIANTS_CHUNK_SIZE,
            submission_output_directory = submission_output_directory
        )

        # Prosess the propagated variants to get ordered arrays of coordinates,
        # velocities, and times
        ordered_variants_coordinates, \
        ordered_variants_velocities, \
        times_isot, \
        ordered_bfo_coordinates = adam_util.processVariants(
            propagated_variants,
            num_time_steps,
            propagated_best_fit_orbit
        )

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

        # Convert the covariance matrix of the propagated best fit orbit to a numpy array
        covariances = propagated_best_fit_orbit.coordinates.covariance.to_matrix()

        # Store the generated data in a data object
        variant_data = adam_util.VariantsData(
            ordered_variants_coordinates,
            ordered_variants_velocities,
            times_isot,
            covariances,
            num_variants,
            num_time_steps,
            ordered_bfo_coordinates
        )
        variants_data.append(variant_data)

    return variants_data


if __name__ == "__main__":

    # Set the time to start propagation and when to end propagation
    end_time = Time("2023-02-16T00:00:00.000", format = "isot") # 2012 DA14
    #end_time = Time("2013-12-01T00:00:00.000", format = "isot") # 2012 DA14 ?
    #end_time = Time("2010-01-01T00:00:00.000", format = "isot")
    #end_time = Time("2075-01-01T00:00:00.000", format = "isot") # 2000 SG344
