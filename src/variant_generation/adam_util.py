import os
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import quivr as qv

from quivr.concat import concatenate

from adam_core.time import Timestamp
from adam_core.orbits import Orbits
from adam_core.coordinates import CartesianCoordinates
from adam_core.orbits import VariantOrbits

# TODO: Set the correct minimum number of timesteps required by the propagator
MINIMUM_TIMESTEPS = 80

# The propagator is slow and cannot handle too amny samples at the same time (even if it
# chunks) so we need to perform some additional chunking manually.
MAX_NUM_VARIANTS_PER_BATCH = 512

# Table to store resulting fitted orbits along with metadata about the result from the
# fitting process.
class FittedOrbits(qv.Table):
    orbit_id = qv.LargeStringColumn(default = lambda: str(uuid.uuid4()))
    object_id = qv.LargeStringColumn()
    coordinates = CartesianCoordinates.as_column()
    included_observations = qv.Int64Column()
    rejected_observations = qv.Int64Column()
    arc_length = qv.Float64Column()

    def to_orbits(self) -> Orbits:
        return Orbits.from_kwargs(
            orbit_id = self.orbit_id,
            object_id = self.object_id,
            coordinates = self.coordinates,
        )


def loadSubmissionsAndOrbits(mpc_directory, orbit_fits_directory):
    """
    Load the observation submission history and the best fit orbit for each submission.
    Input:
        mpc_directory: The directory containing the MPC submission data
        orbit_fits_directory: The directory containing the best fit orbit for each
                              submission
    Output:
        submissions: A pandas dataframe containing the submission history
        submission_orbits: A list of FittedOrbits containing the best fit orbit for each
                           submission
    """
    # Find the input files
    # Observation submission history
    submissions_file = os.path.join(mpc_directory, "submissions.parquet")
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
    #print(submissions)

    print("Loading fitted orbits from", orbits_file)
    submission_orbits = FittedOrbits.from_parquet(orbits_file)
    print("Loaded", len(submission_orbits), "orbits")

    # The number of orbits should be one less than the number of submissions
    if len(submission_orbits) != len(submissions) - 1:
        print("\033[41mWarning:\033[0m "\
            "Missmatch between number of submissions and number of fitted orbits"
        )

    return submissions, submission_orbits


def clacNumTimeSteps(start_interval, end_interval, samples_per_day = 1):
    """
    Calculate the number of time steps to use for the propagation based on the given time
    interval

    Input:
        start_interval: An astropy Time object representing the start of the time interval
        end_interval: An astropy Time object representing the end of the time interval
        samples_per_day: The number of samples to take per day
    Output:
        num_steps: The number of time steps to use for the propagation based on the
                   given time interval. 
    """

    # Calculate the time difference
    time_difference = end_interval - start_interval
    print("Time difference:", time_difference.jd)
    time_interval = time_difference.jd

    # Check the time interval and set the number of timesteps accordingly
    num_steps = MINIMUM_TIMESTEPS
    if time_interval > 1.0:
        # If the number of days are more than 1, then multiply the number of days by the
        # number of samples per day to get the number of timesteps for the propagation.
        num_steps = math.ceil(time_interval)
        num_steps *= samples_per_day

    # Make sure the number of timesteps is at least the minimum required by the propagator
    return max(num_steps, MINIMUM_TIMESTEPS)


def getTimeSteps(start_interval, end_interval, samples_per_day = 1):
    """
    Generate a Timestamp array containing time steps between the start and end interval
    Input:
        start_interval: An astropy Time object representing the start of the time interval
        end_interval: An astropy Time object representing the end of the time interval
        samples_per_day: The number of samples to take per day
    Output:
        time_steps: A Timestamp array containing time steps between the start and end
                    time interval
        num_time_steps: The number of time steps generated
    """
    # Calculate the number of timesteps to use for the propagation
    num_time_steps = clacNumTimeSteps(start_interval, end_interval, samples_per_day)
    print("Number of time steps", num_time_steps)

    # Create a list of time steps between the start and end interval with the desired
    # number of steps in between
    time_steps = np.linspace(
        start_interval.utc.mjd,
        end_interval.utc.mjd,
        num_time_steps,
        endpoint = True
    )

    # Create the Timestamp array from the time steps and return it
    return Timestamp.from_mjd(time_steps.reshape(-1), scale = "utc"), num_time_steps


def findVariantsStartTime(propagated_orbit, desired_start_time):
    """
    Find the index in the propagated orbit that best matches the given desired start
    time, return the time at this index and the index itself.

    Input:
        propagated_orbit: The propagated orbit to search through
        desired_start_time: The desired start time to find in the propagated orbit
    Output:
        actual_start_time: The actual start time found in the propagated orbit
        start_index: The index in the propagated orbit for the actual start time
    """
    # Find the index in the propagated best fit orbit that matches the high
    # ressolution timeframe start time the best
    time_differences = np.abs(
        propagated_orbit.coordinates.time.rescale("utc").mjd() - \
        desired_start_time.utc.mjd
    )
    start_index = np.argmin(time_differences)
    start_index = int(start_index)

    print("Start index matching variants start time:", start_index)
    actual_start_time = propagated_orbit[start_index].coordinates.time.rescale("utc")
    print(
        "Start index cooresponds to time:",
        str(actual_start_time.to_iso8601())[:23]
    )

    return actual_start_time, start_index


def propagateBestFitOrbit(best_fit_orbit, propagator, propagation_times, num_samples,
                         num_threads = 1, chunk_size = 1):
    """
    Propagate the best fit orbit for a submission forward in time to the end time,
    using the propagation sample times.

    Input:
        best_fit_orbit: A util.FittedOrbits object containing the best fit orbit for a 
                        submission
        propagator: The initialized ASSISTPropagator to use for the propagation
        propagation_times: A Timestamp array containing the times to propagate the
                           orbit with. These time steps need to go over the full range of
                           interesting time and cannot be shortened.
        num_samples: The number of samples to draw when creating the covariance matrix
                     using a monte carlo method
        num_threads: The number of threads to use for the propagation. By default no
                     multithreading is used
        chunk_size: The chunk size to use for the propagation. By defalut a chunk size of
                    1 is used
    Output:
        propagated_orbit: The propagated best fit orbit with covariance information
    """

    print("Starting to propagate the best fit orbit")
    print("number of timesteps:", len(propagation_times))
    propagated_orbit = None
    bf_orbit = best_fit_orbit.to_orbits()

    # Do manual batching if the number of samples is too high for the propagator to handle
    if num_samples > MAX_NUM_VARIANTS_PER_BATCH:
        print("The number of samples is larger than the maximum allowed per batch")
        print("Performing manual batching of propagation")

        # Calculate the number of batches needed
        num_batches = math.ceil(num_samples / MAX_NUM_VARIANTS_PER_BATCH)
        print("Number of batches:", num_batches)

        # Propagate each batch separately
        is_first = True
        for batch_index in range(num_batches):
            print("Propagating batch", batch_index + 1, "of", num_batches)

            # Calculate the number of samples for this batch
            start_sample = batch_index * MAX_NUM_VARIANTS_PER_BATCH
            end_sample = min(start_sample + MAX_NUM_VARIANTS_PER_BATCH, num_samples)
            batch_num_samples = end_sample - start_sample
            print("Number of samples in batch:", batch_num_samples)

            # Slice the coordinates of the best fit orbit to only include the samples for
            # this batch
            batch_coordinates = bf_orbit.coordinates
            batch_coordinates.x = bf_orbit.coordinates.x[start_sample:end_sample]
            batch_coordinates.y = bf_orbit.coordinates.y[start_sample:end_sample]
            batch_coordinates.z = bf_orbit.coordinates.z[start_sample:end_sample]
            batch_coordinates.vx = bf_orbit.coordinates.vx[start_sample:end_sample]
            batch_coordinates.vy = bf_orbit.coordinates.vy[start_sample:end_sample]
            batch_coordinates.vz = bf_orbit.coordinates.vz[start_sample:end_sample]

            # Propagate the batch
            propagated_batch = propagator.propagate_orbits(
                Orbits.from_kwargs(
                    orbit_id = bf_orbit.orbit_id,
                    object_id = bf_orbit.object_id,
                    coordinates = batch_coordinates,
                ),
                propagation_times, 
                covariance = True,
                covariance_method = "monte-carlo",
                num_samples = batch_num_samples,
                max_processes = num_threads,
                chunk_size = chunk_size
            )
            
            # Append the propagated batch to the full propagated orbit
            if is_first:
                propagated_orbit = propagated_batch
                is_first = False
            else:
                propagated_orbit = concatenate([propagated_orbit, propagated_batch])

        print("Finished all batches")
        
    else:
        # Propagate the best fit orbit for a submission forward in time using the propagation
        # sample times
        propagated_orbit = propagator.propagate_orbits(
            best_fit_orbit.to_orbits(), 
            propagation_times, 
            covariance = True,
            covariance_method = "monte-carlo",
            num_samples = num_samples,
            max_processes = num_threads,
            chunk_size = chunk_size
        )
        
    print("Finished propagating the best fit orbit")
        
    # Convert the propagated orbit times to UTC
    propagated_orbit = propagated_orbit.set_column(
        "coordinates.time",
        propagated_orbit.coordinates.time.rescale("utc")
    )

    # Return the propagated orbit
    return propagated_orbit


def propagateVariants(propagated_best_fit_orbit, propagator, propagation_times,
                      num_variants, submission_number, submission_id, start_index = 0,
                      num_threads = 1, chunk_size = 1):
    """
    Generate variant orbits based on the propagated best fit orbit for the given
    submission and propagate them forward in time using the given propagation times.

    Input:
        propagated_best_fit_orbit: The propagated best fit orbit to generate variants from
        propagator: The initialized ASSISTPropagator to use for the propagation
        propagation_times: A Timestamp array containing the times to propagate the
                           variants with. These time steps need to start at the same time
                           as the start_index in the given propagated_best_fit_orbit. But
                           after that, they can be different than what was used to
                           propagate the best fit orbit.
        num_variants: The number of variants to generate
        submission_number: The submission number for the current submission
        submission_id: The submission ID for the current submission
        start_index: The index in the propagated_best_fit_orbit to use as the
                     starting point for generating the variants
        num_threads: The number of threads to use for the propagation. By default no
                     multithreading is used
        chunk_size: The chunk size to use for the propagation. By defalut a chunk size of
                    1 is used
    Output:
        propagated_variants: The propagated variant orbits
    """
    # Use the propagated orbit to generate variant samples based on the uncertainty
    # of the orbit. The index of propagated_best_fit_orbit is which timestep to use to
    # seed the variants at.
    print("Generating variant samples at timestep", start_index)
    variants = VariantOrbits.create(
        propagated_best_fit_orbit[start_index],
        method = "monte-carlo", 
        num_samples = num_variants
    )

    # Set the correct ID for each variant
    variants = variants.set_column(
        "orbit_id", 
        pa.array(
            [f"{submission_number:03d}::{submission_id}::{i + 1:06}" \
            for i in range(len(variants))],
            type = pa.large_string()
        )
    )

    # Propagate the variants sample points forward in time to the end time
    print("Starting to propagate variant samples")
    print("number of timesteps:", len(propagation_times))

    # Do manual batching if the number of variants is too high for the propagator to handle
    propagated_variants = None
    if num_variants > MAX_NUM_VARIANTS_PER_BATCH:
        print("The number of variants is larger than the maximum allowed per batch")
        print("Performing manual batching of propagation")

        # Calculate the number of batches needed
        num_batches = math.ceil(num_variants / MAX_NUM_VARIANTS_PER_BATCH)
        print("Number of batches:", num_batches)

        # Propagate each batch separately
        is_first = True
        for batch_index in range(num_batches):
            print("Propagating batch", batch_index + 1, "of", num_batches)

            # Calculate the number of variants for this batch
            start_variant = batch_index * MAX_NUM_VARIANTS_PER_BATCH
            end_variant = min(start_variant + MAX_NUM_VARIANTS_PER_BATCH, num_variants)
            batch_num_variants = end_variant - start_variant
            print("Number of variants in batch:", batch_num_variants)

            # Propagate the batch
            propagated_batch_variants = propagator.propagate_orbits(
                Orbits.from_kwargs(
                    orbit_id = variants[start_variant:end_variant].orbit_id,
                    object_id = variants[start_variant:end_variant].object_id,
                    coordinates = variants[start_variant:end_variant].coordinates,
                ), 
                propagation_times,
                covariance = False,
                max_processes = num_threads,
                chunk_size = chunk_size
            )

            # Append the propagated batch to the full propagated orbit
            if is_first:
                propagated_variants = propagated_batch_variants
                is_first = False
            else:
                propagated_variants = concatenate(
                    [propagated_variants, propagated_batch_variants]
                )

    else:
        propagated_variants = propagator.propagate_orbits(
            Orbits.from_kwargs(
                orbit_id = variants.orbit_id,
                object_id = variants.object_id,
                coordinates = variants.coordinates,
            ), 
            propagation_times,
            covariance = False,
            max_processes = num_threads,
            chunk_size = chunk_size
        )
    print("Finished propagating variant samples")

    # Convert the variants timesteps to UTC
    print("Rescaling time to UTC")
    propagated_variants = propagated_variants.set_column(
        "coordinates.time",
        propagated_variants.coordinates.time.rescale("utc")
    )

    # Return the propagated variants
    return propagated_variants


def saveVariantsToFile(propagated_variants, times, covariances, num_variants,
                       output_directory):
    """
    Save the propagated variants data to files in the given output directory.
    Input:
        propagated_variants: The propagated variants to save
        times: The time steps used for the variant propagation
        covariances: The covariance matricies for the best fit orbit used to generate the
                     variants
        num_variants: The number of variants
        output_directory: The directory to save the files to
    """
    # Coordinates in AU, reference frame Ecliptic 2000 (.r is the position vector)
    print("Saving propagated variants data to files")
    variants_coordinates_filepath = os.path.join(
        output_directory,
        "variants_coordinates_" + str(num_variants)
    )
    np.save(variants_coordinates_filepath, propagated_variants.coordinates.r)

    # Velocities in AU/day (.v is the velocity vector)
    variants_velocity_filepath = os.path.join(
        output_directory,
        "variants_velocity_" + str(num_variants)
    )
    np.save(variants_velocity_filepath, propagated_variants.coordinates.v)

    # Times
    times_filepath = os.path.join(output_directory, "times_isot_" + str(num_variants))
    np.save(times_filepath, times)

    # Covariance matricies
    covariances_filepath = os.path.join(
        output_directory,
        "covariances_ " + str(num_variants)
    )
    np.save(covariances_filepath, covariances)
