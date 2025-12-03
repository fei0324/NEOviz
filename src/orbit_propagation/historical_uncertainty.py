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

from orbit_fitting_util import FittedOrbits
from to_kernel import create_kernels


# TODO: Set the correct minimum number of timesteps required by the propagator
MINIMUM_TIMESTEPS = 80
# TODO: Set this higher when multithreading is supported
MAX_THREADS = 1


def clacAdaptiveTimeSteps(time_interval):
    """
    Calculate the number of time steps to use for the propagation based on the given time
    interval

    Input:
        time_interval: The number of days in jd (Julian date: number of days +
                       fractional days elapsed since January 1, 4713 BC on the Julian
                       calendar) between the start and end time to propagate.
    Output:
        num_steps: The number of time steps to use for the propagation based on the
                   given time interval. 
    """
    # Check the time interval and set the number of timesteps accordingly
    num_steps = MINIMUM_TIMESTEPS
    if time_interval > 1.0:
        # If the number of days are more than 1, then use the number of days as the
        # number of timesteps for the propagation. One time sample per day, rounded
        # upwards.
        num_steps = math.ceil(time_interval)
    else:
         # If the number of days are less than one, then use the minimum number of
         # timesteps requiered for the propagator
        num_steps = MINIMUM_TIMESTEPS

    # Make sure the number of timesteps is at least the minimum required by the propagator
    return max(num_steps, MINIMUM_TIMESTEPS)


def propagateBestFitOrbit(best_fit_orbit, propagator, propagation_times, num_variants):
    """
    Propagate the best fit orbit for a submission forward in time to the end time,
    using the propagation sample times.

    Input:
        best_fit_orbit: A FittedOrbits object containing the best fit orbit for a 
                        submission
        propagator: The initialized ASSISTPropagator to use for the propagation
        propagation_times: A Timestamp array containing the times to propagate the
                           orbit with
        num_variants: The number of variants, this is used as the number of sampels to
                      use when estimating the covariance (uncertainty)
    Output:
        propagated_orbit: The propagated best fit orbit with covariance information
    """
    # Propagate the best fit orbit for a submission forward in time using the propagation
    # sample times
    print("Starting to propagate the best fit orbit")
    propagated_orbit = propagator.propagate_orbits(
        best_fit_orbit.to_orbits(), 
        propagation_times,
        covariance = True,
        covariance_method = "monte-carlo",
        num_samples = num_variants, 
        max_processes = MAX_THREADS,
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
                      num_variants, submission_number, submission_id):
    """
    Generate variant orbits based on the propagated best fit orbit for the given
    submission and propagate them forward in time using the given propagation times.

    Input:
        propagated_best_fit_orbit: The propagated best fit orbit to generate variants from
        propagator: The initialized ASSISTPropagator to use for the propagation
        propagation_times: A Timestamp array containing the times to propagate the
                           orbit with
        num_variants: The number of variants to generate
        submission_number: The submission number for the current submission
        submission_id: The submission ID for the current submission
    Output:
        propagated_variants: The propagated variant orbits
    """
    # Use the propagated orbit to generate variant samples based on the uncertainty
    # of the orbit
    print("Generating variant samples")
    variants = VariantOrbits.create(
        propagated_best_fit_orbit[0], 
        method = "monte-carlo", 
        num_samples = num_variants
    )

    # Set the correct ID for each variant
    variants = variants.set_column(
        "orbit_id", 
        pa.array(
            [f"{submission_number:03d}::{submission_id}::{i + 1:06}" for i in range(len(variants))],
            type = pa.large_string()
        )
    )

    # Propagate the variants sample points forward in time to the end time
    # TODO: Do we need a chunk size? 
    print("Starting to propagate variant samples")
    propagated_variants = propagator.propagate_orbits(
        Orbits.from_kwargs(
            orbit_id = variants.orbit_id,
            object_id = variants.object_id,
            coordinates = variants.coordinates,
        ), 
        propagation_times,
        covariance = False,
        max_processes = MAX_THREADS,
    )
    print("Finished propagating variant samples")

    # Convert the variants timesteps to UTC
    propagated_variants = propagated_variants.set_column(
        "coordinates.time",
        propagated_variants.coordinates.time.rescale("utc")
    )
    return propagated_variants


def calcHistoricalUncertainty(output_directory, submissions, submission_orbits,
                              start_time, end_time, time_of_interest, num_variants,
                              save_kernels = False):
    """
    Sample variants of the asteroid based on the best fitting orbit for each submission.
    When the time of historical interest has been reached, we stop adding more data as
    submissions come in. This effectevly simulates the knowledge we had of the asteroid
    at that point in time. We then propagate sampled variants to the given end time as an
    proximation of the uncertainty of the orbit. The generated variants and some
    additional information is saved to file in the given output directory.

    Input:
        output_directory: The directory to save the generated data to
        submissions: A pandas dataframe containing the observation submission history
        submission_orbits: A FittedOrbits object containing the best fit orbit for each
                           submission
        start_time: An astropy Time object representing the time to start the propagation
        end_time: An astropy Time object representing the time to end the propagation
        time_of_interest: An astropy Time object representing the time of historical
                          interest. From this point onward, we stop adding observation
                          submission data. This simulates the knowledge of the asteroid
                          at that time.
        num_variants: The number of variants to generate for each submission's best
                      fit orbit.
        save_kernels: A boolean flag indicating whether to save SPICE kernel files for
                      each variant or not
    """
    # Initialize the propagator
    if not ray.is_initialized():
        ray.init(num_cpus = MAX_THREADS)
    propagator = ASSISTPropagator()

    # Get submission timestamps
    submission_times = Time(
        submissions["timestamp"].values,
        scale = "utc",
        format = "datetime64"
    )

    # Process each submission. Usually the time of interest is very close to the start
    # time so only a very few number of submissions will be processed. 
    for i, submission_orbit in enumerate(submission_orbits):
        # Get the submission information (ID, timestamp)
        orbit_id = submission_orbit.orbit_id[0].as_py()
        submission_number, submission_id = orbit_id.split("::")
        submission_number = int(submission_number)

        submission_time = submission_times[submission_number]
        submission_time_str = str(submission_time)[:23]
        submission_time_str = submission_time_str.replace(":", ".")
        print("Processing submission", submission_number,
              "at time", submission_time_str)

        # Check the submission time
        if submission_time < start_time:
            # If the time is before the start time, continue to the next submission.
            # This is common since the orbit fitting needs a number of observations to
            # become stable, so many submissions in the beginning needs to be skipped.
            continue
        if submission_time > time_of_interest:
            # If the time is past the time of historical interest, stop processing
            # further submissions
            print("Reached time of historical interest, skipping any further submissions")
            break
        if submission_time > end_time:
            # If the time is past the end time then we stop as well
            print("Reached end time, stopping propagation")
            break

        # Create a new data folder in the output directory for this submission
        submission_output_directory = os.path.join(output_directory, submission_time_str)
        os.makedirs(submission_output_directory, exist_ok = True)

        # Calculate the remaining time to propagate to the end time
        time_remaining = end_time - submission_time

        # Create sample times from now to the end time
        # TODO: We want a way to adapt the number of timesteps to ne much more dense in
        # time frames of high interest. How do we do this?
        print("Remaining time from submission:", time_remaining.jd)
        num_time_steps = clacAdaptiveTimeSteps(time_remaining.jd)
        print("Number of time steps", num_time_steps)

        time_steps = np.linspace(
            submission_time.utc.mjd,
            end_time.utc.mjd,
            num_time_steps,
            endpoint = True
        )
        propagation_times = Timestamp.from_mjd(time_steps, scale = "utc")

        # Propagate the best fit orbit for this submission forward in time to the end
        # time, using the propagation sample times from the previous step
        propagated_best_fit_orbit = propagateBestFitOrbit(
            submission_orbit,
            propagator,
            propagation_times,
            num_variants
        )

        # Generate variants based on the propagated best fit orbit
        propagated_variants = propagateVariants(
            propagated_best_fit_orbit,
            propagator,
            propagation_times,
            num_variants,
            submission_number,
            submission_id
        )

        # Finally, save the variant coordinates and velocities to file
        # The reference frame used by adam is Ecliptic J2000
        # Positions in AU (.r is the position vector)
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
            kernels_output_dir = os.path.join(
                submission_output_directory,
                "variant_kernels"
            )
            create_kernels(propagated_variants, kernels_output_dir)


if __name__ == "__main__":
    """
    """
    # TODO: Parse any input arguments
    # Settings
    num_variants = 10000
    save_kernels = False

    # Choose an asteroid object
    object_id = "2004 MN4"
    print("Calculating historical uncertainty for object:", object_id)

    # Set the time to start propagation and when to end propagation
    start_time = Time("2004-12-27T21:00:00.000", format = "isot")
    print("Start time:", start_time.iso)

    end_time = Time("2005-12-27T21:00:00.000", format = "isot")
    print("End time:", end_time.iso)

    # Set the time of historical interest, from this point onwards we stop adding
    # more observation submission data
    time_of_interest = Time("2004-12-28T00:00:00.000", format = "isot")

    # Setup input directories
    orbit_fits_dir = os.path.join("./data/orbit_fits", object_id)
    submissions_dir = os.path.join("./data/mpc_data", object_id)

    # Setup output directories
    output_directory = os.path.join("./generated_data/historical", object_id)
    os.makedirs(output_directory, exist_ok = True)

    # Find the input files
    # Observation submission history
    submissions_file = os.path.join(submissions_dir, "submissions.parquet")
    if not os.path.isfile(submissions_file):
        print("Could not find submissions file ", submissions_file)
        assert False, "Missing submissions file"

    # Calculated best orbit fit for each submission
    orbits_file = os.path.join(orbit_fits_dir, "orbits.parquet")
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
    assert len(submission_orbits) == len(submissions) - 1, \
        "Missmatch between number of submissions and number of fitted orbits"

    # Calculate the historicla uncertainty of the chosen asteroid over the set timeframe
    calcHistoricalUncertainty(
        output_directory,
        submissions,
        submission_orbits,
        start_time,
        end_time,
        time_of_interest,
        num_variants,
        save_kernels
    )
