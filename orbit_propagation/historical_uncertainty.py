import os
import math
import numpy as np
import pandas as pd

import pyarrow as pa
import quivr as qv

from orbit_fitting_util import FittedOrbits
from astropy.time import Time
from astropy import units as u
from adam_core.time import Timestamp
from adam_core.orbits import Orbits
from adam_core.orbits.variants import VariantOrbits
from adam_core.orbits.query import query_horizons

from adam_core.orbits import VariantOrbits
from adam_assist import ASSISTPropagator

import ray

from to_kernel import create_kernels


def getAdaptiveTimeSteps(time_interval):
    """
    Return the number of time steps we want to propagate for the given time interval.
    If the time interval is less then 1 day, we take two time steps (at the start and end
    of the submission period). If the time interval is bigger than 1 day, we evenly divide
    the time interval over the number of whole days.

    time_interval: Number of days in jd (Julian date: number of days days + fractional
                   days elapsed since January 1, 4713 BC on the Julian calendar)

    Return: The number of time steps we use to propagate between two submissions
    """
    if time_interval > 1:
        num_steps = math.ceil(time_interval)
    else:
        num_steps = 2

    # the propagation does not work if the number of time steps is less than 16
    # TODO What does this mean? it does not match the code that sets it to
    # minimum 80 steps 
    num_steps = max(80, num_steps)

    return num_steps


def getDynamicUncertainty(
    output_dir,
    submissions,
    submission_orbits,
    start_time,
    end_time,
    stop_time,
    num_samples,
    max_processes = 1
):
    """
    Compute tubes from specific submission times to a custom end time.
    The custom_submission_time and stop_time is to control which submission to start the propagation with.
    If there are multiple submissions between custom_submission_time and stop_time, we propagate forward from each of them to the custom_end_time
    
    output_dir: the parent output directory, should contain the object id e.g. "uncertainty_changes/2012 DA14/
    submissions: submission history to the mpc database
    submission_orbits: orbits generated from each submission
    start_time: start propagating from the first submission after this time
    end_time: the time to stop propagating
    stop_time: stop adding more submission data after this time
    num_samples: the number of variants we propagate for each submission orbit
    max_processes: The number of threads to use while multithreading
    """

    # Initialize
    if not ray.is_initialized():
        ray.init(num_cpus = max_processes)
    propagator = ASSISTPropagator()

    # Get timestamps of submissions
    submission_times = Time(
        submissions["timestamp"].values,
        scale = "utc",
        format = "datetime64"
    )
    
    # Process each orbit (What is each orbit?)
    propagated_orbits_total = None
    propagated_variants_total = None
    for i, submission_orbit in enumerate(submission_orbits):
        # Get submission ID from this submission orbit's ID
        orbit_id = submission_orbit.orbit_id[0].as_py()
        submission_number, submission_id = orbit_id.split("::")
        submission_number = int(submission_number)

        # Get the time of this submission
        submission_time = submission_times[submission_number]

        # Check if we have reached time to start gathering submission data or if it is
        # time to stop
        if submission_time < start_time:
            continue 
        if submission_time > stop_time:
            break

        # Create a seperate data folder for the results of this submission
        submission_time_str = str(submission_time)[:23]
        submission_time_str = submission_time_str.replace(":", ".")
        submission_output_dir = os.path.join(output_dir, submission_time_str)
        if not os.path.exists(submission_output_dir):
            os.makedirs(submission_output_dir)

        # Calculate the remaining time from this submission to the end
        time_remaining = end_time - submission_time
        print("Submission time (JD)", submission_time.jd, "End time (JD)", end_time.jd)

        # Create an adaptive number of time steps between the current submission time and
        # the end
        print("Remaining time from submission:", time_remaining.jd)
        num_time_steps = getAdaptiveTimeSteps(time_remaining.jd)
        print("Number of time steps", num_time_steps)

        # Create a set of propagation times between the time of this submission and
        # the end
        time_steps = np.linspace(
            submission_time.utc.mjd,
            end_time.utc.mjd,
            num_time_steps,
            endpoint = True
        )
        print("Time steps:",  time_steps)
        propagation_times = Timestamp.from_mjd(time_steps, scale = "utc")
        print("Propagation times:",  propagation_times)

        # Propagate to get the best-fit orbit at the time of the submission
        # (it might be defined at a different time, typically at the time of the one of
        # the observations within the submission)
        print("Starting to propagate orbit(s)...")
        orbit_at_submission = propagator.propagate_orbits(
            submission_orbit.to_orbits(), 
            propagation_times, 
            covariance = True,
            covariance_method = "monte-carlo",
            num_samples = num_samples, 
            max_processes = max_processes
        )
        print("Finished propagating orbit(s)!")
        
        # Convert the propagated orbit time to UTC
        orbit_at_submission = orbit_at_submission.set_column(
            "coordinates.time",
            orbit_at_submission.coordinates.time.rescale("utc")
        )
        
        # Add the best-orbit to a list of all orbits
        # This will be usefull later to create variant trails in OpenSpace
        if propagated_orbits_total is None:
            propagated_orbits_total = orbit_at_submission
        else:
            propagated_orbits_total = qv.concatenate([
                propagated_orbits_total,
                orbit_at_submission
            ])
            if propagated_orbits_total.fragmented():
                propagated_orbits_total = qv.defragment(propagated_orbits_total)

        # Use this best-fit orbit and its uncertainty to generate variants
        print("Starting to propagate variant(s)...")
        variants = VariantOrbits.create(
            orbit_at_submission[0], 
            method = "monte-carlo", 
            num_samples = num_samples
        )

        # Set the id for each variant
        variants = variants.set_column(
            "orbit_id", 
            pa.array(
                [f"{submission_number:03d}::{submission_id}::{i + 1:06}" for i in range(len(variants))],
                type = pa.large_string()
            )
        )

        # Propagate the variants to the end time
        propagated_variants = propagator.propagate_orbits(
            Orbits.from_kwargs(
                orbit_id = variants.orbit_id,
                object_id = variants.object_id,
                coordinates = variants.coordinates,
            ), 
            propagation_times,
            covariance = False,
            chunk_size = num_samples//max_processes,
            max_processes = max_processes
        )

        # Convert the propagated variant times to UTC
        propagated_variants = propagated_variants.set_column(
            "coordinates.time",
            propagated_variants.coordinates.time.rescale("utc")
        )

        # Add the variants to a list of all variants
        # This will be usefull later to create SPICE kernels for each variant
        if propagated_variants_total is None:
            propagated_variants_total = propagated_variants
        else:
            propagated_variants_total = qv.concatenate([
                propagated_variants_total,
                propagated_variants
            ])
            if propagated_variants_total.fragmented():
                propagated_variants_total = qv.defragment(propagated_variants_total)

        # Save variant coordinates and velocity vectors to file
        variants_coordinates_filepath = os.path.join(
            submission_output_dir,
            "variants_coordinates_" + str(num_samples)
        )
        np.save(variants_coordinates_filepath, propagated_variants.coordinates.r)

        variants_velocity_filepath = os.path.join(
            submission_output_dir,
            "variants_velocity_" + str(num_samples)
        )
        np.save(variants_velocity_filepath, propagated_variants.coordinates.v)

        # Save the propagation times to file
        # TODO can we save the propagation times instead?
        times_isot = orbit_at_submission.coordinates.time.to_astropy().isot
        time_f = os.path.join(submission_output_dir, "times_isot")
        np.save(time_f, times_isot)

        # Save the orbit and variants for file
        orbit_at_submission_filepath = os.path.join(
            submission_output_dir,
            "orbit_at_submission.parquet"
        )
        orbit_at_submission.to_parquet(orbit_at_submission_filepath)

        propagated_variants_filepath = os.path.join(
            submission_output_dir,
            "propagated_variants.parquet"
        )
        propagated_variants.to_parquet(propagated_variants_filepath)

        # Create and save SPICE kernel files (.bsp)
        # create_kernels(propagated_variants_total, os.path.join(submission_output_dir, "openspace_variants"))

    return


if __name__ == "__main__":
    """
    Input:
        Submission history file
        Orbit file with one calculated best-fit orbit for each submission

    Output:
        Variants    

    This function creats an uncertainty tube of the historical uncertainty type. The
    start_time, end_time, and stop_time variables are set to achieve the historical
    uncertainty effect. We start propagation from the fisr submission time after
    "start_time" and before "stop_time" using all historical data before this submission.
    And we end propagtion at "end_time", if ther are any new submissions after the
    "stop_time", they will be ignored. This effectively simulates a reality where no more
    submissions were obtained for this object after the "stop_time".
    """

    # Settings
    # The number of samples to use when sampling the uncertainty region
    num_samples = 10000

    # Choose object here, only one can be used
    # Apophis (start propagating from the last submission on 2004-12-27)
    object_id = "2004 MN4"
    start_time = Time("2004-12-27T21:00:00.000", format="isot")
    end_time = Time("2029-12-31T00:00:00.000", format="isot")
    # end_time = Time("2004-12-29T00:00:00.000", format="isot")
    stop_time = Time("2004-12-28T00:00:00.000", format="isot")

    # 2023 CX1
    # object_id = "2023 CX1"
    # start_time = Time("2023-02-13T02:38:00.000", format="isot")
    # end_time = Time("2023-02-13T03:40:00.000", format="isot")
    # stop_time = Time("2023-02-13T02:39:00.000", format="isot")

    # Setup input data directories
    orbit_fits_dir = os.path.join("./orbit_fits", object_id)
    submissions_dir = os.path.join("./mpc_data", object_id)
    
    # Setup output directories
    output_dir = os.path.join("generated_data/historical", object_id)
    os.makedirs(output_dir, exist_ok=True)

    # Find the input files
    # The file containing the orbits fitted for each submission
    orbit_file = os.path.join(orbit_fits_dir, "orbits.parquet")
    if not os.path.isfile(orbit_file):
        print("Could not find orbit file ", orbit_file)

    # And the submission history file
    submissions_file = os.path.join(submissions_dir, "submissions.parquet")
    if not os.path.isfile(submissions_file):
        print("Could not find submissions file ", submissions_file)

    # Read the data in the input files
    submission_orbits = FittedOrbits.from_parquet(orbit_file)
    print("Loaded ", len(submission_orbits), " orbits")

    submissions = pd.read_parquet(submissions_file)
    print("Loaded ", len(submissions), " submissions:")
    print(submissions)

    # Create and save the tube file to disc
    getDynamicUncertainty(
        output_dir,
        submissions,
        submission_orbits,
        start_time,
        end_time,
        stop_time,
        num_samples
    ) 
"""
    # Get orbits to propagate
    initial_time = Timestamp.from_mjd([53366], scale="tdb")
    print("Initial_time:", initial_time.to_iso8601())
    object_ids = ["Apophis"]
    orbits = query_horizons(object_ids, initial_time)

    # initialize the propagator
    propagator = ASSISTPropagator()

    # Define propagation times
    times = initial_time.from_mjd(initial_time.mjd() + np.arange(0, 100))
    print("Times:", times)

    # Propagate orbits! This function supports multiprocessing for large
    # propagation jobs.
    propagated_orbits = propagator.propagate_orbits(
        orbits,
        times,
        covariance = True,
        covariance_method = "monte-carlo",
        num_samples = 10000, 
        max_processes = 1
    )
    print("Propagated orbits:", propagated_orbits)
"""
    ################### generate apophis kernels around the bifurcation and potential impact
    # apophis_variants_f = "impact_corridor/2004 MN4/2004-12-27T21.28.37.000/propagated_variants_i.parquet"
    # apophis_variants = VariantOrbits.from_parquet(apophis_variants_f)
    # apophis_variants_df = orbits.to_dataframe()
    # new_orbit = qv.concatenate([apophis_variants[0], apophis_variants[1]])
    # print(apophis_variants[0])
    # print(new_orbit.coordinates.v)

    # bifurcation_list = []
    # num_time_steps = 9135
    # for i in range(num_time_steps):
    #     if 8800 <= i < 9000:  # 2029-01-30T23:54:27.865(8800) - 2029-08-18T23:57:46.748(9000)
    #         bifurcation_list.append(apophis_variants[i::num_time_steps])
    # bifurcation_variants = qv.concatenate(bifurcation_list)

    # # need to truncate it from a certian time step and then make kernels
    # print(len(apophis_variants))
    # bifurcation_dir = "impact_corridor/2004 MN4/2004-12-27T21.28.37.000/bifurcation/"
    # os.makedirs(bifurcation_dir, exist_ok=True)
    # create_kernels(bifurcation_variants, os.path.join(bifurcation_dir, "openspace_variants"))
