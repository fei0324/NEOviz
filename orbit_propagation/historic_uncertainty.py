import os
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

from adam_core.orbits import VariantOrbits
from adam_core.propagator import PYOORB

import ray

from to_kernel import create_kernels


def getAdaptiveTimeSteps(submission_interval):
    """
    Return the number of time steps we want to propagate between submissions.
    If two submissions are less than 1 day apart, we take two time steps at the start and end of the submission period.
    If two submissions are more than 1 day apart, we evenly divide the submission period to the number of rounded number of days

    submission_interval: in mjd scale -  Number of (decimal) days elapsed since November 17th, 1858 0h
    Return: num_t_stps: the number of time steps we use to propagate between two submissions
    """
    if submission_interval < 1:
        num_t_steps = 2
    else:
        num_t_steps = round(submission_interval) + 1

    # the propagation does not work if the number of time steps is less than 16
    num_t_steps = max(20, num_t_steps)

    return num_t_steps


def getDynamicUncertainty(out_dir, submissions, orbits, custom_start_time, custom_end_time, break_time, num_samples, max_processes=10):
    """
    Compute tubes from specific submission times to a custom end time.
    The custom_submission_time and break_time is to control which submission to start the propagation with.
    If there are multiple submissions between custom_submission_time and break_time, we propagate forward from each of them to the custom_end_time
    
    out_dir: the parent output directory, should be the object id e.g. "uncertainty_changes/2012 DA14/
    submissions: submissions in the mpc_data directory
    orbits: orbits generated from observations in the orbit_fits directory
    custom_start_time: start propagating from the closest submission after the custom start time
    custom_end_time: the custom end time for the propagated orbits
    break_time: stop propagating for the submissions after this time
    num_samples: the number of variants we propagate for each best-fit orbit
    max_processes: max core for cuda
    """

    if not ray.is_initialized():
        ray.init(num_cpus=max_processes)

    propagator = PYOORB()

    # submission_ids = submissions["id"].values
    submission_times = Time(submissions["timestamp"].values, scale="utc", format="datetime64")
    
    orbits_at_submission = None
    propagated_variants = None
    print(len(orbits))

    for i, orbit in enumerate(orbits):

        # Get submission ID from this orbit's ID
        orbit_id = orbit.orbit_id[0].as_py()
        submission_number, submission_id = orbit_id.split("::")
        submission_number = int(submission_number)

        # Get the time of this submission and the next 
        submission_time = submission_times[submission_number]

        # stop propagating after break_time
        if submission_time > break_time:
            break

        if submission_time < custom_start_time:
            continue

        # Create a folder for this section of the orbits
        submission_time_str = str(submission_time)[:23]
        submission_time_str = submission_time_str.replace(":", ".")
        print(submission_time_str)
        submission_out_dir = os.path.join(out_dir, submission_time_str)
        if not os.path.exists(submission_out_dir):
            os.makedirs(submission_out_dir)

        submission_interval = custom_end_time - submission_time

        # Create a set of propagation times between the time of this submission and the next
        print("submission interval", submission_interval.jd)
        num_t_steps = getAdaptiveTimeSteps(submission_interval.jd)
        print("num time steps", num_t_steps)

        # Create a set of propagation times between the time of this submission and the next
        time_steps = np.linspace(submission_time.utc.mjd, custom_end_time.utc.mjd, num_t_steps, endpoint=True)
        propagation_times = Timestamp.from_mjd(time_steps, scale="utc")
        print(propagation_times)

        # Propagate the best-fit orbit to time of current submission (it might be defined at a different time, typically
        # at the time of the one of the observations within the submission)
        orbit_at_submission_i = propagator.propagate_orbits(
            orbit.to_orbits(), 
            propagation_times, 
            covariance=True,
            covariance_method="monte-carlo",
            num_samples=num_samples, 
            parallel_backend="ray",
            max_processes=max_processes
        )
        # Convert propagated variants to UTC
        orbit_at_submission_i = orbit_at_submission_i.set_column("coordinates.time", orbit_at_submission_i.coordinates.time.rescale("utc"))
        
        if orbits_at_submission is None:
            orbits_at_submission = orbit_at_submission_i
        else:
            orbits_at_submission = qv.concatenate([orbits_at_submission, orbit_at_submission_i])
            if orbits_at_submission.fragmented():
                orbits_at_submission = qv.defragment(orbits_at_submission)

        # Create variants for this propagated best-orbit
        variants = VariantOrbits.create(
            orbit_at_submission_i[0], 
            method="monte-carlo", 
            num_samples=num_samples
        )
        variants = variants.set_column(
            "orbit_id", 
            pa.array([f"{submission_number:03d}::{submission_id}::{i + 1:06}" for i in range(len(variants))], type=pa.large_string())
        )

        # Propagate the variants to the time of the next submission
        propagated_variants_i = propagator.propagate_orbits(
            Orbits.from_kwargs(
                orbit_id=variants.orbit_id,
                object_id=variants.object_id,
                coordinates=variants.coordinates,
            ), 
            propagation_times,
            covariance=False,
            parallel_backend="ray",
            chunk_size=num_samples//max_processes,
            max_processes=max_processes
        )

        # Convert propagated variants to UTC
        propagated_variants_i = propagated_variants_i.set_column("coordinates.time", propagated_variants_i.coordinates.time.rescale("utc"))

        if propagated_variants is None:
            propagated_variants = propagated_variants_i
        else:
            propagated_variants = qv.concatenate([propagated_variants, propagated_variants_i])
            if propagated_variants.fragmented():
                propagated_variants = qv.defragment(propagated_variants)

        # Save variant coordinates and velocity vectors
        variants_coordinates_f = os.path.join(submission_out_dir, "variants_coords_" + str(num_samples))
        variants_velocity_f = os.path.join(submission_out_dir, "variants_velo_" + str(num_samples))
        # print(propagated_variants_i.coordinates.r.shape)
        # print(propagated_variants_i.coordinates.v.shape)
        np.save(variants_coordinates_f, propagated_variants_i.coordinates.r)
        np.save(variants_velocity_f, propagated_variants_i.coordinates.v)

        # Save the propagation times
        times_isot = orbit_at_submission_i.coordinates.time.to_astropy().isot
        time_f = os.path.join(submission_out_dir, "times_isot" )
        np.save(time_f, times_isot)

        # create and save openspace orbit and variants asset files
        orbit_at_submission_i_file = os.path.join(submission_out_dir, "orbit_at_submission_i.parquet")
        propagated_variants_i_file = os.path.join(submission_out_dir, "propagated_variants_i.parquet")

        orbit_at_submission_i.to_parquet(orbit_at_submission_i_file)
        propagated_variants_i.to_parquet(propagated_variants_i_file)
 
        # create_openspace_assets(orbits_at_submission, os.path.join(submission_out_dir, "openspace_orbit"), color="blue")
        # uncomment if you want to create and save kernel files (.bsp)
        # create_kernels(propagated_variants, os.path.join(submission_out_dir, "openspace_variants"))
    
    return


if __name__ == "__main__":

    object_id = "2004 MN4"
    orbit_fits_dir = os.path.join("./orbit_fits", object_id)
    submissions_dir = os.path.join("./mpc_data", object_id)
    out_dir = os.path.join("impact_corridor", object_id)

    os.makedirs(out_dir, exist_ok=True)

    orbit_file = os.path.join(orbit_fits_dir, "orbits.parquet")
    submission_ids_file = os.path.join(orbit_fits_dir, "submission_ids.txt")
    submissions_file = os.path.join(submissions_dir, "submissions.parquet")

    orbits = FittedOrbits.from_parquet(orbit_file)
    submissions = pd.read_parquet(submissions_file)

    num_samples = 10000

    #### impact corridor Apophis (start propagating from the last submission on 2004-12-27)
    custom_start_time = Time("2004-12-27T21:00:00.000", format="isot")
    custom_end_time = Time("2029-12-31T00:00:00.000", format="isot")
    # Only propagate from the submission between custom_start_time and break time (there should only be one)
    break_time = Time("2004-12-28T00:00:00.000", format="isot")

    getDynamicUncertainty(out_dir, submissions, orbits, custom_start_time, custom_end_time, break_time, num_samples)

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
    
    

