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
from adam_assist import ASSISTPropagator

import ray

#from to_asset import create_openspace_assets
from to_kernel import create_kernels


def getAdaptiveTimeSteps(submission_interval, max_time=None):
    """
    Return the number of time steps we want to propagate between submissions. If two
    submissions are less than 1 day apart, we take two time steps at the start and end of
    the submission period. If two submissions are more than 1 day apart, we evenly divide
    the submission period to the number of rounded number of days

    submission_interval: in mjd scale -  Number of (decimal) days elapsed since November
                         17th, 1858 0h
    max_time: the create kernel function does not run for time steps less than 16. This
              is not needed if we are not creating kernels.
    Return: num_t_stps: the number of time steps we use to propagate between two
            submissions
    """
    if submission_interval < 1:
        num_t_steps = 2
    else:
        num_t_steps = round(submission_interval) + 1
    
    if max_time is not None:
        num_t_steps = max(max_time, num_t_steps)
    return num_t_steps


def getSectionedOrbits(out_dir, submissions, orbits, custom_end_time, num_samples, gap_percent=5, max_processes=10):
    """
    Compute sectioned orbits for each submission period (between two adjacent submissions)
    
    out_dir: the parent output directory, should be the object id e.g. "uncertainty_changes/2012 DA14/
    orbits: orbits generated from observations in the orbit_fits directory
    custom_end_time: the custom end time for the last propagated orbit based on the submission time
    num_samples: the number of variants we propagate for each best-fit orbit
    num_t_steps: number of time steps at each submission period (needs to be greater than 15 to propagate)
    gap_percent: a percentage of the period interval. It exists to create the change in uncertainty effect in the visualization
    max_processes: max core for cuda
    """

    if not ray.is_initialized():
        ray.init(num_cpus=max_processes)

    propagator = ASSISTPropagator()

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

        # Create a folder for this section of the orbits
        submission_time_str = str(submission_time)[:23]
        submission_time_str = submission_time_str.replace(":", ".")
        print(submission_time_str)
        submission_out_dir = os.path.join(out_dir, submission_time_str)
        if not os.path.exists(submission_out_dir):
            os.makedirs(submission_out_dir)
        
        # If its not the last submission, use the time of the next submission
        if i < len(orbits) - 1:
            next_submission_time = submission_times[submission_number + 1]

        # If this is the last submission then use a custom end time
        else:
            if object_id == "2000 SG344":
                next_submission_time = Time("2075-01-01T00:00:00.000", format="isot")
            else:
                next_submission_time = Time(custom_end_time, format="isot")

        # Add time gap for the change in uncertainty effect
        # submission_interval = next_submission_time.utc.mjd - submission_time.utc.mjd
        submission_interval = next_submission_time - submission_time

        # Create a set of propagation times between the time of this submission and the next
        print("submission interval", submission_interval.jd)
        num_t_steps = getAdaptiveTimeSteps(submission_interval.jd, max_time=20)
        print("num time steps", num_t_steps)

        # Create a set of propagation times between the time of this submission and the next
        time_steps = np.linspace(submission_time.utc.mjd, next_submission_time.utc.mjd, num_t_steps, endpoint=True)
        time_steps[0] = gap_percent * (time_steps[1] - time_steps[0]) / 100 + time_steps[0]
        time_steps[-1] = time_steps[-1] - gap_percent * (time_steps[-1] - time_steps[-2]) / 100
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
        # times_isot = propagation_times.to_astropy().isot
        # TODO: Why are these two time arrays not the same???
        # print(times_isot)
        time_f = os.path.join(submission_out_dir, "times_isot" )
        np.save(time_f, times_isot)

        # create and save openspace orbit and variants asset files
        orbit_at_submission_i_file = os.path.join(submission_out_dir, "orbit_at_submission_i.parquet")
        propagated_variants_i_file = os.path.join(submission_out_dir, "propagated_variants_i.parquet")

        orbit_at_submission_i.to_parquet(orbit_at_submission_i_file)
        propagated_variants_i.to_parquet(propagated_variants_i_file)
 
        # create_openspace_assets(orbits_at_submission, os.path.join(submission_out_dir, "openspace_orbit"), color="blue")
        # create_openspace_assets(propagated_variants, os.path.join(submission_out_dir, "openspace_variants"))
        # create_kernels(propagated_variants, os.path.join(submission_out_dir, "openspace_variants"))
    return


if __name__ == "__main__":

    object_id = "2012 DA14"
    # object_id = "1998 SG172"
    orbit_fits_dir = os.path.join("orbit_fits", object_id)
    submissions_dir = os.path.join("mpc_data", object_id)
    out_dir = os.path.join("generated_data/sectioned", object_id)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    orbit_file = os.path.join(orbit_fits_dir, "orbits.parquet")
    submission_ids_file = os.path.join(orbit_fits_dir, "submission_ids.txt")
    submissions_file = os.path.join(submissions_dir, "submissions.parquet")

    orbits = FittedOrbits.from_parquet(orbit_file)
    orbits_df = orbits.to_dataframe()
    print(orbits_df)
    submissions = pd.read_parquet(submissions_file)

    num_samples = 5000
    custom_end_time = Time("2023-02-16T00:00:00.000", format="isot")
    # custom_end_time = Time("2013-12-01T00:00:00.000", format="isot")
    # custom_end_time = Time("2010-01-01T00:00:00.000", format="isot")

    getSectionedOrbits(out_dir, submissions, orbits, custom_end_time, num_samples)


    # maybe the order is different:
    # (16000, 3)
    # (16000, 3)
    # Maybe the order is 16 points per orbit, instead of 1000 points per time step