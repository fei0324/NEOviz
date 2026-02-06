import os
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import quivr as qv

from quivr.concat import concatenate
from dataclasses import dataclass
from adam_core.time import Timestamp
from adam_core.orbits import Orbits
from adam_core.coordinates import CartesianCoordinates
from adam_core.orbits import VariantOrbits

import tube_generation.util as util

# TODO: Set the correct minimum number of timesteps required by the propagator
MINIMUM_TIMESTEPS = 80

# The propagator is slow and cannot handle too amny samples at the same time (even if it
# chunks) so we need to perform some additional chunking manually.
MAX_NUM_SAMPLES_PER_BATCH = 512
MAX_NUM_VARIANTS_PER_BATCH = 256

# The number of meters in one Astronomical Unit (AU)
AU = 149597870700.0
SECONDS_PER_DAY = 86400.0

# Data object to hold variants data
@dataclass
class VariantsData:
    variants_coordinates: np.array
    variants_velocities: np.array
    times: np.array
    covariances: np.array
    num_variants: int
    num_time_steps: int
    best_fit_orbit_coordinates: np.array

@dataclass
class ImpactData:
    spice_id: int
    latitude: float
    longitude: float
    time: str

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


def loadImpactData(impact_file, orbit_id):
    """
    """

    # Read the file content
    content = util.readFile(impact_file)

    # Check the first line to see that it is the correct object ID
    object_line = content[0].strip()
    if not object_line.startswith(orbit_id):
        print("Impact file", impact_file, "does not match orbit ID", orbit_id)
        assert False, "Impact file does not match orbit ID"

    # Parse the content into ImpactData objects
    impact_data = []
    start_parsing = False
    for line in content:
        line = line.strip()

        # Skip until the header line
        if not start_parsing and line.startswith("Varient"):
            # This is the header line, the next line is the first data entry
            start_parsing = True
            continue
        elif start_parsing and line == "":
            # Reached the end of the data entries
            break

        if start_parsing:
            # Parse the data line
            parts = line.split()
            spice_id = int(parts[0])
            latitude = float(parts[1])
            longitude = float(parts[2])
            time = parts[3]

            print("Loaded impact data for variant", spice_id,
                  "lat:", latitude, "lon:", longitude, "time:", time)

            # Store it in the list
            impact_data.append(
                ImpactData(
                    spice_id = spice_id,
                    latitude = latitude,
                    longitude = longitude,
                    time = time
                )
            )

    # Return the list of variant impacts
    return impact_data


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
        print("Using samples per day:", samples_per_day)

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
    propagated_orbit = None
    bf_orbit = best_fit_orbit.to_orbits()

    # Do manual batching if the number of samples is too high for the propagator to handle
    if num_samples > MAX_NUM_SAMPLES_PER_BATCH:
        print("The number of samples is larger than the maximum allowed per batch")
        print("Performing manual batching of propagation")

        # Calculate the number of batches needed
        num_batches = math.ceil(num_samples / MAX_NUM_SAMPLES_PER_BATCH)
        print("Number of batches:", num_batches)

        # Propagate each batch separately
        is_first = True
        for batch_index in range(num_batches):
            print("Propagating batch", batch_index + 1, "of", num_batches)

            # Calculate the number of samples for this batch
            start_sample = batch_index * MAX_NUM_SAMPLES_PER_BATCH
            end_sample = min(start_sample + MAX_NUM_SAMPLES_PER_BATCH, num_samples)
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
        # Propagate the best fit orbit for a submission forward in time using the
        # propagation sample times
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
                      num_variants, submission_number, submission_id, configuration,
                      start_index = 0, num_threads = 1, chunk_size = 1,
                      submission_output_directory = ""):
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

    # Check if there already are variant seeds saved for this submission
    variant_seeds_path = os.path.join(
        submission_output_directory,
        "variant_seeds.parquet"
    )
    seeds_exists = os.path.exists(variant_seeds_path)

    # Load existing batch if it exists
    variants = None
    if seeds_exists and configuration["use_existing_variants"]:
        print(
            "Loading existing variant seeds from",
            variant_seeds_path
        )
        variants = Orbits.from_parquet(variant_seeds_path)
    else:
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

        # Save the variant seeds
        if configuration["save_intermediate_results"]:
            print("Saving variant seeds to", variant_seeds_path)
            variants.to_parquet(variant_seeds_path)

    # Create output directory for batches if requested
    batches_output_directory = None
    if configuration["save_intermediate_results"]:
        batches_output_directory = os.path.join(
            submission_output_directory,
            "variant_batches"
        )
        os.makedirs(batches_output_directory, exist_ok = True)

    # Propagate the variants sample points forward in time to the end time
    print("Starting to propagate variant samples")

    # Do manual batching if the number of variants is too high for the propagator
    # to handle
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
            propagated_batch_variants = None

            # Check if we can load an existing propagated batch
            batch_parquet_path = os.path.join(
                batches_output_directory,
                "propagated_batch_"+ str(batch_index) + ".parquet"
            )
            parquet_exists = os.path.exists(batch_parquet_path)

            # Load existing batch if it exists
            if parquet_exists and configuration["use_existing_variants"]:
                print(
                    "Loading existing variants in batch from",
                    batch_parquet_path
                )
                propagated_batch_variants = Orbits.from_parquet(batch_parquet_path)
            else:
                print("Propagating batch", batch_index + 1, "of", num_batches)

                # Calculate the number of variants for this batch
                start_variant = batch_index * MAX_NUM_VARIANTS_PER_BATCH
                end_variant = min(
                    start_variant + MAX_NUM_VARIANTS_PER_BATCH,
                    num_variants
                )
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

                # Save the batch
                if configuration["save_intermediate_results"]:
                    propagated_batch_variants.to_parquet(batch_parquet_path)

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
    print("Rescaling times to UTC")
    propagated_variants = propagated_variants.set_column(
        "coordinates.time",
        propagated_variants.coordinates.time.rescale("utc")
    )

    # Return the propagated variants
    return propagated_variants


def processVariants(propagated_variants, num_time_steps, propagated_best_fit_orbit):
    """
    
    """

    # The coordinates and velocities are orderd per orbit, but we want to find
    # all varaint coordinates and velocities per timestep to create time-slices
    ordered_variants_coordinates = []
    ordered_variants_velocities = []
    ordered_bfo_coordinates = []

    # We only take the coordinate or velocity cooresponding to the t:th timestamp
    # for each orbit. Also rescale the coordinates from AU to meters and velocities
    # from AU/day to m/s.
    print("Processing variants")
    for t in range(num_time_steps):
        # The reference frame used by adam is Ecliptic J2000
        # Positions in AU (.r is the position vector)
        ordered_variants_coordinates.append(
            propagated_variants.coordinates.r[t::num_time_steps] * AU
        )
        ordered_bfo_coordinates.append(
            propagated_best_fit_orbit.coordinates.r[t::num_time_steps] * AU
        )
        # Velocities in AU/day (.v is the velocity vector)
        ordered_variants_velocities.append(
            propagated_variants.coordinates.v[t::num_time_steps] * AU / SECONDS_PER_DAY
        )

    # Create a numpy array from the list of timesteps for the variants
    times_isot = propagated_variants.coordinates.time.to_astropy().isot

    print("Finished processing variants")
    return \
        ordered_variants_coordinates, \
        ordered_variants_velocities, \
        times_isot, \
        ordered_bfo_coordinates


def loadVariantsData(data_directory, propagated_best_fit_orbit, num_variants,
                     configuration):
    """
    
    """

    # Construct the full filepath to the paraquet file
    parquet_path = os.path.join(
        data_directory,
        "propagated_variants_" + str(num_variants) + ".parquet"
    )

    # Load the paraquet file
    print("Loading propagated variants", parquet_path)
    propagated_variants = Orbits.from_parquet(parquet_path)

    # Get the number of time steps
    times_isot = propagated_variants.coordinates.time.to_astropy().isot
    num_time_steps = len(times_isot) / num_variants
    num_time_steps = int(num_time_steps)
    print("Number of time steps", num_time_steps)

    # Process the variants
    ordered_variants_coordinates, \
    ordered_variants_velocities, \
    times_isot, \
    ordered_bfo_coordinates = processVariants(
        propagated_variants,
        num_time_steps,
        propagated_best_fit_orbit
    )

    # TODO: Fix this
    # Recompute covariances of propagated variants, collapse the variants into a
    # single orbit to get one covariance matrix per timestep. Do this last as it will
    # change the variants data structure
    #collapsed_variants = propagated_variants.collapse(propagated_best_fit_orbit)
    #covariances = collapsed_variants.coordinates.covariance.to_matrix()
    covariances = propagated_best_fit_orbit.coordinates.covariance.to_matrix()
    #covariances = propagated_variants.coordinates.covariance.to_matrix()

    # Store in the dataobject 
    return VariantsData(
        ordered_variants_coordinates,
        ordered_variants_velocities,
        times_isot,
        covariances,
        num_variants,
        num_time_steps,
        ordered_bfo_coordinates
    )


def hasVariantsData(data_directory, num_variants):
    """
    
    """

    # Specify the filename that should be present in the directory if the data is present
    parquet_filename = "propagated_variants_" + str(num_variants) + ".parquet"

    # Search for the files in the given directory
    for filename in os.listdir(data_directory):
        if filename == parquet_filename:
            return True

    return False
