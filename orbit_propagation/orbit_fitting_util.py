import os
import numpy as np
import pandas as pd
import pyarrow as pa
from astropy.time import Time


import subprocess
import os
import tempfile
import json
import shutil
import uuid
import quivr as qv

from adam_core.orbits import Orbits
from adam_core.coordinates import CartesianCoordinates, CoordinateCovariances, Origin
from adam_core.time import Timestamp


class FittedOrbits(qv.Table):
    orbit_id = qv.LargeStringColumn(default= lambda: str(uuid.uuid4()))
    object_id = qv.LargeStringColumn()
    coordinates = CartesianCoordinates.as_column()
    included_observations = qv.Int64Column()
    rejected_observations = qv.Int64Column()
    arc_length = qv.Float64Column()

    def to_orbits(self) -> Orbits:
        return Orbits.from_kwargs(
            orbit_id=self.orbit_id,
            object_id=self.object_id,
            coordinates=self.coordinates,
        )

def observations_to_ades(observations, file_out) -> str:
    """
    Writes observations to a reduced MPC ADES file that can be 
    used with find_orb. 

    """
    ades = observations.rename(columns={
        "unpacked_provisional_designation": "provID",
        "timestamp": "obsTime",
        "ra_rms": "rmsRA",
        "dec_rms": "rmsDec",
        "mag_rms": "rmsMag",
        "filter_band": "band",
        "obscode": "stn",
    })

    column_order = ["provID", "obsTime", "ra", "dec", "mag", "rmsRA", "rmsDec", "rmsMag", "band", "stn"]
    ades = ades[column_order]

    observation_times = Time(
        ades["obsTime"].values,
        format="datetime64",
        scale="utc",
        precision=3,
    )
    ades["obsTime"] = np.array([i + "Z" for i in observation_times.utc.isot])

    # Note not necessarily true for all observations 
    ades["astCat"] = np.full(len(ades), "Gaia2")
    ades["mode"] = np.full(len(ades), "CCD")

    col_header = "|".join(ades.columns) + "\n"

    with open(file_out, "w") as f:
        f.write("# version=2017\n")
        f.write(col_header)

    return ades.to_csv(file_out, index=False, header=False, sep="|", mode="a")

def run_find_orb(observations, out_dir=None):
    """
    Runs find_orb on the given observations and returns the calculated
    orbit and covariance matrix. 

    """
    assert observations["unpacked_provisional_designation"].nunique() == 1

    my_env = os.environ.copy()
    my_env["PATH"] = f"{os.path.expanduser('~/bin')}:{my_env['PATH']}"

    with tempfile.TemporaryDirectory() as tempdir:
        ades_file = os.path.join(tempdir, "ades.psv")
        observations_to_ades(observations, ades_file)

        output = subprocess.run(
            ["fo", ades_file, "-O", tempdir, f"-tEjd{Time(observations['timestamp'].max(), scale='utc').utc.jd}", "-j"],
            capture_output=True,
            text=True,
            env=my_env,
        )

        if output.returncode != 0:
            raise RuntimeError(f"find_orb failed: {output.stderr}")
        
        # Read orbit and covariance from covar.json
        try:
            with open(os.path.join(tempdir, "covar.json")) as f:
                result = json.load(f)
                
                state = np.array(result["state_vect"])
                covariance_matrix = np.array(result["covar"]).reshape(1, 6, 6)
                coords = CartesianCoordinates.from_kwargs(
                    time=Timestamp.from_jd([result["epoch"]], scale="tt"),
                    x=state[0:1],
                    y=state[1:2],
                    z=state[2:3],
                    vx=state[3:4],
                    vy=state[4:5],
                    vz=state[5:],
                    covariance=CoordinateCovariances.from_matrix(covariance_matrix),
                    origin=Origin.from_kwargs(code=["SUN"]),
                    frame="ecliptic",
                )
        
        except FileNotFoundError:
            return FittedOrbits.empty()

        # Read information about the observations used from total.json
        with open(os.path.join(tempdir, "total.json")) as f:
            result_total = json.load(f)

            object_id = observations["unpacked_provisional_designation"].values[0]
            included_observations = result_total["objects"][object_id]["observations"]["used"]
            rejected_observations = result_total["objects"][object_id]["observations"]["count"] - included_observations
            arc_length = result_total["objects"][object_id]["observations"]["latest_used"] - result_total["objects"][object_id]["observations"]["earliest_used"]

        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            shutil.copytree(tempdir, out_dir)
            shutil.copy(ades_file, out_dir)
        
    return FittedOrbits.from_kwargs(
        object_id=observations["unpacked_provisional_designation"].values[:1], 
        coordinates=coords,
        included_observations=[included_observations],
        rejected_observations=[rejected_observations],
        arc_length=[arc_length],
    )


def run_object_observations_by_submission(observations, submissions=None):
    """
    Runs find_orb on the observations for a given object, grouped by
    submission. 

    """
    fitted_orbits = []
    submission_ids = []
    last_submission = []

    if submissions is None:
        unique_submission_ids = observations["submission_id"].unique()
    else:
        unique_submission_ids = submissions["id"].unique()

    for i, submission_id in enumerate(unique_submission_ids):
        submission_ids.append(submission_id)
        observations_subset = observations[observations["submission_id"].isin(submission_ids)]
        fitted_orbits_i = run_find_orb(observations_subset)
        if len(fitted_orbits_i) > 0:
            fitted_orbits_i = fitted_orbits_i.set_column("orbit_id", pa.array([f"{i:03d}::{submission_id}"], type=pa.large_string()))

            fitted_orbits.append(fitted_orbits_i)
            last_submission.append(submission_id)
        else:
            print(f"Orbit fit for submissions up to {submission_id} failed.")

    return last_submission, qv.concatenate(fitted_orbits)