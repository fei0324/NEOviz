import os
import spiceypy as spice
import numpy as np
from astropy import units

import pyarrow.compute as pc

from naif_leapseconds import leapseconds
spice.furnsh(leapseconds)


def create_kernels(
        propagated_orbits,
        out_dir,
        id_offset=1000000
    ):

    # Create the our directory
    os.makedirs(out_dir, exist_ok=True)

    # Get the list of variants
    orbit_ids = propagated_orbits.orbit_id.unique().to_numpy(zero_copy_only=False)

    for i, orbit_id in enumerate(orbit_ids):
        # Get the current variant in the list
        mask = pc.equal(propagated_orbits.orbit_id, orbit_id)
        propagated_orbit = propagated_orbits.apply_mask(mask)

        # Sort the given orbit data by time
        propagated_orbit = propagated_orbit.sort_by(["coordinates.time.days", "coordinates.time.nanos"])

        # Create a unique spice id (Cannot collide with any already existing NAIF id)
        target_id = id_offset + i

        # Create the timestamp arrays
        epochs_tdb = propagated_orbit.coordinates.time.rescale("tdb").jd().to_numpy(zero_copy_only=False)
        epochs_et = np.array([spice.str2et(f'JD {i:.15f} TDB'.format(i)) for i in epochs_tdb])

        # Create array of positions and velocities and rescale to km and seconds
        states = propagated_orbit.coordinates.values
        states[:, 0:6] *= units.au.to(units.km)
        states[:, 3:6] /= (units.d).to(units.s)

        # Create and open the out bsp file
        out_bsp = os.path.join(out_dir, f"{orbit_id}.bsp")
        file = spice.spkopn(out_bsp, f"{target_id}", 0)

        # Fill the bsp kernel file
        spice.spkw09(
            file,                           # Handle of an SPK file open for writing
            target_id,                      # Unique SPICE NAIF code for this variant
            10,                             # Sun Center is the observer
            "ECLIPJ2000",                   # Eclip J2000 reference frame
            epochs_et[0],                   # First entry of timestamps
            epochs_et[-1],                  # Last entry of timestamps
            "SPK_STATES_09",                # Segment identifier (?)
            15,                             # Degree of interpolating polynomials (?)
            len(epochs_et),                 # Number of states
            np.ascontiguousarray(states),   # Array of states
            epochs_et                       # Array of timestamps corresponding to states
        )

        # Close the bsp out file
        spice.spkcls(file)

    return
