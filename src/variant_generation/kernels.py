import os
import numpy as np
import spiceypy as spice
import pyarrow.compute as pc

from astropy import units

# Path to the SPICE kernel files, to initialize SPICE
LSK_KERNEL = "../data/kernels/lsk/naif0012.tls.pc"
SPK_KERNEL = "../data/kernels/spk/de432s.bsp"
PCK_KERNEL = "../data/kernels/pck/pck00011.tpc"

def create_kernels(propagated_orbits, output_directory, id_offset = 1000000):
    """
    Create SPICE .bsp kernels for the given propagated orbits. Save the kernel files in
    the given output directory. 

    Input:
        propagated_orbits: The propagated orbits to create SPICE kernels for
        output_directory: The directory to save the created kernels in
        id_offset: The offset to add to each orbit index to make sure that the created  
                   SPICE id is unique. Default is 1000000.

    """
    # Initialize SPICE
    spice.furnsh(LSK_KERNEL)
    spice.furnsh(SPK_KERNEL)
    spice.furnsh(PCK_KERNEL)

    # Create the output directory
    os.makedirs(output_directory, exist_ok = True)

    # Get the ids of the oribts to create kernels for
    orbit_ids = propagated_orbits.orbit_id.unique().to_numpy(zero_copy_only = False)
    num_orbits = len(orbit_ids)

    # Create the kernel filenames. The filenames are different from the SPICE ids, which
    # in turn is different from its list index. See below for details:
    # Index in list (such as index number 0 or 153) = index
    # SPICE id (such as 1000000 or 1000153) = id_offset + index
    # Filename (such as 000001.bsp or 000154.bsp) =
    #     (K-width zero-padded string of index + 1).bsp (K is by default 6)
    #
    # The SPICE ids start from the id_offset to ensure that the created kernel ids do not
    # conflict with any pre-existing SPICE ids.
    K = 6
    filenames = [str(i + 1).zfill(K) for i in range(num_orbits)]

    # Process each orbit and create a kernesl file for it
    for i, orbit_id in enumerate(orbit_ids):
        # Get the current orbit
        mask = pc.equal(propagated_orbits.orbit_id, orbit_id)
        propagated_orbit = propagated_orbits.apply_mask(mask)

        # Sort the given orbit data by time
        propagated_orbit = propagated_orbit.sort_by([
            "coordinates.time.days", "coordinates.time.nanos"
        ])

        # Create the unique SPICE id for this orbit
        SPICE_target_id = id_offset + i

        # Create a list of time steps
        epochs_tdb = propagated_orbit.coordinates.time.rescale(
            "tdb"
        ).jd().to_numpy(zero_copy_only = False)
        epochs_et = np.array([
            spice.str2et(f'JD {i:.15f} TDB'.format(i)) for i in epochs_tdb
        ])

        # Create a list of states (positions and velocities) that coorrespond to the time
        # steps. coordinates.values is a 6-element array: [x, y, z, vx, vy, vz]
        states = propagated_orbit.coordinates.values

        # Rescale the states to km and km/s from AU and AU/day
        states[:, 0:6] *= units.au.to(units.km)
        states[:, 3:6] /= (units.d).to(units.s)

        # Create and open the .bsp file for writing
        out_filepath = os.path.join(output_directory, f"{filenames[i]}.bsp")
        out_file = spice.spkopn(out_filepath, f"{SPICE_target_id}", 0)

        # Fill the kernel file with the orbit data
        spice.spkw09(
            out_file,                       # Handle of an SPK file open for writing
            SPICE_target_id,                # Unique SPICE NAIF code for this variant
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

        # Be nice and close the kernel file
        spice.spkcls(out_file)

    return


def saveKernels(propagated_variants, num_variants, output_directory, configuration):
    """
    Save SPICE kernels for the given propagated variants in the given output directory.
    If there already are kernels in the output directory, and the configuration allows
    overriding existing results, the existing kernels will be removed before creating
    new ones.

    Input:
        propagated_variants: The propagated variants to create SPICE kernels for
        num_variants: The number of propagated variants
        output_directory: The directory to save the kernels in
        configuration: The configuration
    """

    # Check if there already are kernels
    has_existing_kernels = False
    if len(os.listdir(output_directory)) > 0:
        has_existing_kernels = True

    if has_existing_kernels and configuration["override_existing_results"]:
        # SPICE cannot generate kernels if the files already exist, therefor we
        # need to remove the old kernels before making new ones
        print("Removing existing kernels in", output_directory)
        for filename in os.listdir(output_directory):
            file_path = os.path.join(output_directory, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print("Failed to delete file:", file_path, "Reason:", e)

        os.rmdir(output_directory)
        has_existing_kernels = False

    if not has_existing_kernels:
        print("Creating SPICE kernels for the propagated variants")
        create_kernels(propagated_variants, output_directory)
