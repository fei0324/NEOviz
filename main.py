import os

import spiceypy as spice
METAKERNEL = './data/kernels/meta-kernel.tm'

import src.getEllipse


if __name__ == "__main__":
    # Initialize SPICE
    spice.furnsh(METAKERNEL)

    # Select the data
    input_dir = "./src/orbit_propagation/generated_data/historical/2023 CX1/2023-02-13T02.38.19.001/"
    out_dir = "./src/generated_data/historical/2023 CX1/2023-02-13T02.38.19.001/"


    os.makedirs(out_dir, exist_ok = True)

    src.getEllipse.main(input_dir, out_dir)

