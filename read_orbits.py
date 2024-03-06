import numpy as np
import os
import spiceypy as spice


if __name__ == "__main__":

    input_path = "../adam_core/uncertainty_changes/2023 CX1/"
    dir_list = os.listdir(input_path)
    dir_list.sort()
    
    for submission_dir in dir_list:
        submission_path = os.path.join(input_path, submission_dir) + "/"

        variants_coord_f = [filename for filename in os.listdir(submission_path) if filename.startswith("variants_coords_")]
        variants_velo_f = [filename for filename in os.listdir(submission_path) if filename.startswith("variants_velo_")]
        time_f = os.path.join(submission_path, "times_isot.npy")
        print(time_f)
        # There should only be one file of each submission
        assert len(variants_coord_f) == 1
        assert len(variants_velo_f) == 1
        variants_coords = np.load(os.path.join(submission_path, variants_coord_f[0]))
        variants_velo = np.load(os.path.join(submission_path, variants_velo_f[0]))
        time_arr = np.load(time_f)
        print(variants_coord_f)
        print(variants_velo_f)

        num_samples = int(variants_coord_f[0].split("_")[-1].split(".")[0])
        print("num_samples", num_samples)
        num_time_steps = len(time_arr)
        print("num_time_steps")
        print(num_time_steps)
        print(variants_coords.shape)

        variants_coords_list = []
        variants_velo_list = []
        for i in range(num_samples):
            variants_coords_list.append(variants_coords[i::num_time_steps])
            variants_velo_list.append(variants_velo[i::num_time_steps])