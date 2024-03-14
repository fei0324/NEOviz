import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances
import json
import os
import copy

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN

from mvee import mvee2
from plotting import plot_ellipse
from matplotlib.patches import Ellipse
from astropy import units as astrounit

def loadJSON(input_dir):

    input_path = os.path.join(input_dir, "tube_data.json")

    with open(input_path) as f:
        tube_data = json.load(f)
        num_time_steps = len(tube_data['polygons'])
        num_sample_per_t = len(tube_data['polygons'][0]['points'])
        # print(num_sample_per_t)

        tube_arr = []

        for i in range(num_time_steps):
            ellipse_i = tube_data['polygons'][i]['points']
            ellipse_arr = np.zeros((num_sample_per_t, 3))
            for j in range(num_sample_per_t):
                ellipse_arr[j, 0] = ellipse_i[j]['x']
                ellipse_arr[j, 1] = ellipse_i[j]['y']
                ellipse_arr[j, 2] = ellipse_i[j]['z']
            tube_arr.append(ellipse_arr)
        
        f.close()

    return tube_arr


def subsetTube(input_dir, start_t, end_t):

    input_path = os.path.join(input_dir, "tube_data.json")

    with open(input_path) as f:
        tube_data = json.load(f)
        tube_data_subset = copy.deepcopy(tube_data)
        print(tube_data_subset["polygons"][8800])
        polygons_subset = tube_data_subset['polygons'][start_t:end_t]
        print(len(polygons_subset))
        tube_data_subset['polygons'] = polygons_subset

        tube_subset_path = os.path.join(input_dir, "tube_data_subset.json")
        
        with open(tube_subset_path, 'w') as fp:
            json.dump(tube_data_subset, fp)


def cluster2dPts(ori_pts_2d, t_to_cluster):
    # print(ori_pts_2d)
    cutplane_pts_t = ori_pts_2d[t_to_cluster].T
    print(cutplane_pts_t.shape)

    # spectral = SpectralClustering(n_clusters=2, affinity="nearest_neighbors").fit(cutplane_pts_t)
    # print(spectral.labels_)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(cutplane_pts_t)
    print(kmeans.labels_)
    # db = DBSCAN(eps=10, min_samples=100).fit(cutplane_pts_t)
    # labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # n_noise_ = list(labels).count(-1)

    # print("Estimated number of clusters: %d" % n_clusters_)
    # print("Estimated number of noise points: %d" % n_noise_)

    fig, ax = plt.subplots()
    cutplane_x = cutplane_pts_t[:, 0]
    cutplane_y = cutplane_pts_t[:, 1]
    ax.scatter(cutplane_x, cutplane_y, c=kmeans.labels_)

    for i in range(len(cutplane_x)):
        ax.annotate(i, (cutplane_x[i], cutplane_y[i]))
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    plt.show()


def plotImpact(ori_pts_2d, impact_ids, time_steps):
    
    num_t_steps = ori_pts_2d.shape[0]
    print(num_t_steps)
    num_pts = ori_pts_2d.shape[2]
    indicator = np.zeros(num_pts)
    indicator[impact_ids] = 1

    for i, t in enumerate(time_steps):
        cutplane_pts_i = ori_pts_2d[t].T

        fig, ax = plt.subplots()
        cutplane_x = cutplane_pts_i[:, 0]
        cutplane_y = cutplane_pts_i[:, 1]
        ax.scatter(cutplane_x, cutplane_y, c=indicator)

        for j in range(len(cutplane_x)):
            ax.annotate(j, (cutplane_x[j], cutplane_y[j]))
        ax.grid(True)
        ax.set_aspect('equal')
        plt.show()
        plt.close()



def makeSubTube(variants_coords, variants_velo, time_arr, orb_subset_ids, time_range, out_dir):
    
    num_t_steps = len(time_arr)
    time_steps = np.arange(time_range[0], time_range[1])
    print(time_arr.shape)
    time_arr_subset = time_arr[time_steps]
    print(time_arr_subset.shape)

    print(variants_coords.shape)  # (9,135,000, 3)
    print(variants_velo.shape)
    variants_coords_subset = None
    variants_velo_subset = None
    for i, orb_id in enumerate(orb_subset_ids):
        print(i)
        start_index = orb_id*num_t_steps + time_range[0]
        end_index = orb_id*num_t_steps + time_range[1]

        if variants_coords_subset is None:
            variants_coords_subset = variants_coords[start_index:end_index]
            variants_velo_subset = variants_velo[start_index:end_index]
        else:
            variants_coords_subset = np.concatenate((variants_coords_subset, variants_coords[start_index:end_index]), axis=0)
            variants_velo_subset = np.concatenate((variants_velo_subset, variants_velo[start_index:end_index]), axis=0)
    print(variants_coords_subset.shape)

    np.save(os.path.join(out_dir, "times_isot"), time_arr_subset)
    np.save(os.path.join(out_dir, "variants_coords_" + str(len(orb_subset_ids))), variants_coords_subset)
    np.save(os.path.join(out_dir, "variants_velo_" + str(len(orb_subset_ids))), variants_velo_subset)

    return





if __name__ == "__main__":
    input_dir = "../adam_core/impact_corridor/2004 MN4/2004-12-27T21.28.37.000/"
    variants_coord_f = [filename for filename in os.listdir(input_dir) if filename.startswith("variants_coords_")]
    variants_velo_f = [filename for filename in os.listdir(input_dir) if filename.startswith("variants_velo_")]
    time_f = os.path.join(input_dir, "times_isot.npy")
    print(time_f)
    # There should only be one file of each submission
    assert len(variants_coord_f) == 1
    assert len(variants_velo_f) == 1
    variants_coords = np.load(os.path.join(input_dir, variants_coord_f[0]))
    variants_velo = np.load(os.path.join(input_dir, variants_velo_f[0]))
    time_arr = np.load(time_f)
    print(variants_coord_f)
    print(variants_velo_f)
    print(len(time_arr))

    # Bifurcation happens
    print(time_arr[8800])
    print(time_arr[8872])  # 2029-04-12T23:55:39.463
    print(time_arr[8873])  # 2029-04-13T23:55:40.457
    print(time_arr[8874])  # 2029-04-14T23:55:41.452
    print(time_arr[8950])
    print(time_arr[9000])

    # orb_subset_ids = [949, 733, 993, 88, 881]
    # out_dir = "../adam_core/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/"
    # os.makedirs(out_dir, exist_ok=True)
    # makeSubTube(variants_coords, variants_velo, time_arr, orb_subset_ids, [8800, 9000], out_dir)

    json_dir = "./sampled_data/impact_corridor/2004 MN4/2004-12-27T21.28.37.000/"
    # subsetTube(json_dir, 8800, 9000)

    ori_pts_2d = np.load(os.path.join(json_dir, "ori_points_2d.npy"))
    cluster2dPts(ori_pts_2d, 150)
    impact_ids = [88, 218, 372, 426, 498, 733, 738, 881, 893, 949, 993]
    # impact_ids = [88, 87, 218, 217, 372, 371, 426, 425, 498, 497, 733, 732, 738, 737, 881, 880, 893, 892, 949, 948, 993, 992]
    plotImpact(ori_pts_2d, impact_ids, [150])
    # plotImpact(ori_pts_2d, impact_ids)
    
    # tube_arr = loadJSON(json_dir)

    # num_time_steps = len(tube_arr)
    # num_samples_per_t = len(tube_arr[0])
    # print(num_time_steps)
    # print(num_samples_per_t)
    # for i in range(num_time_steps):
    #     ellipse_i = tube_arr[i]
    #     ellipse_x = []
    

