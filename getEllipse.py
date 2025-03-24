import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances
import json
import os

from mvee import mvee2
from plotting import plot_ellipse
from matplotlib.patches import Ellipse
from astropy import units as astrounit

import spiceypy as spice
METAKERNEL = 'meta-kernel.tm'
spice.furnsh(METAKERNEL)


def computeEllipsoid(points: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper for the function mvee2(): compute the minimum enclosing ellipsoid from input point cloud
    points: input point cloud in 2d or 3d
    
    Output
    L:
    H: the matrix that represents the ellipsoid
    c: the center of the ellipsoid
    """
    obj = mvee2(points)
    L = obj["L"]
    c = obj["c"]
    H = L @ L.T

    return L, H, c


def computePlane(n, c, xr, yr):
    """
    Compute a plane to plot from the normal vector and a point on the plane
    n: the normal vector which is the average velocity of the orbits
    c: the center of the ellipse
    xr: range vector of x
    yr: range vectr of y
    """

    xx, yy = np.meshgrid(xr, yr)
    n0 = n[0]
    n1 = n[1]
    n2 = n[2]
    z = (np.dot(c, n) - np.multiply(n0, xx) - np.multiply(n1, yy))/n2

    return xx, yy, z


def computePlaneLineIntersection(n: npt.ArrayLike, c: npt.ArrayLike, lp: npt.ArrayLike, lv: npt.ArrayLike):
    """
    Compute the intersection point between a plane and a line
    n: the normal vector of the plane (average velocity of the orbits)
    c: the point on the plane (the center of the ellipse)
    lp: the point on the line (an orbit coordinate at a time t)
    lv: the directional vector of the line (the velocity vector of the point at time t)
    
    Output:
    t: the t when the intersection happens
    intersection_coordinate: the coordinate of the intersection
    """

    t = (np.dot(c, n) - np.dot(lp, n))/np.dot(lv, n)
    intersection_coordinate = np.multiply(t, lv) + lp

    return t, intersection_coordinate


def getTranslationZ(n, c):
    """
    Get the magnitude of the translation vector so the new plane passes through (0, 0, 0)
    n: normal of the old plane
    c: point on the old plane (center of the ellipsoid)

    Output:
    z: the translation amount, the translation vector would be (0, 0, z)
    """
    return np.dot(c, n)/n[2]


def getRotationalMatrix(n, k):
    """
    Get the rotational matrix when transforming an old plane to a new plane
    n: the normal of the original 3d plane
    k: the normal of the new plane

    Output:
    r: the rotational matrix (3 x 3)
    """

    # theta is the angle betweeen n and k
    cosTheta = n[2]/np.linalg.norm(n)
    sinTheta = np.sqrt(1 - cosTheta**2)

    # u is the axis or rotation that is orthogonal to both n and k
    nCrossk = np.cross(n, k)
    u = nCrossk/np.linalg.norm(nCrossk)

    # when k is [0, 0, 1], u should be [_, _, 0]
    if np.array_equal(k, np.array([0, 0, 1])):
        assert np.isclose(u[2], 0)
    u1, u2 = u[0], u[1]

    # construct rotational matrix
    r = np.array([
        [cosTheta + u1**2*(1 - cosTheta), u1*u2*(1 - cosTheta), u2*sinTheta],
        [u1*u2*(1 - cosTheta), cosTheta + u2**2*(1 - cosTheta), -u1*sinTheta],
        [-u2*sinTheta, u1*sinTheta, cosTheta]
    ])

    return r


def getNormalSolarSystem(utctime: list[str]):
    """
    Get the normal of the solar system.
    utctime: a time stamp used to compute the transformation matrix between the reference frames. Shouln't matter to the computation.
        e.g. ["Jan 1, 2015"]
    Output:
    nss: normal of the solar system
    """

    ettime = spice.str2et(utctime[0])

    # get the transformation matrix between two reference frames
    rf_trans_mat = spice.sxform(instring='ECLIPJ2000', \
                    tostring='GALACTIC', \
                    et=ettime)
    
    # get the z direction of the rf_trans_mat
    nss = rf_trans_mat[:3, :3] @ np.array([0, 0, 1])

    # compute the unit vector
    nss = nss/np.linalg.norm(nss)

    return nss


def projectVec2Plane(u, n):
    """
    Projoect a vector onto a plane.
    u: vector
    n: normal of the plane

    Output: new vector on the plane
    """
    return u - (np.dot(u, n)/np.linalg.norm(n)**2)*n


def _getMonPlane(n_3d, vec_csun, nss):
    """
    Redundant function used for plotting (Fig 1). Part of getStartingPoint()
    Get m that is perpendicular to the vec_csun and nss. Then project it onto the plane.

    Output:
    m: the vector perpendicular to vec_csun and nss
    proj_m: m projected onto the plane
    """
    m = np.cross(vec_csun, nss)

    # check if m is on the right side because we want m to have consistent orientation as the asteroid traverses around its orbit
    # here we pick np.dot(m, nss) to always be > 0
    # if np.dot(m, nss) < 0:
    #     m = -m

    # vector m might not be on the plane, project m onto the plane
    proj_m = projectVec2Plane(m, n_3d)

    return m, proj_m


def _getMP2d(proj_m, c_3d, trans_z, rotate_mat, c_2d):
    """
    Redundant function used for plotting (Fig 2 and Fig 3). Part of getStartingPoint()
    Get the transformed point of mp_3d -> mp_2d
    mp_3d is a point on the plane starting from c in the direction of proj_m

    Output:
    mp_2d: point on the x-y plane
    c_mp_2d: vector from c_2d to mp_2d on the x-y plane
    """

    # get point mp_3d on the plane using proj_m and c_3d
    mp_3d = c_3d + proj_m

    # transform mp_3d to the x-y plane
    mp_2d = rotate_mat @ (mp_3d - np.array([0, 0, trans_z]))

    # get vector from c_2d to mp_2d
    c_2d_3 = np.array([c_2d[0], c_2d[1], 0])
    c_mp_2d = mp_2d - c_2d_3

    return mp_2d, c_mp_2d


def getRotationMat2D(theta):
    """
    Get the rotational matrix in 2d
    
    Output:
    rot_theta: rotation matrix by theta
    rot_neg_theta: rotation matrix by negative theta
    """
    rot_theta = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rot_neg_theta = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])

    return rot_theta, rot_neg_theta


def getEllipseParam(H_2d):
    """
    Get the parameters for the general ellipse in 2d
    """
    s, u = np.linalg.eigh(H_2d)
    idxs = np.argsort(s)
    s, u = s[idxs], u[:, idxs]
    
    # get radii of the ellipse
    a = np.sqrt(s[0]*2)
    b = np.sqrt(s[1]*2)

    # rotation angle for the ellipse
    theta = np.arctan2(u[1, 0], u[0, 0])

    return a, b, theta


def getRayEllipseIntersection(a, b, ray):
    """
    Get the intersection between the ellipse and rotated_c_mp_2d
    a, b: parameters of the ellipse centered at the origin

    Output: the coordinate of the intersection point
    """
    vx, vy = ray[0], ray[1]
    t = 1/np.sqrt((vx/a)**2 + (vy/b)**2)
    intersection_coord = np.array([vx*t, vy*t])
    return intersection_coord


def getStartingPoint(vec_csun, nss, trans_z, rotate_mat, c_2d, H_2d, c_3d, n_3d):
    """
    Get the starting point to sample from the 2d ellipse centered at the origin
    vec_csun: vector from c_3d to the sun
    c_3d: center of the ellipsoid in 3d
    nss: normal of the solar system
    rotate_mat: rotation matrix from the original 3d plane to the x-y plane
    H_2d: matrix that describes the ellipse
    """
    m = np.cross(vec_csun, nss)
    # print("dot product", np.dot(m, nss))

    # check if m is on the right side because we want m to have consistent orientation as the asteroid traverses around its orbit
    # here we pick np.dot(m, nss) to always be >= 0
    # if np.dot(m, nss) < 0:
    #     print("dot product is less than 0")
    #     print("original m", m)
    #     m = m
    #     print("updated m", m)

    # vector m might not be on the plane, project m onto the plane
    proj_m = projectVec2Plane(m, n_3d)

    # get point mp_3d on the plane using proj_m and c_3d
    mp_3d = c_3d + proj_m

    # transform mp_3d to the x-y plane
    mp_2d = rotate_mat @ (mp_3d - np.array([0, 0, trans_z]))
    assert np.isclose(mp_2d[2], 0, rtol=1e-08)
    mp_2d[2] = 0  # force the z coordinate to be 0

    # get vector from c_2d to mp_2d
    c_mp_2d = mp_2d[:2] - c_2d

    # get 2d ellipse parameters
    a, b, theta = getEllipseParam(H_2d)
    # print("theta", theta)
    # theta_degrees = theta * 180 / np.pi
    # print("theta degrees", theta_degrees)

    # rotate mp_2d by -1*theta
    _, rot_neg_theta = getRotationMat2D(theta)
    rotated_c_mp_2d = rot_neg_theta @ c_mp_2d[:2]

    # compute intersection point on the 2d ellipse centered at the origin
    elli_r_o = getRayEllipseIntersection(a, b, rotated_c_mp_2d)

    return rotated_c_mp_2d, elli_r_o


def angle2phi(angle, a, b):
    """
    From the actual angle to the parameter phi of the parameterized equation of an ellipse
    x = a*sin(phi), y = a*cos(phi)
    a, b are the semi-major axes radii of the ellipse
    """
    # phi = np.arctan2(a*np.tan(angle), b)  # this only gives results from -pi/2 to pi/2 need the whole 2pi
    phi = angle - np.arctan2((b-a)*np.tan(angle), b + a*np.tan(angle)**2)
    
    return phi


def ellipse_arc(a, b, theta_sample, n):
    """Cumulative arc length of ellipse with given dimensions"""

    # Divide the interval [theta_sample , theta_sample + 2*pi] into n steps at regular angles
    t = np.linspace(theta_sample, theta_sample + 2*np.pi, n)

    # Using parametric form of ellipse, compute ellipse coord for each t
    x, y = np.array([a * np.cos(t), b * np.sin(t)])

    # Compute vector distance between each successive point
    x_diffs, y_diffs = x[1:] - x[:-1], y[1:] - y[:-1]

    cumulative_distance = [0]
    c = 0

    # Iterate over the vector distances, cumulating the full arc
    for xd, yd in zip(x_diffs, y_diffs):
        c += np.sqrt(xd**2 + yd**2)
        cumulative_distance.append(c)
    cumulative_distance = np.array(cumulative_distance)

    # Return theta-values, distance cumulated at each theta,
    # and total arc length for convenience
    return t, cumulative_distance, c


def theta_from_arc_length_constructor(a, b, theta_sample, precision):
    """
    Inverse arc length function: constructs a function that returns the
    angle associated with a given cumulative arc length for given ellipse."""

    # Get arc length data for this ellipse
    t, cumulative_distance, total_distance = ellipse_arc(a, b, theta_sample, precision)

    # Construct the function
    def f(s):
        assert np.all(s <= total_distance), "s out of range"
        # Can invert through interpolation since monotonic increasing
        return np.interp(s, cumulative_distance, t)

    # return f and its domain
    return f, total_distance


def sampleEllipse2D(a, b, theta_sample=0, sample_size=50, precision=1000):
    """
    Sample points from the 2d ellipse centered atthe origin.
    a, b: parameters of the 2d ellipse centered at (0, 0)
    theta: the angle to start sampling. We sample in the interval [theta, theta+2*pi]
    n: the number of points to sample
    precision: controls the precision of the arc length calculation.
    """
    theta_from_arc_length, domain = theta_from_arc_length_constructor(a, b, theta_sample, precision)
    # sample_size+1 to fix the issue that the first and the last points overlap
    s = np.linspace(0, 1, sample_size+1) * domain
    t = theta_from_arc_length(s)
    x, y = np.array([a * np.cos(t), b * np.sin(t)])
    # take away the last point that overlap. We now have the correct number of points without overlap
    x = x[:-1]
    y = y[:-1]
    assert len(x) == sample_size

    return x, y


def transformPts3D(sampled_x: np.array, sampled_y: np.array, c_2d, rot_theta, rotate_mat, trans_z):
    """
    Transform the sampled points from the 2d ellipse centered at (0, 0) back to the original 3d 
    
    sampled_x, sampled_y: sampled 2d points on the ellipse centered at (0, 0) with no rotation
    c_2d: center of the ellipse in 2d
    rot_theta: 2d rotational matrix (to rotate back to the original rotation of the ellipse)
    rot_mat: 3d tranformation matrix (from the xy plane to the 3d space)
    trans_z: translation amount in the z direction
    """
    
    # Note: the order of rotation and translation is important. Here we have to rotate first and then translate
    # rotate the points by theta (the angle of the 2d ellipse)
    sampled_xy = np.stack((sampled_x, sampled_y))
    sampled_xy = rot_theta @ sampled_xy

    # shift the 2d points on the ellipse by c_2d
    x_shift = np.full(sampled_x.shape, c_2d[0])
    x_2d = sampled_xy[0] + x_shift
    y_shift = np.full(sampled_y.shape, c_2d[1])
    y_2d = sampled_xy[1] + y_shift
    sampled_trans_xy = np.stack((x_2d, y_2d))

    # transform all the points back into the original 3d space
    z_zeros = np.zeros((1, sampled_trans_xy.shape[1]))
    sampled_pts_3d = np.concatenate((sampled_trans_xy, z_zeros), axis=0)
    rot_mat_inv = np.linalg.inv(rotate_mat)
    sampled_pts_3d = rot_mat_inv @ sampled_pts_3d
        
    # get the inverse translation z
    num_samples = len(sampled_x)
    inverse_trans_vec = np.tile(np.array([0, 0, -trans_z]).T, (num_samples, 1)).T
    sampled_pts_3d -= inverse_trans_vec

    return sampled_trans_xy, sampled_pts_3d


def getSampledPtVals(a, b, c_2d, sampled_x, sampled_y, transformed_2d_x, transformed_2d_y, rot_neg_theta, r=None):
    
    # rotate the original 2d points centered at (0, 0) by neg_theta
    x_shift_ori = np.full(transformed_2d_x.shape, c_2d[0])
    y_shift_ori = np.full(transformed_2d_y.shape, c_2d[1])
    trans_rot_2d_x = transformed_2d_x - x_shift_ori
    trans_rot_2d_y = transformed_2d_y - y_shift_ori
    trans_rot_2d_xy = np.stack((trans_rot_2d_x, trans_rot_2d_y))
    trans_rot_2d_xy = rot_neg_theta @ trans_rot_2d_xy
    
    intersections_2d_all = np.zeros(trans_rot_2d_xy.shape)
    for i in range(trans_rot_2d_xy.shape[1]):
        ray_i = trans_rot_2d_xy[:, i]
        intersections_2d_all[:, i] = getRayEllipseIntersection(a, b, ray_i)

    # count the number of intersections within a certain radius of each sampled point on the ellipse
    # compute distance table (num sample points, num of intersection points)
    sampled_xy = np.stack((sampled_x, sampled_y))
    num_sampled_pts = sampled_xy.shape[1]
    num_intersection_pts = intersections_2d_all.shape[1]
    lookup_table = np.zeros((num_sampled_pts, num_intersection_pts))
    sampled_xy = np.stack((sampled_x, sampled_y))
    for i in range(num_sampled_pts):
        for j in range(num_intersection_pts):
            lookup_table[i, j] = np.linalg.norm(sampled_xy[:, i] - intersections_2d_all[:, j])
    
    
    # select threshold r (take the min of 2a, 2b and half of the distance between two adjacent sample points)
    if r is None:
        sample_pts_dist = 0.5*np.linalg.norm(sampled_xy[:, 0] - sampled_xy[:, 1])
        r = min(2*a, 2*b, sample_pts_dist)

    # distance table -> indicator matrix
    indicator_mat = np.where(lookup_table < r, 1, 0)

    # count the number of points for each sample point
    sampled_pt_vals = np.sum(indicator_mat, axis=1, dtype=int)
    sampled_pt_vals = sampled_pt_vals.astype(int).tolist()

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.set_aspect('equal')
    # ax.scatter(trans_rot_2d_xy[0], trans_rot_2d_xy[1])
    # ax.scatter(intersections_2d_all[0], intersections_2d_all[1], alpha=0.3, color="green")
    # sample_i_circle_index = np.where(indicator_mat[0] == 1)[0]
    # sample_i_circle = intersections_2d_all[:, sample_i_circle_index]
    # ax.scatter(sample_i_circle[0], sample_i_circle[1], color="deepskyblue")
    # ax.scatter(sampled_x, sampled_y, c=sampled_pt_vals, cmap="magma_r")
    # for i, val in enumerate(sampled_pt_vals):
    #     ax.annotate(val, (sampled_x[i], sampled_y[i]))
    # plt.show()

    return sampled_pt_vals


def getTextureCoordinates(c_2d, sampled_x, sampled_y, transformed_2d_x, transformed_2d_y, rot_neg_theta, img_name=None, img_dir=None, p_radius=5, res=500, save_textures=False):
    """
    Generate texture coordinates and the image for the cut plane of the tube.
    c_2d: center of the 2d ellipse.
    sampled_x, sampled_y: sampled points on the 2d ellipse sentered at (0, 0)
    transformed_2d_x, transformed_2d_y: original points after the 3d to 2d transformation (not centered at (0, 0))
    rot_neg_theta: the tranformation matrix for rotating by -theta
    time_lag: the time it takes for each orbit to reach the 3d plane (used as the second channel of the texture)
    p_radius: odd integer, the radius of the points in the image of the cut plane, e.g. 3 -> 3 x 3 patch
    """

    # rotate the original 2d points centered at (0, 0) by neg_theta. All points are now centered at (0, 0) without rotation
    x_shift_ori = np.full(transformed_2d_x.shape, c_2d[0])
    y_shift_ori = np.full(transformed_2d_y.shape, c_2d[1])
    trans_rot_2d_x = transformed_2d_x - x_shift_ori
    trans_rot_2d_y = transformed_2d_y - y_shift_ori
    trans_rot_2d_xy = np.stack((trans_rot_2d_x, trans_rot_2d_y))
    trans_rot_2d_xy = rot_neg_theta @ trans_rot_2d_xy

    # find max x and y difference in sampled points
    max_x_diff = np.max(pairwise_distances(sampled_x.reshape(-1,1)))
    max_y_diff = np.max(pairwise_distances(sampled_y.reshape(-1,1)))

    # there might be some error with the differences, continue to adjust to make sure all points are non-negative
    if min(sampled_x + max_x_diff/2) < 0:
        max_x_diff = max_x_diff + abs(min(sampled_x + max_x_diff))
    if min(sampled_y + max_y_diff/2) < 0:
        max_y_diff = max_y_diff + abs(min(sampled_y + max_y_diff))

    # use the larger difference directin as the one to compute img_ratio with
    if max_x_diff > max_y_diff:
        max_diff = max_x_diff
        img_ratio = (res - 1)/max_x_diff
    else:
        max_diff = max_y_diff
        img_ratio = (res - 1)/max_y_diff

    # draw the sampled points on an image
    img_mat = np.zeros((res, res, 3))
    adjusted_x = sampled_x + max_diff/2
    adjusted_y = sampled_y + max_diff/2
    # print(adjusted_x)
    # print(adjusted_y)
    sampled_u = adjusted_x*img_ratio
    sampled_v = adjusted_y*img_ratio

    # img_mat[sampled_u] = np.array([1, 0, 0])
    # img_mat[sampled_v] = np.array([1, 0, 0])

    assert all(x >= 0 for x in sampled_u) and all(x < res for x in sampled_u)
    assert all(x >= 0 for x in sampled_v) and all(x < res for x in sampled_v)

    sampled_u_percent = sampled_u/res
    sampled_v_percent = sampled_v/res

    # sampled_u = np.round(sampled_u).astype(int)
    # sampled_v = np.round(sampled_v).astype(int)

    # for i in range(len(sampled_u)):
    #     img_mat[sampled_u[i], sampled_v[i]] = np.array([1, 1, 1])

    # compute the u (trans_rot_2d_xy[1]), v pixel (trans_rot_2d_xy[0]) for each point
    trans_rot_2d_u = np.round((trans_rot_2d_xy[0] + max_diff/2)*img_ratio).astype(int)
    trans_rot_2d_v = np.round((trans_rot_2d_xy[1] + max_diff/2)*img_ratio).astype(int)
    
    # make points visible by the specified p_radius
    for i in range(len(trans_rot_2d_u)):
        pu = trans_rot_2d_u[i]
        pv = trans_rot_2d_v[i]
        # t = time_lag[i]  # TODO: implement time lag later
        
        # ignore the points that outside of the image since they are outside of the ellipse
        if pu < 0 or pv < 0 or pu > res-1 or pv > res-1:
            continue

        img_mat[pu, pv] = np.array([1, 0, 0])
        offset = p_radius//2
        if p_radius%2 == 1:  # odd
            for j in range(offset):
                img_mat[max(0, pu-j), min(pv+j, res-1)] = np.array([1, 0, 0])
                img_mat[min(pu, res-1), min(pv+j, res-1)] = np.array([1, 0, 0])
                img_mat[min(pu+j, res-1), min(pv+j, res-1)] = np.array([1, 0, 0])
                img_mat[max(0, pu-j), min(pv, res-1)] = np.array([1, 0, 0])
                img_mat[min(pu+j, res-1), min(pv, res-1)] = np.array([1, 0, 0])
                img_mat[max(0, pu-j), max(0, pv-j)] = np.array([1, 0, 0])
                img_mat[min(pu, res-1), max(0, pv-j)] = np.array([1, 0, 0])
                img_mat[min(pu+j, res-1), max(0, pv-j)] = np.array([1, 0, 0])

    assert np.max(img_mat) <= 1
    assert np.min(img_mat) >= 0

    if save_textures is True:
        print("saving texture...")
        img_path = os.path.join(img_dir, img_name)
        print("img path", img_path)
        plt.imshow(img_mat, interpolation="spline16")
        plt.axis('off')
        plt.imsave(img_path, img_mat)
        plt.close()

    
    # print(np.max(img_mat))
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # plt.imshow(img_mat, cmap='gray', vmin=0, vmax=1)
    # plt.axis("off")
    # ax2 = fig.add_subplot()
    # ax2.set_aspect('equal')
    # ax2.scatter(adjusted_x, adjusted_y)
    # ax2.scatter(trans_rot_2d_xy[0], trans_rot_2d_xy[1])
    # plt.show()
    
    if save_textures is True:
        return sampled_u_percent, sampled_v_percent
    else:
        return sampled_u_percent, sampled_v_percent, img_mat


def dumpJSON(sampled_pts_all_list, c_3d_all_list, time_arr_list, sampled_u_all_list, sampled_v_all_list, axes_length_all, axes_direction_all, out_dir, sampled_pt_vals_all=None, img_mat_all_list=None, starting_time_index=None):
    
    """
    Save tube data to JSON
    starting_time_index: the starting time step index of the tube, if it is not 0.
    """
    
    data_dict = {"version": {"major": 0, "minor": 1},
                 "texture-channels": ["density", "time-delta"],
                 "polygons": []}
    num_pts_per_t = sampled_pts_all_list[0].shape[1]

    # t_min = time_lag_range[0]
    # t_max = time_lag_range[1]
    # half_range = np.max([abs(t_min), abs(t_max)])
    # print("t_min", t_min)
    # print("t_max", t_max)
    # print("half_range", half_range)

    for t, time_step in enumerate(time_arr_list):
        data_dict["polygons"].append({"time": time_step})
        c_3d_t_meters = c_3d_all_list[t]
        c_3d_t_meters *= astrounit.au.to(astrounit.m)
        data_dict["polygons"][t]["center"] = {"x": c_3d_t_meters[0],
                                       "y": c_3d_t_meters[1],
                                       "z": c_3d_t_meters[2]}
        
        data_dict["polygons"][t]["axes-length"] = {"a": axes_length_all[t][0],
                                                   "b": axes_length_all[t][1],
                                                   "c": axes_length_all[t][2]}
        
        axes_direction_all_t = axes_direction_all[t]
        data_dict["polygons"][t]["axes-direction"] = {"x1": axes_direction_all_t[0][0],
                                                      "x2": axes_direction_all_t[0][1],
                                                      "x3": axes_direction_all_t[0][2],
                                                      "y1": axes_direction_all_t[1][0],
                                                      "y2": axes_direction_all_t[1][1],
                                                      "y3": axes_direction_all_t[1][2],
                                                      "z1": axes_direction_all_t[2][0],
                                                      "z2": axes_direction_all_t[2][1],
                                                      "z3": axes_direction_all_t[2][2]}
        
        if starting_time_index is not None:
            data_dict["polygons"][t]["texture"] = str(t + starting_time_index) + ".png"
        else:
            data_dict["polygons"][t]["texture"] = str(t) + ".png"
        
        if sampled_pt_vals_all is not None:
            sampled_pt_val_t = sampled_pt_vals_all[t]

        # img_t_dir = os.path.join(out_dir, str(t))
        # os.makedirs(img_t_dir, exist_ok=True)
        points_arr_per_t = []
        for i in range(num_pts_per_t):
            # add point coordinates
            pt = sampled_pts_all_list[t][:, i]
            pt *= astrounit.au.to(astrounit.m)
            # add texture coordinates
            texture_coord_u = sampled_u_all_list[t][i]
            texture_coord_v = sampled_v_all_list[t][i]
            points_arr_per_t.append({"x": pt[0], "y": pt[1], "z": pt[2],
                                     "u": texture_coord_v, "v": texture_coord_u, # the u and v are flipped
                                     "data": {"density": sampled_pt_val_t[i]}})
            
        # save the texture coordinate images
        if img_mat_all_list is not None:
            img_mat_t = img_mat_all_list[t]  # (res, res, 3)
            img_dir = os.path.join(out_dir, "textures")
            os.makedirs(img_dir, exist_ok=True)

            if starting_time_index is not None:
                img_name = str(t + starting_time_index) + ".png"
            else:
                img_name = str(t) + ".png"
            img_path = os.path.join(img_dir, img_name)
            # fig = plt.figure()
            plt.imshow(img_mat_t, interpolation="spline16")
            plt.axis('off')
            plt.imsave(img_path, img_mat_t)
            plt.close()
            # plt.show()
            
        # img_mat_shape = img_mat_t.shape

        # Normalize channel g based on time_lag_range so that 0 -> 0.5
        # We split the time_lag_range to positive and negative side. The larger side desides (half_range) decides the ratio
        # We use the ratio to rescale the values in g channel
        # for i in range(img_mat_shape[0]):
        #     for j in range(img_mat_shape[1]):
                # if img_mat_t[i, j, 0] != 0:
                #     print("r channel", img_mat_t[i, j, 0])
                # if img_mat_t[i, j, 1] != 0:
                #     print("g channel", img_mat_t[i, j, 1])
                # img_mat_t[i, j, 1] = img_mat_t[i, j, 1]*0.5/half_range + 0.5
                # print(img_mat_t[i, j, 1])
        # print(np.min(img_mat_t))
        # print(np.max(img_mat_t))
        # assert np.max(img_mat_t) <= 1
        # assert np.min(img_mat_t) >= 0

        data_dict["polygons"][t]["points"] = points_arr_per_t
    
    jsonpath = os.path.join(out_dir, "tube_data.json")

    with open(jsonpath, 'w') as fp:
        json.dump(data_dict, fp)



def getEllipsePerSubmission(variants_dir, num_sample_ellipse, out_dir, sectioned_uncertainty=True, plotEllipse=False):
    """
    Get the ellipses for orbits from the same submission period
    
    variants_dir: should end with "/"
    num_sample_ellipse: the number of sample points we get from the circumference of the ellipse
    """
    # Get positions of astroid w.r.t. the sun for a given time
    # use adam_core to get the orbit positions at a specific time
    variants_coord_f = [filename for filename in os.listdir(variants_dir) if filename.startswith("variants_coords_")]
    variants_velo_f = [filename for filename in os.listdir(variants_dir) if filename.startswith("variants_velo_")]
    time_f = os.path.join(variants_dir, "times_isot.npy")
    print(time_f)
    # There should only be one file of each submission
    assert len(variants_coord_f) == 1
    assert len(variants_velo_f) == 1
    variants_coords = np.load(os.path.join(variants_dir, variants_coord_f[0]))
    variants_velo = np.load(os.path.join(variants_dir, variants_velo_f[0]))
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
    # variants_coords and variants_velo are ordered per orbit through all time steps
    # to make an ellipse slice, we need all orbits at time i
    for i in range(num_time_steps):
        variants_coords_list.append(variants_coords[i::num_time_steps])
        variants_velo_list.append(variants_velo[i::num_time_steps])
    # for i in range(num_time_steps):
    #     print(i)
    #     variants_coords_list.append(variants_coords[i*num_samples:(i+1)*num_samples])
    #     variants_velo_list.append(variants_velo[i*num_samples:(i+1)*num_samples])

    print("length of variants coords list", len(variants_coords_list))
    print("length of variants coords list [0]", len(variants_coords_list[0]))
    print(variants_coords_list[0].shape)
    sampled_pts_all = []
    c_3d_all = []
    time_lag_all = []
    sampled_pt_vals_all = []
    sampled_u_all = []
    sampled_v_all = []
    axes_length_all = []
    axes_direction_all = []

    # for Apophis analysis
    transformed_2d_all = []

    # make directory to save the texture coordinates
    img_dir = os.path.join(out_dir, "textures")

    if sectioned_uncertainty is True:
        img_mat_all = []
    else:
        os.makedirs(img_dir, exist_ok=True)

    for i in range(num_time_steps):
        print("time step", i)
        print(time_arr[i])

        # for Apophis analysis
        # if i < 8870:
        #     continue
        # if i >= 9000:
        #     break

        # Compute ellipsoid and center of the ellipsoid using mvee
        Xi = variants_coords_list[i].T
        print("Xi shape", Xi.shape)
        L_3d, H_3d, c_3d = computeEllipsoid(Xi)
        
        # The eigenvectors of H_3d are the orientation of the semi-axes
        # Can compute the length of the semi-axes a, b, c from the eigenvalues of H_3d
        eigenvalues, eigenvectors = np.linalg.eig(H_3d)
        axes_length = np.sqrt(np.reciprocal(eigenvalues))
        print(axes_length)
        print(eigenvectors)
        axes_length_all.append(axes_length)
        axes_direction_all.append(eigenvectors)  # columns are the eigenvectors

        # Get unit vector from c_3d to the sun -> vec_csun
        vec_csun = np.array([0, 0, 0]) - c_3d
        vec_csun = vec_csun/np.linalg.norm(vec_csun)

        # Get the normal of the solar system
        utctime = ["Jan 1, 2015"]
        nss = getNormalSolarSystem(utctime)

        # use the average velocity vector (n_3d) as the normal of the plane
        Vi = variants_velo_list[i].T
        n_3d = np.mean(Vi, axis=1)

        # compute orbit plane intersection points and time lag for all orbits
        orbit_plane_intersection = np.zeros(Xi.shape)
        time_lag = np.zeros(Xi.shape[1])
        for j in range(Xi.shape[1]):
            lp = Xi[:, j]
            lv = Vi[:, j]
            t, intersection_coor = computePlaneLineIntersection(n_3d, c_3d, lp, lv)
            time_lag[j] = t
            orbit_plane_intersection[:, j] = intersection_coor

        if plotEllipse is True:
            # Fig 1 (3d): plot points and ellipsoid center
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            x_3d = Xi[0, :]
            y_3d = Xi[1, :]
            z_3d = Xi[2, :]
            ax.scatter(x_3d, y_3d, z_3d)
            ax.scatter(c_3d[0], c_3d[1], c_3d[2], s=50)
            print("x_3d shape", x_3d.shape)

            # Fig 1 (3d): plot unit vector vec_csun
            # ax.quiver(c_3d[0], c_3d[1], c_3d[2], vec_csun[0], vec_csun[1], vec_csun[2], color='red')

            # Fig 1 (3d): plot the unit vector of nss
            # ax.quiver(c_3d[0], c_3d[1], c_3d[2], nss[0], nss[1], nss[2], color='darkorchid')

            # Fig 1 (3d): plot n_3d and the plane
            xr = np.linspace(c_3d[0] - 3e-08, c_3d[0] + 3e-08, num=20)
            yr = np.linspace(c_3d[1] - 3e-08, c_3d[1] + 3e-08, num=20)
            xx, yy, pz = computePlane(n_3d, c_3d, xr, yr)
            # ax.plot_surface(xx, yy, pz, color="green", alpha=0.5)
            ax.quiver(c_3d[0], c_3d[1], c_3d[2], n_3d[0], n_3d[1], n_3d[2], color='green')
            ax.set_aspect('equal')
        
            # Fig 1 (3d): plot orbit plane intersection points on the plan
            intersectX = orbit_plane_intersection[0, :]
            intersectY = orbit_plane_intersection[1, :]
            intersectZ = orbit_plane_intersection[2, :]
            ax.scatter(intersectX, intersectY, intersectZ)

            # (for plotting) get the m vector and the projected m vector
            m, proj_m = _getMonPlane(n_3d, vec_csun, nss)

            # Fig 1 (3d): plot m and proj_m from the c_3d
            # ax.quiver(c_3d[0], c_3d[1], c_3d[2], m[0], m[1], m[2], color='gold')
            ax.quiver(c_3d[0], c_3d[1], c_3d[2], proj_m[0], proj_m[1], proj_m[2], color='tab:orange')
            plt.show()

        # Note that the intersection points are on a 3d plane
        # Need to transform the plane onto the xy-plane with a translation (trans_vec) and a rotation (rotate_mat)
        # n_2d is the normal of the x-y plane (0, 0, 1)
        n_2d = np.array([0, 0, 1])
        trans_z = getTranslationZ(n_3d, c_3d)
        rotate_mat = getRotationalMatrix(n_3d, n_2d)
        
        # get the translated and rotated points. All points should be on the x-y plane
        expanded_trans_vec = np.tile(np.array([0, 0, trans_z]).T, (num_samples, 1)).T
        translated_points = orbit_plane_intersection - expanded_trans_vec
        transformed_points = rotate_mat @ translated_points

        # Force the z coordinates of the transformed points to be 0
        transformed_points[2, :] = 0

        # Then this becomes a 2d problem
        # Compute the minimum enclosing ellipse of the intersection points on the xy-plane
        transformed_2d = transformed_points[:2, :]
        print(transformed_2d.shape)
        transformed_2d_x = transformed_2d[0, :]
        transformed_2d_y = transformed_2d[1, :]
        L_2d, H_2d, c_2d = computeEllipsoid(transformed_2d)

        # Compute ellipse ray intersection centered at origin
        a, b, theta = getEllipseParam(H_2d)
        rot_theta, rot_neg_theta = getRotationMat2D(theta)
        rotated_c_mp_2d, elli_r_o = getStartingPoint(vec_csun, nss, trans_z, rotate_mat, c_2d, H_2d, c_3d, n_3d)

        # Compute the angle in radiant of the intersection point
        rad = np.arctan2(elli_r_o[1], elli_r_o[0])
        # Transform it to the parameter for the ellipse
        phi = angle2phi(rad, a, b)

        # Compute new p from phi
        new_p = a*np.cos(phi), b*np.sin(phi)
        assert np.isclose(new_p[0], elli_r_o[0])
        assert np.isclose(new_p[1], elli_r_o[1])

        if plotEllipse is True:
            # Fig 2 (3d): plot the new plane and the transformed points
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            transformedX = transformed_points[0, :]
            transformedY = transformed_points[1, :]
            transformedZ = transformed_points[2, :]
            ax.scatter(transformedX, transformedY, transformedZ)
            # ax.set_aspect('equal')  # super important: ensures consistent scale for the axes!!!

            # Fig 2 (3d): plot the center of the ellipse
            ax.scatter(c_2d[0], c_2d[1], 0, s=50, c="red")

            # Get point mp_2d and vector c_mp_3d
            m, proj_m = _getMonPlane(n_3d, vec_csun, nss)
            mp_2d, c_mp_2d = _getMP2d(proj_m, c_3d, trans_z, rotate_mat, c_2d)

            # Rotaet c_mp_2d based on the rotation angle of the ellipse
            a, b, theta = getEllipseParam(H_2d)
            rot_theta, rot_neg_theta = getRotationMat2D(theta)
            rotated_c_mp_2d = rot_neg_theta @ c_mp_2d[:2]
            # TODO: Not sure why sometimes the rotated_c_mp_2d is on the other size of the ellipse

            # Fig 2 (3d): plot mp_2d and c_mp_2d
            ax.quiver(c_2d[0], c_2d[1], 0, c_mp_2d[0], c_mp_2d[1], c_mp_2d[2], color="tab:orange")
            plt.show()

            # Fig 3 (2d): plot a 2D version of the problem, points and min ellipse and c_2d
            fig = plt.figure()
            ax = fig.add_subplot()
            # transformed_2d_x = transformed_2d[0, :]
            # transformed_2d_y = transformed_2d[1, :]
            # ax.scatter(c_2d[0], c_2d[1], s=50, c='red')
            ax.scatter(transformed_2d_x, transformed_2d_y)
            plot_ellipse(H_2d, c_2d, ax=ax)

            # Fig 3 (2d): plot c_mp_2d
            ax.quiver(c_2d[0], c_2d[1], c_mp_2d[0], c_mp_2d[1], scale=1, width=0.02, color="tab:orange")

            # Fig 3 (2d): plot rotated_c_mp_2d and the ellipse without rotation
            ax.quiver(c_2d[0], c_2d[1], rotated_c_mp_2d[0], rotated_c_mp_2d[1], scale=1, width=0.02, color="tab:pink")
            kwrg = {'facecolor': 'none', 'edgecolor':'tab:pink', 'alpha':1, 'linewidth':2}
            ellip = Ellipse(xy=c_2d, width=2*a, height=2*b, angle=0, **kwrg)
            ax.set_aspect('equal')
            plt.xticks([])
            plt.yticks([])
            ax.axis("off")
            ax.add_artist(ellip)
            plt.show()

            # Fig 4 (2d): plot the rotated array and the ellipse at the origin
            fig = plt.figure()
            ax = fig.add_subplot()
            kwrg = {'facecolor': 'none', 'edgecolor':'gray', 'alpha':1, 'linewidth':2}
            ellip_origin = Ellipse(xy=np.array([0, 0]), width=2*a, height=2*b, angle=0, **kwrg)
            ax.set_aspect('equal')
            ax.add_artist(ellip_origin)
            ax.relim()
            ax.autoscale_view()

            ax.quiver(0, 0, rotated_c_mp_2d[0], rotated_c_mp_2d[1], scale=0.3, color="gray")
        
            # Fig 4 (2d): plot the intersection point
            ax.scatter(elli_r_o[0], elli_r_o[1])

            # Fig 4 (2d): plot the new point computed from phi. it should overlap with elli_r_o
            ax.scatter(new_p[0], new_p[1], color="red")
            plt.show()
        
        # Sample points from the 2d ellipse
        x_elli_2d, y_elli_2d = sampleEllipse2D(a, b, phi, num_sample_ellipse, 1000)

        # Get sample points density value
        sampled_pt_vals = getSampledPtVals(a, b, c_2d, x_elli_2d, y_elli_2d, transformed_2d_x, transformed_2d_y, rot_neg_theta)
        sampled_pt_vals_all.append(sampled_pt_vals)

        # Get texture coordinates
        if sectioned_uncertainty is True:
            sampled_u, sampled_v, img_mat = getTextureCoordinates(c_2d, x_elli_2d, y_elli_2d, transformed_2d_x, transformed_2d_y, rot_neg_theta, save_textures=False)
        else:
            img_name = str(i) + ".png"
            sampled_u, sampled_v = getTextureCoordinates(c_2d, x_elli_2d, y_elli_2d, transformed_2d_x, transformed_2d_y, rot_neg_theta, img_name, img_dir, save_textures=True)

        # Transform the sampled points back to the original 3d space
        sampled_trans_xy, sampled_pts_3d = transformPts3D(x_elli_2d, y_elli_2d, c_2d, rot_theta, rotate_mat, trans_z)
        print("sampled 2d points shape")
        print(sampled_trans_xy.shape)

        if plotEllipse is True:
            # Fig 5 (2d): plot the sampled points from the ellipse
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_aspect('equal')
            ax.scatter(x_elli_2d, y_elli_2d, alpha=0.5, color="forestgreen")
            # plot the first and the last points
            ax.scatter(x_elli_2d[0], y_elli_2d[0], alpha=0.5, color="tab:pink")
            # ax.scatter(x_elli_2d[1], y_elli_2d[1], alpha=0.5, color="blue")
            # ax.scatter(x_elli_2d[-1], y_elli_2d[-1], alpha=0.5, color="crimson")
            ax.quiver(0, 0, rotated_c_mp_2d[0], rotated_c_mp_2d[1], scale=0.8, width=0.05, color="tab:pink")
            ax.axis("off")
            plt.xticks([])
            plt.yticks([])
            plt.show()

            # Fig 5.5 (2d): plot the sampled points colored by the number of original points assigned to each sample point
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_aspect('equal')
            s = ax.scatter(x_elli_2d, y_elli_2d, c=sampled_pt_vals, cmap="magma_r")
            for i, val in enumerate(sampled_pt_vals):
                ax.annotate(val, (x_elli_2d[i], y_elli_2d[i]))
            ax.axis("off")
            plt.xticks([])
            plt.yticks([])
            fig.colorbar(s, ax=ax, cmap="magma_r", orientation='vertical')
            plt.show()

            # Fig 6 (2d): Plot the original ellipse centered at c_2d and the sampled points centered at c_2d
            fig = plt.figure()
            ax = fig.add_subplot()
            transformed_2d_x = transformed_2d[0, :]
            transformed_2d_y = transformed_2d[1, :]
            ax.scatter(c_2d[0], c_2d[1], s=50, c='red')
            ax.scatter(transformed_2d_x, transformed_2d_y)
            ax.set_aspect('equal')
            plot_ellipse(H_2d, c_2d, ax=ax)
            

            # Transform the sampled points back to the original 3d space, plot the 3d points in the xy plane
            plt.scatter(sampled_trans_xy[0], sampled_trans_xy[1], color="orange")
            plt.show()

            # Fig 7 (3d): plot the original points in 3d and the sampled ellipse in 3d
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(sampled_pts_3d[0], sampled_pts_3d[1], sampled_pts_3d[2], color="tab:orange")
            ax.scatter(Xi[0, :], Xi[1, :], Xi[2, :], color="tab:blue")
            ax.scatter(c_3d[0], c_3d[1], c_3d[2], s=50, color="red")
            ax.set_aspect('equal')
            plt.show()

            # fig 8: plot the texture coordinates
            # fig = plt.figure()
            # texture_img = plt.imshow(img_mat, interpolation='none')
            # plt.axis('off')
            # plt.colorbar(texture_img, orientation='horizontal')
            # plt.show()
            # plt.close("all")

        sampled_pts_all.append(sampled_pts_3d)
        c_3d_all.append(c_3d)
        time_lag_all.append(time_lag)
        sampled_u_all.append(sampled_u)
        sampled_v_all.append(sampled_v)

        # for Apophis analysis
        transformed_2d_all.append(transformed_2d)
        # print(len(transformed_2d_all))

        if sectioned_uncertainty is True:
            img_mat_all.append(img_mat)
    
    # save transformed_2d points for Apophis analysis
    print("Saving transformed 2d points")
    transformed_2d_f = os.path.join(out_dir, "ori_points_2d")
    np.save(transformed_2d_f, transformed_2d_all)
    # time_arr = time_arr[8800:9000]
            
    if sectioned_uncertainty is True:
        return time_arr, time_lag_all, sampled_pts_all, c_3d_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all, axes_length_all, axes_direction_all, img_mat_all
    else:
        return time_arr, time_lag_all, sampled_pts_all, c_3d_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all, axes_length_all, axes_direction_all


if __name__ == "__main__":

    # For 2023 CX1 (impact corridor)...
    input_dir = "./input_data/2023 CX1/2023-02-13T02.38.19.001/"
    out_dir = "./sampled_data/2023 CX1/2023-02-13T02.38.19.001/"
    os.makedirs(out_dir, exist_ok=True)
    time_arr, time_lag_all, sampled_pts_all, c_3d_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all, axes_length_all, axes_direction_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=True)
    print(time_arr)
    print(time_lag_all)
    dumpJSON(sampled_pts_all, c_3d_all, time_arr, sampled_u_all, sampled_v_all, axes_length_all, axes_direction_all, out_dir, sampled_pt_vals_all)

    # For figures...
    # input_dir = "../adam_core/dynamic_uncertainty/2012 DA14/2013-02-10T04.54.49.000/"
    # out_dir = "./sampled_data/dynamic_uncertainty/2012 DA14/for_figures/"
    # os.makedirs(out_dir, exist_ok=True)
    # time_arr, time_lag_all, sampled_pts_all, c_3d_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=True)

    # Increase Apophis resolution...
    # input_dir = "../adam_core/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/adaptive/"
    # out_dir = "./sampled_data/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/adaptive/"
    # os.makedirs(out_dir, exist_ok=True)
    # time_arr, time_lag_all, sampled_pts_all, c_3d_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=False)
    # dumpJSON(sampled_pts_all, c_3d_all, time_arr, sampled_u_all, sampled_v_all, out_dir, sampled_pt_vals_all)

    # For 2023 CX1 (impact corridor)...
    # input_dir = "../adam_core/impact_corridor/2023 CX1/2023-02-13T02.38.19.001/"
    # out_dir = "./sampled_data/impact_corridor/2023 CX1/2023-02-13T02.38.19.001/"
    # os.makedirs(out_dir, exist_ok=True)
    # time_arr, time_lag_all, sampled_pts_all, c_3d_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=False)
    # dumpJSON(sampled_pts_all, c_3d_all, time_arr, sampled_u_all, sampled_v_all, out_dir, sampled_pt_vals_all)

    # For Apophis...
    # input_dir = "../adam_core/impact_corridor/2004 MN4/2004-12-27T21.28.37.000/"
    # out_dir = "./sampled_data/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/"
    # os.makedirs(out_dir, exist_ok=True)
    # time_arr, time_lag_all, sampled_pts_all, c_3d_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=False)
    # dumpJSON(sampled_pts_all, c_3d_all, time_arr, sampled_u_all, sampled_v_all, out_dir, sampled_pt_vals_all, starting_time_index=8800)

    # For Apophis subtube...
    # input_dir = "../adam_core/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/adaptive/"
    # out_dir = "./sampled_data/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/adaptive/"
    # os.makedirs(out_dir, exist_ok=True)
    # time_arr, time_lag_all, sampled_pts_all, c_3d_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=False)
    # dumpJSON(sampled_pts_all, c_3d_all, time_arr, sampled_u_all, sampled_v_all, out_dir, sampled_pt_vals_all)

    # For dynamic uncertainty (nested tube)...
    # input_path = "../adam_core/dynamic_uncertainty/2012 DA14/"
    # dir_list = os.listdir(input_path)
    # dir_list.sort()

    # out_parent_dir = "./sampled_data/dynamic_uncertainty/2012 DA14/"
    # os.makedirs(out_parent_dir, exist_ok=True)
    
    # for submission_dir in dir_list:
    #     submission_path = os.path.join(input_path, submission_dir) + "/"
    #     print(submission_path)
    #     out_dir = os.path.join(out_parent_dir, submission_dir)
    #     os.makedirs(out_dir, exist_ok=True)
    #     time_arr, time_lag_all, sampled_pts_all, c_3d_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(submission_path, 50, out_dir, plotEllipse=False)
    #     dumpJSON(sampled_pts_all, c_3d_all, time_arr, sampled_u_all, sampled_v_all, out_dir, sampled_pt_vals_all)

    # For sectioned uncertainty...
    # input_path = "../adam_core/sectioned_uncertainty/2012 DA14/"
    # input_path = "../adam_core/sectioned_uncertainty/2023 CX1_tube/"
    # input_path = "../adam_core/uncertainty_changes/1998 SG172_2007/"
    # dir_list = os.listdir(input_path)
    # dir_list.sort()
    # time_arr_list = []
    # time_lag_all_list = []
    # sampled_pts_all_list = []
    # c_3d_all_list = []
    # sampled_pt_vals_all_list = []
    # sampled_u_all_list = []
    # sampled_v_all_list = []
    # img_mat_all_list = []

    # out_dir = "./sampled_data/sectioned_uncertainty/2012 DA14/"
    # out_dir = "./sampled_data/sectioned_uncertainty/2023 CX1_tube/"
    # os.makedirs(out_dir, exist_ok=True)
    
    # for submission_dir in dir_list:
    #     submission_path = os.path.join(input_path, submission_dir) + "/"
    #     print(submission_path)
    #     time_arr, time_lag_all, sampled_pts_all, c_3d_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all, img_mat_all = getEllipsePerSubmission(submission_path, 50, out_dir, sectioned_uncertainty=True, plotEllipse=False)
    #     time_arr_list += list(time_arr)
    #     time_lag_all_list += time_lag_all
    #     sampled_pts_all_list += sampled_pts_all
    #     c_3d_all_list += c_3d_all
    #     sampled_pt_vals_all_list += sampled_pt_vals_all
    #     sampled_u_all_list += sampled_u_all
    #     sampled_v_all_list += sampled_v_all
    #     img_mat_all_list += img_mat_all

    # print(len(time_arr_list))
    # print(len(time_lag_all_list))
    # print(len(sampled_pts_all_list))
    # print(len(c_3d_all_list))
    # print(len(sampled_pt_vals_all_list))
    # print(len(sampled_u_all_list))
    # print(len(img_mat_all_list))

    # # time_lag_range = (np.min(time_lag_all_list), np.max(time_lag_all_list))
    # # print("time lag range", time_lag_range)

    # # out_dir = "./sampled_data/1998 SG172_2007"
    # dumpJSON(sampled_pts_all_list, c_3d_all_list, time_arr_list, sampled_u_all_list, sampled_v_all_list, out_dir, sampled_pt_vals_all_list, img_mat_all_list)