import os
import json
import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from matplotlib.patches import Ellipse

from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances
from astropy import units as astrounit

from src.plotting import plot_ellipse
from src.ext.mvee import mvee2

import spiceypy as spice


def computeEllipsoid(points: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wrapper for the mvee2() function which compute the minimum enclosing ellipsoid from
    the input point cloud.
    Input: 
        points: The input point cloud in 2D or 3D
    
    Output:
        L:
        hermitian: The matrix that represents the ellipsoid.
           A complex Hermitian (conjugate symmetric) or a real symmetric matrix
        center: The center of the ellipsoid
    """

    obj = mvee2(points)
    L = obj["L"]
    center = obj["c"]
    hermitian = L @ L.T

    return L, hermitian, center


def computePlane(n, c, xr, yr):
    """
    Compute a plane from the normal vector and a point on the plane
    Input:
        n: The normal vector of the plane
        c: A point on the plane
        xr: Range vector of x
        yr: Range vectr of y
    """

    xx, yy = np.meshgrid(xr, yr)
    n0 = n[0]
    n1 = n[1]
    n2 = n[2]
    z = (np.dot(c, n) - np.multiply(n0, xx) - np.multiply(n1, yy))/n2

    return xx, yy, z


def computePlaneLineIntersection(n: npt.ArrayLike, c: npt.ArrayLike, lp: npt.ArrayLike,
                                 lv: npt.ArrayLike):
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
    ssb_normal: normal of the solar system
    """

    ettime = spice.str2et(utctime[0])

    # get the transformation matrix between two reference frames
    rf_trans_mat = spice.sxform(instring='ECLIPJ2000', \
                    tostring='GALACTIC', \
                    et=ettime)
    
    # get the z direction of the rf_trans_mat
    ssb_normal = rf_trans_mat[:3, :3] @ np.array([0, 0, 1])

    # compute the unit vector
    ssb_normal = ssb_normal/np.linalg.norm(ssb_normal)

    return ssb_normal


def projectVectorToPlane(vector, normal):
    """
    Projoect a vector onto a plane.
    Input:
        vector: The vector to project onto the plane
        normal: The normal of the plane to project the vector onto

    Output: The input vector projected onto the input plane
    """
                    
    return vector - ((np.dot(vector, normal) / np.dot(normal, normal)) * normal)


def _getMonPlane(normal, center_to_ssb, ssb_normal):
    """
    Redundant function used for plotting (Fig 1). Part of getStartingPoint()
    Get m that is perpendicular to the center_to_ssb and ssb_normal. Then project it onto the plane.

    Output:
    m: the vector perpendicular to center_to_ssb and ssb_normal
    proj_m: m projected onto the plane
    """

    m = np.cross(center_to_ssb, ssb_normal)

    # check if m is on the right side because we want m to have consistent orientation as the asteroid traverses around its orbit
    # here we pick np.dot(m, ssb_normal) to always be > 0
    # if np.dot(m, ssb_normal) < 0:
    #     m = -m

    # vector m might not be on the plane, project m onto the plane
    proj_m = projectVectorToPlane(m, normal)

    return m, proj_m


def _getMP2d(proj_m, center, translation_z, rotation_matrix, center_2d):
    """
    Redundant function used for plotting (Fig 2 and Fig 3). Part of getStartingPoint()
    Get the transformed point of mp_3d -> mp_2d
    mp_3d is a point on the plane starting from c in the direction of proj_m

    Output:
    mp_2d: point on the x-y plane
    c_mp_2d: vector from center_2d to mp_2d on the x-y plane
    """

    # get point mp_3d on the plane using proj_m and center
    mp_3d = center + proj_m

    # transform mp_3d to the x-y plane
    mp_2d = rotation_matrix @ (mp_3d - np.array([0, 0, translation_z]))

    # get vector from center_2d to mp_2d
    center_2d_3 = np.array([center_2d[0], center_2d[1], 0])
    c_mp_2d = mp_2d - center_2d_3

    return mp_2d, c_mp_2d


def getRotationMat2D(theta):
    """
    Get a rotational matrix in 2D
    
    Output:
    rotation_theta: rotation matrix by the angle theta
    neg_rotation_theta: rotation matrix by the negative angle theta
    """

    rotation_theta = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    neg_rotation_theta = np.array([
        [np.cos(-theta), -np.sin(-theta)],
        [np.sin(-theta), np.cos(-theta)]
    ])

    return rotation_theta, neg_rotation_theta


def getEllipseParam(hermitian_2d):
    """
    Get the parameters for the general ellipse in 2d
    """

    eigenvalues, eigenvectors = np.linalg.eigh(hermitian_2d)
    # TODO: Do we really want to sort here?
    idxs = np.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[idxs], eigenvectors[:, idxs]
    
    # Get the ellipse axes lengths (radius)
    # TODO: Is the eigenvalue really the size of the ellipse side? Radius or diameter?
    a = np.sqrt(eigenvalues[0]*2)
    b = np.sqrt(eigenvalues[1]*2)

    # rotation angle for the ellipse
    # TODO: This is not the angle of the ellipse? We want the angle between (1, 0) or
    # (0, 1) and the largest of the axes? Or consistently the same axes? DO we get the
    # same axes everytime from the mvee library?
    theta = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

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

def transformToXYPlane(point, plane_center, plane_normal):
    """
    Transform the input point (or vector) on the plane defined by the given center and normal, to the XY plane with normal

    Input:
        point: A point (or a vector) on the input plane (or originating on the input
               plane) to transform onto the XY-plane
        plane_center: The center point of the ellipse. Or the origin of the input vector
                      if it is a vector. This point must be part of the input plane.
        plane_normal: The normal of the input plane

    Output: The transformed point (or vector) on the XY plane
    """

    # Start by translating the point to the origin 
    translated_point = point - plane_center
    translated_plane_normal = plane_normal - plane_center

    # Then find the rotation matrix to rotate the input plane to the XY plane
    # with a normal of (0, 0, 1)

    # Rotation matrix formula:
    # matrix = [
    #   [cos(theta) + u1^2*(1 - cos(theta)), u1*u2*(1 - cos(theta)), u2*sin(theta)], 
    #   [u1*u2*(1 - cos(theta)), cos(theta) + u2^2*(1 - cos(theta)), -u1*sin(theta)],
    #   [-u2*sin(theta), u1*sin(theta), cos(theta)]    
    # ]

    # Where:
    # translated_plane_normal = (a, b, c)
    a = translated_plane_normal[0]
    b = translated_plane_normal[1]
    c = translated_plane_normal[2]

    # theta is the angle betweeen the translated_plane_normal and the normal of the XY plane
    # cos(theta) = c/sqrt(a^2+b^2+c^2) -> c/|translated_plane_normal|
    cos_theta = c/np.linalg.norm(translated_plane_normal)

    # sin(theta) = sqrt((a^2+b^2)/(a^2+b^2+c^2)) -> sqrt(1 - cos(theta)^2)
    sin_theta = np.sqrt(1 - cos_theta**2)

    # u1 = b/sqrt(a^2+b^2)
    u1 = b/np.sqrt(a**2 + b**2)

    # u2 = −a/sqrt(a^2+b^2)
    u2 = -a/np.sqrt(a**2 + b**2)

    # Construct the rotation matrix
    rotation_matrix = np.array([
        [cos_theta + u1^2*(1 - cos_theta), u1*u2*(1 - cos_theta), u2*sin_theta], 
        [u1*u2*(1 - cos_theta), cos_theta + u2^2*(1 - cos_theta), -u1*sin_theta],
        [-u2*sin_theta, u1*sin_theta, cos_theta] 
    ])

    return rotation_matrix @ translated_point


def getStartingPoint(center_to_ssb, ssb_normal, center_2d, hermitian_2d, center, normal):
    """
    Get the starting point to sample from the 2D ellipse centered at the origin
    Input:
        center_to_ssb: vector from center to the sun
        center: center of the ellipsoid in 3D
        ssb_normal: normal of the solar system
        rotation_matrix: rotation matrix from the original 3D plane to the x-y plane
        hermitian_2d: matrix that describes the ellipse
    Output:
        rotated_c_mp_2d:
        elli_r_o:
    """

    # TODO: This vector creates a right handed coordinate system
    # It is almost directed in the direction of movement af the asteroid, normal of the ellipse plane
    center_cross_normal = np.cross(center_to_ssb, ssb_normal)

    # The center_cross_normal vector might not be on the plane, project center_cross_normal onto the plane
    # TODO: This makes no sense, it is almost the normal of the plane
    # TODO: Don't we want to project the ssb_normal onto the plane instead? 
    projected_center_cross_normal = projectVectorToPlane(center_cross_normal, normal)

    # get point mp_3d on the plane using projected_center_cross_normal and center
    mp_3d = center + projected_center_cross_normal

    # transform mp_3d to the x-y plane
    mp_2d = transformToXYPlane(mp_3d, center, normal)

    # get vector from center_2d to mp_2d
    c_mp_2d = mp_2d[:2] - center_2d

    # get 2d ellipse parameters
    a, b, theta = getEllipseParam(hermitian_2d)
    # print("theta", theta)
    # theta_degrees = theta * 180 / np.pi
    # print("theta degrees", theta_degrees)

    # rotate mp_2d by -1*theta
    _, neg_rotation_theta = getRotationMat2D(theta)
    rotated_c_mp_2d = neg_rotation_theta @ c_mp_2d[:2]

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
    # TODO: No we do not need to whole 2pi, the ellipse is symetrical and it doesnt matter
    phi = angle - np.arctan2((b-a)*np.tan(angle), b + a*np.tan(angle)**2)
    
    return phi


def ellipse_arc(a, b, theta_sample, n):
    """
    Cumulative arc length of ellipse with given dimensions
    """

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
    angle associated with a given cumulative arc length for given ellipse.
    """

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


def transformPts3D(sampled_x: np.array, sampled_y: np.array, center_2d, rotation_theta, rotation_matrix, translation_z):
    """
    Transform the sampled points from the 2d ellipse centered at (0, 0) back to the original 3d 
    
    sampled_x, sampled_y: sampled 2d points on the ellipse centered at (0, 0) with no rotation
    center_2d: center of the ellipse in 2d
    rotation_theta: 2d rotational matrix (to rotate back to the original rotation of the ellipse)
    rot_mat: 3d tranformation matrix (from the xy plane to the 3d space)
    translation_z: translation amount in the z direction
    """
    
    # Note: the order of rotation and translation is important. Here we have to rotate first and then translate
    # rotate the points by theta (the angle of the 2d ellipse)
    sampled_xy = np.stack((sampled_x, sampled_y))
    sampled_xy = rotation_theta @ sampled_xy

    # shift the 2d points on the ellipse by center_2d
    x_shift = np.full(sampled_x.shape, center_2d[0])
    x_2d = sampled_xy[0] + x_shift
    y_shift = np.full(sampled_y.shape, center_2d[1])
    y_2d = sampled_xy[1] + y_shift
    sampled_trans_xy = np.stack((x_2d, y_2d))

    # transform all the points back into the original 3d space
    z_zeros = np.zeros((1, sampled_trans_xy.shape[1]))
    sampled_pts_3d = np.concatenate((sampled_trans_xy, z_zeros), axis=0)
    rotation_matrix_inv = np.linalg.inv(rotation_matrix)
    sampled_pts_3d = rot_mat_inv @ sampled_pts_3d
        
    # get the inverse translation z
    num_samples = len(sampled_x)
    inverse_translation = np.tile(np.array([0, 0, -translation_z]).T, (num_samples, 1)).T
    sampled_pts_3d -= inverse_translation

    return sampled_trans_xy, sampled_pts_3d


def getSampledPtVals(a, b, center_2d, sampled_x, sampled_y, transformed_2d_x, transformed_2d_y, neg_rotation_theta, r=None):

    # rotate the original 2d points centered at (0, 0) by neg_theta
    x_shift_ori = np.full(transformed_2d_x.shape, center_2d[0])
    y_shift_ori = np.full(transformed_2d_y.shape, center_2d[1])
    trans_rot_2d_x = transformed_2d_x - x_shift_ori
    trans_rot_2d_y = transformed_2d_y - y_shift_ori
    trans_rot_2d_xy = np.stack((trans_rot_2d_x, trans_rot_2d_y))
    trans_rot_2d_xy = neg_rotation_theta @ trans_rot_2d_xy
    
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


def getTextureCoordinates(center_2d, sampled_x, sampled_y, transformed_2d_x, transformed_2d_y, neg_rotation_theta, img_name=None, img_dir=None, p_radius=5, res=500, save_textures=False):
    """
    Generate texture coordinates and the image for the cut plane of the tube.
    center_2d: center of the 2d ellipse.
    sampled_x, sampled_y: sampled points on the 2d ellipse sentered at (0, 0)
    transformed_2d_x, transformed_2d_y: original points after the 3d to 2d transformation (not centered at (0, 0))
    neg_rotation_theta: the tranformation matrix for rotating by -theta
    time_lag: the time it takes for each orbit to reach the 3d plane (used as the second channel of the texture)
    p_radius: odd integer, the radius of the points in the image of the cut plane, e.g. 3 -> 3 x 3 patch
    """

    # rotate the original 2d points centered at (0, 0) by neg_theta. All points are now centered at (0, 0) without rotation
    x_shift_ori = np.full(transformed_2d_x.shape, center_2d[0])
    y_shift_ori = np.full(transformed_2d_y.shape, center_2d[1])
    trans_rot_2d_x = transformed_2d_x - x_shift_ori
    trans_rot_2d_y = transformed_2d_y - y_shift_ori
    trans_rot_2d_xy = np.stack((trans_rot_2d_x, trans_rot_2d_y))
    trans_rot_2d_xy = neg_rotation_theta @ trans_rot_2d_xy

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


def dumpJSON(sampled_pts_all_list, center_all_list, time_data_list, sampled_u_all_list, sampled_v_all_list, axes_length_all, axes_direction_all, out_dir, sampled_pt_vals_all=None, img_mat_all_list=None, starting_time_index=None):
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

    for t, time_step in enumerate(time_data_list):
        data_dict["polygons"].append({"time": time_step})
        center_t_meters = center_all_list[t]
        center_t_meters *= astrounit.au.to(astrounit.m)
        data_dict["polygons"][t]["center"] = {"x": center_t_meters[0],
                                       "y": center_t_meters[1],
                                       "z": center_t_meters[2]}
        
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


def getEllipsePerSubmission(data_directory, num_sample_ellipse, out_dir, sectioned_uncertainty=True, plotEllipse=False):
    """
    Get the ellipses for orbits from the same submission period
    
    data_directory: Should end with "/"
    num_sample_ellipse: the number of sample points we get from the circumference of the
    ellipse
    """

    # Extract the variants coordinates and velocities files from the given data 
    # directory. The positions and velocities are given with respect to the Sun.
    # There should also only be one coordinate and velocity file, if there is more than
    # one then only the first that is found is considered
    variants_coordinates_file = ""
    for filename in os.listdir(data_directory):
        if filename.startswith("variants_coordinates_"):
            variants_coordinates_file = filename
            break
    print(variants_coordinates_file)
    
    variants_velocities_file = ""
    for filename in os.listdir(data_directory):
        if filename.startswith("variants_velocity_"):
            variants_velocities_file = filename
            break
    print(variants_velocities_file)
    
    # Get the time file from the directory. All variants use the same timesteps
    time_f = os.path.join(data_directory, "times_isot.npy")
    time_data = np.load(time_f)

    # Load the coordinate and velocity data
    # TODO: How does the data look like? Matrix? Long flat flist? What order of items?
    variants_coordinates = np.load(
        os.path.join(data_directory, variants_coordinates_file[0])
    )
    variants_velocities = np.load(
        os.path.join(data_directory, variants_velocities_file[0])
    )
    print("Coordinate numpy shape", variants_coordinates.shape)
    print("Velocity numpy shape", variants_velocities.shape)
    
    # Get meta data
    num_samples = int(variants_coordinates_file[0].split("_")[-1].split(".")[0])
    num_time_steps = len(time_data)
    print("Number of samples", num_samples)
    print("Number of time steps", num_time_steps)
    
    # The coordinates and velocities are orderd per orbit in the data, but we want to find
    # all varaint cooridnates and velocities per timestep to create time-slices
    timed_variants_coordinates = []
    ordered_variants_velocities = []
    for t in range(num_time_steps):
        # We only take the coordinate or velocity cooresponding to the t:th timestamp
        # for each orbit
        timed_variants_coordinates.append(variants_coordinates[t::num_time_steps])
        ordered_variants_velocities.append(variants_velocities[t::num_time_steps])

    print("Size of timed_variants_coordinates", len(timed_variants_coordinates))
    print("Size of timed_variants_coordinates[0]", len(timed_variants_coordinates[0]))
    print("Shape of timed_variants_coordinates[0]", timed_variants_coordinates[0].shape)

    sampled_pts_all = []
    center_all = []
    time_lag_all = []
    sampled_pt_vals_all = []
    sampled_u_all = []
    sampled_v_all = []
    axes_length_all = []
    axes_direction_all = []
    img_mat_all = []

    # for Apophis analysis
    transformed_2d_all = []

    # Create a directory to store textures
    texture_dir = os.path.join(out_dir, "textures")
    os.makedirs(texture_dir, exist_ok = True)        

    # Get the normal of the solar system
    # TODO: Why this date?
    utctime = ["Jan 1, 2015"]
    ssb_normal = getNormalSolarSystem(utctime)

    for t in range(num_time_steps):
        print("Time step", t)
        print("Time data", time_data[t])

        # for Apophis analysis
        # if t < 8870:
        #     continue
        # if t >= 9000:
        #     break

        # Compute ellipsoid and center of the ellipsoid using mvee
        # TODO: Why do we transpose this array? Isnt the first elemet all coordinates for
        # the first timestep?
        coordinates_t = timed_variants_coordinates[t].T
        print("coordinates_t shape", coordinates_t.shape)
        L_3d, hermitian_3d, center = computeEllipsoid(coordinates_t)
        
        # The eigenvectors of hermitian_3d are the orientation of the semi-axes
        eigenvalues, eigenvectors = np.linalg.eig(hermitian_3d)

        # We can compute the length of the semi-axes a, b, and c from the eigenvalues
        # TODO: Is this math correct? 
        # TODO: Ask KJ about this 
        axes_lengths = np.sqrt(np.reciprocal(eigenvalues))
        axes_lengths_all.append(axes_lengths)
        print("Axes lengths", axes_lengths)

        # Store the ellipse rotation. The columns in this matrix is the eigenvectors
        axes_direction_all.append(eigenvectors)
        print("Eigenvectors", eigenvectors)

        # Get unit vector from the center of ellipse to SSB -> center_to_ssb
        center_to_ssb = np.array([0, 0, 0]) - center
        center_to_ssb = center_to_ssb/np.linalg.norm(center_to_ssb)

        # We use the average velocity vector as the normal of the ellipse plane
        # TODO: Why transpose here?
        velocities_t = ordered_variants_velocities[i].T
        mean_velocity = np.mean(velocities_t, axis = 1)

        # Compute the intersection between the ellipse plane and the orbit for all orbits
        # This gives a number to how far behind or ahead each variant orbit is compared
        # to the average movement
        plane_orbit_intersection = np.zeros(coordinates_t.shape)
        time_lag = np.zeros(coordinates_t.shape[1])
        for orbit in range(coordinates_t.shape[1]):
            coordinates_t_o = coordinates_t[:, orbit]
            velocities_t_o = velocities_t[:, orbit]
            time, intersection = computePlaneLineIntersection(
                mean_velocity,
                center,
                coordinates_t_o,
                velocities_t_o
            )
            time_lag[orbit] = time
            plane_orbit_intersection[:, orbit] = intersection

        # TODO: Clean up plotting code
        if plotEllipse is True:
            # Fig 1 (3d): plot points and ellipsoid center
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            x_3d = coordinates_t[0, :]
            y_3d = coordinates_t[1, :]
            z_3d = coordinates_t[2, :]
            ax.scatter(x_3d, y_3d, z_3d)
            ax.scatter(center[0], center[1], center[2], s=50)
            print("x_3d shape", x_3d.shape)

            # Fig 1 (3d): plot unit vector center_to_ssb
            # ax.quiver(center[0], center[1], center[2], center_to_ssb[0], center_to_ssb[1], center_to_ssb[2], color='red')

            # Fig 1 (3d): plot the unit vector of ssb_normal
            # ax.quiver(center[0], center[1], center[2], ssb_normal[0], ssb_normal[1], ssb_normal[2], color='darkorchid')

            # Fig 1 (3d): plot mean_velocity and the plane
            xr = np.linspace(center[0] - 3e-08, center[0] + 3e-08, num=20)
            yr = np.linspace(center[1] - 3e-08, center[1] + 3e-08, num=20)
            xx, yy, pz = computePlane(mean_velocity, center, xr, yr)
            # ax.plot_surface(xx, yy, pz, color="green", alpha=0.5)
            ax.quiver(center[0], center[1], center[2], mean_velocity[0], mean_velocity[1], mean_velocity[2], color='green')
            ax.set_aspect('equal')
        
            # Fig 1 (3d): plot orbit plane intersection points on the plan
            intersectX = plane_orbit_intersection[0, :]
            intersectY = plane_orbit_intersection[1, :]
            intersectZ = plane_orbit_intersection[2, :]
            ax.scatter(intersectX, intersectY, intersectZ)

            # (for plotting) get the m vector and the projected m vector
            m, proj_m = _getMonPlane(mean_velocity, center_to_ssb, ssb_normal)

            # Fig 1 (3d): plot m and proj_m from the center
            # ax.quiver(center[0], center[1], center[2], m[0], m[1], m[2], color='gold')
            ax.quiver(center[0], center[1], center[2], proj_m[0], proj_m[1], proj_m[2], color='tab:orange')
            plt.show()

        # Note that the intersection points are on a plane in 3D space
        # We need to transform this plane onto the xy-plane with a translation and a
        # rotation to make this into a 2D problem
        # The normal of the x-y plane is (0, 0, 1)
        # TODO: This is wrong?
        normal = np.array([0, 0, 1])
        translation_z = np.dot(center, mean_velocity)/mean_velocity[2]
        rotation_matrix = getRotationalMatrix(mean_velocity, normal)
        
        expanded_translation = np.tile(
            np.array([0, 0, translation_z]).T,
            (num_samples, 1)
        ).T
        translated_points = plane_orbit_intersection - expanded_translation
        transformed_points = rotation_matrix @ translated_points

        # All points should now be on the x-y plane

        # Force the z coordinates of the transformed points to be 0
        # TODO: This should already be the case if the rotation and translation was
        # correct. If this is run, then we are skewing the plane and changeing its shape.
        transformed_points[2, :] = 0

        # Compute the minimum enclosing ellipse of the intersection points on the xy-plane
        transformed_points_2d = transformed_points[:2, :]
        print(transformed_points_2d.shape)
        transformed_points_2d_x = transformed_points_2d[0, :]
        transformed_points_2d_y = transformed_points_2d[1, :]
        L_2d, hermitian_2d, center_2d = computeEllipsoid(transformed_points_2d)

        # Compute ellipse ray intersection centered at origin
        a, b, theta = getEllipseParam(hermitian_2d)
        rotation_theta, neg_rotation_theta = getRotationMat2D(theta)
        rotated_c_mp_2d, elli_r_o = getStartingPoint(
            center_to_ssb,
            ssb_normal,
            translation_z,
            rotation_matrix,
            center_2d,
            hermitian_2d,
            center,
            mean_velocity
        )

        # Compute the angle in radiant of the intersection point
        rad = np.arctan2(elli_r_o[1], elli_r_o[0])
        # Transform it to the parameter for the ellipse
        phi = angle2phi(rad, a, b)

        # Compute new p from phi
        new_p = a*np.cos(phi), b*np.sin(phi)
        assert np.isclose(new_p[0], elli_r_o[0])
        assert np.isclose(new_p[1], elli_r_o[1])

        # TODO: Clean up plotting code
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
            ax.scatter(center_2d[0], center_2d[1], 0, s=50, c="red")

            # Get point mp_2d and vector c_mp_3d
            m, proj_m = _getMonPlane(mean_velocity, center_to_ssb, ssb_normal)
            mp_2d, c_mp_2d = _getMP2d(proj_m, center, translation_z, rotation_matrix, center_2d)

            # Rotaet c_mp_2d based on the rotation angle of the ellipse
            a, b, theta = getEllipseParam(hermitian_2d)
            rotation_theta, neg_rotation_theta = getRotationMat2D(theta)
            rotated_c_mp_2d = neg_rotation_theta @ c_mp_2d[:2]
            # TODO: Not sure why sometimes the rotated_c_mp_2d is on the other size of the ellipse

            # Fig 2 (3d): plot mp_2d and c_mp_2d
            ax.quiver(center_2d[0], center_2d[1], 0, c_mp_2d[0], c_mp_2d[1], c_mp_2d[2], color="tab:orange")
            plt.show()

            # Fig 3 (2d): plot a 2D version of the problem, points and min ellipse and center_2d
            fig = plt.figure()
            ax = fig.add_subplot()
            # transformed_2d_x = transformed_2d[0, :]
            # transformed_2d_y = transformed_2d[1, :]
            # ax.scatter(center_2d[0], center_2d[1], s=50, c='red')
            ax.scatter(transformed_2d_x, transformed_2d_y)
            plot_ellipse(hermitian_2d, center_2d, ax=ax)

            # Fig 3 (2d): plot c_mp_2d
            ax.quiver(center_2d[0], center_2d[1], c_mp_2d[0], c_mp_2d[1], scale=1, width=0.02, color="tab:orange")

            # Fig 3 (2d): plot rotated_c_mp_2d and the ellipse without rotation
            ax.quiver(center_2d[0], center_2d[1], rotated_c_mp_2d[0], rotated_c_mp_2d[1], scale=1, width=0.02, color="tab:pink")
            kwrg = {'facecolor': 'none', 'edgecolor':'tab:pink', 'alpha':1, 'linewidth':2}
            ellip = Ellipse(xy=center_2d, width=2*a, height=2*b, angle=0, **kwrg)
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
        sampled_pt_vals = getSampledPtVals(a, b, center_2d, x_elli_2d, y_elli_2d, transformed_2d_x, transformed_2d_y, neg_rotation_theta)
        sampled_pt_vals_all.append(sampled_pt_vals)

        # Get texture coordinates
        if sectioned_uncertainty is True:
            sampled_u, sampled_v, img_mat = getTextureCoordinates(center_2d, x_elli_2d, y_elli_2d, transformed_2d_x, transformed_2d_y, neg_rotation_theta, save_textures=False)
        else:
            img_name = str(i) + ".png"
            sampled_u, sampled_v = getTextureCoordinates(center_2d, x_elli_2d, y_elli_2d, transformed_2d_x, transformed_2d_y, neg_rotation_theta, img_name, texture_dir, save_textures=True)

        # Transform the sampled points back to the original 3d space
        sampled_trans_xy, sampled_pts_3d = transformPts3D(x_elli_2d, y_elli_2d, center_2d, rotation_theta, rotation_matrix, translation_z)
        print("sampled 2d points shape")
        print(sampled_trans_xy.shape)

        # TODO: Clean up plotting code
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

            # Fig 6 (2d): Plot the original ellipse centered at center_2d and the sampled points centered at center_2d
            fig = plt.figure()
            ax = fig.add_subplot()
            transformed_2d_x = transformed_2d[0, :]
            transformed_2d_y = transformed_2d[1, :]
            ax.scatter(center_2d[0], center_2d[1], s=50, c='red')
            ax.scatter(transformed_2d_x, transformed_2d_y)
            ax.set_aspect('equal')
            plot_ellipse(hermitian_2d, center_2d, ax=ax)
            

            # Transform the sampled points back to the original 3d space, plot the 3d points in the xy plane
            plt.scatter(sampled_trans_xy[0], sampled_trans_xy[1], color="orange")
            plt.show()

            # Fig 7 (3d): plot the original points in 3d and the sampled ellipse in 3d
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(sampled_pts_3d[0], sampled_pts_3d[1], sampled_pts_3d[2], color="tab:orange")
            ax.scatter(coordinates_t[0, :], coordinates_t[1, :], coordinates_t[2, :], color="tab:blue")
            ax.scatter(center[0], center[1], center[2], s=50, color="red")
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
        center_all.append(center)
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
    # time_data = time_data[8800:9000]
            
    if sectioned_uncertainty is True:
        return time_data, time_lag_all, sampled_pts_all, center_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all, axes_length_all, axes_direction_all, img_mat_all
    else:
        return time_data, time_lag_all, sampled_pts_all, center_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all, axes_length_all, axes_direction_all


def main(input_dir, out_dir):
    # For 2023 CX1
    time_data, time_lag_all, sampled_pts_all, center_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all, axes_length_all, axes_direction_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=False)
    print(time_data)
    print(time_lag_all)
    dumpJSON(sampled_pts_all, center_all, time_data, sampled_u_all, sampled_v_all, axes_length_all, axes_direction_all, out_dir, sampled_pt_vals_all)

    # For figures...
    # input_dir = "../adam_core/dynamic_uncertainty/2012 DA14/2013-02-10T04.54.49.000/"
    # out_dir = "./sampled_data/dynamic_uncertainty/2012 DA14/for_figures/"
    # os.makedirs(out_dir, exist_ok=True)
    # time_data, time_lag_all, sampled_pts_all, center_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=True)

    # Increase Apophis resolution...
    # input_dir = "../adam_core/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/adaptive/"
    # out_dir = "./sampled_data/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/adaptive/"
    # os.makedirs(out_dir, exist_ok=True)
    # time_data, time_lag_all, sampled_pts_all, center_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=False)
    # dumpJSON(sampled_pts_all, center_all, time_data, sampled_u_all, sampled_v_all, out_dir, sampled_pt_vals_all)

    # For 2023 CX1 (impact corridor)...
    # input_dir = "../adam_core/impact_corridor/2023 CX1/2023-02-13T02.38.19.001/"
    # out_dir = "./sampled_data/impact_corridor/2023 CX1/2023-02-13T02.38.19.001/"
    # os.makedirs(out_dir, exist_ok=True)
    # time_data, time_lag_all, sampled_pts_all, center_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=False)
    # dumpJSON(sampled_pts_all, center_all, time_data, sampled_u_all, sampled_v_all, out_dir, sampled_pt_vals_all)

    # For Apophis...
    # input_dir = "../adam_core/impact_corridor/2004 MN4/2004-12-27T21.28.37.000/"
    # out_dir = "./sampled_data/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/"
    # os.makedirs(out_dir, exist_ok=True)
    # time_data, time_lag_all, sampled_pts_all, center_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=False)
    # dumpJSON(sampled_pts_all, center_all, time_data, sampled_u_all, sampled_v_all, out_dir, sampled_pt_vals_all, starting_time_index=8800)

    # For Apophis subtube...
    # input_dir = "../adam_core/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/adaptive/"
    # out_dir = "./sampled_data/impact_corridor/2004 MN4/2004-12-27T21.28.37.000_8800-9000/adaptive/"
    # os.makedirs(out_dir, exist_ok=True)
    # time_data, time_lag_all, sampled_pts_all, center_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(input_dir, 50, out_dir, sectioned_uncertainty=False, plotEllipse=False)
    # dumpJSON(sampled_pts_all, center_all, time_data, sampled_u_all, sampled_v_all, out_dir, sampled_pt_vals_all)

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
    #     time_data, time_lag_all, sampled_pts_all, center_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all = getEllipsePerSubmission(submission_path, 50, out_dir, plotEllipse=False)
    #     dumpJSON(sampled_pts_all, center_all, time_data, sampled_u_all, sampled_v_all, out_dir, sampled_pt_vals_all)

    # For sectioned uncertainty...
    # input_path = "../adam_core/sectioned_uncertainty/2012 DA14/"
    # input_path = "../adam_core/sectioned_uncertainty/2023 CX1_tube/"
    # input_path = "../adam_core/uncertainty_changes/1998 SG172_2007/"
    # dir_list = os.listdir(input_path)
    # dir_list.sort()
    # time_data_list = []
    # time_lag_all_list = []
    # sampled_pts_all_list = []
    # center_all_list = []
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
    #     time_data, time_lag_all, sampled_pts_all, center_all, sampled_pt_vals_all, sampled_u_all, sampled_v_all, img_mat_all = getEllipsePerSubmission(submission_path, 50, out_dir, sectioned_uncertainty=True, plotEllipse=False)
    #     time_data_list += list(time_data)
    #     time_lag_all_list += time_lag_all
    #     sampled_pts_all_list += sampled_pts_all
    #     center_all_list += center_all
    #     sampled_pt_vals_all_list += sampled_pt_vals_all
    #     sampled_u_all_list += sampled_u_all
    #     sampled_v_all_list += sampled_v_all
    #     img_mat_all_list += img_mat_all

    # print(len(time_data_list))
    # print(len(time_lag_all_list))
    # print(len(sampled_pts_all_list))
    # print(len(center_all_list))
    # print(len(sampled_pt_vals_all_list))
    # print(len(sampled_u_all_list))
    # print(len(img_mat_all_list))

    # # time_lag_range = (np.min(time_lag_all_list), np.max(time_lag_all_list))
    # # print("time lag range", time_lag_range)

    # # out_dir = "./sampled_data/1998 SG172_2007"
    # dumpJSON(sampled_pts_all_list, center_all_list, time_data_list, sampled_u_all_list, sampled_v_all_list, out_dir, sampled_pt_vals_all_list, img_mat_all_list)