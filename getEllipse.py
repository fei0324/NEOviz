import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    Get the translation vector so the new plane passes through (0, 0, 0)
    n: normal of the old plane
    c: point on the old plane (center of the ellipsoid)
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
    if np.dot(m, nss) < 0:
        m = -m

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


def getStartingPoint(vec_csun, nss, trans_z, rotate_mat, H_2d):
    """
    Get the starting point to sample from the 2d ellipse centered at the origin
    vec_csun: vector from c_3d to the sun
    c_3d: center of the ellipsoid in 3d
    nss: normal of the solar system
    rotate_mat: rotation matrix from the original 3d plane to the x-y plane
    H_2d: matrix that describes the ellipse
    """
    m = np.cross(vec_csun, nss)

    # check if m is on the right side because we want m to have consistent orientation as the asteroid traverses around its orbit
    # here we pick np.dot(m, nss) to always be > 0
    if np.dot(m, nss) < 0:
        m = -m

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


def ellipse_arc(a, b, theta, n):
    """Cumulative arc length of ellipse with given dimensions"""

    # Divide the interval [theta , theta + 2*pi] into n steps at regular angles
    t = np.linspace(theta, theta + 2*np.pi, n)

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


def theta_from_arc_length_constructor(a, b, theta, precision):
    """
    Inverse arc length function: constructs a function that returns the
    angle associated with a given cumulative arc length for given ellipse."""

    # Get arc length data for this ellipse
    t, cumulative_distance, total_distance = ellipse_arc(a, b, theta, precision)

    # Construct the function
    def f(s):
        assert np.all(s <= total_distance), "s out of range"
        # Can invert through interpolation since monotonic increasing
        return np.interp(s, cumulative_distance, t)

    # return f and its domain
    return f, total_distance


def sampleEllipse2D(a, b, theta=0, sample_size=50, precision=1000):
    """
    Sample points from the 2d ellipse centered atthe origin.
    a, b: parameters of the 2d ellipse centered at (0, 0)
    theta: the angle to start sampling. We sample in the interval [theta, theta+2*pi]
    n: the number of points to sample
    precision: controls the precision of the arc length calculation.
    """
    theta_from_arc_length, domain = theta_from_arc_length_constructor(a, b, theta, precision)
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


def dumpJSON(sampled_pts_all, c_3d_all, observer, time_arr, outpath):
    data_dict = {"version": {"major": 0, "minor": 1}, "observer": observer, "polygons": []}
    num_pts_per_t = sampled_pts_all[0].shape[1]
    for t, time_step in enumerate(time_arr):
        data_dict["polygons"].append({"time": time_step})
        c_3d_t_meters = c_3d_all[t]
        c_3d_t_meters *= astrounit.au.to(astrounit.m)
        data_dict["polygons"][t]["center"] = {"x": c_3d_t_meters[0],
                                       "y": c_3d_t_meters[1],
                                       "z": c_3d_t_meters[2]}
        points_arr_per_t = []
        for i in range(num_pts_per_t):
            pt = sampled_pts_all[t][:, i]
            pt *= astrounit.au.to(astrounit.m)
            points_arr_per_t.append({"x": pt[0], "y": pt[1], "z": pt[2]})
        data_dict["polygons"][t]["points"] = points_arr_per_t

    with open(outpath, 'w') as fp:
        json.dump(data_dict, fp)


if __name__ == "__main__":

    # Get positions of astroid w.r.t. the sun for a given time
    # use adam_core to get the orbit positions at a specific time
    # variants_coord_f = "../adam_core/2012 DA14_t720_10/variants_coords_10.npy"
    # variants_velo_f = "../adam_core/2012 DA14_t720_10/variants_velo_10.npy"
    # time_f = "../adam_core/2012 DA14_t720_10/times_isot.npy"
    variants_coord_f = "../adam_core/2022 SF289_t66_150/variants_coords_150.npy"
    variants_velo_f = "../adam_core/2022 SF289_t66_150/variants_velo_150.npy"
    time_f = "../adam_core/2022 SF289_t66_150/times_isot.npy"
    variants_coords = np.load(variants_coord_f)
    variants_velo = np.load(variants_velo_f)
    print(variants_coords.shape)  # (600, 3)
    print(variants_coords)
    num_rows = variants_coords.shape[0]
    print(variants_velo.shape)  # (600, 3)
    time_arr = np.load(time_f)
    num_samples = 150
    # num_time_steps = num_rows//num_samples
    num_time_steps = len(time_arr)
    print("num_time_steps")
    print(num_time_steps)
    variants_coords_list = []
    variants_velo_list = []
    for i in range(num_time_steps):
        variants_coords_list.append(variants_coords[i*num_samples:(i+1)*num_samples])
        variants_velo_list.append(variants_velo[i*num_samples:(i+1)*num_samples])

    sampled_pts_all = []
    c_3d_all = []
    for i in range(num_time_steps):
        print("time step", i)

        # Compute ellipsoid and center of the ellipsoid using mvee
        Xi = variants_coords_list[i].T
        L_3d, H_3d, c_3d = computeEllipsoid(Xi)

        # Fig 1 (3d): plot points and ellipsoid center
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x_3d = Xi[0, :]
        y_3d = Xi[1, :]
        z_3d = Xi[2, :]
        ax.scatter(x_3d, y_3d, z_3d)
        ax.scatter(c_3d[0], c_3d[1], c_3d[2], s=50)

        # Get unit vector from c_3d to the sun -> vec_csun
        vec_csun = np.array([0, 0, 0]) - c_3d
        vec_csun = vec_csun/np.linalg.norm(vec_csun)

        # Fig 1 (3d): plot unit vector vec_csun
        ax.quiver(c_3d[0], c_3d[1], c_3d[2], vec_csun[0], vec_csun[1], vec_csun[2], color='red')

        # Get the normal of the solar system
        utctime = ["Jan 1, 2015"]
        nss = getNormalSolarSystem(utctime) 

        # Fig 1 (3d): plot the unit vector of nss
        ax.quiver(c_3d[0], c_3d[1], c_3d[2], nss[0], nss[1], nss[2], color='darkorchid')

        # use the average velocity vector (n_3d) as the normal of the plane
        Vi = variants_velo_list[i].T
        n_3d = np.mean(Vi, axis=1)

        # Fig 1 (3d): plot n_3d and the plane
        xr = np.linspace(c_3d[0] - 1e-08, c_3d[0] + 1e-08, num=20)
        yr = np.linspace(c_3d[1] - 1e-08, c_3d[1] + 1e-08, num=20)
        xx, yy, pz = computePlane(n_3d, c_3d, xr, yr)
        ax.plot_surface(xx, yy, pz, alpha=0.5)
        ax.set_aspect('equal')

        # compute orbit plane intersection points for all orbits
        orbit_plane_intersection = np.zeros(Xi.shape)
        for j in range(Xi.shape[1]):
            lp = Xi[:, j]
            lv = Vi[:, j]
            _, intersection_coor = computePlaneLineIntersection(n_3d, c_3d, lp, lv)
            orbit_plane_intersection[:, j] = intersection_coor
        
        # Fig 1 (3d): plot orbit plane intersection points on the plan
        intersectX = orbit_plane_intersection[0, :]
        intersectY = orbit_plane_intersection[1, :]
        intersectZ = orbit_plane_intersection[2, :]
        ax.scatter(intersectX, intersectY, intersectZ)

        # (for plotting) get the m vector and the projected m vector
        m, proj_m = _getMonPlane(n_3d, vec_csun, nss)

        # Fig 1 (3d): plot m and proj_m from the c_3d
        ax.quiver(c_3d[0], c_3d[1], c_3d[2], m[0], m[1], m[2], color='gold')
        ax.quiver(c_3d[0], c_3d[1], c_3d[2], proj_m[0], proj_m[1], proj_m[2], color='gold')
        # plt.show()

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

        # Fig 2 (3d): plot the new plane and the transformed points
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        transformedX = transformed_points[0, :]
        transformedY = transformed_points[1, :]
        transformedZ = transformed_points[2, :]
        ax.scatter(transformedX, transformedY, transformedZ)
        ax.set_aspect('equal')  # super important: ensures consistent scale for the axes!!!

        # Force the z coordinates of the transformed points to be 0
        transformed_points[2, :] = 0

        # Then this becomes a 2d problem
        # Compute the minimum enclosing ellipse of the intersection points on the xy-plane
        transformed_2d = transformed_points[:2, :]
        print(transformed_2d.shape)
        L_2d, H_2d, c_2d = computeEllipsoid(transformed_2d)
        
        # Fig 2 (3d): plot the center of the ellipse
        ax.scatter(c_2d[0], c_2d[1], 0, s=50, c="red")

        # Get point mp_2d and vector c_mp_3d
        mp_2d, c_mp_2d = _getMP2d(proj_m, c_3d, trans_z, rotate_mat, c_2d)

        # Fig 2 (3d): plot mp_2d and c_mp_2d
        ax.quiver(c_2d[0], c_2d[1], 0, c_mp_2d[0], c_mp_2d[1], c_mp_2d[2], color="gold")

        # plt.show()

        # Fig 3 (2d): plot a 2D version of the problem, points and min ellipse and c_2d
        fig = plt.figure()
        ax = fig.add_subplot()
        transformed_2d_x = transformed_2d[0, :]
        transformed_2d_y = transformed_2d[1, :]
        ax.scatter(c_2d[0], c_2d[1], s=50, c='red')
        ax.scatter(transformed_2d_x, transformed_2d_y)
        plot_ellipse(H_2d, c_2d, ax=ax)

        # Fig 3 (2d): plot c_mp_2d
        ax.quiver(c_2d[0], c_2d[1], c_mp_2d[0], c_mp_2d[1], scale=2, color="gold")

        # Rotaet c_mp_2d based on the rotation angle of the ellipse
        a, b, theta = getEllipseParam(H_2d)
        rot_theta, rot_neg_theta = getRotationMat2D(theta)
        rotated_c_mp_2d = rot_neg_theta @ c_mp_2d[:2]
        # TODO: Not sure why sometimes the rotated_c_mp_2d is on the other size of the ellipse

        # Fig 3 (2d): plot rotated_c_mp_2d and the ellipse without rotation
        ax.quiver(c_2d[0], c_2d[1], rotated_c_mp_2d[0], rotated_c_mp_2d[1], scale=2, color="limegreen")
        kwrg = {'facecolor': 'none', 'edgecolor':'darkgray', 'alpha':1, 'linewidth':2}
        ellip = Ellipse(xy=c_2d, width=2*a, height=2*b, angle=0, **kwrg)
        ax.set_aspect('equal')
        ax.add_artist(ellip)
        # plt.show()

        # Fig 4 (2d): plot the rotated array and the ellipse at the origin
        fig = plt.figure()
        ax = fig.add_subplot()
        kwrg = {'facecolor': 'none', 'edgecolor':'darkgray', 'alpha':1, 'linewidth':2}
        ellip_origin = Ellipse(xy=np.array([0, 0]), width=2*a, height=2*b, angle=0, **kwrg)
        ax.set_aspect('equal')
        ax.add_artist(ellip_origin)
        ax.relim()
        ax.autoscale_view()

        ax.quiver(0, 0, rotated_c_mp_2d[0], rotated_c_mp_2d[1], scale=0.3, color="limegreen")
        
        # Compute ellipse ray intersection centered at origin
        rotated_c_mp_2d, elli_r_o = getStartingPoint(vec_csun, nss, trans_z, rotate_mat, H_2d)
        
        # Fig 4 (2d): plot the intersection point
        ax.scatter(elli_r_o[0], elli_r_o[1])

        # Compute the angle in radiant of the intersection point
        rad = np.arctan2(elli_r_o[1], elli_r_o[0])
        # Transform it to the parameter for the ellipse
        phi = angle2phi(rad, a, b)

        # Fig 4 (2d): plot the new point computed from phi. it should overlap with elli_r_o
        new_p = a*np.cos(phi), b*np.sin(phi)
        ax.scatter(new_p[0], new_p[1], color="red")
        # plt.show()
        assert np.isclose(new_p[0], elli_r_o[0])
        assert np.isclose(new_p[1], elli_r_o[1])

        # Sample points from the 2d ellipse
        x_elli_2d, y_elli_2d = sampleEllipse2D(a, b, phi, 20, 1000)

        # Fig 5 (2d): plot the sampled points from the ellipse
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_aspect('equal')
        ax.scatter(x_elli_2d, y_elli_2d, alpha=0.5, color="forestgreen")
        # plot the first and the last points
        ax.scatter(x_elli_2d[0], y_elli_2d[0], alpha=0.5, color="orange")
        ax.scatter(x_elli_2d[1], y_elli_2d[1], alpha=0.5, color="blue")
        ax.scatter(x_elli_2d[-1], y_elli_2d[-1], alpha=0.5, color="crimson")
        # plt.show()

        # Fig 6 (2d): Plot the original ellipse centered at c_2d and the sampled points centered at c_2d
        fig = plt.figure()
        ax = fig.add_subplot()
        transformed_2d_x = transformed_2d[0, :]
        transformed_2d_y = transformed_2d[1, :]
        ax.scatter(c_2d[0], c_2d[1], s=50, c='red')
        ax.scatter(transformed_2d_x, transformed_2d_y)
        plot_ellipse(H_2d, c_2d, ax=ax)

        # Transform the sampled points back to the original 3d space
        sampled_trans_xy, sampled_pts_3d = transformPts3D(x_elli_2d, y_elli_2d, c_2d, rot_theta, rotate_mat, trans_z)
        print("sampled 2d points shape")
        print(sampled_trans_xy.shape)
        plt.scatter(sampled_trans_xy[0], sampled_trans_xy[1], color="orange")
        # plt.show()

        # Fig 7 (3d): 
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_aspect('equal')
        ax.scatter(sampled_pts_3d[0], sampled_pts_3d[1], sampled_pts_3d[2], color="orange")
        ax.scatter(Xi[0, :], Xi[1, :], Xi[2, :], color="blue")
        ax.scatter(c_3d[0], c_3d[1], c_3d[2], s=50, color="red")
        # plt.show()
        plt.close("all")

        sampled_pts_all.append(sampled_pts_3d)
        c_3d_all.append(c_3d)

    outdir = "./sampled_data"
    os.makedirs(outdir, exist_ok=True)
    # json_file = "2012_DA14_t" + str(num_time_steps) + "_" + str(Xi.shape[1]) + ".json"
    json_file = "2022_SF289_t" + str(num_time_steps) + "_" + str(Xi.shape[1]) + ".json"
    outpath = os.path.join(outdir, json_file)
    print(outpath)
    dumpJSON(sampled_pts_all, c_3d_all, "SSB", time_arr, outpath)