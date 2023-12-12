import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mvee import mvee2
from plotting import plot_ellipse
from matplotlib.patches import Ellipse

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

    return elli_r_o


def sampleEllipse2D():
    """
    Sample the 2D ellipse centered at the origin.
    """
    pass




if __name__ == "__main__":

    # Get positions of astroid w.r.t. the sun for a given time
    # use adam_core to get the orbit positions at a specific time
    variants_coord_f = "../adam_core/2012 DA14/variants_coords_150.npy"
    variants_velo_f = "../adam_core/2012 DA14/variants_velo_150.npy"
    variants_coords = np.load(variants_coord_f)
    variants_velo = np.load(variants_velo_f)
    print(variants_coords.shape)  # (600, 3)
    num_rows = variants_coords.shape[0]
    print(variants_velo.shape)  # (600, 3)
    num_samples = 150
    num_time_steps = num_rows//num_samples
    variants_coords_list = []
    variants_velo_list = []
    for i in range(num_time_steps):
        variants_coords_list.append(variants_coords[i*num_samples:(i+1)*num_samples])
        variants_velo_list.append(variants_velo[i*num_samples:(i+1)*num_samples])

    for i in range(num_time_steps):

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
        L_2d, H_2d, c_2d = computeEllipsoid(transformed_2d)
        
        # Fig 2 (3d): plot the center of the ellipse
        ax.scatter(c_2d[0], c_2d[1], 0, s=50, c="red")

        # Get point mp_2d and vector c_mp_3d
        mp_2d, c_mp_2d = _getMP2d(proj_m, c_3d, trans_z, rotate_mat, c_2d)

        # Fig 2 (3d): plot mp_2d and c_mp_2d
        ax.quiver(c_2d[0], c_2d[1], 0, c_mp_2d[0], c_mp_2d[1], c_mp_2d[2], color="gold")

        plt.show()

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
        _, rot_neg_theta = getRotationMat2D(theta)
        rotated_c_mp_2d = rot_neg_theta @ c_mp_2d[:2]

        # Fig 3 (2d): plot rotated_c_mp_2d and the ellipse without rotation
        ax.quiver(c_2d[0], c_2d[1], rotated_c_mp_2d[0], rotated_c_mp_2d[1], scale=2, color="limegreen")
        kwrg = {'facecolor': 'none', 'edgecolor':'darkgray', 'alpha':1, 'linewidth':2}
        ellip = Ellipse(xy=c_2d, width=2*a, height=2*b, angle=0, **kwrg)
        ax.set_aspect('equal')
        ax.add_artist(ellip)
        plt.show()

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
        elli_r_o = getRayEllipseIntersection(a, b, rotated_c_mp_2d)
        print(elli_r_o)
        
        # Fig 4 (2d): plot the intersection point
        ax.scatter(elli_r_o[0], elli_r_o[1])
        
        plt.show()
