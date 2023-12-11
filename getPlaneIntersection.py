import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from mvee import mvee2
from plotting import plot_ellipse

import spiceypy as spice
METAKERNEL = 'getsta.tm'
spice.furnsh(METAKERNEL)


def computePlane(n, c, xr, yr):
    """
    Compute a plane from the normal vector and a point on the plane
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


def computePlaneLineIntersection(n, c, lp, lv):
    """
    Compute the intersection point between a plane and a line
    n: the normal vector of the plane (average velocity of the orbits)
    c: the point on the plane (the center of the ellipse)
    lp: the point on the line (an orbit coordinate at a time t)
    lv: the directional vector of the line (the velocity vector of the point at time t)
    """

    t = (np.dot(c, n) - np.dot(lp, n))/np.dot(lv, n)
    intersection_coordinate = np.multiply(t, lv) + lp

    return t, intersection_coordinate


def getRotationalMatrix(n, k):
    """
    Get the rotational matrix when transforming an old plane to a new plane
    n: the normal of the original 3d plane
    k: the normal of the new plane
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


def getTranslationZ(n, c):
    """
    Get the translation vector so the new plane passes through (0, 0, 0)
    n: normal of the old plane
    c: point on the old plane (center of the ellipsoid)
    """
    return np.dot(c, n)/n[2]


def projectVec2Plane(u, n):
    """
    Projoect a vector onto a plane.
    u: vector
    n: normal of the plane
    """
    return u - (np.dot(u, n)/np.linalg.norm(n)**2)*n


def getRayEllipseIntersection(H, ray):
    """
    Get the intersection point of a ray and an ellipse
    H: matrix that represents the general ellipse
    """
    s, u = np.linalg.eigh(H)
    idxs = np.argsort(s)
    s, u = s[idxs], u[:, idxs]
    print(s, u)
    # rotation angle for the ellipse
    theta = np.degrees(np.arctan2(u[1, 0], u[0, 0]))
    # get rotation matrix for -1*theta
    r = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
    rotated_ray = r @ ray
    return rotated_ray


def getStartingPoint(n, a, c, nss):
    """
    Get the starting point to sample from the ellipse
    n: normal of the plane (mean velocity direction)
    a: vector from c (center of the ellipsoid that's on the plane) to the sun
    c: center of the ellipsoid
    nss: normal of the solar system
    """
    m = np.cross(n, a)

    # check if m is on the right side because we want m to have consistent orientation as the asteroid traverses around its orbit
    # here we pick np.dot(m, nss) to always be > 0
    if np.dot(m, nss) < 0:
        m = -m

    # vector m might not be on the plane, project m onto the plane
    proj_m_vec = projectVec2Plane(m, n)
    # get point mp on plane using projected vector m and c
    mp = c + proj_m_vec
    print("mp")
    print(mp)
    # transform m_p to 2d: translate then rotate
    translation_z = getTranslationZ(n, c)
    new_n = np.array([0, 0, 1])
    r = getRotationalMatrix(n, new_n)
    # mp_2d should be 0 for z
    mp_2d = r @ (mp - translation_z)
    print("mp_2d")
    print(mp_2d)

    return mp_2d




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

    # Compute ellipsoid and center of the ellipsoid using mvee
    for i in range(num_time_steps):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        Xi = variants_coords_list[i].T
        print(Xi.shape)
        x = Xi[0, :]
        y = Xi[1, :]
        z = Xi[2, :]
        ax.scatter(x, y, z)

        obj = mvee2(Xi)
        L = obj["L"]
        c = obj["c"]  # center of the ellipsoid
        ax.scatter(c[0], c[1], c[2], s=50)

        # Get the unit vector of the center from the sun
        c_vector = np.array([0, 0, 0]) - c
        c_unit = c_vector/np.linalg.norm(c_vector)
        print(c_vector)
        print(c_unit)
        ax.quiver(c[0], c[1], c[2], c_vector[0], c_vector[1], c_vector[2], color='red')

        # ax.quiver(Xi[0,:], Xi[1,:], Xi[2,:], Vi[0,:], Vi[1,:], Vi[2,:], color='blue')

        # Compute the positive definite matrix that represents the 3d ellipsoid
        # H = L@L.T
        # print(H)

        # use average velocity vector instead of looking for th one that's the closest
        Vi = variants_velo_list[i].T
        print(Vi.shape)
        v_mean = np.mean(Vi, axis=1)
        ax.quiver(c[0], c[1], c[2], v_mean[0], v_mean[1], v_mean[2], color='blue')
        print(v_mean)

        # Compute the plane using the normal vector (average velocity)
        xr = np.linspace(c[0] - 1e-08, c[0] + 1e-08, num=20)
        yr = np.linspace(c[1] - 1e-08, c[1] + 1e-08, num=20)
        xx, yy, pz = computePlane(v_mean, c, xr, yr)
        ax.plot_surface(xx, yy, pz, alpha=0.5)
        ax.set_aspect('equal')
        # plt.show()

        # Compute orbit plane intersection points
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        intersection_list = np.zeros(Xi.shape)
        for j in range(Xi.shape[1]):
            lp = Xi[:, j]
            lv = Vi[:, j]
        
            _, intersection_coor = computePlaneLineIntersection(v_mean, c, lp, lv)
            intersection_list[:, j] = intersection_coor
        print(intersection_list.shape)
        intersectX = intersection_list[0, :]
        intersectY = intersection_list[1, :]
        intersectZ = intersection_list[2, :]
        ax.scatter(intersectX, intersectY, intersectZ)
        plt.show()

        # Note that the intersection points are on a 3d plane
        # Need to transform the plane onto the xy-plane with a translation and a rotation
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        k = np.array([0, 0, 1])
        r = getRotationalMatrix(v_mean, k)
        
        # Translate first so the plane passes through the origin
        translation_z = getTranslationZ(v_mean, c)
        print(translation_z)
        translation_vec = np.tile(np.array([0, 0, translation_z]).T, (num_samples, 1)).T
        translated_list = intersection_list - translation_vec
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # ax.scatter(intersectX, intersectY, intersectZ, c='orange')
        # ax.scatter(translated_list[0, :], translated_list[1, :], translated_list[2, :], c='orange')
        # ax.scatter(0, 0, 0, c='red')
        # plt.show()

        # Then rotate, so the plane is the x-y plane
        transformed_list = r @ translated_list
        print("v_mean", v_mean)
        print("c", c)
        transformedX = transformed_list[0, :]
        transformedY = transformed_list[1, :]
        transformedZ = transformed_list[2, :]
        # print(transformedZ)
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        ax.scatter(transformedX, transformedY, transformedZ)
        ax.set_aspect('equal')  # super important: ensures consistent scale for the axes!!!
        # plt.show()

        # Then this becomes a 2d problem
        # Compute the minimum enclosing ellipse of the intersection points on the xy-plane
        transformed_2d = transformed_list[:2, :]
        print(transformed_2d.shape)
        obj_2d = mvee2(transformed_2d)
        L_2d = obj_2d["L"]
        c_2d = obj_2d["c"]  # center of the ellipse
        H_2d = L_2d@L_2d.T
        print(c_2d)
        ax.scatter(c_2d[0], c_2d[1], 0, s=50, c="red")
        plt.show()

        # Draw a 2D version with the ellipse
        fig = plt.figure()
        ax = fig.add_subplot()
        transformed_2d_x = transformed_2d[0, :]
        transformed_2d_y = transformed_2d[1, :]
        ax.scatter(c_2d[0], c_2d[1], s=50, c='red')
        ax.scatter(transformed_2d_x, transformed_2d_y)
        plot_ellipse(H_2d, c_2d, ax=ax)
        # plt.show()

        # Need to sample from the ellipse
        # Convert the general ellipse matrix H to the cartesian formula of an ellipse
        # Use the parameterized representation
        # Figure out where to start sampling

        # Compute an approximate normal of the solar system
        utctime = ["Jan 1, 2015"]
        ettime = spice.str2et(utctime[0])
        # get the transformation matrix between two reference frames
        rf_trans_mat = spice.sxform(instring='ECLIPJ2000', \
                      tostring='GALACTIC', \
                      et=ettime)
        # get the z direction of the rf_trans_mat
        nss = rf_trans_mat[:3, :3] @ np.array([0, 0, 1])
        print(nss)
        mp_2d = getStartingPoint(v_mean, c_unit, c, nss)
        ax.quiver(c_2d[0], c_2d[1], mp_2d[0], mp_2d[1], color='orange')

        # rotate mp_2d
        print("mp_2d")
        print(mp_2d)
        rotated_mp = getRayEllipseIntersection(H_2d, mp_2d[:2])
        # TODO: This rotated mp doesn't make any sense!!! How is the theta defined? Check Matplotlib!
        ax.quiver(c_2d[0], c_2d[1], rotated_mp[0], rotated_mp[1], color='red')
        plt.show()
        # Sample equal amount of points from each time step


        # TODO: Add time step as the title of the plot

    


    # Get positions of earth w.r.t. the sun at that given time


    # TODO: the translation isn't working. Need to figure out why
    # Example that works
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # X = np.array([[1, 1, 1], [1, -2, 4], [-2, 2, 3], [0, 1, 2], [-3, 2, 4], [0, -4, 7], [2, -3, 4]])
    # print(X)
    # Xi = X.T
    # print(Xi)
    # print(Xi.shape)
    # x = Xi[0, :]
    # y = Xi[1, :]
    # z = Xi[2, :]
    # ax.scatter(x, y, z)

    # c = [0, 0, 3]
    # ax.scatter(0, 0, 3, c='red')

    # xr = np.linspace(c[0] - 10, c[0] + 10, num=21)
    # print(xr)
    # yr = np.linspace(c[1] - 10, c[1] + 10, num=21)
    # v_mean = np.array([1, 1, 1])
    # xx, yy, pz = computePlane(v_mean, c, xr, yr)
    # ax.plot_surface(xx, yy, pz, alpha=0.5)
    # ax.set_aspect('equal')
    # plt.show()

    # k = np.array([0, 0, 1])
    # r = getRotationalMatrix(v_mean, k)
        
    # translation_z = getTranslationZ(v_mean, c)
    # print(translation_z)
    # num_samples = 7
    # translation_vec = np.tile(np.array([0, 0, translation_z]).T, (num_samples, 1)).T
    # translated_list = Xi - translation_vec
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x, y, z, c='green')
    # ax.scatter(translated_list[0, :], translated_list[1, :], translated_list[2, :], c='orange')
    # # ax.scatter(0, 0, 0, c='red')
    # # plt.show()
    # rotated_list = r @ translated_list
    # # rotated_list = r @ intersection_list
    # print("v_mean", v_mean)
    # print("c", c)
    # transformedX = rotated_list[0, :]
    # transformedY = rotated_list[1, :]
    # transformedZ = rotated_list[2, :]
    # print(transformedZ)
    # # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # ax.scatter(0, 0, 0, c='red')
    # ax.scatter(transformedX, transformedY, transformedZ)
    # plt.show()

        