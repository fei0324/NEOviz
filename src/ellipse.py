# TODO: Verify if these are all needed
import os
import json
import numpy as np
import numpy.typing as npt
import spiceypy as spice
from astropy import units as astrounit

import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from matplotlib.patches import Ellipse

from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances

from src.ext.mvee import mvee2

import src.util as util
import src.plotting as plotting


def createDebugEllipse(a, b, theta):
    """
    Create a ellipse matrix from the ellipse parameters. These can then be compared to
    the results from the calcEllipseParameters function to see if the math is correct.

    Input:
        a: The length of the major axis of the ellipse
        b: The length of the minor axis of the ellipse
        theta: The angle between the horizontal possitive x axis and the major axis of
               the ellipse
    Output: The ellipse matrix that represents an ellipse with the given charectaristics
    """

    # Use the fomulas in https://en.wikipedia.org/wiki/Ellipse to create a ellipse matrix
    # from the ellipse characteristics. 
    A = a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2
    B = 2*(b**2 - a**2) * np.sin(theta) * np.cos(theta)
    C = a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2

    matrix = np.array([
        [A, B/2.0], 
        [B/2.0, C]
    ])

    return matrix


def calcEllipsoid(points):
    """
    Compute the minimum enclosing ellipsoid from the input point cloud using the
    mvee2 function. If the input points are all in 2D, the result will instead be an
    ellipse.

    Input: 
        points: The input point cloud in 2D or 3D
    Output:
        matrix: The matrix that represents the ellipsoid. A symmetric positive
                definite matrix.
        center: The center point of the ellipsoid (or ellipse)
    """
    # TODO: The function prints if it is feasable and/or optimal to create the ellipsoid
    # What does this mean? Do we need to consider this? 
    result = mvee2(points)

    center = result["c"]

    # TODO: What is this? What is L and why do we do this? 
    L = result["L"]
    matrix = L @ L.T

    return matrix, center


def calcEllipseParameters(center, matrix):
    """
    Calculate the ellipse parameters given the matrix representation of a 2D ellipse

    Input:
        matrix: Matrix representation of a 2D ellipse
    Output:
        axes: The normalized axes directions of the ellipse, starting with the major axis
              and then the minor axis
        axes_lengths: The lengths of the ellipse axes in decending order (same as axes)
        theta: The angle between the possitive horizontal x-axis and the major axis of
               the ellipse (in radians)
    """

    # TODO: Need to verify these calculations
    # TODO: Does this work with all types of values? what is theta == 0?

    # The matrix cannot represent an ellipse if the determinant is not bigger than 0
    daterminant = np.linalg.det(matrix)
    assert daterminant > 0, "Matrix does not represent an ellipse"
    print("Matrix representation of ellipse", matrix)

    # The eigenvectors of the ellipse matrix are always gonna be in the same direction as
    # the major and minor axis of the ellipse. The axis with the smallest absolute
    # eigenvalue is the major axis. This is according to the principal axis theorem.
    #
    # A standard form ellipse has its center at the origin and its axes align with the
    # coordinate x and y axes. Our ellipse will likely be a general ellipse and not a
    # standard one.
    #
    # From: https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # The np.linalg.eigh function returns the eigenvalues in ascending order. Meaning
    # that the smallest eigenvalue is the first item, meaning that the cooresponding
    # major axis would then be the first one
    print("Eigenvalues 2D", eigenvalues)
    print("Eigenvectors 2D", eigenvectors)
    assert np.abs(eigenvalues[0]) < np.abs(eigenvalues[1]), "The first eigenvector must be the major axis"
    major_axis = eigenvectors[:, 0]
    minor_axis = eigenvectors[:, 1]

    # The center point is (Xc, Yc)
    Xc = center[0]
    Yc = center[1]

    # The ellipse equation is:
    # Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    # Where the ellipse matrix looks like:
    # [ [A, B/2],
    #   [B/2, C] ]
    # Formulas from: https://en.wikipedia.org/wiki/Ellipse
    A = matrix[0, 0]
    B = 2 * matrix[0, 1]
    C = matrix[1, 1]

    # The D and E parameters can be caluclated using the center point
    # (The F parameter does not matter in the following calculations)
    D = -2*A*Xc - B*Yc
    E = -B*Xc - 2*C*Yc

    # The angle theta is from the positive horizontal X axis to the ellipse's major axis
    theta = np.arctan2(-B, C - A)
    theta = theta/2.0
    print("Angle theta", theta, "(", np.degrees(theta), ")")

    # Calculate the lengths of the major and minor-axis (a is the major-axis and b is the
    # minor-axis)
    # TODO: What is the unit of this length? Is this radius or diameter?
    a = np.sqrt(
        (A*np.cos(theta)**4 - A*np.cos(2*theta) - C * np.cos(theta)**2) /
        (np.sin(theta)**2 * -np.cos(2*theta))
    )
    b = np.sqrt(
        (C - A*np.cos(theta)**2)/
        (-np.cos(2*theta))
    ) 
    print("Major axis length", a)
    print("Minor axis length", b)
    assert a > b, "a must be the major axis"

    return  np.array([major_axis, minor_axis]), np.array([a, b]), theta


def calcEllipsoidParameters(center, matrix):
    """
    Calculate the ellipsoid parameters given the matrix representation of a 3D ellipsoid

    Input:
        matrix: Matrix representation of a 3D ellipsoid

    Output:
        axes: The normalized axes directions of the ellipsoid, starting with the major
              axis, the semi-major axis and lastly the minor axis
        axes_lengths: The lengths of the ellipsoid axes in decending order (same as axes)
        theta: The angle between the possitive horizontal x-axis and the major axis of
               the ellipsoid (in radians)
        phi: TODO: We might need another angle to characterise the ellipsoid. Or is is 
             better with a rotation matrix?
    """

    # TODO: Implement this
    major_axis = np.array([1, 0, 0])
    semi_major_axis = np.array([0, 1, 0])
    minor_axis = np.array([0, 0, 1])

    a = 3
    b = 2
    c = 1

    theta = 0

    return np.array([major_axis, semi_major_axis, minor_axis]), np.array([a, b, c]), theta


# TODO: Clean up
def computePlaneLineIntersection(n, c, lp, lv):
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

    # TODO: Where does this formula come from?
    t = (np.dot(c, n) - np.dot(lp, n))/np.dot(lv, n)
    intersection_coordinate = np.multiply(t, lv) + lp

    return t, intersection_coordinate


def calcSamplingStartingPoint(ssb_normal, plane_center_2d_3d, plane_normal_3d,
                              major_axis_2d):
    """
    Get the starting point to sample from the 2D ellipse centered at the origin

    Input:
        ssb_normal: The normal of the Solar System 
        plane_center_2d_3d: The center point of the 2D plane extrapolated to be a 3D 
                            vector (z axis is 0)
        plane_normal_3d: The normal of the 3D ellipsoid plane 
        major_axis_2d: The major axis of the 2D ellipse
    Output: The angle from the major axis in the 2D ellipse to the sampling starting
            point in radians
    """

    # Project the Solar System normal to the 2D ellipse plane
    projected_ssb_normal = util.projectVectorToPlane(ssb_normal, plane_normal_3d)

    # Move the projected vector to the ellipse center (we know that the ssb_normal is in
    # the origin)
    translated_ssb_normal = projected_ssb_normal + plane_center_2d_3d

    # We now know that the ssb_normal vector is on the ellipse plane, but we need it on
    # the XY plane to do 2D calculations
    transformed_ssb_normal = util.transformPointsToXYPlane(
        np.array([translated_ssb_normal]).T,
        plane_center_2d_3d,
        plane_normal_3d
    )
    transformed_ssb_normal = transformed_ssb_normal[:2]

    # Find the angle between the major axis and the ssb normal (in the transformed space)
    # We know that both vectors originate in the origin of the xy plane. And thye are 
    # both located in the xy plane. Simple 2D problem
    # Formula: u*v = |u|*|v|*cos(angle) -> angle = acos((u*v)/(|u|*|v|))
    dot_prod = np.dot(transformed_ssb_normal[0], major_axis_2d[0])
    norm_prod = np.linalg.norm(transformed_ssb_normal) * np.linalg.norm(major_axis_2d)
    angle = np.acos(dot_prod/norm_prod)

    return angle[0]


# TODO: Add DEBUG plotting code
# TODO: Split up into smaller steps and functions
def createEllipse(data, time, ssb_normal, do_plotting):
    """
    Create just one ellipse in the tube

    Input:
        data: A dataobject
        time: The timestep
        ssb_normal: The solar system normal
        do_plotting: Whether or not to show plots during the calculations
    Output:
        ellipse: One ellipse data object for the given timestep
    """

    print("\nTime", data.time_data[time])
    print("Time step number", time)
    
    # Get the coordinate list for this timestep
    # TODO: Why do we transpose this?
    coordinates_T = data.variants_coordinates[time].T

    # TODO:  Plot the points for this timestep
    if do_plotting:
        plotting.plotPoints(coordinates_T)

    # Compute a minimum encasing ellipsoid for the coordinates at this timestep using mvee
    ellipsoid_matrix, center_3d = calcEllipsoid(coordinates_T)

    # TODO: Plot the points together with the generated ellipsoid

    # TODO: Get the ellipsoid characteristics to be able to "draw" it in OpenSpace
    # TODO: Double check if we can use the eigenvalues for this 
    # The eigenvectors of the ellipsoid matrix are the orientation of the axis in the
    # ellipsoid
    eigenvalues, eigenvectors = np.linalg.eig(ellipsoid_matrix)
    print("Eigenvalues 3D", eigenvalues)
    print("Eigenvectors 3D", eigenvectors)

    axes_3d, axes_lengths_3d, theta_3d = calcEllipsoidParameters(
        center_3d,
        ellipsoid_matrix
    )

    # We use the mean velocity vector as the normal of the 2D ellipse plane
    # TODO: Why transpose here?
    velocities_T = data.variants_velocities[time].T
    mean_velocity = np.mean(velocities_T, axis = 1)

    # TODO: Use the actual propagation to get the proper position of the variant on the
    # 2D ellipse.
    # TODO: We could step in the direction of their individual velocity to get an
    # inbetween solution? It will be more accurate than pure projection but more cost
    # efficient than propagation. If the distance is rather short it can also be rather
    # accurate. 

    # Project the points onto the plane of the ellipsoid center point and the mean
    # velocity as the normal
    plane_variant_intersections = np.zeros(coordinates_T.shape)
    time_lags = np.zeros(data.num_variants)
    for variant in range(data.num_variants):
        variant_coordinate_T = coordinates_T[:, variant]
        variant_velocity_T = velocities_T[:, variant]
        intersection_time, intersection_coordinate = computePlaneLineIntersection(
            mean_velocity,
            center_3d,
            variant_coordinate_T,
            variant_velocity_T
        )
        time_lags[variant] = intersection_time
        plane_variant_intersections[:, variant] = intersection_coordinate

    # Transform the plots on the 3D plane to the XY plane to convert this to a 2D problem
    transformed_intersections_2d = util.transformPointsToXYPlane(
        plane_variant_intersections,
        center_3d,
        mean_velocity
    )

    # Solve the 2D problem and create a minimum enclosing ellipse around the 2D points on
    # the XY plane
    transformed_intersections_2d = transformed_intersections_2d[:2, :]
    ellipse_matrix, center_2d = calcEllipsoid(transformed_intersections_2d)
    axes_2d, axes_lengths_2d, theta_2d = calcEllipseParameters(center_2d, ellipse_matrix)

    # TODO: DEBUG
    #matrix = createDebugEllipse(8.5, 2.6, 1.68)
    #calcEllipseParameters(np.array([20, 15]), matrix)

    # Sample the ellipse to create the polygon
    # Find a (semi)-stable starting point
    center_2d_3d = np.array([center_2d[0], center_2d[1], 0])
    center_2d_3d = np.array([center_2d_3d]).T
    transformed_center_2d_3d = util.invTransformPointsToXYPlane(
        center_2d_3d,
        center_3d,
        mean_velocity
    )
    start_angle = calcSamplingStartingPoint(
        ssb_normal,
        transformed_center_2d_3d[0],
        mean_velocity,
        axes_2d[0]
    )
    print("Start angle", start_angle)

    # Compute ellipse meta data
    # Texture coordinates and position texture
    # timelag texture
    # Density on the edges of the ellipse
    # etc.

    # Transform the ellipse back to the original 3D space

    # Return the ready ellipse
    # Dummy
    return np.array([0, 0, 1])


def createEllipses(data, out_directory, do_plotting):
    """
    Take the input data and create a list of all ellipses that will be the base for the
    tube

    Input:
        data: Data object with the data from the input files
        out_directory: The outpur directory to store results. Here only textures will be 
                       created and saved.
        do_plotting: Whether or not to show plots during the calculations
    Output: A list of all ellipses to create the tube for the input data
    """
    
    # Create a directory to store the textures for each ellipse
    texture_directory = os.path.join(out_directory, "textures")
    os.makedirs(texture_directory, exist_ok = True)        

    # Get the normal of the solar system
    # The date should not make any difference
    # NOTE: There might be some slight rotation of the normal over time but this should
    # not affect the overall results
    ssb_normal = util.getSolarSystemNormal(["Jan 1, 2015"])

    # Loop over all timesteps
    ellipses = []
    for t in range(data.num_time_steps):
        ellipses.append(createEllipse(data, t, ssb_normal, do_plotting))

    return ellipses
    