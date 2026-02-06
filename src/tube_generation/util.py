import numpy as np
import pingouin as pg
import spiceypy as spice
import matplotlib.pyplot as plt

import tube_generation.plotting as plotting

EPSILON = 1e-4

# SPICE ID offset
SPICE_OFFSET = 1000000

# General functions
def readFile(filename):
    """
    Read the content of a file and return it as a string.
    
    Input:
        filename: The path to the file to read
    Output:
        content: The content of the file as a string
    """

    # Open the file, read it and return its content
    file = open(filename, 'r')
    content = file.read()
    return content


def writeFile(filename, content):
    """
    Write the given content to a file.
    Input:
        filename: The path to the file to write to
        content: The content to write to the file
    """

    # Open the file, write the content and close it
    file = open(filename, "w")
    file.write(content)
    file.close()


# Math related functions
def normalizePoints(points):
    """
    Normalize the input points to be between 0 and 1 for each dimention and calculate the
    scaling factor for each dimension

    Input:
        points: The points to normalize
    Output:
        normalized_points: The normalized points
        offset: The offset for each dimension (minimum value)
        scaling_factors: The scaling factor for each dimension
    """

    # Find the maximum and minimum of each axis pf the given points
    min_x = np.min(points[0, :])
    max_x = np.max(points[0, :])
    min_y = np.min(points[1, :])
    max_y = np.max(points[1, :])
    min_z = np.min(points[2, :])
    max_z = np.max(points[2, :])

    # Normalize the points to be between 0 and 1 for each dimention
    normalized_points = np.zeros(points.shape)
    normalized_points[0, :] = (points[0, :] - min_x) / (max_x - min_x)
    normalized_points[1, :] = (points[1, :] - min_y) / (max_y - min_y)
    normalized_points[2, :] = (points[2, :] - min_z) / (max_z - min_z)

    # Store the scaling factor for each dimesion
    scaling_factors = np.array([max_x - min_x, max_y - min_y, max_z - min_z])
    offsets = np.array([min_x, min_y, min_z])

    # Return new set of points that are normalized and the scaling factors
    return normalized_points, offsets, scaling_factors


def invNormalizePoints(normalized_points, offsets, scaling_factors):
    """
    Inverse normalize the input points from being between 0 and 1 for each dimention

    Input:
        normalized_points: The points to inverse normalize
        offset: The offset for each dimension (minimum value)
        scaling_factors: The scaling factor for each dimension
                         (maximum value - minimum value)
    Output:
        points: The inverse normalized points
    """

    # Inverse normalize the points from being between 0 and 1 for each dimention
    points = np.zeros(normalized_points.shape)
    points[0, :] = normalized_points[0, :] * scaling_factors[0] + offsets[0] 
    points[1, :] = normalized_points[1, :] * scaling_factors[1] + offsets[1]
    points[2, :] = normalized_points[2, :] * scaling_factors[2] + offsets[2] 

    # Return new set of points that are normalized and the scaling factors
    return points


def invNormalizePoint(normalized_point, offsets, scaling_factors):
    """
    Inverse normalize the input point from being between 0 and 1 for each dimention

    Input:
        normalized_point: The point to inverse normalize
        offset: The offset for each dimension (minimum value)
        scaling_factors: The scaling factor for each dimension (maximum value - minimum value)
    Output:
        point: The inverse normalized point
    """

    # Inverse normalize the point from being between 0 and 1 for each dimention
    point = normalized_point
    point[0] = normalized_point[0] * scaling_factors[0] + offsets[0] 
    point[1] = normalized_point[1] * scaling_factors[1] + offsets[1]
    point[2] = normalized_point[2] * scaling_factors[2] + offsets[2] 

    # Return new set of point that are normalized and the scaling factors
    return point


def projectVectorToPlane(vector, normal):
    """
    Projoect a vector onto a plane.

    Input:
        vector: The vector to project onto the plane
        normal: The normal of the plane to project the vector onto
    Output: The input vector projected onto the input plane
    """
    
    # Normalize the vectors
    vector = vector / np.linalg.norm(vector)
    normal = normal / np.linalg.norm(normal)

    return vector - (np.dot(vector, normal) / np.dot(normal, normal) * normal)


def transformPointToXYPlane(point, translation, rotation_matrix):
    """
    Transform the input point with the given translation and rotation onto the XY plane.

    Input:
        point: A point to transform 
        translation: The translation to get the point to the XY plane
        rotation_matrix: The rotation matrix to rotate the point to the XY plane
    Output: The transformed point on the XY plane
    """

    # First translate the point to the origin
    translated_point = point - translation  

    # And then rotate it to the XY plane
    return rotation_matrix @ translated_point


def invTransformPointToXYPlane(point, translation, rotation_matrix):
    """
    Transform the input point with the given translation and rotation onto the XY plane.

    Input:
        point: A point to transform in an inverse manner
        translation: The translation to get the point from the XY plane to the input plane
        rotation_matrix: The rotation matrix to rotate the point from the XY plane to the
                         input plane. The inversion happens before callign this function.
    Output: The transformed point on the input plane
    """

    # First rotate the point around the origin
    rotated_point = rotation_matrix @ point

    # And then translate the point from the origin
    return rotated_point + translation


def calcRotationMatrix(vector, target_vector):
    """
    Calculate the rotation matrix that rotates the input vector to be aligned with the
    target vector using the Rodrigues rotation formula.

    Input:
        vector: The vector to rotate
        target_vector: The target vector to align the input vector with 
    Output: The rotation matrix that rotates the input vector to be aligned with the
            target vector
    """
    
    # Normalize the input vectors
    vector = vector / np.linalg.norm(vector)
    target_vector = target_vector / np.linalg.norm(target_vector)

    # Find the axis of rotation
    rotation_axis = np.cross(vector, target_vector)

    # Caclulate the skew-symmetric cross-product matrix of the rotation axis
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ]) / np.linalg.norm(rotation_axis)

    # Prepare the rotation matrix parameters
    cos_a = np.dot(vector, target_vector)
    sin_a = np.linalg.norm(rotation_axis)

    # Calculate the rotation matrix 
    # Formula from: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    return np.eye(3) + K * sin_a + K @ K * (1 - cos_a)


def transformPointsToXYPlane(points, plane_center, plane_normal, do_plotting):
    """
    Transform the input points on the given plane to the XY plane. The input plane is
    defined by its center point and normal. The XY plane have a normal of (0, 0, 1)

    Input:
        points: A list of points on the input plane to transform onto the XY-plane
        plane_center: The center point of the input plane. This point must be part of the 
                      input plane.
        plane_normal: The normal of the input plane
        do_plotting: Whether to do plotting or not

    Output: The transformed points on the XY plane in 2D
    """

    # The translation is the same as the plane center vector
    translation = plane_center

    # Find the rotation matrix to rotate the plane normal to be aligned with the Z axis
    rotation_matrix = calcRotationMatrix(plane_normal, np.array([0, 0, 1]))

    # Apply the transformation to all points
    transformed_points = np.zeros(points.shape)
    for p in range(points.shape[1]):
        transformed_points[:, p] = transformPointToXYPlane(
            points[:, p],
            translation,
            rotation_matrix
        )
    
    # Plot the result if requested
    if do_plotting:
        figure = plt.figure(figsize = plt.figaspect(1))
        figure_axes = figure.add_subplot(projection = '3d')
        plotting.plotPoints3D(figure_axes, transformed_points, is_normalized = True)
        plt.title("Plot of transformed points onto the XY plane")
        plt.show()

    # Check that all points are on the XY plane
    z_value = transformed_points[2, 0]
    for p in range(points.shape[1]):
        if np.abs(transformed_points[2, p] - z_value) > EPSILON:
            print('\033[41mNot on XY plane\033[0m')
            print('difference:', np.abs(transformed_points[2, p] - z_value))

    # Return the XY coordinates of the transformed points
    return transformed_points[:2, :]


def invTransformPointsToXYPlane(points, plane_center, plane_normal, do_plotting):
    """
    Inverse transform the input points from the XY plane to the given plane. The input
    plane is defined by its center point and normal. The XY plane have a normal of
    (0, 0, 1)

    Input:
        points: A list of points on the XY plane to transform to the input plane in 2D
        plane_center: The center point of the input plane. This point must be part of the 
                      input plane.
        plane_normal: The normal of the input plane
        do_plotting: Whether to do plotting or not

    Output: The transformed points on the input plane in 3D
    """

    # Normalize the vector
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # The translation is the same as the plane center vector
    translation = plane_center

    # Find the rotation matrix to rotate the plane normal to be aligned with the Z axis
    rotation_matrix = calcRotationMatrix(plane_normal, np.array([0, 0, 1]))
    rotation_matrix = np.linalg.inv(rotation_matrix)
    
    # First convert all 2D points to 3D by adding a Z coordinate of 0
    points_3D = np.zeros((3, points.shape[1]))
    points_3D[0, :] = points[0, :]
    points_3D[1, :] = points[1, :]

    # Apply the transformation to all points
    transformed_points = np.zeros(points_3D.shape)
    for p in range(points.shape[1]):
        transformed_points[:, p] = invTransformPointToXYPlane(
            points_3D[:, p],
            translation,
            rotation_matrix
        )
    
    # Plot the result if requested
    if do_plotting:
        figure = plt.figure(figsize = plt.figaspect(1))
        figure_axes = figure.add_subplot(projection = '3d')
        plotting.plotPoints3D(figure_axes, transformed_points, is_normalized = True)
        plotting.plotPlane(figure_axes, plane_center, plane_normal)
        plt.title("Plot of transformed points from the XY plane to the input plane")
        plt.show()

    # Check that all points are on the input plane. Formula from:
    # https://stackoverflow.com/questions/17227149/using-dot-product-to-determine-if-point-lies-on-a-plane
    for p in range(points.shape[1]):
        dot_product = np.dot(plane_center - transformed_points[:, p], plane_normal)
        if np.abs(dot_product) > EPSILON:
            print('\033[41mNot on input plane\033[0m')
            print('difference:', np.abs(dot_product))

    # Return the transformed points on the input plane
    return transformed_points


def calcPlaneLineIntersection(normal, center, line_start, line_direction):
    """
    Compute the intersection point between the given plane and the given line

    Input:
        normal: The normal of the plane (normalized)
        center: The center point of the plane (a point on the plane)
        line_start: The starting point of the line
        line_direction: The direction of the line (normalized)
    
    Output:
        multiplier: The multiplier of the line_direction that is requiered for the 
                    line_start to end up on the plane
        intersection: the coordinate of the intersection
    """
    
    # First normalize the input vectors
    normal = normal / np.linalg.norm(normal)
    line_direction = line_direction / np.linalg.norm(line_direction)

    # Check if there ever will be an intersection
    assert np.abs(np.dot(line_direction, normal)) > EPSILON, "No intersection"
    
    # Calculate the multiplier of the line direction to makes the point be on the plane
    # Formula from: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    multiplier = (np.dot(normal, center) - np.dot(normal, line_start)) / np.dot(normal, line_direction)

    # The calculate the coordinate of the intersection point
    intersection = line_start + line_direction * multiplier
    
    return multiplier, intersection


def isGaussian(points):
    """
    Check if the given points (2D or 3D) are Gaussian distributed using the Henze-Zirkler
    multivariate normality test from the package pingouin

    Input:
        points: input points in the shape (n_dimensions, n_points)
    Returns:
        True if the points are Gaussian distributed, False otherwise
    """
    
    # Translate the points to be in the shape (n_points, n_dimensions)
    points = points.T

    # Check the dimension
    assert points.shape[1] in [2, 3], "points must be in 2D or 3D"

    # Perform the Henze-Zirkler multivariate normality test
    return pg.multivariate_normality(points).normal


# Space specific functions
def getSolarSystemNormal(utc_time):
    """
    Get the normal of the solar system.
        utc_time: a timestamp used to compute the transformation matrix between the
                 reference frames. Shouln't matter to the computation.
                 For example ["Jan 1, 2015"]
    Output:
        ssb_normal: The normal of the solar system
    """
    # Convert time fomr utc to et
    et_time = spice.str2et(utc_time[0])

    # The normal of the solar system in ECLIPJ2000 is (0, 0, 1)
    # The ECLIPJ2000 reference frame is the plane of Earth's motion around the Sun
    eclip_normal = np.array([0, 0, 1])

    # Get the transformation matrix from ECLIPJ2000 to GALACTIC reference frame
    # OpenSpace uses GALACTIC for coordinates per default
    transform_matrix = spice.sxform(
        "ECLIPJ2000",
        "GALACTIC",
        et_time
    )

    # The transformation matrix contain both x, y, and z, as well as vx, vy, and vz
    # We only need the position part
    cropped_transform_matrix = transform_matrix[:3, :3]
      
    # Get the solar system normal in the GALACTIC reference frame
    ssb_normal = cropped_transform_matrix @ eclip_normal

    # TODO: Why are we not using the transformation matrix? 
    # Return the normalized vector
    return eclip_normal/np.linalg.norm(eclip_normal)


def getImpactedVariantsAtTime(impact_data, time_astrop):
    """
    """

    # Loop over all impact and check if any of them impact before the given time
    impacted_variants = []
    for impact in impact_data:
        if impact.time <= time_astrop:
            impacted_variants.append(impact.spice_id)

    return impacted_variants


def excludeVariants(coordinates, excluded_variants):
    """
    """

    # Get the spice id form the excluded variants and convert it into a set of indicies
    # of the coordinates
    excluded_indices = []
    for variant in excluded_variants:
        # Index in list (such as index number 0 or 153) = index
        # SPICE id (such as 1000000 or 1000153) = SPICE_OFFSET + index
        # Filename (such as 000001.bsp or 000154.bsp) =
        #     (K-width zero-padded string of index + 1).bsp (K is by default 6)
        spice_id = variant
        index = spice_id - SPICE_OFFSET
        excluded_indices.append(index)

    # Create a mask for the coordinates to exclude the excluded indices
    mask = np.ones(coordinates.shape[1], dtype = bool)
    mask[excluded_indices] = False
    return coordinates[:, mask]