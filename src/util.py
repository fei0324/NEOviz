import numpy as np
import spiceypy as spice

import src.plotting as plotting

EPSILON = 1e-10


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


# Common math functions
def projectVectorToPlane(vector, normal):
    """
    Projoect a vector onto a plane.

    Input:
        vector: The vector to project onto the plane
        normal: The normal of the plane to project the vector onto
    Output: The input vector projected onto the input plane
    """
                    
    return vector - ((np.dot(vector, normal) / np.dot(normal, normal)) * normal)


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


def calcRotationMatrix(vector, target_vector):
    """
    Calculate the rotation matrix that rotates the input vector to be aligned with the
    target vector using the Rodrigues rotation formula.

    Input:
        vector: The vector to rotate
        target_vector: The target vector to align the input vector with (normalized)
    Output: The rotation matrix that rotates the input vector to be aligned with the
            target vector
    """
    
    # Normalize the input vector
    vector = vector / np.linalg.norm(vector)

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


# TODO: This function is not used at the moment, but I will keep it around for some time
# In case the Rodrigues rotation matrix approach doesn't work as expected
def transformPointToXYPlaneOrtho(point, plane_center, plane_normal):
    # Put the camera stright in front of the plane
    camera_pos = plane_center + (plane_normal / np.linalg.norm(plane_normal))

    # Set the camera to look at the plane center
    camera_direction = -plane_normal / np.linalg.norm(plane_normal)

    # Get the right direction of the camera
    # We get this by crossing the camera direction with the world up vector
    camera_right = np.cross(camera_direction, np.array([0, 1, 0]))

    # Now get the actual up vector of the camera
    camera_up = np.cross(camera_right, camera_direction)

    # Create a look at matrix, i.e. the view matrix
    look_at = np.array([
        [camera_right[0], camera_right[1], camera_right[2], 0],
        [camera_up[0], camera_up[1], camera_up[2], 0],
        [camera_direction[0], camera_direction[1], camera_direction[2], 0],
        [0, 0, 0, 1]
    ])
    view_matrix = look_at @ np.array([
        [ 1, 0, 0, -camera_pos[0]],
        [ 0, 1, 0, -camera_pos[1]],
        [ 0, 0, 1, -camera_pos[2]],
        [ 0, 0, 0, 1]
    ])

    # Setup parameters for the orthografic projection matrix
    top = 1.5
    bottom = -1.5
    right = 1.5
    left = -1.5
    near = 0.1
    far = 5.0

    # Calculate the orthografic projection matrix
    ortho_matrix = np.array([
        [2 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2/(top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2/(far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ])
    
    # Apply the camera transformations to the point
    point_clip = ortho_matrix @ view_matrix @ np.array([point[0], point[1], point[2], 1])

    # Perform perspective division to get the normalized device coordinates
    # For orthographic projection this doesn't change the coordinates
    # Therefore we just return the x, y, and z coordinates and ignore the w coordinate
    return point_clip[:3]


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
        plotting.plotPoints(transformed_points)

    # Check that all points are on the XY plane
    for p in range(points.shape[1]):
        assert np.abs(transformed_points[2, p]) < EPSILON, "Not on XY plane"

    # Return the XY coordinates of the transformed points
    return transformed_points[:2, :]


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

    # First check if there ever will be an intersection
    assert np.abs(np.dot(line_direction, normal)) > EPSILON, "No intersection"
    
    # Calculate the multiplier of the line direction to makes the point be on the plane
    # Formula from: https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    multiplier = (np.dot(normal, center) - np.dot(normal, line_start)) / np.dot(normal, line_direction)

    # The calculate the coordinate of the intersection point
    intersection = line_start + line_direction * multiplier
    
    return multiplier, intersection


# Space specific functions
def getSolarSystemNormal(utc_time: list[str]):
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

    # Return the normalized vector
    return ssb_normal/np.linalg.norm(ssb_normal)
