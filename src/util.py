import numpy as np
import spiceypy as spice


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


def transformPoint(point, translation, rotation_matrix):
    """
    Transform the input point with the given translation and rotation

    Input:
        point: A point to transform 
        translation: The translation to apply for the transform
        rotation_matrix: The rotation to apply for the transform

    Output: The transformed point
    """

    # First translate the point 
    translated_point = point - translation

    # Then apply the rotation
    return rotation_matrix @ translated_point


def invTransformPoint(point, original_translation, original_rotation_matrix):
    """
    Inversely transform the input point with the given translation and rotation.
    The given point has previouslt been transformed with the same input, this function
    reverese that transformation

    Input:
        point: A point to inverse the previous transform
        original_translation: The translation that has already been applied to the input
                              point. This is the translation to reverse to apply the
                              inverse transform. 
        original_rotation_matrix: The rotation that has previously been applies and 
                                  should be revered.
    Output: The inversely transformed point
    """

    # First rotate the point 
    rotated_point = np.linalg.inv(original_rotation_matrix) @ point

    # Then translate it
    return rotated_point + original_translation


def calcXYPlaneRotationMatrix(normal):
    """
    Get the rotation matrix to rotate a plane with the given input normal to the XY plane
    with a normal of (0, 0, 1)

    Input:
        normal: The normal of the 3D plane that should be rotated to the XY plane
    Output:
        rotation_matrix: The 3D rotation matrix to rotate the input plane to the XY plane
    """

    # Rotation matrix formula:
    # matrix = [
    #   [cos(theta) + u1^2*(1 - cos(theta)), u1*u2*(1 - cos(theta)), u2*sin(theta)], 
    #   [u1*u2*(1 - cos(theta)), cos(theta) + u2^2*(1 - cos(theta)), -u1*sin(theta)],
    #   [-u2*sin(theta), u1*sin(theta), cos(theta)]    
    # ]
    # From:
    # https://math.stackexchange.com/questions/1167717/transform-a-plane-to-the-xy-plane

    # Where:
    # normal = (a, b, c)
    a = normal[0]
    b = normal[1]
    c = normal[2]

    # theta is the angle betweeen the translated_plane_normal and the normal of the XY plane
    # cos(theta) = c/sqrt(a^2+b^2+c^2) -> c/|translated_plane_normal|
    cos_theta = c/np.linalg.norm(normal)

    # sin(theta) = sqrt((a^2+b^2)/(a^2+b^2+c^2)) -> sqrt(1 - cos(theta)^2)
    sin_theta = np.sqrt(1 - cos_theta**2)

    # u1 = b/sqrt(a^2+b^2)
    u1 = b/np.sqrt(a**2 + b**2)

    # u2 = −a/sqrt(a^2+b^2)
    u2 = -a/np.sqrt(a**2 + b**2)

    # Construct the rotation matrix
    rotation_matrix = np.array([
        [cos_theta + u1**2 * (1 - cos_theta), u1*u2*(1 - cos_theta), u2*sin_theta], 
        [u1*u2*(1 - cos_theta), cos_theta + u2**2 * (1 - cos_theta), -u1*sin_theta],
        [-u2*sin_theta, u1*sin_theta, cos_theta] 
    ])

    return rotation_matrix


def transformPointsToXYPlane(points, plane_center, plane_normal):
    """
    Transform the input points on the given plane to the XY plane. The input plane is
    defined by the its center point and normal. The XY plane have a normal of (0, 0, 1)

    Input:
        points: A list of points on the input plane to transform onto the XY-plane
        plane_center: The center point of the input plane. This point must be part of the 
                      input plane.
        plane_normal: The normal of the input plane

    Output: The transformed points on the XY plane
    """

    # The translation is the same as the plane center vector
    translation = plane_center
    translated_plane_normal = plane_normal - translation

    # Find the rotation matrix to rotate the input plane to the XY plane
    rotation_matrix = calcXYPlaneRotationMatrix(translated_plane_normal)

    # Apply the transformation to all points
    for p in range(points.shape[1]):
        transformPoint(points[:, p], translation, rotation_matrix)

    return points


def invTransformPointsToXYPlane(points, original_plane_center, original_plane_normal):
    """
    Inversely transform the input points on the XY plane to the original plane in 3D.
    The original plane is defined by the its center point and normal. The XY plane have a
    normal of (0, 0, 1)

    Input:
        points: A list of points on XY plane that previously have been transformed from
                the original plane
        original_plane_center: The center point of the original plane. This point must be 
                            part of the original plane.
        plane_normal: The normal of the original plane

    Output: The transformed points on the XY plane
    """

    # The translation is the same as the original plane center vector
    translation = original_plane_center
    translated_plane_normal = original_plane_normal - translation

    # Find the rotation matrix to rotate the original plane to the XY plane
    rotation_matrix = calcXYPlaneRotationMatrix(translated_plane_normal)

    # Apply the inverse transformation to all points
    for p in range(points.shape[1]):
        invTransformPoint(points[:, p], translation, rotation_matrix)

    return points


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