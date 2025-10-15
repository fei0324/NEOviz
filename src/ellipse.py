# TODO: Verify if these are all needed
import os
import json
import numpy as np
import numpy.typing as npt
import spiceypy as spice
from astropy import units as astrounit
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from matplotlib.patches import Ellipse

from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances

from src.ext.mvee import mvee2

import src.util as util
import src.plotting as plotting


# Data type to hold information about a 3D ellipsoid
@dataclass
class Ellipsoid:
    ellipsoid_matrix: list
    center: list
    axes: list
    axes_lengths: list
    rotation_matrix: list 


def calcMveeEllipsoid(points):
    """
    Compute the minimum enclosing ellipsoid from the input point cloud using the
    mvee2 function. If the input points are all in 2D, the result will instead be an
    ellipse.

    Input: 
        points: The input point cloud in 2D or 3D
    Output:
        H_matrix: The symmetric positive definite (SPD) matrix that represents an 
                  ellipsoid centered at the origin.
        center: The center point of the ellipsoid (or ellipse)
    """

    # TODO: This function sometimes fail to get the ellipsoid. There is some adjustment
    # we have put in but we might still get some strange result
    result = mvee2(points)

    # The center point of the ellipsoid. H describes an ellipsoid in the origin and then 
    # it can be moved to this center point instead.
    center = result["c"]

    # H is a symmetric positive definite (SPD) matrix that represents an ellipsoid
    # L is the Cholesky factor of H such as: H = L*LT
    # TODO: Source? Do we need to invert it? 
    L = result["L"]
    H_matrix = L @ L.T

    return H_matrix, center


def calcEllipsoidParameters(ellipsoid_matrix):
    """
    Calculate the ellipsoid parameters given the ellipsoid_matrix representation of a
    3D ellipsoid. The function also works for a 2D ellipse.

    Input:
        ellipsoid_matrix: Matrix representation of a 3D ellipsoid (or 2D ellipse)

    Output:
        axes: The normalized axes directions of the ellipsoid, starting with the 
              semi-major axis, and lastly the semi-minor axis
        axes_lengths: The lengths of the ellipsoid axes in decending order (same order as
                      the axes)
        rotation_matrix: The rotation of the ellipsoid in matrix form
    """

    # The ellipsoid_matrix must be symetric
    assert np.allclose(ellipsoid_matrix, ellipsoid_matrix.T), "Matrix not symetric"

    # The eigenvalues and eigenvectors of the ellipsoid matrix can be used to calculate
    # the axes of the principle directions of the ellipsoid and their lengths.
    # NOTE: We use the eigh function instead of eig since we know the matrix is symetric
    eigenvalues, eigenvectors = np.linalg.eigh(ellipsoid_matrix)

    # The eigenvectors of the ellipsoid matrix are always gonna be in the same direction 
    # as the axes of the ellipsoid. From: https://en.wikipedia.org/wiki/Ellipsoid and
    # https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
    axes = eigenvectors.T

    # Flip the axes so that they are in the order from largest to smallest
    # This is most likely needed due to the transposing of the axes matrix.
    # According to the documentation of np.linalg.eigh they should be in the right order 
    # but they are not, so we flip them.
    axes = np.flip(axes, axis = 0)
    
    # The lengths of the axes can be claulated using the eigenvalues with the formula:
    # length = sqrt(eigenvalue) * 2 (since we want the "diameter" not the "radius")
    # (From: https://math.stackexchange.com/questions/185954/finding-the-major-and-minor-axes-of-an-n-dimensional-ellipse)
    axes_lengths = np.sqrt(eigenvalues) * 2.0

    # Since we flip the axes we should flip the values too. Largest eigenvalue is the
    # major axis
    axes_lengths = np.flip(axes_lengths, axis = 0)
    
    # The rotation matrix can be constructed by putting the axes into a matrix, i.e. the 
    # axes matrix transposed. 
    rotation_matrix = axes.T

    return axes, axes_lengths, rotation_matrix


def createEllipsoid(points, do_plotting, is_3d = True):
    """
    Create an ellipsoid that encases all of the given points. This function also work fr
    2D input and will then instead create an ellipse that encases all the input points. 

    Input: 
        points: The input point cloud in 3D (or 2D)
        do_plotting: Whether debug plots should be made or not
        is_3d: Whether the inout data is in 3D or 2D
    Output:
        Ellipsoid:
            ellipsoid_matrix: The matrix that representa this ellipsoid (or ellipse in
                              case of 2D data)
            center: The center point of the ellipsoid
            axes: The normalized vectors that represent the directions of the axes of the 
                  ellipsoid
            axes_lengths: The lengths of the axes of the ellipsoid
            rotation_matrix: The rotation matrix of the ellipsoid
    """

    # Compute a minimum encasing ellipsoid for the points using mvee
    ellipsoid_matrix, center = calcMveeEllipsoid(points)

    # Get the ellipsoid shape characteristics
    axes, axes_lengths, rotation_matrix = calcEllipsoidParameters(ellipsoid_matrix)
    
    # Plot the points together with the generated ellipsoid
    if do_plotting:
        if is_3d:
            print("3D ellipsoid axes", axes)
            print("3D ellipsoid axes lengths", axes_lengths)
            print("3D ellipsoid rotation", rotation_matrix)

            # Plot the original points and the ellipsoid axes. Red should be the largest
            # axis and blue should be the smallest axis
            plotting.plotPointsAndAxes(points, center, axes)

            # Plot the original points and the ellipsoid as a transparent 3D shape
            plotting.plotPointsAndEllipsoid(
                points,
                center,
                axes,
                axes_lengths,
                rotation_matrix
            )
        else:
            print("2D ellipse axes", axes)
            print("2D ellipse axes lengths", axes_lengths)
            print("2D ellipse rotation", rotation_matrix)

            # Plot the points and the ellipse axes in 2D. Red should be the largest axis
            # and blue should be the smallest axis
            plotting.plotPointsAndAxes2D(points, center, axes)

            # Plot the original points and the ellipse in 2D
            plotting.plotPointsAndEllipse(
                points,
                center,
                axes,
                axes_lengths,
                rotation_matrix
            )

    return Ellipsoid(
        ellipsoid_matrix,
        center, 
        axes,
        axes_lengths,
        rotation_matrix 
    )


def getPointsOnSlice(coordinates, velocities, mean_velocity, ellipsoid_center,
                     do_plotting):
    """
    Transform all points to be on a slice of the tube that is perpendicular to the 
    direction towards the Sun. Then transform this plane to the XY plane.

    TODO: Instead, use the actual propagation to get the proper position of the variant 
    # on the 2D ellipse.

    Input:
        coordinates: The coordinates for the variants of this timestep
        velocities: The velocities for the variants of this timestep
        mean_velocity: The normalized mean velocity of the variants. This is the normal 
                       of the plane that is perpendicular to the direction towards the 
                       Sun 
        ellipsoid_center: The center of the ellipsoid that encases all the points
        do_plotting: Whether debug plots should be done or not
    Output:
        transformed_points: The points on the 2D plane slice and transformed to the XY 
                            plane
        time_lags: The amount each variant is ahead/behind the plane slice
    """

    # To store results
    plane_variant_intersections = np.zeros(coordinates.shape)
    time_lags = np.zeros(coordinates.shape[1])

    # Calculate the intersection point for each variant with the plane using their 
    # velocity
    for variant in range(coordinates.shape[1]):
        variant_coordinate = coordinates[:, variant]
        variant_velocity = velocities[:, variant]

        # Find the intersection of this variant with the plane using its current velocity
        # As long as the distance to the plane is rather short and no major body
        # gravitationally interact with the variant, this should be a good approximation
        intersection_multiplier, intersection_coordinate = util.calcPlaneLineIntersection(
            mean_velocity,
            ellipsoid_center,
            variant_coordinate,
            variant_velocity
        )
        plane_variant_intersections[:, variant] = intersection_coordinate

        # The intersection_multiplier is an abstract value that tells how far behind or 
        # ahead a variant are of the plane  
        time_lags[variant] = intersection_multiplier

    # Plot the original points and the new points to compare
    if do_plotting:
        plotting.plotPointsCompAndPlane(
            coordinates,
            plane_variant_intersections,
            mean_velocity,
            ellipsoid_center
        )

    # Transform the points on this 3D plane to the 2D XY plane to convert this to a
    # 2D problem
    transformed_intersections_2d = util.transformPointsToXYPlane(
        plane_variant_intersections,
        ellipsoid_center,
        mean_velocity,
        do_plotting
    )

    # Plot the points in 2D
    if do_plotting:
        plotting.plotPoints2D(transformed_intersections_2d)

    # Return the transformed points
    return transformed_intersections_2d, time_lags


def findStartingPoint(ssb_normal, center_3D, mean_velocity):
    """
    Find the starting point on the ellipse where the given vector intersects the ellipse.
    The vector should be given in 2D space.
    Input:
        ssb_normal: The vector to find the intersection point with the ellipse
        center: The center of the ellipse
        axes: The normalized axes of the ellipse
        axes_lengths: The lengths of the axes of the ellipse
    Output:
        intersection_point: The intersection point on the ellipse
    """

    # Normalize the SSB vector
    ssb_vector = ssb_normal / np.linalg.norm(ssb_normal)

    # Project this 3D vector onto the 3D plane of the mean velocity
    # TODO: Debug the projection function to make sure it creates a vector that is on the plane
    ssb_vector_on_plane = util.projectVectorToPlane(ssb_vector, mean_velocity)

    # Transform this 3D vector on the mean velocity plane to the XY plane (2D)
    # TODO: Debug this. The assetion for being on the XY plane fails
    ssb_vector_2d = util.transformPointsToXYPlane(
        np.array([ssb_vector_on_plane]).T,
        center_3D,
        mean_velocity,
        False
    )

    # Find the starting point on the 2D ellipse. I.e where this vector meets the ellipse.
    # This needs to consistently chosse the intersection point in the positive direction
    # of the SSB vector.


def sampleEllipse(num_samples, center_2D, axes_lengths, rotation_matrix, ssb_normal,
                  center_3D, mean_velocity, do_plotting, precision = 1000):
    """
    Sample the 2D ellipse on the 3D mean velocity plane in a consistent manner so that
    the points are always in the same order. The sampling should start from the
    intersection point of the projected SSB vector onto the plane and then go clockwise
    around the ellipse with each sample being equal arc distance between each other.

    Input:
        num_samples: The number of samples to take on the ellipse
        center_2D: The center of the ellipse in 2D space
        axes_lengths: The lengths of the axes of the ellipse in 2D
        rotation_matrix: The rotation matrix of the ellipse in 2D
        ssb_normal: The SSB vector in 3D world space
        center: The center of the ellipsoid in 3D world space (should be very similar to
                center_2D in 3D world space)
        mean_velocity: The normal vector of the mean velocity plane
        do_plotting: Whether or not to do debug plotting
        precision: The precision to use when calculating the circumference of the 
                   ellipse. Higher values give better precision (default is 1000).

    Output:
        sampled_points_3D: The sampled points in order on the 2D ellipse in 3D world space
        meta_data: Meta data about the sampled points
    """

    # 1. Sample the ellipse in 2D starting from the top of the Y axis on the standard
    # ellipse and going clockwise with equal arc distance between each points
    
    # To simplify the formulas we define a and b
    a = axes_lengths[0]
    b = axes_lengths[1]

    # Calculate the total circumference of the ellipse to then divide it into points with
    # equal arc distance between each point
    circumference = 4.0*a

    # Need to compute the integral numerically, therefore we divide the circumfrance into
    # many smaller parts to approximate it. We only need to compute a quarter of the
    # circumfrance and then multiply it by 4 since ellipses are symetric
    theta_theta = np.linspace(0, np.pi/2.0, precision)
    d_theta = np.pi/(2.0 * precision)

    # Approximate the circumference by integrating the complete elliptic integral of the
    # second kind (Ee). From:
    # https://en.wikipedia.org/wiki/Elliptic_integral#Complete_elliptic_integral_of_the_second_kind and
    # https://math.stackexchange.com/questions/172766/calculating-equidistant-points-around-an-ellipse-arc
    Ee = 0.0
    for theta in theta_theta:
        Ee += np.sqrt(1 - np.sin(theta)**2 * (1 - b**2/a**2)) * d_theta
    circumference *= Ee

    # To sample the ellipse we need to walk along the full ellipse and not just a quarter
    # of it but lets keep the precision the same as before. i.e. 4 times the previous
    # precision to keep the same step size
    theta_theta = np.linspace(0, 2.0*np.pi, 4*precision)
    
    # The formula for where to put a sample is: iC/n = a * Em(phi) where i is the sample
    # index [0, n[, C is the total circumference, and n is the total number of
    # samples. We simplify this to iC/na = Em(phi). We need then to solve the incomplete
    # elliptic integral of the second kind for the angle phi, Em(phi). From:
    # https://en.wikipedia.org/wiki/Elliptic_integral#Incomplete_elliptic_integral_of_the_second_kind and
    # https://math.stackexchange.com/questions/172766/calculating-equidistant-points-around-an-ellipse-arc
    target_Em = circumference / (num_samples * a)
    max_Em = target_Em * num_samples

    # Start at i = 0 (phi = 0) and accumulate the Em until we reach the desired Em and
    # save a sample point. Next increment the target Em and repeat until we have all
    # samples. Make sure we do not go outside the bounds of the ellipse
    # TODO: Store the x and y in one 2D array instead of two separate ones
    sampled_points_x = []
    sampled_points_y = []
    Em_accumulated = 0.0
    i = 0
    for theta in theta_theta:
        Em_accumulated += np.sqrt(1 - np.sin(theta)**2 * (1 - b**2/a**2)) * d_theta

        # Stop if we have reached the maximum Em
        if (Em_accumulated > max_Em):
            break

        if (Em_accumulated >= target_Em * i):
            # We have reached the desired Em, take this angle as a desired angle to
            # sample (phi) and sample a point along the ellipse here
            phi = theta
            x = a*np.sin(phi)
            y = b*np.cos(phi)
            sampled_points_x.append(x)
            sampled_points_y.append(y)

            # Increment to target teh arc lnegth of the next sample
            i += 1

            # Stop if we have reached the desired number of samples
            if (i >= num_samples):
                break
    
    # The samples are now located with equal arc distance between them around a standard
    # ellipse at the origin. The semi-major axis is along the X axis and the semi-minor
    # axis along the Y axis. We need to rotate and translate these points to be in the
    # correct position to select the starting point correctly
    sampled_points = np.array([sampled_points_x, sampled_points_y])
    for point in range(sampled_points.shape[1]):
        # Rotate the point using the rotation matrix of the ellipse
        sampled_points[:, point] = rotation_matrix @ sampled_points[:, point]

        # Translate the point using the center of the ellipse
        sampled_points[:, point] = sampled_points[:, point] + center_2D

    # Plot the samples. The plotting code will translate and rotate the ellipse, but we
    # have to translate and rotate the points ourselves (which we have done above)
    plotting.plotEllipseSamples(
        sampled_points,
        center_2D,
        axes_lengths, 
        rotation_matrix
    )

    # 2. Find the starting point on the ellipse using the SSB vector and reorder the
    # samples accordingly

    # 6. Compute meta data of the ellipse in 2D space
    # 7. Inverse transform the sampled 2D points back to 3D space
    # 8. Return the sampled 3D points and their meta data


    # Ellipse meta data:
    # Texture coordinates
    # Density
    # Timelag
    # etc.

    return None


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
    
    # Get the variant coordinate list for this timestep. The coordinates are in meters 
    # and relative the SUN (TODO: Or SSB need to check that)
    coordinates = data.variants_coordinates[time].T

    # Plot the points for this timestep
    if do_plotting:
        plotting.plotPoints(coordinates)

    # Normalize the points to be between 0 and 1 for better numerical stability
    normalized_coordinates, offset, scaling_factors = util.normalizePoints(coordinates)

    # Plot the normalized points for this timestep
    if do_plotting:
        plotting.plotPoints(normalized_coordinates)

    # Create an ellipsoid data objects with all of the ellipsoid features
    ellipsoid = createEllipsoid(normalized_coordinates, do_plotting, True)

    # Transform all points to be on the a plane that is perpendicular to the direction
    # towards the Sun. The mean velocity vector is used as the normal of this plane.
    # TODO: Should I normalize the velocities too? Or does it matter? 
    velocities = data.variants_velocities[time].T
    mean_velocity = np.mean(velocities, axis = 1)

    # Plot the points and the axes of the ellipsoid
    if do_plotting:
        plotting.plotPointsAndVectors(
            normalized_coordinates,
            ellipsoid.center,
            velocities,
            mean_velocity
        )

    # Transform all points to be on this plane, i.e. a slice of the tube going around the
    # Sun. Then transform this plane to be on the XY plane, giving a 2D problem
    intersections_2d, time_lags = getPointsOnSlice(
        normalized_coordinates,
        velocities,
        mean_velocity,
        ellipsoid.center,
        do_plotting
    )
    
    # Solve the 2D problem and create a minimum enclosing ellipse around the 2D points on
    # the XY plane
    ellipse = createEllipsoid(intersections_2d, do_plotting, False)

    # Sample the ellipse to create the polygon that make up the tube
    num_samples = 80
    samples = sampleEllipse(
        num_samples,
        ellipse.center,
        ellipse.axes_lengths,
        ellipse.rotation_matrix,
        ssb_normal,
        ellipsoid.center,
        mean_velocity,
        do_plotting
    )

    # Transform the ellipse back to the original 3D space

    # Return the samples of the ellipse in their correct 3D position
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
    