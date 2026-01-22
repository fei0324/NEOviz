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
from adam_core.coordinates.covariances import make_positive_semidefinite
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import pairwise_distances
from copy import deepcopy

from ext.mvee import mvee2

import tube_generation.util as util
import tube_generation.texture as texture
import tube_generation.plotting as plotting

EPSILON = 1e-4
AU = 149597870700.0

# Data types to hold information
# An ellipsoid data object from MVEE and other parameters
@dataclass
class Ellipsoid:
    ellipsoid_matrix: np.array
    center: np.array
    axes: np.array
    axes_lengths: np.array
    rotation_matrix: np.array

# An data object for each sample that will be written in the tube file
@dataclass
class EllipseSamplePoint:
    position_2D: np.array
    position_3D: np.array
    texture_coordinate: np.array
    density: float
    standard_deviation: np.array
    variance: np.array
    is_gaussian: int
    is_deviating: int

# Data object for the full ellipse that will be written in the tube file
@dataclass
class TubeEllipse:
    time: str
    center: np.array
    texture: str
    samples: list
    ellipsoid: Ellipsoid


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
            figure = plt.figure(figsize = plt.figaspect(1))
            figure_axes = figure.add_subplot(projection = '3d')
            plotting.plotPoints3D(figure_axes, points, alpha = 0.6)
            plotting.plotPoint3D(figure_axes, center, 250, "red", 1.0, 'x')
            plotting.plotAxes3D(figure_axes, center, axes)
            plt.title("Plot of ellipsoid points and the 3 axes")
            plt.show()

            # Plot the original points and the ellipsoid as a transparent 3D shape
            figure.clear()
            figure = plt.figure(figsize = plt.figaspect(1))
            figure_axes = figure.add_subplot(projection = '3d')
            plotting.plotPoints3D(figure_axes, points, alpha = 0.6)
            plotting.plotPoint3D(figure_axes, center, 250, "red", 1.0, 'x')
            plotting.plotEllipsoid(
                figure_axes,
                center,
                axes_lengths,
                rotation_matrix
            )
            plt.title("Plot of the ellipsoid and the points it encases")
            plt.show()
        else:
            print("2D ellipse axes", axes)
            print("2D ellipse axes lengths", axes_lengths)
            print("2D ellipse rotation", rotation_matrix)

            # Plot the points and the ellipse axes in 2D. Red should be the largest axis
            # and blue should be the smallest axis
            figure, figure_axes = plt.subplots(figsize = plt.figaspect(1))
            plotting.plotPoints2D(figure_axes, points, alpha = 0.6)
            plotting.plotPoint2D(figure_axes, center, 250, "red", 1.0, 'x')
            plotting.plotAxes2D(figure_axes, center, axes)
            plt.title("Plot of ellipse points and the 2 axes")
            plt.show()

            # Plot the original points and the ellipse in 2D
            figure.clear()
            figure, figure_axes = plt.subplots(figsize = plt.figaspect(1))
            plotting.plotPoints2D(figure_axes, points, alpha = 0.6)
            plotting.plotPoint2D(figure_axes, center, 250, "red", 1.0, 'x')
            plotting.plotEllipse(
                figure_axes,
                center,
                axes_lengths,
                rotation_matrix
            )
            plt.title("Plot of the ellipse and the points it encases")
            plt.show()

    return Ellipsoid(
        ellipsoid_matrix,
        center, 
        axes,
        axes_lengths,
        rotation_matrix 
    )

 
def createEllipsoidFromMatrix(points, ellipsoid_matrix, do_plotting, is_3d = True):
    """
    Create an ellipsoid from a given ellipsoid matrix. This function also work for
    2D input and will then instead create an ellipse from the matrix.

    Input: 
        points: The input point cloud in 3D (or 2D)
        ellipsoid_matrix: The matrix representation of the ellipsoid (or ellipse in 2D)
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

    # Slice the ellipsoid matrix so we only work with the positional uncertainty
    #ellipsoid_matrix = make_positive_semidefinite(ellipsoid_matrix)
    ellipsoid_matrix = ellipsoid_matrix[:3, :3]
    
    # Compute a minimum encasing ellipsoid for the points using mvee. Use this only to
    # estimate the center of the ellipoid
    center = np.mean(points, axis = 1)
    print("Center", center)
    
    # Get the ellipsoid shape characteristics
    axes, axes_lengths, rotation_matrix = calcEllipsoidParameters(ellipsoid_matrix)

    # Rescale the axes to be in meters and not AU
    axes_lengths = axes_lengths * AU
    
    # Plot the points together with the generated ellipsoid
    if do_plotting:
        if is_3d:
            print("3D ellipsoid axes", axes)
            print("3D ellipsoid axes lengths", axes_lengths)
            print("3D ellipsoid rotation", rotation_matrix)

            # Plot the original points and the ellipsoid axes. Red should be the largest
            # axis and blue should be the smallest axis
            figure = plt.figure(figsize = plt.figaspect(1))
            figure_axes = figure.add_subplot(projection = '3d')
            plotting.plotPoints3D(figure_axes, points, alpha = 0.6, is_normalized = False)
            plotting.plotPoint3D(
                figure_axes,
                center,
                250,
                "red",
                1.0,
                'x',
                is_normalized = False
            )
            plotting.plotAxes3D(figure_axes, center, axes, is_normalized = False)
            plt.title("Plot of ellipsoid points and the 3 axes")
            plt.show()

            # Plot the original points and the ellipsoid as a transparent 3D shape
            figure.clear()
            figure = plt.figure(figsize = plt.figaspect(1))
            figure_axes = figure.add_subplot(projection = '3d')
            plotting.plotPoints3D(figure_axes, points, alpha = 0.6, is_normalized = False)
            plotting.plotPoint3D(
                figure_axes,
                center,
                250,
                "red",
                1.0,
                'x',
                is_normalized = False
            )
            plotting.plotEllipsoid(
                figure_axes,
                center,
                axes_lengths,
                rotation_matrix,
                is_normalized = False
            )
            plt.title("Plot of the ellipsoid and the points it encases")
            plt.show()
        else:
            print("2D ellipse axes", axes)
            print("2D ellipse axes lengths", axes_lengths)
            print("2D ellipse rotation", rotation_matrix)

            # Plot the points and the ellipse axes in 2D. Red should be the largest axis
            # and blue should be the smallest axis
            figure, figure_axes = plt.subplots(figsize = plt.figaspect(1))
            plotting.plotPoints2D(figure_axes, points, alpha = 0.6, is_normalized = False)
            plotting.plotPoint2D(
                figure_axes,
                center,
                250,
                "red",
                1.0,
                'x',
                is_normalized = False
            )
            plotting.plotAxes2D(figure_axes, center, axes, is_normalized = False)
            plt.title("Plot of ellipse points and the 2 axes")
            plt.show()

            # Plot the original points and the ellipse in 2D
            figure.clear()
            figure, figure_axes = plt.subplots(figsize = plt.figaspect(1))
            plotting.plotPoints2D(figure_axes, points, alpha = 0.6, is_normalized = False)
            plotting.plotPoint2D(
                figure_axes,
                center,
                250,
                "red",
                1.0,
                'x',
                is_normalized = False
            )
            plotting.plotEllipse(
                figure_axes,
                center,
                axes_lengths,
                rotation_matrix,
                is_normalized = False
            )
            plt.title("Plot of the ellipse and the points it encases")
            plt.show()

    return Ellipsoid(
        ellipsoid_matrix,
        center, 
        axes,
        axes_lengths,
        rotation_matrix 
    )


def getPointsOnSlice(coordinates, velocities, median_velocity, ellipsoid_center,
                     do_plotting):
    """
    Transform all points to be on a slice of the tube that is perpendicular to the 
    direction towards the Sun. Then transform this plane to the XY plane.

    TODO: Instead, use the actual propagation to get the proper position of the variant 
    # on the 2D ellipse.

    Input:
        coordinates: The coordinates for the variants of this timestep
        velocities: The velocities for the variants of this timestep
        median_velocity: The median velocity of the variants. This is the normal 
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
            median_velocity,
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
        figure = plt.figure(figsize = plt.figaspect(1))
        figure_axes = figure.add_subplot(projection = '3d')
        plotting.plotPoints3D(figure_axes, coordinates)
        plotting.plotPoints3D(
            figure_axes,
            plane_variant_intersections,
            color = "red"
        )
        #plotting.plotPlane(figure_axes, ellipsoid_center, median_velocity)
        plotting.plotVector3D(figure_axes, ellipsoid_center, median_velocity, "red",)
        plt.title(
            "Plot of original points and the projections on the median velocity plane"
        )
        plt.show()


    # Transform the points on this 3D plane to the 2D XY plane to convert this to a
    # 2D problem
    transformed_intersections_2d = util.transformPointsToXYPlane(
        plane_variant_intersections,
        ellipsoid_center,
        median_velocity,
        do_plotting
    )

    # Plot the points in 2D
    if do_plotting:
        figure, axes = plt.subplots(figsize = plt.figaspect(1))
        plotting.plotPoints2D(
            axes,
            transformed_intersections_2d
        )
        plt.title("Plot of transformed points on the XY plane")
        plt.show()

    # Return the transformed points
    return transformed_intersections_2d, time_lags


def calcEllipseCircumference(axes_lengths, precision):
    # To simplify the formulas we define a and b
    a = axes_lengths[0]
    b = axes_lengths[1]
    
    # Calculate the total circumference of the ellipse using numerical integration
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
        Ee += np.emath.sqrt(1 - np.sin(theta)**2 * (1 - b**2/a**2)) * d_theta

    return circumference * Ee


def sampleEqualArcLength(num_samples, axes_lengths, circumference, precision):
    # Sample the ellipse in 2D starting from the top of the Y axis on the standard
    # ellipse and going clockwise with equal arc distance between each points
    # To simplify the formulas we define a and b
    a = axes_lengths[0]
    b = axes_lengths[1]
    
    # To sample the ellipse we need to walk along the full ellipse and not just a quarter
    # of it but lets keep the precision the same as before. i.e. 4 times the previous
    # precision to keep the same step size
    theta_theta = np.linspace(0, 2.0*np.pi, 4*precision)
    d_theta = np.pi/(2.0 * precision)
    
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
    # samples. Make sure we do not go outside the bounds of the ellipse.
    # TODO: Store the x and y together instead of in seperate lists
    sampled_points_x = []
    sampled_points_y = []
    Em_accumulated = 0.0
    i = 0
    for theta in theta_theta:
        Em_accumulated += np.emath.sqrt(1 - np.sin(theta)**2 * (1 - b**2/a**2)) * d_theta

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
    
    return np.array([sampled_points_x, sampled_points_y])


def findStartingPoint(ssb_normal, center_3D, median_velocity, center_2D, 
                      ellipse_samples_2D, do_plotting = False):
    """
    Find the index to one of the sample points on the ellipse that is closest to the
    intersection point of the SSB vector projected ontot he 2D ellipse. This is the index
    to the sample point that should go first in the list of samples.

    Input:
        ssb_normal: The SSB vector in 3D world space
        center_3D: The center of the ellipsoid in 3D world space
        median_velocity: The normal vector of the median velocity plane
        center_2D: The center of the ellipse in 2D space
        ellipse_samples_2D: The sampled points along the ellipse in 2D space
                            (non-standard ellipse)
        do_plotting: Whether or not to do debug plotting
    Output:
        intersection_index: The index of the sample point that should be first in the
                            list of samples of the ellipse
    """

    # Normalize the vectors
    ssb_normal = ssb_normal / np.linalg.norm(ssb_normal)
    median_velocity = median_velocity / np.linalg.norm(median_velocity)
    
    # Project the ssb 3D vector onto the 3D plane of the median velocity
    ssb_on_plane = util.projectVectorToPlane(ssb_normal, median_velocity)

    # Transform the projected 3D ssb vector on the median velocity plane, to the XY plane
    ssb_vector_2d = util.transformPointsToXYPlane(
        np.array([ssb_on_plane]).T,
        center_3D,
        median_velocity,
        False
    )

    # Plot the 2D projection of the ssb vector
    if do_plotting:
        figure, figure_axes = plt.subplots(figsize = plt.figaspect(1))
        plotting.plotPoint2D(figure_axes, center_2D, 250, "red", 1.0, 'x')
        plotting.plotVector2D(figure_axes, center_2D, ssb_vector_2d, color = "red")
        plotting.plotPoints2D(figure_axes, ellipse_samples_2D)
        plt.title("Projection of SSB vector onto the median velocity plane")
        plt.show()

    # Get the smallest angle from the transformed ssb vector to one of the ellipse sample
    # points
    intersection_index = -1
    min_angle = np.inf
    for point in range(ellipse_samples_2D.shape[1]):
        sample = ellipse_samples_2D[:, point]
        ssb_normalized = ssb_vector_2d / np.linalg.norm(ssb_vector_2d)
        sample_normalized = sample / np.linalg.norm(sample)
        angle = np.arccos(np.dot(ssb_normalized.reshape(-1), sample_normalized))

        # Check if this is the new smallest angle
        if angle < min_angle:
            min_angle = angle
            intersection_index = point

    # Did we find any intersection point
    assert intersection_index != -1, "Could not find starting point on ellipse"
    intersection_point = ellipse_samples_2D[:, intersection_index]

    # Plot the intersection point in the ellipse
    if do_plotting:
        figure, figure_axes = plt.subplots(figsize = plt.figaspect(1))
        plotting.plotPoint2D(figure_axes, intersection_point, 250, "green")
        plotting.plotVector2D(figure_axes, center_2D, ssb_vector_2d, color = "red")
        plotting.plotPoints2D(figure_axes, ellipse_samples_2D)
        plt.title("Intersection point on the sampled ellipse")
        plt.show()

    return intersection_index
    

def sampleEllipse(num_samples, center_2D, axes_lengths, rotation_matrix, ssb_normal,
                  center_3D, median_velocity, do_plotting, precision = 1000):
    """
    Sample the 2D ellipse on the 3D median velocity plane in a consistent manner so that
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
        median_velocity: The normal vector of the median velocity plane
        do_plotting: Whether or not to do debug plotting
        precision: The precision to use when calculating the circumference of the 
                   ellipse. Higher values give better precision (default is 1000).
    Output:
        sampled_points_3D: The sampled points in order on the 2D ellipse in 3D world space
        meta_data: Meta data about the sampled points
    """

    # Calculate the total circumference of the ellipse to then divide it into points with
    # equal arc distance between each point
    circumference = calcEllipseCircumference(axes_lengths, precision)

    # Sample the ellipse with equal arc distance between each point
    sampled_points = sampleEqualArcLength(
        num_samples,
        axes_lengths,
        circumference,
        precision
    )

    # The samples are now located with equal arc distance between them around a standard
    # ellipse at the origin. The semi-major axis is along the X axis and the semi-minor
    # axis along the Y axis. We need to rotate and translate these points to be in the
    # correct position so we can select the starting point consistently
    for point in range(sampled_points.shape[1]):
        # Rotate the point using the rotation matrix of the ellipse
        sampled_points[:, point] = rotation_matrix @ sampled_points[:, point]

        # Translate the point using the center of the ellipse
        sampled_points[:, point] = sampled_points[:, point] + center_2D

    # Plot the samples. The plotting code will translate and rotate the ellipse, but we
    # have to translate and rotate the points ourselves (which we have done above)
    if do_plotting:
        figure, figure_axes = plt.subplots(figsize = plt.figaspect(1))
        plotting.plotPoints2D(figure_axes, sampled_points, size = 50, alpha = 0.2)
        plotting.plotPoint2D(figure_axes, center_2D, 250, "red", 1.0, 'x')
        plotting.plotEllipse(
            figure_axes,
            center_2D,
            axes_lengths,
            rotation_matrix
        )
        plt.title("Sampled points along the ellipse")
        plt.show()

    # Find the starting point on the ellipse using the SSB vector
    starting_point_index = findStartingPoint(
        ssb_normal,
        center_3D,
        median_velocity,
        center_2D,
        sampled_points,
        do_plotting
    )

    # Reorder the samples so that the starting point is first in the list
    # First copy the points from the starting point to the end of the old samples list to
    # the start of the new list
    reordered_sampled_points = np.zeros(sampled_points.shape)
    new = 0
    for old in range(starting_point_index, sampled_points.shape[1]):
        reordered_sampled_points[:, new] = sampled_points[:, old]
        new += 1

    # Next copy the points from the start of the old samples list to the starting point
    # of the new samples list
    for old in range(0, starting_point_index):
        reordered_sampled_points[:, new] = sampled_points[:, old]
        new += 1

    # Plot the reordered samples, marking the first and last sample points
    if do_plotting:
        figure, figure_axes = plt.subplots(figsize = plt.figaspect(1))
        plotting.plotPoint2D(
            figure_axes,
            reordered_sampled_points[:, 0],
            size = 80,
            color = "red",
            edgecolor = "black",
            alpha = 0.6
        )
        plotting.plotPoint2D(
            figure_axes,
            reordered_sampled_points[:, reordered_sampled_points.shape[1] - 1],
            size = 80,
            color = "green",
            edgecolor = "black",
            alpha = 0.6
        )
        plotting.plotPoints2D(
            figure_axes,
            reordered_sampled_points,
            size = 50,
            alpha = 0.2
        )
        plotting.plotPoint2D(figure_axes, center_2D, 250, "red", 1.0, 'x')
        plotting.plotEllipse(
            figure_axes,
            center_2D,
            axes_lengths,
            rotation_matrix
        )
        plt.title("Reordered sampled points along the ellipse. First sample is red and last is green")
        plt.show()

    # Return the ellipse samples points in their correct order in 2D space
    return reordered_sampled_points


def calcTextureCoordinates(ellipse_samples):
    """
    Calculate the texture coordinates for each sample point on the ellipse

    Input:
        ellipse_samples: The sampled points along the ellipse in 2D space
    Output:
        texture_coordinates: The texture coordinates for each sample point
        range_x: The min and max x value of the ellipse sample points
        range_y: The min and max y value of the ellipse sample points
    """

    # Find the largest x and y value to set the texture coordinates between 0 and 1
    max_x = np.max(ellipse_samples[0, :])
    min_x = np.min(ellipse_samples[0, :])
    range_x = [min_x, max_x]

    max_y = np.max(ellipse_samples[1, :])
    min_y = np.min(ellipse_samples[1, :])
    range_y = [min_y, max_y]

    # Calculate the texture coordinate for each sample point
    texture_coordinates = np.zeros((2, ellipse_samples.shape[1]))
    for sample in range(ellipse_samples.shape[1]):
        # Texture coordinates
        u = (ellipse_samples[0, sample] - min_x) / (max_x - min_x)
        v = (ellipse_samples[1, sample] - min_y) / (max_y - min_y)

        # Invert u coordinate since OpenSpace uses a different coordinate system for
        # textures
        u = 1.0 - u

        texture_coordinates[:, sample] = np.array([u, v])

    return texture_coordinates, range_x, range_y


def calcDensities(ellipse_samples, intersection_points):
    """
    Calculate a density value for each sample point on the ellipse based on the distance
    to all intersection points.
    Input:
        ellipse_samples: The sampled points along the ellipse in 2D space
        intersection_points: The intersection points of the variants on the 2D plane
    Output:
        densities: The density value for each sample point
    """

    # For the density calculation, we accumilate the distances from each sample point to
    # all intersection points. The density is then the inverse of this accumilated
    # distance and normalized over this ellipse.
    min_accumilated_distance = np.inf
    max_accumilated_distance = -np.inf

    # Calculate the density for each sample point
    densities = []
    for sample in range(ellipse_samples.shape[1]):
        # Accumilate the distance from this sample point to all intersection points
        accumilated_distance = 0.0
        for point in range(intersection_points.shape[1]):
            accumilated_distance += np.linalg.norm(
                ellipse_samples[:, sample] - intersection_points[:, point]
            )
        densities.append(accumilated_distance)

        # Update min and max values
        if (accumilated_distance > max_accumilated_distance):
            max_accumilated_distance = accumilated_distance
        if (accumilated_distance < min_accumilated_distance):
            min_accumilated_distance = accumilated_distance

    # Loop over all samples again to normalize the density values and invert it, so
    # larger distances gives a smaller density
    for d in range(len(densities)):
        # Normalize the density value between 0 and 1
        densities[d] = (densities[d] - min_accumilated_distance) / (
            max_accumilated_distance - min_accumilated_distance
        )

        # Invert the density value
        densities[d] = 1.0 - densities[d]

    return densities


def calcStatistics(intersection_points, ellipse_rotation):
    """
    Caclulate some statistics (standard deviation and variance) for the points on the 2D
    plane. Make sure to use double floats for the calculations to get higher accuracy. 
    Input:
        intersection_points: The intersection points of the variants on the 2D plane
        ellipse_rotation: The rotation matrix of the ellipse on the 2D plane
    Output:
        standard_deviation: The standard deviation of the intersection points along
                            each ellipse axis
        variance: The variance of the intersection points along each ellipse axis
    """

    # Rotate the intersection points, so the ellipse axes are aligned with the x and
    # y axis. TODO: Invert the rotation matrix
    rotated_intersection_points = ellipse_rotation.T @ intersection_points

    # Standard deviation of the intersection point positions
    standard_deviation = np.std(rotated_intersection_points, axis = 1, dtype = np.float64)
    print("Standard deviation:", standard_deviation)

    # Variance of the intersection point positions
    variance = np.var(rotated_intersection_points, axis = 1, dtype = np.float64)
    print("Variance:", variance)

    # Check if the points are in a gaussian distribution or not
    is_gaussian = util.isGaussian(intersection_points)
    print("Is Gaussian:", is_gaussian)
    if is_gaussian:
        is_gaussian = int(1)
    else:
        is_gaussian = int(0)

    return standard_deviation, variance, is_gaussian


def checkForOutliers(intersection_points, ellipse_range_x, ellipse_range_y):
    """
    Check if there are any outliers of the variant intersection points on the 2D plane. An
    outlier is defined as a point that lays outside of the bounding box of the ellipse.
    An outlier can occur when the MVEE calculation fails to encase all points.

    Input:
        intersection_points: The intersection points of the variants on the 2D plane
        ellipse_range_x: The min and max x value of the ellipse sample points
        ellipse_range_y: The min and max y value of the ellipse sample points
    Output:
        has_outlier: 1 if there is at least one outlier, 0 otherwise
    """

    # Check if any point is outside of the bounding box of the ellipse
    for point in range(intersection_points.shape[1]):
        x = intersection_points[0, point]
        y = intersection_points[1, point]

        has_outlier = int(0)
        num_outliers = 0
        if (x < ellipse_range_x[0] or
            x > ellipse_range_x[1] or
            y < ellipse_range_y[0] or
            y > ellipse_range_y[1]):
            print("Point outside ellipse found at [x, y]", intersection_points[:, point])
            has_outlier = int(1)
            num_outliers += 1

    if has_outlier:
        print("Number of outliers found:", num_outliers)

    return has_outlier


def normalizeEllipsoid(ellipsoid, offsets, scaling_factors):
    """
    
    """

    # Start with the center point
    center = ellipsoid.center
    center[0] = (ellipsoid.center[0] - offsets[0]) / scaling_factors[0]
    center[1] = (ellipsoid.center[1] - offsets[1]) / scaling_factors[1]
    center[2] = (ellipsoid.center[2] - offsets[2]) / scaling_factors[2]

    # The axes of the ellispoid needs to be scaled according to the scaling factors
    # applied during normalization of the full foint cloud. This is a bit more complicated
    # since each axis needs to be scaled in each direction seperatly.
    # First make sure the axes are in their acurate length
    major_axis = ellipsoid.axes[0]
    major_axis = major_axis / np.linalg.norm(major_axis)
    major_axis = major_axis * ellipsoid.axes_lengths[0]

    middle_axis = ellipsoid.axes[1]
    middle_axis = middle_axis / np.linalg.norm(middle_axis)
    middle_axis = middle_axis * ellipsoid.axes_lengths[1]

    minor_axis = ellipsoid.axes[2]
    minor_axis = minor_axis / np.linalg.norm(minor_axis)
    minor_axis = minor_axis * ellipsoid.axes_lengths[2]

    # Then scale each axis in the x, y and z direction seperatly with the scaling factors
    major_axis[0] = major_axis[0] / scaling_factors[0]
    major_axis[1] = major_axis[1] / scaling_factors[1]
    major_axis[2] = major_axis[2] / scaling_factors[2]

    middle_axis[0] = middle_axis[0] / scaling_factors[0]
    middle_axis[1] = middle_axis[1] / scaling_factors[1]
    middle_axis[2] = middle_axis[2] / scaling_factors[2]

    minor_axis[0] = minor_axis[0] / scaling_factors[0]
    minor_axis[1] = minor_axis[1] / scaling_factors[1]
    minor_axis[2] = minor_axis[2] / scaling_factors[2] 

    # Then measure the new length of the axes and that is the normalized axes lengths
    axes_lengths = ellipsoid.axes_lengths
    axes_lengths[0] = np.linalg.norm(major_axis)
    axes_lengths[1] = np.linalg.norm(middle_axis)
    axes_lengths[2] = np.linalg.norm(minor_axis)

    # Then the new rotation matrix can be constructed with the new axes in unit length
    major_axis = major_axis / np.linalg.norm(major_axis)
    middle_axis = middle_axis / np.linalg.norm(middle_axis)
    minor_axis = minor_axis / np.linalg.norm(minor_axis)

    rotaion_matrix = np.array([major_axis, middle_axis, minor_axis])
    rotation_matrix = rotaion_matrix.T

    return Ellipsoid(
        ellipsoid.ellipsoid_matrix,
        center, 
        ellipsoid.axes,
        axes_lengths,
        rotation_matrix 
    )


def createEllipse(data, time_step, ssb_normal, texture_directory, configuration):
    """
    Create just one ellipse in the tube

    Input:
        data: The VariantData object with all of the variant data
        time_step: The timestep index to create the ellipse for
        num_ellipse_samples: The number of samples to take on the ellipse
        ssb_normal: The SSB vector in 3D world space
        save_textures: Whether to save textures for this timestep or not
        do_plotting: Whether to do debug plotting or not
    Output:
        ellipse: One ellipse data object for the given timestep
        ellipsoid: One ellipsoid that encapsulated the 3D points for the given timestep
    """
    # Configuration parameters
    do_plotting = configuration["do_plotting"]
    num_ellipse_samples = configuration["num_polygon_samples"]
    save_textures = configuration["save_textures"]
    texture_resolution = configuration["texture_resolution"]

    print("\nTime", data.times[time_step])
    print("Time step number", time_step)
    
    # Get the variant coordinate list for this timestep. The coordinates are in meters 
    # and relative the SUN (TODO: Or SSB need to check that)
    coordinates = data.variants_coordinates[time_step].T

    # Plot the points for this timestep
    if do_plotting:
        figure = plt.figure(figsize = plt.figaspect(1))
        axes = figure.add_subplot(projection = '3d')
        plotting.plotPoints3D(axes, coordinates, is_normalized = False)
        plt.title("Plot of non-normalized coordinates")
        plt.show()

    # Normalize the points to be between 0 and 1 for better numerical stability
    normalized_coordinates, offsets, scaling_factors = util.normalizePoints(coordinates)

    # Plot the normalized points for this timestep
    if do_plotting:
        figure = plt.figure(figsize = plt.figaspect(1))
        axes = figure.add_subplot(projection = '3d')
        plotting.plotPoints3D(axes, normalized_coordinates)
        plt.title("Plot of normalized coordinates")
        plt.show()

    # Create an ellipsoid data objects with all of the ellipsoid features
    ellipsoid = createEllipsoidFromMatrix(
        coordinates,
        data.covariances[time_step],
        do_plotting,
        True
    )

    # Make sure the original ellipsoid does not get normalized, create a deep copy
    normalized_ellipsoid = deepcopy(ellipsoid)
    normalized_ellipsoid = normalizeEllipsoid(
        normalized_ellipsoid,
        offsets,
        scaling_factors
    )

    if do_plotting:
        # Plot the non-normalized points and the ellipsoid as a transparent 3D shape
        figure = plt.figure(figsize = plt.figaspect(1))
        figure_axes = figure.add_subplot(projection = '3d')
        plotting.plotPoints3D(figure_axes, coordinates, alpha = 0.6)
        plotting.plotPoint3D(
            figure_axes,
            ellipsoid.center,
            250,
            "red",
            1.0,
            'x',
            is_normalized = False
        )
        plotting.plotEllipsoid(
            figure_axes,
            ellipsoid.center,
            ellipsoid.axes_lengths,
            ellipsoid.rotation_matrix,
            is_normalized = False
        )
        plt.title("Plot of the non-normalized ellipsoid and the points it encases")
        figure_axes.set_xlim(offsets[0], scaling_factors[0] + offsets[0])
        figure_axes.set_ylim(offsets[1], scaling_factors[1] + offsets[1])
        figure_axes.set_zlim(offsets[2], scaling_factors[2] + offsets[2])
        plt.show()
        
        # Plot the normalized points and the ellipsoid as a transparent 3D shape
        figure.clear()
        figure = plt.figure(figsize = plt.figaspect(1))
        figure_axes = figure.add_subplot(projection = '3d')
        plotting.plotPoints3D(figure_axes, normalized_coordinates, alpha = 0.6)
        plotting.plotPoint3D(
            figure_axes,
            normalized_ellipsoid.center,
            250,
            "red",
            1.0,
            'x'
        )
        plotting.plotEllipsoid(
            figure_axes,
            normalized_ellipsoid.center,
            normalized_ellipsoid.axes_lengths,
            normalized_ellipsoid.rotation_matrix
        )
        plt.title("Plot of the normalized ellipsoid and the points it encases")
        plt.show()

    # Transform all points to be on the a plane that is perpendicular to the direction
    # towards the Sun. The median velocity vector is used as the normal of this plane.
    velocities = data.variants_velocities[time_step].T
    median_velocity = np.median(velocities, axis = 1)

    # Make a rough check if there are outliers in the data for this timestep by comparing
    # the mean and median velocity directions
    # TODO: Make this angle tolerance configurable
    angle_tolerance = 0.5
    is_deviating = int(0)
    median_velocity_normalized = median_velocity / np.linalg.norm(median_velocity)
    mean_velocity = np.mean(velocities, axis = 1)
    mean_velocity_normalized = mean_velocity / np.linalg.norm(mean_velocity)
    angle = np.arccos(np.dot(mean_velocity_normalized, median_velocity_normalized))
    if angle > np.deg2rad(angle_tolerance):
        is_deviating = int(1)
        print("\033[41mWarning:\033[0m " \
            "Mean and median velocities differ by more than the specified angle " \
            "tolerance"
        )
        print("Mean velocity:", mean_velocity_normalized)
        print("Median velocity:", median_velocity_normalized)
        print("Angle difference (degrees):", np.rad2deg(angle))

    # Plot the points and the axes of the ellipsoid
    if do_plotting:
        figure = plt.figure(figsize = plt.figaspect(1))
        axes = figure.add_subplot(projection = '3d')
        plotting.plotPoints3D(axes, normalized_coordinates)
        plotting.plotPoint3D(axes, normalized_ellipsoid.center, 250, "red", 1.0, 'x')
        plotting.plotVector3D(
            axes,
            normalized_ellipsoid.center,
            mean_velocity,
            "red"
        )
        plotting.plotVectors3D(
            axes,
            normalized_coordinates,
            velocities
        )
        plt.title("Plot of the velocities and the mean velocity")
        plt.show()

    # Transform all points to be on this plane, i.e. a slice of the tube going around the
    # Sun. Then transform this plane to be on the XY plane, giving a 2D problem
    intersections_2D, time_lags = getPointsOnSlice(
        normalized_coordinates,
        velocities,
        median_velocity,
        normalized_ellipsoid.center,
        do_plotting
    )
    
    # Solve the 2D problem and create a minimum enclosing ellipse around the 2D points on
    # the XY plane
    ellipse = createEllipsoid(intersections_2D, do_plotting, False)

    # Sample the ellipse to create the polygon that make up the tube
    samples_2D = sampleEllipse(
        num_ellipse_samples,
        ellipse.center,
        ellipse.axes_lengths,
        ellipse.rotation_matrix,
        ssb_normal,
        normalized_ellipsoid.center,
        median_velocity,
        do_plotting
    )

    # Texture coordinate for the textures on the cutplane of the tube
    texture_coordinates, range_x, range_y = calcTextureCoordinates(samples_2D)

    # Check if there are any points on the 2D plane that are outside of the ellipse
    has_outlier = checkForOutliers(intersections_2D, range_x, range_y)
    is_deviating = max(is_deviating, has_outlier)

    # Density values for each sample point on the ellipse
    densities = calcDensities(samples_2D, intersections_2D)

    # Calculate standard deviation and variance of the intersection points along the
    # ellipse axes
    standard_deviation, variance, is_gaussian = calcStatistics(
        intersections_2D,
        ellipse.rotation_matrix
    )

    # Create textures with more detailed data for this timestep
    saved_texture = ""
    if save_textures:
        # Generate all types of textures into one
        saved_texture = texture.generateTexture(
            range_x,
            range_y,
            texture_resolution,
            texture_directory,
            time_step,
            texture_coordinates,
            intersections_2D,
            time_lags            
        )

    # Transform the ellipse samples back to the original 3D space
    samples_3D = util.invTransformPointsToXYPlane(
        samples_2D,
        normalized_ellipsoid.center,
        median_velocity,
        do_plotting
    )

    # Plot the final ellipse samples in 3D space related to the original ellipsoid
    if do_plotting:
        figure = plt.figure(figsize = plt.figaspect(1))
        figure_axes = figure.add_subplot(projection = '3d')
        plotting.plotPoints3D(figure_axes, samples_3D, alpha = 0.6)
        plotting.plotPoint3D(
            figure_axes,
            normalized_ellipsoid.center,
            250,
            "red",
            1.0,
            'x'
        )
        plotting.plotEllipsoid(
            figure_axes,
            normalized_ellipsoid.center,
            normalized_ellipsoid.axes_lengths,
            normalized_ellipsoid.rotation_matrix
        )
        plt.title("Plot of the sampled 2D ellipse and the original 3D ellipsoid")
        plt.show()

    # Then inverse normalize everything to be in the correct positions in the original
    # space
    samples_3D_non_normalized = util.invNormalizePoints(
        samples_3D,
        offsets,
        scaling_factors
    )

    # Plot the original points and the ellipse sample points in non-normalized space
    # together with the non-normalized ellipsoid
    if do_plotting:
        figure = plt.figure(figsize = plt.figaspect(1))
        figure_axes = figure.add_subplot(projection = '3d')
        plotting.plotPoints3D(figure_axes, coordinates, is_normalized = False)
        plotting.plotPoints3D(
            figure_axes,
            samples_3D_non_normalized,
            color = "red",
            is_normalized = False
        )
        plotting.plotEllipsoid(
            figure_axes,
            ellipsoid.center,
            ellipsoid.axes_lengths,
            ellipsoid.rotation_matrix,
            is_normalized = False
        )
        plt.title("Plot of the coordinates, the ellipse samples and the ellipsoid in non-normalized space")
        figure_axes.set_xlim(offsets[0], scaling_factors[0] + offsets[0])
        figure_axes.set_ylim(offsets[1], scaling_factors[1] + offsets[1])
        figure_axes.set_zlim(offsets[2], scaling_factors[2] + offsets[2])
        plt.show()

    # Create a list of EllipseSamplePoint data objects for each sample point to store the
    # data in an organized manner
    ellipse_sample_points = []
    for sample in range(samples_3D.shape[1]):
        # Create the EllipseSamplePoint data object
        ellipse_sample_points.append(EllipseSamplePoint(
            samples_2D[:, sample],
            samples_3D_non_normalized[:, sample],
            texture_coordinates[:, sample],
            densities[sample],
            standard_deviation,
            variance,
            is_gaussian,
            is_deviating
        ))

    # Return the full ellipse for this timestep, with its samples points and meta data
    return ellipse_sample_points, ellipsoid, saved_texture


def createEllipses(variants_data, output_directory, configuration):
    """
    Take the input data and create a list of all ellipses that will be the base for the
    tube

    Input:
        data: Data object with the data from the input files
        num_ellipse_samples: The number of samples to take on each ellipse
        out_directory: The output directory to store the generated textures
        save_textures: Whether or not to save the generated textures
        texture_resolution: The resolution of the generated textures
        do_plotting: Whether or not to show plots during the calculations
    Output:
        A list of all ellipses to create the tube for the input data
    """

    # Create a directory to store the textures for each ellipse
    texture_directory = os.path.join(output_directory, "textures")
    os.makedirs(texture_directory, exist_ok = True)        

    # Get the normal of the solar system
    # The date should not make any difference
    # NOTE: There might be some slight rotation of the normal over time but this should
    # not affect the overall results
    ssb_normal = util.getSolarSystemNormal(["Jan 1, 2015"])

    # Loop over all timesteps
    time_ellipses = []
    for tube_part in variants_data:
        for t in range(tube_part.num_time_steps):
            # Create one ellipse and ellipsoid for this timestep
            ellipse_sample_points, ellipsoid, saved_texture = createEllipse(
                tube_part,
                t,
                ssb_normal,
                texture_directory,
                configuration
            )
            # TODO: Make the number of generated textures configurable and automatically
            # adjust when it comes to witing the tube file

            # Create the TubeEllipse data object for this timestep
            time_ellipse = TubeEllipse(
                tube_part.times[t],
                ellipsoid.center,
                saved_texture,
                ellipse_sample_points,
                ellipsoid
            )

            # Store the ellipse and ellipsoid
            time_ellipses.append(time_ellipse)

    return time_ellipses
    