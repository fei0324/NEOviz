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


def calcEllipsoidParameters(center, ellipsoid_matrix):
    """
    Calculate the ellipsoid parameters given the ellipsoid_matrix representation of a
    3D ellipsoid. The function would also work for a 2D ellipse.

    Input:
        ellipsoid_matrix: Matrix representation of a 3D ellipsoid (or 2D ellipse)

    Output:
        axes: The normalized axes directions of the ellipsoid, starting with the 
              semi-major axis, and lastly the semi-minor axis
        axes_lengths: The lengths of the ellipsoid axes in decending order (same as axes)
        rotation_matrix: The rotation matrix of the ellipsoid
    """

    # The eigenvalues and vectors of the ellipsoid matrix can be used to calculate the 
    # axes of the principle directions of the ellipsoid and their lengths. 
    eigenvalues, eigenvectors = np.linalg.eigh(ellipsoid_matrix)
    print("eigenvalues", eigenvalues)
    print("eigenvectors", eigenvectors)

    # The eigenvectors of the ellipsoid matrix are always gonna be in the same direction 
    # as the axes of the ellipsoid. The axis with the smallest absolute eigenvalue is the # semi-major axis. This is according to the principal axis theorem.
    # From: https://en.wikipedia.org/wiki/Ellipsoid and
    # https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
    axes = eigenvectors
    
    # The lengths of the axes can be claulated using the eigenvalues with the formula:
    # a = 1/(sqrt(eigenvalue[0]))
    # (From: https://en.wikipedia.org/wiki/Ellipsoid)
    axes_lengths = np.reciprocal(np.sqrt(eigenvalues))
    
    # Get the rotation
    rotation_matrix = axes.T

    # The np.linalg.eigh function returns the eigenvalues in ascending order. Meaning 
    # that the smallest eigenvalue is the first item, meaning that the cooresponding
    # major axis is the first item
    return axes, axes_lengths, rotation_matrix


def createEllipsoid(points, do_plotting, is_3d = True):
    """
    Create an ellipsoid that encases all of the given points.

    Input: 
        points: The input point cloud in 3D
    Output:
        Ellipsoid:
            ellipsoid_matrix: The matrix that representa this ellipsoid
            center: The center point of the ellipsoid
            axes: The normalized vectors that represent the directions of the axes of the 
                  ellipsoid
            axes_lengths: The lengths of the axes of the ellipsoid
            rotation_matrix: The rotation matrix of the ellipsoid
    """

    # Compute a minimum encasing ellipsoid for the points using mvee
    ellipsoid_matrix, center = calcMveeEllipsoid(points)

    # Get the ellipsoid characteristics to be able to draw the shape
    axes, axes_lengths, rotation_matrix = calcEllipsoidParameters(
        center,
        ellipsoid_matrix
    )
    
    # Plot the points together with the generated ellipsoid
    # TODO: The ellipsoid parameters seem off. Ellipsoid is much bigger than the range of the points
    if do_plotting:
        if is_3d:
            print("3D ellipsoid axes", axes)
            print("3D ellipsoid axes lengths", axes_lengths)
            print("3D ellipsoid rotation", rotation_matrix)

            plotting.plotPointsAndEllipsoid(
                points,
                center,
                axes,
                axes_lengths,
                rotation_matrix
            )
        else:
            print("2D ellipsoid axes", axes)
            print("2D ellipsoid axes lengths", axes_lengths)
            print("2D ellipsoid rotation", rotation_matrix)

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
        mean_velocity: The mean velocity of the variants. This is the normal of the plane 
                       that is perpendicular to the direction towards the Sun 
        ellipsoid_center: The center of the ellipsoid that encases all the points
        do_plotting: Whether debug plots should be done or not
    Output:
        transformed_points: The points on the 2D plane slice and transformed to the XY 
                            plane
        time_lags: The amount each variant is ahead/behind the plane slice
    """
    # Calculate the intersection point for each variant with the mean plane using their
    # velocity
    plane_variant_intersections = np.zeros(coordinates.shape)
    time_lags = np.zeros(coordinates.shape[0])
    for variant in range(coordinates.shape[0]):
        variant_coordinate = coordinates[:, variant]
        variant_velocity = velocities[:, variant]

        intersection_multiplier, intersection_coordinate = util.calcPlaneLineIntersection(
            mean_velocity,
            ellipsoid_center,
            variant_coordinate,
            variant_velocity
        )

        time_lags[variant] = intersection_multiplier
        plane_variant_intersections[:, variant] = intersection_coordinate

    # Plot the old and new points
    if do_plotting:
        plotting.plotPointsComp(coordinates, plane_variant_intersections)

    # Transform the 3D plane to the XY plane to convert this to a 2D problem
    transformed_intersections_2d = util.transformPointsToXYPlane(
        plane_variant_intersections,
        ellipsoid_center,
        mean_velocity
    )

    return transformed_intersections_2d, time_lags


def sampleEllipse(center, ):
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

    return None


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
    coordinates = data.variants_coordinates[time].T

    # Plot the points for this timestep
    if do_plotting:
        plotting.plotPoints(coordinates)

    # Create ellipsoid
    ellipsoid = createEllipsoid(coordinates, do_plotting, True)

    # Transform the points to be on the a plane that is perpendicular to the direction
    # towards the Sun. The mean velocity vector is used as the normal of this plane
    velocities = data.variants_velocities[time].T
    mean_velocity = np.mean(velocities, axis = 1)

    if do_plotting:
        plotting.plotPointsAndPlane(coordinates, ellipsoid.center, mean_velocity)

    intersections_2d, time_lags = getPointsOnSlice(
        coordinates,
        velocities,
        mean_velocity,
        ellipsoid.center,
        do_plotting
    )
    
    # Solve the 2D problem and create a minimum enclosing ellipse around the 2D points on
    # the XY plane
    ellipse = createEllipsoid(intersections_2d, do_plotting, False)

    # Sample the ellipse to create the polygon
    #sampleEllipse()

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
    