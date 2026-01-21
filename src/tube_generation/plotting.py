import matplotlib.pyplot as plt
import numpy as np


def plotPoint3D(axes, point, size = 10, color = "blue", alpha = 1.0, marker = 'o',
                edgecolor = "face", is_normalized = True):
    """
    Plot the given 3D point onto the given figure coordinates. The caller can then show
    or save the figure. Or they can add more plotting to the same figure.

    Input:
        axes: The figure axes to plot onto
        point: A 3D point to plot
        size: The size of the point to plot
        color: The color of the point to plot
        alpha: The alpha transparency of the point to plot
        marker: The marker style of the point to plot
        edgecolor: The edge color of the point to plot
        is_normalized: Whether the point is normalized between 0 and 1 or not
    """
    
    # Plot the point onto the figure
    points = np.zeros((3, 1))
    points[0, :] = point[0]
    points[1, :] = point[1]
    points[2, :] = point[2]
    plotPoints3D(
        axes,
        points,
        size,
        color,
        alpha,
        marker,
        edgecolor,
        is_normalized
    )


def plotPoints3D(axes, points, size = 10, color = "blue", alpha = 1.0, marker = 'o',
                 edgecolor = "face", is_normalized = True):
    """
    Plot the given 3D points onto the given figure axes. The caller can then show
    or save the figure. Or add more plots to the same figure. 

    Input:
        axes: The figure axes to plot onto
        points: A 3D point cloud with points to plot
        size: The size of the points to plot
        color: The color of the points to plot
        alpha: The alpha transparency of the points to plot
        marker: The marker style of the points to plot
        edgecolor: The edge color of the points to plot
        is_normalized: Whether the points are normalized between 0 and 1 or not
    """

    # Plot the points onto the figure
    axes.scatter(
        points[0, :],
        points[1, :],
        points[2, :],
        s = int(size),
        c = color,
        alpha = alpha,
        marker = marker,
        edgecolors = edgecolor
    )

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_zlabel("Z-Axis")

    # If we know that teh points are normalized we can set the axes limits directly
    if is_normalized:
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_zlim(0, 1)


def plotPoint2D(axes, point, size = 10, color = "blue", alpha = 1.0, marker = 'o',
                edgecolor = "face", is_normalized_neg = True):
    """
    Plot the given 2D point onto the given figure coordinates. The caller can then show
    or save the figure.

    Input:
        axes: The figure axes to plot onto
        point: A 2D point to plot
        size: The size of the point to plot
        color: The color of the point to plot
        alpha: The alpha transparency of the point to plot
        marker: The marker style of the point to plot
        edgecolor: The edge color of the point to plot
        is_normalized_neg: Whether the point is normalized between -1 and 1 or not
    """

    # Plot the point onto the figure
    points = np.zeros((2, 1))
    points[0, :] = point[0]
    points[1, :] = point[1]
    plotPoints2D(
        axes,
        points,
        size,
        color,
        alpha,
        marker,
        edgecolor,
        is_normalized_neg
    )
    

def plotPoints2D(axes, points, size = 10, color = "blue", alpha = 1.0, marker = 'o',
                 edgecolor = "face", is_normalized_neg = True):
    """
    Plot the given 2D points onto the given figure coordinates. The caller then can show
    or save the figure.

    Input:
        axes: The figure axes to plot onto
        points: A 2D point could of points to plot
        size: The size of the points to plot
        color: The color of the points to plot
        alpha: The alpha transparency of the points to plot
        marker: The marker style of the points to plot
        edgecolor: The edge color of the points to plot
        is_normalized_neg: Whether the points are normalized between -1 and 1 or not
    """

    # Plot the points onto the figure
    axes.scatter(
        points[0, :],
        points[1, :],
        s = int(size),
        c = color,
        alpha = alpha,
        marker = marker,
        edgecolors = edgecolor
    )

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")

    if is_normalized_neg:
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)


def plotAxes3D(figure_axes, origin, coordinate_axes, is_normalized = True):
    """
    Plot the given axes at the origin point in 3D

    Input:
        figure_axes: The figure axes to plot onto
        origin: A 3D point that is the origin of the axes
        coordinate_axes: A set of 3 axes vectors to plot
        is_normalized: Whether the sizes of the vectors are normalized between 0 and
        1 or not
    """

    # Plot the axes at the given origin point
    # The first axes is red
    figure_axes.quiver(
        *origin,
        coordinate_axes[0, 0],
        coordinate_axes[0, 1],
        coordinate_axes[0, 2],
        color = 'r'
    )
    # The second axes is green
    figure_axes.quiver(
        *origin,
        coordinate_axes[1, 0],
        coordinate_axes[1, 1],
        coordinate_axes[1, 2],
        color = 'g'
    )
    # The third axes is blue
    figure_axes.quiver(
        *origin,
        coordinate_axes[2, 0],
        coordinate_axes[2, 1],
        coordinate_axes[2, 2],
        color = 'b'
    )

    # Name the axes
    figure_axes.set_xlabel("X-Axis")
    figure_axes.set_ylabel("Y-Axis")
    figure_axes.set_zlabel("Z-Axis")

    # If we know that the points are normalized we can set the axes limits directly
    if is_normalized:
        figure_axes.set_xlim(0, 1)
        figure_axes.set_ylim(0, 1)
        figure_axes.set_zlim(0, 1)


def plotAxes2D(figure_axes, origin, coordinate_axes, is_normalized_neg = True):
    """
    Plot the given axes at the origin point in 2D

    Input:
        figure_axes: The figure axes to plot onto
        origin: A 2D point that is the origin of the axes
        coordinate_axes: A set of 2 axes vectors to plot
        is_normalized_neg: Whether the sizes of the vectors are normalized between -1 and
        1 or not
    """

    # Plot the axes at the given origin point
    # The first axes is red
    figure_axes.quiver(
        *origin,
        coordinate_axes[0, 0],
        coordinate_axes[0, 1],
        color = 'r'
    )
    # The second axes is green
    figure_axes.quiver(
        *origin,
        coordinate_axes[1, 0],
        coordinate_axes[1, 1],
        color = 'g'
    )

    # Name the axes and title
    figure_axes.set_xlabel("X-Axis")
    figure_axes.set_ylabel("Y-Axis")

    # If we know that the points are normalized we can set the axes limits directly
    if is_normalized_neg:
        figure_axes.set_xlim(-1, 1)
        figure_axes.set_ylim(-1, 1)


def plotVector2D(axes, origin, vector, color = "green", is_normalized = True):
    """
    Plot the given 2D vectors at the given origin onto the given figure
    axes.

    Input:
        axes: The figure axes to plot onto
        origin: A 3D point cloud of origins for the vectors
        vector: A list of 3D vectors to plot
        color: The color to plot the vectors with
        is_normalized: Whether the sizes of the vectors are normalized between 0 and
                       1 or not
    """

    # Plot the vectors
    axes.quiver(
        origin[0],
        origin[1],
        vector[0],
        vector[1],
        color = color
    )

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")

    # If we know that the vectors are in normalized range we can set the axes limits
    # directly
    if is_normalized:
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)


def plotVector3D(axes, origin, vector, color = "green", is_normalized = True):
    """
    Plot the given 3D vector at the given origin onto the given figure axes.

    Input:
        axes: The figure axes to plot onto
        origin: A 3D point that is the origin of the vector
        vector: A 3D vector to plot
        color: The color to plot the vector as
        is_normalized: Whether the sizes of the vectors are normalized between 0 and
                       1 or not
    """

    # Convert to the shape that the plotVectors3D function expects
    origins = np.zeros((3, 1))
    origins[0, :] = origin[0]
    origins[1, :] = origin[1]
    origins[2, :] = origin[2]

    vectors = np.zeros((3, 1))
    vectors[0, :] = vector[0]
    vectors[1, :] = vector[1]
    vectors[2, :] = vector[2]

    # Plot the vector using the general function
    plotVectors3D(axes, origins, vectors, color, 1, is_normalized)


def plotVectors3D(axes, origins, vectors, color = "green", every_nth = 100,
                  is_normalized = True):
    """
    Plot the given 3D vectors ar their respective given origin onto the given figure
    axes. If there are too many vectors to plot, only every nth element can be plotted.

    Input:
        axes: The figure axes to plot onto
        origins: A 3D point cloud of origins for the vectors
        vectors: A list of 3D vectors to plot
        color: The color to plot the vectors with
        every_nth: Plot only every nth vector to reduce clutter
        is_normalized: Whether the sizes of the vectors are normalized between 0 and
                       1 or not
    """

    # Plot the vectors
    axes.quiver(
        origins[0, ::int(every_nth)],
        origins[1, ::int(every_nth)],
        origins[2, ::int(every_nth)],
        vectors[0, ::int(every_nth)],
        vectors[1, ::int(every_nth)],
        vectors[2, ::int(every_nth)],
        color = color
    )

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_zlabel("Z-Axis")

    # If we know that the vectors are in normalized range we can set the axes limits
    # directly
    if is_normalized:
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_zlim(0, 1)


def plotEllipsoid(axes, center, axes_lengths, rotation_matrix, is_normalized = True):
    """
    Plot the ellipsoid with the given parameters

    Input:
        axes: The figure axes to plot onto
        center: The center point in 3D 
        axes_lengths: The axes lengths of the ellipsoid, semi-major axis first,
                      semi-minor axis last
        rotation_matrix: The rotation matrix for the whole ellipsoid
    """

    # Create a set of all spherical angles
    theta_theta = np.linspace(0, 2*np.pi, 100)
    phi_phi = np.linspace(0, np.pi, 100)

    # Cartesian coordinates that correspond to the spherical angles on the standard 
    # ellipsoid. Following the equation of an ellipsoid, from:
    # https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib)
    # and https://en.wikipedia.org/wiki/Ellipsoid
    xx = axes_lengths[0] * np.outer(np.sin(phi_phi), np.cos(theta_theta))
    yy = axes_lengths[1] * np.outer(np.sin(phi_phi), np.sin(theta_theta))
    zz = axes_lengths[2] * np.outer(np.cos(phi_phi), np.ones_like(theta_theta))

    # Since we have a standard ellipsoid we need to transform it to have the parameters
    # that were given. First apply the rotation
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            v = np.array([xx[i][j], yy[i][j], zz[i][j]])
            new_v = rotation_matrix @ v

            xx[i][j] = new_v[0]
            yy[i][j] = new_v[1]
            zz[i][j] = new_v[2]      

    # Then translate it to the center point
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            v = np.array([xx[i][j], yy[i][j], zz[i][j]])
            new_v = v + center

            xx[i][j] = new_v[0]
            yy[i][j] = new_v[1]
            zz[i][j] = new_v[2]  

    # Plot the ellipsoid
    axes.plot_surface(xx, yy, zz, rstride = 4, cstride = 4, color = 'g', alpha = 0.3)

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_zlabel("Z-Axis")

    # Normalize the axes if we know the ellipsoid is within normalized range
    if is_normalized:
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.set_zlim(0, 1)


def plotEllipse(axes, center, axes_lengths, rotation_matrix, is_normalized_neg = True):
    """
    Plot the given ellipse with the given parameters

    Input:
        axes: The figure axes to plot onto
        center: The center point in 2D 
        axes_lengths: The axes lengths of the ellipse, semi-major axis first, semi-minor
                      axis last
        rotation_matrix: The rotation matrix for the whole ellipse
    """

    # Set of all polar angles:
    rr = np.linspace(0, 2*np.pi, 100)

    # Cartesian coordinates that correspond to the polar angles on the ellipse:
    # This is the equation of a standard ellipse, from:
    # https://stackoverflow.com/questions/10952060/plot-ellipse-with-matplotlib-pyplot
    # and https://en.wikipedia.org/wiki/Ellipse
    xx = axes_lengths[0]*np.cos(rr)
    yy = axes_lengths[1]*np.sin(rr)

    # The ellipse is now in standard form, we need to transform it in order the have the
    # given parameters. First apply the rotation
    for i in range(xx.shape[0]):
        v = np.array([xx[i], yy[i]])
        new_v = rotation_matrix @ v

        xx[i] = new_v[0]
        yy[i] = new_v[1]

    # Then translate
    for i in range(xx.shape[0]):
        v = np.array([xx[i], yy[i]])
        new_v = v + center

        xx[i] = new_v[0]
        yy[i] = new_v[1]

    # Plot the ellipse
    plt.plot(xx, yy, color = 'g', alpha = 0.3)

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")

    # Normalize the axes if we know the ellipse is within normalized range
    if is_normalized_neg:
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)


def plotPlane(axes, point, normal):
    """
    Plot a plane defined by the given point and normal vector. The plane is assumed to be
    within normalized range [0, 1]

    Input:
        axes: The figure axes to plot onto
        point: A 3D point on the plane
        normal: The normal vector of the plane
    """

    # Prepare the plane, from:
    # https://stackoverflow.com/questions/3461869/plot-a-plane-based-on-a-normal-vector-and-a-point-in-matlab-or-matplotlib
    # The plane formula is a*x + b*y + c*z + d = 0
    # The normal is (a, b, c), we need to calculate d
    a = normal[0]
    b = normal[1]
    c = normal[2]
    d = -(a * point[0] + b * point[1] + c * point[2])

    # Set of all x and y values for the plane
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100))

    # Calculate the corresponding z-value for each xx and yy
    zz = point[2] - (a * (xx - point[0]) + b * (yy - point[1])) / c

    # plot the plane
    axes.plot_surface(xx, yy, zz)

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_zlabel("Z-Axis")

    # Normalize the axes if we know the plane is within normalized range
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_zlim(0, 1)
