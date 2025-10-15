import matplotlib.pyplot as plt
import numpy as np


def plotPoints(points):
    """
    Plot the given points

    Input:
        points: A 3D point could of points to plot
    """

    # Prepare the figure
    figure = plt.figure(figsize = plt.figaspect(1))
    axes = figure.add_subplot(projection = '3d')

    # Plot the points onto the figure
    axes.scatter(points[0, :], points[1, :], points[2, :], c = 'blue')

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_zlabel("Z-Axis")
    plt.title("Plot of points")

    plt.show()


def plotPoints2D(points):
    """
    Plot the given points

    Input:
        points: A 3D point could of points to plot
    """

    # Prepare the figure
    figure, axes = plt.subplots(figsize = plt.figaspect(1))

    # Plot the points onto the figure
    axes.scatter(points[0, :], points[1, :], c = 'blue')

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    plt.title("Plot of points")

    plt.show()


def plotPointsAndAxes(points, center, axes_in):
    """
    Plot the given points, the center of the generated ellipsoid and its eigenvectors

    Input:
        points: A 3D point could of points to plot
        center: A 3D point that is the center of the ellipsoid
        points: A set of 3 axes vectors that are the eigenvectors of the ellipsoid
    """

    # Prepare the figure
    figure = plt.figure(figsize = plt.figaspect(1))
    axes = figure.add_subplot(projection = '3d')

    # Plot the points onto the figure
    axes.scatter(
        points[0, :],
        points[1, :],
        points[2, :],
        s = 10,
        c = 'blue',
        alpha = 0.6
    )

    # Plot the center point to be destinct from the other points
    axes.scatter(
        center[0],
        center[1],
        center[2],
        s = 250,
        c = 'red',
        marker = 'x',
        alpha = 1.0
    )

    # Plot the axes at the center point
    axes.quiver(
        *center,
        axes_in[0, 0],
        axes_in[0, 1],
        axes_in[0, 2],
        color = 'r'
    )

    axes.quiver(
        *center,
        axes_in[1, 0],
        axes_in[1, 1],
        axes_in[1, 2],
        color = 'g'
    )

    axes.quiver(
        *center,
        axes_in[2, 0],
        axes_in[2, 1],
        axes_in[2, 2],
        color = 'b'
    )

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_zlabel("Z-Axis")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_zlim(0, 1)
    plt.title("Plot of points and the 3 axes vectors")

    plt.show()


def plotPointsAndAxes2D(points, center, axes_in):
    """
    Plot the given points, the center of the generated ellipse and its eigenvectors

    Input:
        points: A 2D point could of points to plot
        center: A 2D point that is the center of the ellipsoid
        points: A set of 2 axes vectors that are the eigenvectors of the ellipsoid
    """

    # Prepare the figure
    figure, axes = plt.subplots(figsize = plt.figaspect(1))

    # Plot the points onto the figure
    axes.scatter(
        points[0, :],
        points[1, :],
        s = 10,
        c = 'blue',
        alpha = 0.6
    )

    # Plot the center point to be destinct from the other points
    axes.scatter(
        center[0],
        center[1],
        s = 250,
        c = 'red',
        marker = 'x',
        alpha = 1.0
    )

    # Plot the axes at the center point
    axes.quiver(
        *center,
        axes_in[0, 0],
        axes_in[0, 1],
        color = 'r'
    )

    axes.quiver(
        *center,
        axes_in[1, 0],
        axes_in[1, 1],
        color = 'b'
    )

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    plt.title("Plot of points and the 2 axes vectors")

    plt.show()


def plotPointsAndVectors(points, center, vectors, mean_vector):
    """
    Plot the given points, the center of the generated ellipsoid and its eigenvectors

    Input:

    """

    # Prepare the figure
    figure = plt.figure(figsize = plt.figaspect(1))
    axes = figure.add_subplot(projection = '3d')

    # Plot the points onto the figure
    axes.scatter(
        points[0, :],
        points[1, :],
        points[2, :],
        s = 10,
        c = 'blue',
        alpha = 0.6
    )

    # Plot the vectors
    axes.quiver(
        points[0, ::100],
        points[1, ::100],
        points[2, ::100],
        vectors[0, ::100],
        vectors[1, ::100],
        vectors[2, ::100],
        color = 'g'
    )

    # Plot the center point to be destinct from the other points
    axes.scatter(
        center[0],
        center[1],
        center[2],
        s = 250,
        c = 'r',
        alpha = 1.0
    )

    # Plot the mean vector at the center
    axes.quiver(
        *center,
        mean_vector[0],
        mean_vector[1],
        mean_vector[2],
        color = 'r'
    )

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_zlabel("Z-Axis")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_zlim(0, 1)
    plt.title("Plot of points and vectors with mean vector at the center")

    plt.show()


def plotPointsAndEllipsoid(points, center, ellipsoid_axes, ellipsoid_axes_lengths,
                           rotation_matrix):
    """
    Plot the given points with the ellipsoid that encases them

    Input:
        points: A 3D point could of points to plot
        center: The center point of the point cloud
        ellipsoid_axes: The axes directions of the ellipsoid to plot
        ellipsoid_axes_lengths: The axes lengths of the ellipsoid to plot
        rotation_matrix: The rotation of the ellipsoid
    """

    # Prepare the figure
    figure = plt.figure(figsize = plt.figaspect(1))
    axes = figure.add_subplot(projection = '3d')

    # Plot the points
    axes.scatter(
        points[0, :],
        points[1, :],
        points[2, :],
        s = 10,
        c = 'blue',
        alpha = 0.6
    )

    # Plot the center point to be destinct from the other points
    axes.scatter(
        center[0],
        center[1],
        center[2],
        s = 250,
        c = 'red',
        marker = 'x',
        alpha = 1.0
    )

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # TODO: What if we take a unit circle and use the ellipsoid matrix to transform it?
    # Will it create our ellipsoid?

    # Cartesian coordinates that correspond to the spherical angles on the ellipsoid:
    # This is the equation of an ellipsoid, from:
    # https://stackoverflow.com/questions/7819498/plotting-ellipsoid-with-matplotlib)
    # and https://en.wikipedia.org/wiki/Ellipsoid
    x = ellipsoid_axes_lengths[0] * np.outer(np.sin(v), np.cos(u))
    y = ellipsoid_axes_lengths[1] * np.outer(np.sin(v), np.sin(u))
    z = ellipsoid_axes_lengths[2] * np.outer(np.cos(v), np.ones_like(u))

    # Apply the rotation of the ellipsoid
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v = np.array([x[i][j], y[i][j], z[i][j]])
            new_v = rotation_matrix @ v

            x[i][j] = new_v[0]
            y[i][j] = new_v[1]
            z[i][j] = new_v[2]      

    # Translate the ellipsoid
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            v = np.array([x[i][j], y[i][j], z[i][j]])
            new_v = v + center

            x[i][j] = new_v[0]
            y[i][j] = new_v[1]
            z[i][j] = new_v[2]  

    # Plot the ellipsoid
    axes.scatter(center[0], center[1], center[2], s = 100, c = 'red')
    axes.plot_surface(x, y, z, rstride = 4, cstride = 4, color = 'g', alpha = 0.3)

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_zlabel("Z-Axis")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_zlim(0, 1)
    axes.set_title("Plot of points and the ellipsoid that encases them")

    plt.show()


def plotPointsAndEllipse(points, center, ellipse_axes, ellipse_axes_lengths,
                        rotation_matrix):
    """
    Plot the given points with the ellipsoid that encases them

    Input:
        points: A 3D point could of points to plot
        center: The center point of the point cloud
        ellipsoid_axes: The axes directions of the ellipsoid to plot
        ellipsoid_axes_lengths: The axes lengths of the ellipsoid to plot
        rotation_matrix: The rotation of the ellipsoid
    """

    # Prepare the figure
    figure, axes = plt.subplots(figsize = plt.figaspect(1))

    # Plot the points
    axes.scatter(
        points[0, :],
        points[1, :],
        s = 10,
        c = 'blue',
        alpha = 0.6
    )

    # Plot the center point to be destinct from the other points
    axes.scatter(
        center[0],
        center[1],
        s = 250,
        c = 'red',
        marker = 'x',
        alpha = 1.0
    )

    # Set of all radial angles:
    rr = np.linspace(0, 2 * np.pi, 100)

    # Cartesian coordinates that correspond to the radial angles on the ellipse:
    # This is the equation of an ellipse, from:
    # https://stackoverflow.com/questions/10952060/plot-ellipse-with-matplotlib-pyplot
    # and https://en.wikipedia.org/wiki/Ellipse
    xx = ellipse_axes_lengths[0]*np.cos(rr)
    yy = ellipse_axes_lengths[1]*np.sin(rr)

    # Apply the rotation of the ellipsoid
    for i in range(xx.shape[0]):
        v = np.array([xx[i], yy[i]])
        new_v = rotation_matrix @ v

        xx[i] = new_v[0]
        yy[i] = new_v[1]

    # Translate the ellipse
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
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)
    axes.set_title("Plot of points and the ellipse that encases them")

    plt.show()


def plotPointsCompAndPlane(points_1, points_2, normal, center):
    """
    Plot both sets of given points in different style to be able to compare them. Also
    plot the plane that one set of points should be located on.

    Input:
        points_1: The first 3D point cloud of points to plot
        points_2: The second 3D point cloud to plot
        normal: The normal of the plane that the second point cloud should be located on
        center: The center point of the plane
    """

    # Prepare the figure
    figure = plt.figure(figsize = plt.figaspect(1))
    axes = figure.add_subplot(projection = '3d')

    # Plot the first set of points onto the figure
    axes.scatter(points_1[0, :], points_1[1, :], points_1[2, :], c = 'blue')

    # Plot the second set of points onto the figure
    axes.scatter(points_2[0, :], points_2[1, :], points_2[2, :], c = 'red')

    # Prepare the plane, from:
    # https://stackoverflow.com/questions/3461869/plot-a-plane-based-on-a-normal-vector-and-a-point-in-matlab-or-matplotlib
    # The plane formula is a*x + b*y + c*z + d = 0
    # The normal is [a, b, c], we need to calculate d
    a = normal[0]
    b = normal[1]
    c = normal[2]
    d = -(a * center[0] + b * center[1] + c * center[2])

    # Set of all x and y values for the plane
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 100), np.linspace(0.0, 1.0, 100))

    # Calculate corresponding z-value for each xx and yy
    zz = center[2] - (a * (xx - center[0]) + b * (yy - center[1])) / c

    # plot the plane
    #axes.plot_surface(xx, yy, zz)

    # Plot the normal vector at the center point
    axes.quiver(
        center[0],
        center[1],
        center[2],
        normal[0],
        normal[1],
        normal[2],
        color = 'red',
        length = 0.2
    )

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_zlabel("Z-Axis")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_zlim(0, 1)
    plt.title("Two sets of points and the plane that the red points should be on")

    plt.show()


def plotEllipseSamples(samples, center, ellipse_axes_lengths, rotation_matrix):
    """
    Plot the given samples with the ellipse that they sample

    Input:
        samples: A 2D point could of points on the ellipse
        center: The center point of the ellipse
        ellipse_axes_lengths: The axes lengths of the ellipse to plot
        rotation_matrix: The rotation of the ellipse
    """

    # Prepare the figure
    figure, axes = plt.subplots(figsize = plt.figaspect(1))

    # Plot the samples
    axes.scatter(
        samples[0, :],
        samples[1, :],
        s = 100,
        linewidths = 3.0,
        c = 'blue',
        alpha = 0.2
    )

    # Plot the center point to be destinct from the other points
    axes.scatter(
        center[0],
        center[1],
        s = 250,
        
        c = 'red',
        marker = 'x',
        alpha = 1.0
    )

    # Set of all radial angles:
    rr = np.linspace(0, 2 * np.pi, 100)

    # Cartesian coordinates that correspond to the radial angles on the ellipse:
    # This is the equation of an ellipse, from:
    # https://stackoverflow.com/questions/10952060/plot-ellipse-with-matplotlib-pyplot
    # and https://en.wikipedia.org/wiki/Ellipse
    xx = ellipse_axes_lengths[0]*np.cos(rr)
    yy = ellipse_axes_lengths[1]*np.sin(rr)

    # Apply the rotation of the ellipse
    for i in range(xx.shape[0]):
        v = np.array([xx[i], yy[i]])
        new_v = rotation_matrix @ v

        xx[i] = new_v[0]
        yy[i] = new_v[1]

    # Translate the ellipse
    for i in range(xx.shape[0]):
        v = np.array([xx[i], yy[i]])
        new_v = v + center

        xx[i] = new_v[0]
        yy[i] = new_v[1]

    # Plot the ellipse
    plt.plot(xx, yy, color = 'g')

    # Name the axes and title
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_xlim(-2, 2)
    axes.set_ylim(-2, 2)
    axes.set_title("Plot of samples along an ellipse")

    plt.show()                       
