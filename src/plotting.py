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
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Plot of points")

    plt.show()


def plotPointsAndPlane(points, center, normal):
    """
    Plot the given points and the given plane

    Input:
        points: A 3D point could of points to plot
        center: The center point of the plane 
        normal: The 3D vector normal of the plane
    """

    # Prepare the figure
    figure = plt.figure(figsize = plt.figaspect(1))
    axes = figure.add_subplot(projection = '3d')

    # Plot the points onto the figure
    axes.scatter(points[0, :], points[1, :], points[2, :], c = 'blue')

    # Prepare the plane, from:
    # https://stackoverflow.com/questions/3461869/plot-a-plane-based-on-a-normal-vector-and-a-point-in-matlab-or-matplotlib
    # The plane formula is a*x + b*y + c*z + d = 0
    # The normal is [a, b, c], we need to calculate d
    d = -np.dot(normal, center)

    # Set of all x and y values for the plane
    xx, yy = np.meshgrid(range(100), range(100))

    # Calculate corresponding z-value for each xx and yy
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1.0/normal[2]

    # plot the surface
    axes.plot_surface(xx, yy, zz)

    # Plot the plane

    # Name the axes and title
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Plot of points")

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
    figure, axes = plt.subplots(
        nrows = 1,
        ncols = 2,
        #sharex = True,
        #sharey = True,
        figsize = plt.figaspect(1),
        subplot_kw = dict(projection = '3d')
    )

    # Plot the points
    axes[0].scatter(
        points[0, :],
        points[1, :],
        points[2, :],
        s = 10,
        c = 'blue',
        alpha = 0.6
    )

    # Plot the center point to be destinct from the other points
    axes[0].scatter(
        center[0],
        center[1],
        center[2],
        s = 250,
        c = 'red',
        marker = 'x',
        alpha = 1.0
    )

    # Name the axes and title
    axes[0].set_xlabel("X-Axis")
    axes[0].set_ylabel("Y-Axis")
    axes[0].set_title("Plot of points and the ellipsoid center point")

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

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
    axes[1].scatter(center[0], center[1], center[2], s = 100, c = 'red')
    axes[1].plot_surface(x, y, z,  rstride = 4, cstride = 4, color = 'g', alpha = 0.3)

    # Name the axes and title
    axes[1].set_xlabel("X-Axis")
    axes[1].set_ylabel("Y-Axis")
    axes[1].set_title("Plot of ellipsoid and the center point")

    plt.show()


def plotPointsAndEllipse(points, center, ellipse_axes, ellipse_axes_lengths,
                           rotation_matrix):
    # Prepare the figure
    figure, axes = plt.subplots(
        nrows = 1,
        ncols = 2,
        #sharex = True,
        #sharey = True,
        figsize = plt.figaspect(1),
        subplot_kw = dict(projection = '2d')
    )

    # Plot the points
    axes[0].scatter(
        points[0, :],
        points[1, :],
        s = 10,
        c = 'blue',
        alpha = 0.6
    )

    # Plot the center point to be destinct
    axes[0].scatter(
        center[0],
        center[1],
        s = 250,
        c = 'red',
        marker = 'x',
        alpha = 1.0
    )

    # Name the axes and title
    axes[0].set_xlabel("X-Axis")
    axes[0].set_ylabel("Y-Axis")
    axes[0].set_title("Plot of points and the ellipse center point")

    # Create the ellipse
    ellipse = Ellipse(
        xy = (center[0], center[1]),
        height = ellipse_axes_lengths[0],
        width = ellipse_axes_lengths[1],
        edgecolor = 'g',
        alpha = 0.3
    )
    
    # Apply the rotation of the ellipse

    
    # Plot the ellipse
    axes[1].scatter(center[0], center[1], s = 100, c = 'red')
    axes[1].add_patch(ellipse)

    # Name the axes and title
    axes[1].set_xlabel("X-Axis")
    axes[1].set_ylabel("Y-Axis")
    axes[1].set_title("Plot of ellipse and the center point")

    plt.show()

def plotPointsComp(points_1, points_2):
    """
    Plot both sets of given points in different style to be able to compare them

    Input:
        points_1: The first 3D point cloud of points to plot
        points_2: The second 3D point cloud of points to plot
    """

    # Prepare the figure
    figure = plt.figure(figsize = plt.figaspect(1))
    axes = figure.add_subplot(projection = '3d')

    # Plot the first set of points onto the figure
    axes.scatter(points_1[0, :], points_1[1, :], points_1[2, :], c = 'red')

    # Plot the second set of points onto the figure
    axes.scatter(points_2[0, :], points_2[1, :], points_2[2, :], c = 'blue')

    # Name the axes and title
    plt.xlabel("X-Axis")
    plt.ylabel("Y-Axis")
    plt.title("Two sets of points")

    plt.show()
