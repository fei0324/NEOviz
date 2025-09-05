import matplotlib.pyplot as plt


def plotPoints(points):
    """
    Plot the given points

    Input:
        points: A 3D point could of points to plot
    """

    # Prepare the figure
    figure = plt.figure()
    axes = figure.add_subplot(projection = '3d')

    # Seperate the dimensions of the points
    x_3d = points[0, :]
    y_3d = points[1, :]
    z_3d = points[2, :]


    axes.scatter(x_3d, y_3d, z_3d)
    axes.scatter(center[0], center[1], center[2], s = 50)

    # Fig 1 (3d): plot unit vector center_to_ssb
    # axes.quiver(center[0], center[1], center[2], center_to_ssb[0], center_to_ssb[1], center_to_ssb[2], color='red')

    # Fig 1 (3d): plot the unit vector of ssb_normal
    # axes.quiver(center[0], center[1], center[2], ssb_normal[0], ssb_normal[1], ssb_normal[2], color='darkorchid')

    # Fig 1 (3d): plot mean_velocity and the plane
    xr = np.linspace(center[0] - 3e-08, center[0] + 3e-08, num=20)
    yr = np.linspace(center[1] - 3e-08, center[1] + 3e-08, num=20)
    xx, yy, pz = computePlane(mean_velocity, center, xr, yr)
    # axes.plot_surface(xx, yy, pz, color="green", alpha=0.5)
    axes.quiver(center[0], center[1], center[2], mean_velocity[0], mean_velocity[1], mean_velocity[2], color='green')
    axes.set_aspect('equal')

    # Fig 1 (3d): plot orbit plane intersection points on the plan
    intersectX = plane_orbit_intersection[0, :]
    intersectY = plane_orbit_intersection[1, :]
    intersectZ = plane_orbit_intersection[2, :]
    axes.scatter(intersectX, intersectY, intersectZ)

    # (for plotting) get the m vector and the projected m vector
    m, proj_m = _getMonPlane(mean_velocity, center_to_ssb, ssb_normal)

    # Fig 1 (3d): plot m and proj_m from the center
    # axes.quiver(center[0], center[1], center[2], m[0], m[1], m[2], color='gold')
    axes.quiver(center[0], center[1], center[2], proj_m[0], proj_m[1], proj_m[2], color='tab:orange')
    plt.show()


def plotEllipsoid(ellipsoid_matrix):
    return None


def plotEllipse(ellipse_matrix):
    return None


def plot_ellipse(mat, pos=None, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse based on the specified positive-definite matrix (*mat*) 
    and center (*pos*). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        mat : The 2x2 matrix to base the ellipse on
        pos : The (2,) array that gives the center of the ellipse. Defaults
            to (0, 0)
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
    """

    import numpy as np
    from matplotlib.patches import Ellipse

    def get_sorted_eig(mat):
        import numpy.linalg as la
        s, u = la.eigh(mat)
        idxs = np.argsort(s)
        return s[idxs], u[:, idxs]

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    if pos is None:
        pos = np.array([0, 0])

    # Width and height are "full" widths, not radius
    s, u = get_sorted_eig(mat)
    theta = np.degrees(np.arctan2(u[1, 0], u[0, 0]))
    width = 2*np.sqrt(s[0]*2)
    height = 2*np.sqrt(s[1]*2)

    print(width/2)
    print(height/2)

    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwrg)

    if ax is None:
        ax = plt.gca()
    ax.add_artist(ellip)
    ax.relim()
    ax.autoscale_view()


def applySettings(xlabel=None, ylabel=None, ylimits=None,
                  legend=False, labspace=0.85):
    '''
    Generic pyplot settings from Erin to make plots prettier
    '''
    # Increase the sizes of labels and ticks
    plt.tick_params(axis='both', labelsize=9)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=9)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=9)

    if ylimits is not None:
        # Change y-axis limits
        ymin = ylimits[0]
        ymax = ylimits[1]
        if ymin is not None:
            plt.ylim(ymin=ymin)
        if ymax is not None:
            plt.ylim(ymax=ymax)

    if legend:
        # Change location and text size of legend
        lgd = plt.legend(loc='upper left', labelspacing=labspace)
        plt.setp(lgd.get_texts(), fontsize='9')
        return lgd
