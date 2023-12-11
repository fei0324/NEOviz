'''
Plotting functions related to ellipsoids.
'''
import matplotlib.pyplot as plt


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
