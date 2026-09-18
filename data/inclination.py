import matplotlib.pyplot as plt


if __name__=="__main__":
    f = open("MPCORB.DAT.txt", "r")
    # Available from https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT
    lines = f.readlines()

    inclinationColumn = 7
    numberColumn = 21

    maxInclination = 0.0
    maxIndex = -1

    hasReachedData = False
    i = 0
    inclinations = []
    for line in lines:
        if not hasReachedData and line[0] == "-":
            hasReachedData = True
            i += 1
            continue
        elif not hasReachedData:
            i += 1
            continue

        columns = line.split()
        if len(columns) < 1:
            break

        inclination = float(columns[inclinationColumn])
        if inclination > 90:
            # Retrograde orbits (https://en.wikipedia.org/wiki/Orbital_elements)
            inclination = inclination - 180

        if inclination < 0:
            print("Negative inclination: ", inclination)
            inclination = abs(inclination)
            print("Corrected inclination: ", inclination)
            print("Name: ", columns[numberColumn], columns[numberColumn + 1], columns[numberColumn + 2])

        inclinations.append(inclination)

        if inclination > maxInclination:
            maxInclination = inclination
            maxIndex = i

        #print(columns[inclinationColumn], columns[numberColumn], columns[nameColumn], '\n')

        i += 1

    # Plot of the inclinations to see where most inclinations are located
    font = {
        'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16
    }
    plt.rc('font', **font)
    figure, figure_axes = plt.subplots(figsize = plt.figaspect(1))

    # Violin plot
    #plt.violinplot(inclinations, facecolor = "xkcd:azure", showmeans = False, orientation = "horizontal", side = "high", showextrema = False)
    #plt.axis([0, 90, 1, 1.25])
    #plt.autoscale(enable=True, axis='y', tight=True)
    #plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    #plt.yticks([])
    #figure_axes.set_ylabel("Amount of Asteroids")
    #figure_axes.set_xlabel("Inclination (Degrees)")
    #plt.show()

    #plt.violinplot(inclinations, facecolor = "xkcd:azure", linecolor = "black", showmedians = True, showextrema = True)
    #plt.axis([0, 1, 0, 90])
    #plt.autoscale(enable=True, axis='x', tight=True)
    #plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    #plt.xticks([])
    #figure_axes.set_xlabel("Amount of Asteroids")
    #figure_axes.set_ylabel("Inclination (Degrees)")
    #plt.show()

    # Histogram
    plt.hist(inclinations, color = "xkcd:azure", bins = range(90))
    plt.axis([0, 90, 0, 80000])
    #plt.autoscale(enable=True, axis='y', tight=True)
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    figure_axes.set_ylabel("Number of Asteroids")
    figure_axes.set_xlabel("Inclination (Degrees)")
    plt.show()

    # Box plot
    #plt.boxplot(inclinations, widths = 0.75, showfliers = True, sym='k+', patch_artist=True, capprops = dict(linewidth = 2), whiskerprops = dict(linewidth = 2), boxprops=dict(facecolor = "xkcd:azure", color = "black"), medianprops=dict(color="black"))
    #plt.axis([0, 1, 0, 90])
    #plt.autoscale(enable=True, axis='x', tight=True)
    #plt.autoscale(enable=True, axis='y', tight=True)
    #plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    #plt.xticks([])
    #figure_axes.set_ylabel("Inclination (Degrees)")
    #plt.show()

    maxLine = lines[maxIndex]
    maxColumns = maxLine.split()
    maxName = ""

    for c in range(numberColumn, len(maxColumns) - 1):
        maxName += " " + maxColumns[c]

    print("Max: ", maxInclination, maxName)
    print("Of", i, "items")
