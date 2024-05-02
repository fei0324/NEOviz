# NEOviz

NEOviz is an interactive visualization system that uses the software [OpenSpace](http://openspaceproject.com) to visualize the trajectory uncertainties of near-Earth objects  (NEO) in the Solar system. This repository contains the source code for data generation (_i.e._ orbit propagation), preprocessing (_i.e._ uncertainty tube computation), and analysis (_i.e._ impact corridor computation, cut-plane plotting) of the uncertainty visualiation pipeline.

The implementation is described in "NEOviz: Uncertainty-Driven Visual Analysis of Asteroid Trajectories" (under review). This project is a collaboration between the [Immersive Visualization](https://immvis.github.io/) group at Linköping University in Sweden and the [B612 Foundation](https://b612foundation.org/).

[![License](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)

<!-- [![System Paper](https://img.shields.io/badge/System%20Paper-10.1109%2FTVCG.2019.2934259-blue?style=flat-square)]() -->

## Installation
For the uncertainty tube computation and analysis (the /src directory), the code is tested with Python 3.11 and Windows-64. You can download or clone the repository and run the following,

```
$ pip install -r requirements.txt
```

Important python dependencies:
 - [numpy](https://numpy.org/)
 - [matplotlib](https://matplotlib.org/)
 - [spiceypy](https://pypi.org/project/spiceypy/)
 - [astropy](https://www.astropy.org/)

The code uses [SPICE kernels](https://naif.jpl.nasa.gov/naif/data.html) for many computations. The file `/src/getsta.tm` details the three generic kernels we used. You need to visit the [Generic Kernels](https://naif.jpl.nasa.gov/naif/data_generic.html) on the SPICE website, download each of the text files, and create the file structure as described in the `KERNEL_TO_LOAD` section in the `/src/getsta.tm` file.
The code then loads the SPICE data for later use. In general, if you want to learn about what data contained in each kernel file, read the `aareadme.txt` files in each directory on the SPICE website. 

For orbit propagation (the /apophis_propagation directory), the code is tested with Python 3.11 and Ubuntu 22.04.4. You can run the following:

```
$ pip install -r requirements-linux.txt
```

Important python dependencies:
 - [pandas](https://pandas.pydata.org/)
 - [adam_core](https://b612.ai/opensource/adam_core/)
 - [mpcq](https://github.com/B612-Asteroid-Institute/mpcq)
 - [astropy](https://www.astropy.org/)
 - [pyarrow](https://pypi.org/project/pyarrow/)
 - [quivr](https://pypi.org/project/quivr/)



## License
This repository is provided under the [3-Clause BSD License](https://github.com/fei0324/NEOviz/blob/main/LICENSE).
