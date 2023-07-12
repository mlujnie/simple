# Simple Intensity Map Producer for Line Emission (SIMPLE)

## Introduction

![400](SIMPLE_pipeline.png)

The latest documentation can be found here: https://simple-intensity-mapping-simulator.readthedocs.io/en/latest/.

## Setup
* install intensity-mapping branch of lognormal_galaxies code from https://bitbucket.org/komatsu5147/lognormal_galaxies/src/master/:
      `git clone -b intensity-mapping https://bitbucket.org/komatsu5147/lognormal_galaxies/`
      and follow installation instructions.
* clone this repo 
    `git clone https://github.com/mlujnie/simple`.
* modify the `simple/config.py` file: change the path to the path of your lognormal_galaxies installation.
* type `pip install .` in the root directory of this repo.

### add following information: 
* Brief description of features
* Usage (with examples)
* Build and install (with examples)
* Dependencies
    * python 3.8 (for pmesh)
    * cython
    * scipy
    * numpy
    * astropy
    * h5py
    * dask
    * pmesh
* Status of the code and how it is maintained
