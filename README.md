# Simple Intensity Map Producer for Line Emission (SIMPLE)

## Introduction

Thank you for considering to use the SIMPLE code for your intensity map mocks!

The Simple Intensity Map Producer for Line Emission (SIMPLE) is meant as a versatile tool to quickly generate intensity maps. It is introduced in this paper https://arxiv.org/abs/2307.08475 and follows this basic pipeline:

<img src="docs/SIMPLE_pipeline.png" alt="simple_pipeline" width="600"/>

While you can specify everything necessary in the input file or dictionary and run this pipeline in one step (`lim.run()`), the code is structured in a modular way so that you can freely use components of the code to calculate whatever you want.

The latest documentation can be found here: https://simple-intensity-mapping-simulator.readthedocs.io/en/latest/.

## Installation
1. Make sure you have a python environment with all the required packages (specified in the `environments.yaml` file.)
2. install intensity-mapping branch of lognormal_galaxies code from https://bitbucket.org/komatsu5147/lognormal_galaxies/src/master/:
      `git clone -b intensity-mapping https://bitbucket.org/komatsu5147/lognormal_galaxies/`
      and follow installation instructions.
3. clone this repo 
    `git clone https://github.com/mlujnie/simple`.
4. modify the `simple/config.py` file: change the path to the path of your lognormal_galaxies installation.
5. type `pip install .` in the root directory of this repo.

## Dependencies
See `environment.yaml`:
* python 3.8 (for pmesh)
* cython
* scipy
* numpy
* astropy
* h5py
* dask
* pmesh
