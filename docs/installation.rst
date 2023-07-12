============
Installation
============

#. Make sure you have a python environment with all the required packages (specified in the ``environment.yaml`` file.)
#. Install intensity-mapping branch of lognormal_galaxies code from https://bitbucket.org/komatsu5147/lognormal_galaxies/src/master/: ``git clone -b intensity-mapping https://bitbucket.org/komatsu5147/lognormal_galaxies/`` and follow installation instructions.
#. Clone the SIMPLE repo ``git clone https://github.com/mlujnie/simple``.
#. Modify the ``simple/config.py`` file: change the path to the path of your ``lognormal_galaxies`` installation.
#. Type ``pip install .`` in the root directory of this repo.
