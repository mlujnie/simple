from setuptools import setup
from Cython.Build import cythonize

setup(
    name='simple',
    version='0.1.0',    
    description='Simple Intensity Map Producer for Line Emission',
    url='https://github.com/mlujnie/simple',
    author='Maja Lujan Niemeyer',
    author_email='maja@mpa-garching.mpg.de',
    license='???',
    packages=['simple'],
    install_requires=['mpi4py>=2.0', # modify!!!
                      'numpy',                     
                      ],
 
    ext_modules = cythonize("simple/tools.pyx"),

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  # change !!!
        'Operating System :: POSIX :: Linux',   # why do I have to specify this?
        'Programming Language :: Python :: 3.10.9',
    ],
)