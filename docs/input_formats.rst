====================
Input format options
====================

Here I am generally describing the options for the input to a ``LognormalIntensityMock`` instance, especially the parameters that have different options.
For details on the other parameters, refer to the documentation of the ``LognormalIntensityMock`` class: https://simple-intensity-mapping-simulator.readthedocs.io/en/latest/simple_reference.html#simple.simple.LognormalIntensityMock.

Input file or dictionary
=========================

In general, you can initiate a ``LognormalIntensityMock`` instance from a dictionary or a yaml file that contains the information of the dictionary.

If you want to initiate a ``LognormalIntensityMock`` instance from a dictionary called ``input_dict``, you can run

.. code-block:: python

    lim = LognormalIntensityMock(input_dict)

An example dictionary could look like this, where ``luminosity_function`` is a function defined in python and ``cosmo`` is an astropy cosmology object:

.. code-block:: python

    import astropy.units as u
    from astropy.cosmology import Planck18 as cosmo

    input_dict = {"verbose" : False,
              "bias" : 1.5,
              "redshift" : 2.0,
              "single_redshift" : False,
              "box_size" : np.array([400,400,400]) * u.Mpc,
              "N_mesh" : np.array([128,128,128]),
              "luminosity_unit" : luminosity_unit,
              "Lmin" : 2e41 * u.erg/u.s,
              "Lmax" : np.inf * u.erg/u.s,
              "galaxy_selection" : {"intensity" : "all",
                                    "n_gal" : "detected"},
              "lambda_restframe" : 1215.67 * u.angstrom,
              "brightness_temperature" : False,
              "do_spectral_smooth" : True,
              "do_spectral_tophat_smooth" : False,
              "do_angular_smooth" : True,
              "sigma_beam" : 6 * u.arcsec,
              "dlambda" : 5 * u.angstrom,
              "footprint_radius" : 9 * u.arcmin,
              "luminosity_function" : luminosity_function,
              "run_pk" : {"intensity": True,
                        "n_gal": True,
                        "cross": True,
                        "sky_subtracted_cross": True
                            },
              "dk" : 0.04,
              "kmin" : 0.04,
              "kmax" : 1.0,
              "seed_lognormal" : 100,
              "outfile_prefix" : 'mock',
              "cosmology" : cosmo,
              "lnAs" : 3.094,
              "n_s" : 0.9645,
              "RSD" : True,
              "out_dir" : "../tmp/mocks/",
              "min_flux" : 3e-17 * u.erg/u.s/u.cm**2,
              "sigma_noise" : 2e-22 * u.erg/u.s/u.cm**2/u.angstrom/u.arcsec**2,
    }

If you want to initiate the ``LognormalIntensityMock`` instance from a yaml file called ``input_file.yaml``, you can run

.. code-block:: python

    lim = LognormalIntensityMock("input_file.yaml")

An example yaml file could look like this, where ``np`` will be interpreted as ``numpy``, ``u`` will be interpreted as ``astropy.units`` and ``cosmo`` will be interpreted as ``lim.astropy_cosmo``, which is given by the ``cosmology`` item in the input dictionary:

.. code-block:: yaml

    verbose : False
    bias : 1.5
    redshift : 2.0
    single_redshift : False
    box_size : np.array([400,400,400]) * u.Mpc / cosmo.h
    N_mesh : np.array([128,128,128])
    luminosity_unit : luminosity_unit
    Lmin : 2e41 * u.erg/u.s
    Lmax : np.inf * u.erg/u.s
    galaxy_selection : 
    intensity : all
    n_gal : detected
    lambda_restframe : 1215.67 * u.angstrom
    brightness_temperature : False
    do_spectral_smooth : True
    do_spectral_tophat_smooth : False
    do_angular_smooth : True
    sigma_beam : 6 * u.arcsec
    dlambda : 5 * u.angstrom
    footprint_radius : 9 * u.arcmin
    luminosity_function : "luminosity_function_example.csv"
    luminosity_unit : 1e42 * u.erg/u.s
    run_pk : 
    intenisty : True
    n_gal : True
    cross : True
    sky_subtracted_cross : True
    dk : 0.04
    kmin : 0.04
    kmax : 1.0
    seed_lognormal : 100
    outfile_prefix : 'mock'
    cosmology : Planck18
    n_s : 0.9645
    lnAs : 3.094
    RSD : True
    out_dir : "../tmp/mocks/"
    min_flux : 3e-17 * u.erg/u.s/u.cm**2
    sigma_noise : 2e-22 * u.erg/u.s/u.cm**2/u.angstrom/u.arcsec**2

Cosmology
==========

If your input version is a yaml file, the possible options for the ``cosmology`` keyword are 

#. the name of a cosmology already built into ``astropy.cosmology``, e.g. ``Planck18`` (see https://docs.astropy.org/en/stable/cosmology/index.html#built-in-cosmologies) as a string,
#. the name of a file containing a saved cosmology object saved with astropy (see https://docs.astropy.org/en/stable/cosmology/io.html#cosmology-io) as a string,
#. or a dictionary that can be interpreted by astropy as a ``astropy.cosmology.FlatwCDM`` object (see https://docs.astropy.org/en/stable/api/astropy.cosmology.FlatwCDM.html#astropy.cosmology.FlatwCDM).

If your input is a dictionary, you can also define your astropy cosmology within python and use this cosmology in the dictionary.

You will have to provide a value for the spectral index :math:`n_s` and for :math:`\ln(10^{10}A_s)` separately in the input dictionary or file because they are not part of astropy cosmology.

Luminosity function
====================

The possible options for the ``luminosity_function`` keyword are 

#. a function that takes the luminosity in units of ``luminosity_unit`` and outputs the luminosity function :math:`\frac{\mathrm{d}n}{\mathrm{d}L}` in units of :math:`\mathrm{luminosity\_unit}^{-1}\, \mathrm{Mpc}^{-3}` (only possible if you use a dictionary as input), 
#. or the name of a file that contains the tabulated luminosity function in a csv format, where

    * the ``L`` column contains the luminosity values in units of ``luminosity_unit`` and
    * the ``dn/dL`` column contains the corresponding values of the luminosity function in :math:`\mathrm{luminosity\_unit}^{-1}\, \mathrm{Mpc}^{-3}` units.

Power spectrum
===============

The default of SIMPLE is to generate the power spectrum with the Eisenstein & Hu fitting function given the input cosmology.
However, you can also input your own power spectrum with the key ``input_pk_filename``.
Then, the Eisenstein & Hu power spectrum version will still be generated, but not used for the simulation. This is just an artifact of the code.

If you specify ``input_pk_filename``, it should be the name of the file containing a tabulated input matter power spectrum. 
The file should be ascii-formatted with
    
    * first column: wavenumber in units of [:math:`h\mathrm{Mpc}^{-1}`], 
    * second column: matter power spectrum in units of [:math:`h^{-3}\mathrm{Mpc}^{3}`].
The first column should not contain names for the columns because it will be read by lognormal_galaxies code. See https://bitbucket.org/komatsu5147/lognormal_galaxies/src/intensity-mapping/.

Logarithmic growth factor
===========================

The default of SIMPLE is to generate the logarithmic using the lognormal_galaxies code.
However, if you want to input your own growth parameter, you can do so by specifying ``f_growth_filename``.
It should then be the name of the file containing a tabulated logarithmic growth function. 
The file should be ascii formatted with

    * 1st column: wavenumber [:math:`h\mathrm{Mpc}^{-1}`], 
    * 2nd column: fnu.

See https://bitbucket.org/komatsu5147/lognormal_galaxies/src/intensity-mapping/.

Mesh size
==========

You either have to input the ``N_mesh`` parameter, which specifies the number of cells in each dimension used for any mesh (np.array with integers),
or the ``voxel_length`` parameter, which specifies the size of a voxel in each dimension (np.array with length units.)
If ``voxel_length`` is given, ``N_mesh`` will be inferred as an integer and ``voxel_length`` will be adjusted accordingly.

Selection function
===================

There are two ways of specifying a selection function, which will decide which galaxies are detected or not: specifying ``min_flux`` or ``limit_ngal``.
If ``limit_ngal`` is given, the required ``min_flux`` will be inferred so that the detected galaxy number density matches the ``limit_ngal`` value.
Those galaxies that have a higher flux than the ``min_flux`` will be detected.

If neither ``min_flux`` nor ``limit_ngal`` are given, all galaxies are detected.

min_flux
---------

This parameter can be given in these ways:

#. As an astropy quantity with flux units (e.g. ``u.erg / u.s``), which denotes the universal minimum flux above which a galaxy is detected,
#. or as an array of astropy quantity with flux units in the shape of ``N_mesh``, which denotes the ``min_flux`` at each voxel,
#. or as a function that takes the redshift as an input and outputs the minimum flux at that redshift (as an astropy quantity with flux units, only possible when using a dictionary as input),
#. or the name of a file (as a string) containing a table in ecsv format readable by astropy.table with

    * ``redshift`` column: redshift,
    * ``min_flux`` column: ``min_flux`` at that redshift,
    * and the unit of ``min_flux`` as the unit of the ``min_flux`` column (automatic if saved using astropy.table).

    In this case, it will interpolate between redshifts (in case a redshift is out of bounds, the border values are used.)

#. or the name of a file (as a string) in hdf5 format with

    * ``ff[“min_flux”]`` : ``min_flux`` mesh with the same shape as ``N_mesh``
    * ``ff[“min_flux”].attrs[“unit”]`` should be the string of the astropy unit (e.g. ``erg / (cm2 s)``),

    which will specify the ``min_flux`` for each cell in the mesh.

limit_ngal
------------

This parameter can be given in three ways:

#. As an astropy quantity (in inverse volume units), which is the desired total galaxy number density of the detected galaxies in the entire box. From this, a universal minimum flux will be calculated for the selection function, so that the galaxy number density is not flat, but changes with redshift because the minimum flux is constant.
#. Or as a function, which takes the redshift as input and outputs the desired galaxy number density at that redshift (as an astropy quantity in inverse volume dimensions, only possible if using a dictionary as input),
#. or the name of the file (as a string) containing a table in ecsv format readable by astropy.table with

    * ``redshift`` column: redshift,
    * ``limit_ngal`` column: limit_ngal at that redshift,
    * unit of ``limit_ngal``: unit of the ``limit_ngal`` column (automatic if saved using astropy.table).
    In this case it will interpolate between redshifts (in case a redshift is out of bounds, the border values are used.)

Intensity noise 
================

The sum of all the noise contributions in your mock intensity mapping project is given in the ``self.noise_mesh`` property. This is generated using a Gaussian ``sigma_noise`` parameter, which can be specified in the following ways:

#. astropy quantity, which denotes the universal sigma of the Gaussian noise in intensity units. 
#. array of astropy quantities, it denotes the sigma of the Gaussian noise in intensity units at that position in the array.
#. a function that takes the redshift as input and outputs the ``sigma_noise`` at that redshift (as an astropy quantity in intensity units; only possible if using a dictionary as input).
#. the name of the file (as a string) in an ecsv format readable by astropy.table, containing
    
    * ``redshift`` column: redshift,

    * ``sigma_noise`` column: sigma_noise at that redshift,

    * unit of ``sigma_noise`` (automatic if saved using astropy.table). Then it will be interpolated between redshifts (in case a redshift is out of bounds, the border values are used.)

#. or the name of an hdf5 file containing the ``sigma_noise`` mesh (same shape as ``N_mesh``) under the name ``sigma_noise`` with the unit as a string under ``ff[‘sigma_noise’].attrs[‘unit’]``.

Survey mask
============

The survey mask is an optional parameter under the name ``obs_mask``.

This should be either an array containing integers or floats with the same shape of ``N_Mesh``, if you are using a dictionary as input,
or the name of a file (as a string) in the hdf5 format containing this array under the name ``mask``.
