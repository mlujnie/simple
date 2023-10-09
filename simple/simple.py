import copy
import functools
import logging
import os
import sys

import astropy.constants as const
import astropy.units as u
# have to keep it to initiate cosmology in LIM.
import astropy.cosmology as astropy_cosmology
import dask.array as da
import h5py
import numpy as np
import pmesh
import random

from astropy.cosmology import Cosmology, z_at_value, FlatwCDM
from astropy.table import Table
from scipy.integrate import quad
from scipy.interpolate import interp1d

from simple.run_lognormal_galaxies import *
from simple.tools_python import (
    transform_bin_to_h5,
    bin_scipy,
    jinc,
    yaml_file_to_dictionary,
    make_map,
    log_interp1d
)
from simple.tools import catalog_to_mesh_cython
from simple.tools import get_kspec_cython
from simple.tools import apply_selection_function_by_position


def aniso_filter(k, v):
    """
    Filter for k_perp and k_par modes separately.
    Applies to an pmesh mesh object as a regular filter.

    Uses globally defined variables:
        sigma_perp - 'angular' smoothing in the flat sky approximation
        sigma_par - 'radial' smoothing from number of channels.

    Usage:
        mesh.apply(perp_filter, mode='complex', kind='wavenumber')

    NOTES:
    k[0] *= modifies the next iteration in the loop.
    Coordinates are fixed except for the k[1] which are
    the coordinate that sets what slab is being altered?

    """

    los_axis = 0  # np.where(LOS)[0][0]  # int
    perp_axes = [1, 2]  # np.where(LOS == 0)[0]  # list
    rper = sigma[perp_axes[0]]
    rpar = sigma[los_axis]
    newk = copy.deepcopy(k)

    # kk = sum(ki**2 for ki in newk) ** 0.5
    kk2_perp = newk[perp_axes[0]] ** 2 + newk[perp_axes[1]] ** 2

    if rpar > 0:
        w = np.exp(-0.5 * kk2_perp * rper**2) * np.sinc(
            0.5 * newk[los_axis] * rpar / np.pi
        )
    else:
        w = np.exp(-0.5 * kk2_perp * rper**2)

    # w[newk[0] == 0] = 1.0
    return w * v


def aniso_filter_gaussian(k, v):
    """
    Filter for k_perp and k_par modes separately.
    Applies to an pmesh mesh object as a regular filter.

    Uses globally defined variables:
        sigma_perp - 'angular' smoothing in the flat sky approximation
        sigma_par - 'radial' smoothing from number of channels.

    Usage:
        mesh.apply(perp_filter, mode='complex', kind='wavenumber')

    NOTES:
    k[0] *= modifies the next iteration in the loop.
    Coordinates are fixed except for the k[1] which are
    the coordinate that sets what slab is being altered?

    """

    los_axis = 0  # np.where(LOS)[0][0]  # int
    perp_axes = [1, 2]  # np.where(LOS == 0)[0]  # list
    rper = sigma[perp_axes[0]]
    rpar = sigma[los_axis]
    newk = copy.deepcopy(k)

    # kk = sum(ki**2 for ki in newk) ** 0.5
    kk2_perp = newk[perp_axes[0]] ** 2 + newk[perp_axes[1]] ** 2

    if rpar > 0:
        w = np.exp(-0.5 * kk2_perp * rper**2) * \
            np.exp(-0.5 * newk[los_axis]**2 * rpar**2)
    else:
        w = np.exp(-0.5 * kk2_perp * rper**2)

    # w[newk[0] == 0] = 1.0
    return w * v


def angular_tophat_filter(k, v):
    """
    Top-hat filter for k_perp.
    Applies to an pmesh mesh object as a regular filter.

    Uses globally defined variables:
        size_perp - 'angular' size of the top-hat filter in the flat sky approximation

    Usage:
        mesh.apply(angular_tophat_filter, mode='complex', kind='wavenumber')

    NOTES:
    k[0] *= modifies the next iteration in the loop.
    Coordinates are fixed except for the k[1] which are
    the coordinate that sets what slab is being altered?

    """

    los_axis = 0  # np.where(LOS)[0][0]  # int
    perp_axes = [1, 2]  # np.where(LOS == 0)[0]  # list
    rper = tophat_size
    newk = copy.deepcopy(k)
    k_perp = np.sqrt(newk[perp_axes[0]] ** 2 + newk[perp_axes[1]] ** 2)
    w = jinc(k_perp * rper) * 2
    return w * v


class BasicBoxCalculator:

    def __init__(self, redshift, box_size, N_mesh, single_redshift, cosmology):
        self.redshift = redshift
        self.box_size = box_size
        self.N_mesh = N_mesh
        self.single_redshift = single_redshift
        self.astropy_cosmo = cosmology

        self.Mpch = u.Mpc / self.astropy_cosmo.h

        self.voxel_length = self.box_size / self.N_mesh

        self.LOS = np.array([1, 0, 0])

    def get_comoving_distance_per_delta_z(self, mean_z, delta_z):
        """
        Calculates the comoving distance between (z + delta_z) and (z - delta_z).

        Parameters
        ----------
        mean_z : float
            Mean redshift of the box: mean_z = (z_min + z_max) / 2.
        delta_z : float
            Half of the redshift difference: delta_z = (z_max - z_min) / 2.

        Returns
        -------
        Quantity
            Comoving distance between (z + delta_z) and (z - delta_z).

        """

        return self.astropy_cosmo.comoving_distance(
            mean_z + delta_z
        ) - self.astropy_cosmo.comoving_distance(mean_z - delta_z)

    @functools.cached_property
    def delta_redshift(self):
        """
        Calculates the redshift difference corresponding to the comoving distance of the LOS length.

        Returns
        -------
        float
            Redshift difference corresponding to the comoving distance of the line-of-sight (LOS) length.
            Cached like a property.

        Raises
        ------
        ValueError
            If the LOS is not aligned with any of the axes [1,0,0], [0,1,0], or [0,0,1].

        """

        axis = np.where(self.LOS == 1)[0]
        if len(axis) != 1:
            raise ValueError("LOS must be [1,0,0] or [0,1,0] or [0,0,1].")
        axis = axis[0]
        LOS_LENGTH = (
            self.box_size[axis].to(u.Mpc).value
        )  # box length in Mpc, 0 -> Lx, 1 -> Ly, 2 -> Lz
        delta_zs = np.linspace(0.0, self.redshift, 1000)
        comoving_distances = (
            self.get_comoving_distance_per_delta_z(self.redshift, delta_zs)
            .to(u.Mpc)
            .value
        )
        delta_z_per_comoving_distance = interp1d(comoving_distances, delta_zs)
        return delta_z_per_comoving_distance(LOS_LENGTH)

    @functools.cached_property
    def minimum_distance(self):
        """ Calculates the minimum distance of a cell within the mesh (cached)."""
        return self.astropy_cosmo.comoving_distance(
            self.redshift - self.delta_redshift
        ).to(self.Mpch)

    @functools.cached_property
    def redshift_mesh_axis(self):
        """ Calculates the redshift coordinates along the LOS axis (cached)."""
        distances = (
            self.minimum_distance
            + self.voxel_length[0] * (np.arange(self.N_mesh[0]) + 0.5)
        ).to(u.Mpc)
        return z_at_value(self.astropy_cosmo.comoving_distance, distances).value

    @functools.cached_property
    def mean_redshift(self):
        " Calculates the mean of the redshifts of the cells (cached)."
        return np.mean(self.redshift_mesh_axis)

    @functools.cached_property
    def k_Nyquist(self):
        """ Calculates the Nyquist frequency (cached)."""
        return np.pi / self.voxel_length

    @functools.cached_property
    def box_volume(self):
        """ Calculates the volume of the simulation box.
            Cached like a property self.box_volume."""
        return self.box_size[0] * self.box_size[1] * self.box_size[2]

    @functools.cached_property
    def voxel_volume(self):
        """ Calculates the volume of a voxel.
            Cached like a property self.voxel_volume."""
        return self.voxel_length[0] * self.voxel_length[1] * self.voxel_length[2]

    def get_kspec(self, dohalf=True, doindep=True):
        """
        Compute the power spectrum in Fourier space.

        Parameters
        ----------
        dohalf : bool, optional
            Flag indicating whether to generate only the half of the Fourier transforms due to symmetry. Default is True.
        doindep : bool, optional
            Flag indicating whether to calculate only the independent modes of the Fourier transform. Default is True.

        Returns
        -------
        kspec : ndarray
            The norm of the wavevector k in the Fourier transform of our meshes.
        muspec : ndarray
            The corresponding values of mu (cosine of the angle between the LOS and the wavevector k).
        indep : ndarray (bool)
            Indicates whether the mode is independent or not.

        """

        logging.info("Getting k_spec...")
        nx, ny, nz = self.N_mesh
        lx, ly, lz = self.box_size.to(self.Mpch).value

        kspec, muspec, indep, kx, ky, kz, k_par, k_perp = get_kspec_cython(
            nx, ny, nz, lx, ly, lz, dohalf, doindep
        )
        logging.info("Done with the biggest part including k_par and k_perp.")

        self.kspec = kspec
        self.muspec = muspec
        self.indep = indep
        self.k_par = k_par
        self.k_perp = k_perp
        self.kx = kx
        self.ky = ky
        self.kz = kz
        logging.info("Done")

        return kspec, muspec, indep

    def get_mesh_positions(self, voxel_length, N_mesh, save_to_self=True):
        """
        Returns an array of mesh positions with dimensions (N_mesh, N_mesh, N_mesh, 3),
        where mesh_position[i, j, k] is the position vector of the center of cell (i, j, k) in Mpc/h.

        Parameters
        ----------
        voxel_length : Quantity array (3,)
            Length of each voxel in the three spatial dimensions with length units.
        N_mesh : array (3,)
            Number of cells in each dimension (dimensionless array).
        save_to_self : bool, optional
            Save mesh_position to self.mesh_position. Default: True.

        """

        logging.info("Getting mesh positions.")

        position = np.transpose(
            np.transpose([np.arange(N_mesh[i])
                         for i in range(N_mesh.shape[0])])
            * voxel_length
            + voxel_length / 2.0
        )
        position = position.to(self.Mpch).value
        x = np.array(
            [[position[0] for i in range(N_mesh[1])] for j in range(N_mesh[2])]
        )
        y = np.transpose(
            [[position[1] for i in range(N_mesh[2])]
             for j in range(N_mesh[0])],
            axes=[1, 2, 0],
        )
        z = np.transpose(
            [[position[2] for i in range(N_mesh[0])]
             for j in range(N_mesh[1])],
            axes=[2, 0, 1],
        )
        mesh_position = np.transpose([x, y, z], axes=[3, 2, 1, 0])
        if save_to_self:
            self.mesh_position = mesh_position
        return (mesh_position * self.Mpch).to(self.Mpch)


class LognormalIntensityMock(BasicBoxCalculator):
    """
    The LognormalIntensityMock class takes cosmological and survey parameters as input
    and produces a intensity and galaxy density meshes from a lognormal galaxy catalog.
    It can apply a selection function, smoothing, masks, add noise, and model the sky subtraction.
    Finally, it can calculate the power spectra.

    The following parameter list will contain the input parameters in the input_dict.
    input_dict can be a dictionary or string. 
    As a dictionary, it should contain the following input parameters.
    If it is a string, it should be the filename of a yaml file version of the dictionary.

    Strings as arguments in the dictionary or yaml file can be evaluated in python, e.g. '10 * u.Mpc'.
    Here, 'u' is the imported name for 'astropy.units'.
    Other possible packages are numpy ('np') and astropy.constants ('const').
    If a string contains 'cosmo', it will be converted to the class' astropy cosmology.
    Therefore expressions such as 'cosmo.h' can be evaluated self-consistently.


    Parameters
    ----------

    verbose: bool, optional
        Flag stating whether to print lots of output.
        If False, it will only print errors.
        Default: True.

    cosmology: string, dict, or astropy cosmology object
        If astropy cosmology object, it will directly use this.
        If string, it can either be a named cosmology in astropy.cosmology 
        (see https://docs.astropy.org/en/stable/cosmology/index.html#built-in-cosmologies)
        or the name of an ecsv-format file generated by saving an astropy.cosmology object 
        (see https://docs.astropy.org/en/stable/cosmology/io.html#cosmology-io).
        If dict, it will attempt to initiate an astropy.cosmology.FlatwCDM object from the dict 
        (see https://docs.astropy.org/en/stable/api/astropy.cosmology.FlatwCDM.html#astropy.cosmology.FlatwCDM).

    n_s: float
        Value of the spectral index :math:`n_s` (part of the cosmology).

    lnAs: float
        Value of :math:`\\mathrm{ln}(10^{10} A_s)` (part of the cosmology).

    luminosity_function: string or function
        If function, it should take the luminosity as an input (in units of luminosity_unit, but as a float)
        and return dn/dL as output (in units of 1/luminosity_unit * 1/Mpc**3).
        If string, it should be the name of the file that contains a table (csv format readable by astropy.table)
        where 'L' is the column containing the luminosity (in units of luminosity_unit, but as a float) 
        and 'dn/dL' is the column containing the corresponding dn/dL (in units of 1/luminosity_unit * 1/Mpc**3).
        Then the class will interpolate between the values and extrapolate at the borders.

    luminosity_unit: astropy quantity
        This is the unit of the luminosity in the luminosity function.

    Lmin: astropy quantity
        This is the minimum luminosity to consider for the luminosity function.
        This is necesssary to get a finite number of galaxies to generate.

    Lmax: astropy quantity, optional
        This is the maximum luminosity to consider for the luminosity function.
        Default: infinity.

    input_pk_filename: string, optional
        Name of the file containing a tabulated input matter power spectrum.
        This power spectrum will be used to generate the lognormal galaxy catalog.
        The file should be ascii-formatted with
            first column: wavenumber in units of [h/Mpc],
            second column:  matter power spectrum in units of [Mpc^3/h^3].
        The first column should not contain names for the columns (should be read by lognormal_galaxies code).
        Default: Use the file generated by lognormal_galaxies from the input cosmological parameters 
        and the Eisenstein & Hu fitting function (see Agrawal et al. 2017 arxiv:1706.09195).

    f_growth_filename: string, optional
        Name of the file containing a tabulated logarithmic growth function.
        This will be used to generate the lognormal galaxy catalog 
        (lognormal_galaxies: "logarithmic growth rate fnu for massive (massless) 
        neutrinos which varies (does not vary) as a function of k").
        The file should be ascii formatted with 
            1st column: wavenumber [h Mpc^-1],
            2nd column: fnu.
        Default: Use the file generated by lognormal_galaxies from 
        the input cosmological parameters (see Agrawal et al. 2017 arxiv:1706.09195).

    redshift: float
        Central redshift of the box so that z_max - redshift = redshift - z_min
        where z_min and z_max are the minimum and maximum redshift of the box, respectively.
        Will be assigned to the entire box if single_redshift == True.

    bias: float
        Galaxy bias.

    box_size: astropy quantity array
        This should be a list or array containing the three box side lengths 
        in the dimension of length using astropy.units.
        It will be converted to units of Mpc/h.

    N_mesh: array-like, optional
        List containing the mesh size (3 numbers) used to generate the galaxy catalog, meshes, and Fourier transforms.
        This is optional if voxel_length is given instead.

    voxel_length: astropy quantity array, optional
        List containing the side lengths of the voxels (3 astropy quantities).
        It will be slightly adjusted to make sure than N_mesh is an integer.
        This is optional if N_mesh is given and will be ignored if N_mesh is given.

    single_redshift: bool, optional
        Flag stating whether the 'redshift' parameter should be assigned to the entire box.
        If False, the redshift will evolve along the line of sight (LOS).
        Default: False.

    RSD: bool
        Flag stating whether to calculate the power spectra in redshift space.
        This is only important when using the run() function, which runs everything automatically.
        If this class is used in a modular way, you can choose in each function.

    min_flux: astropy quantity, array of astropy quantities, function, or string, optional
        If it is an astropy quantity, it denotes the minimum flux above which a galaxy is detected.
        If it is an astropy quantity array in the shape of N_mesh, it denotes the minimum flux above which a galaxy is detected in each voxel.
        If it is a function, it should take the redshift as input and should output the minimum flux
        (as an astropy quantity in flux units).
        If it is a string, it should be the name of the file
            * either containing a table in ecsv format readable by astropy.table with 
                - 'redshift' column: redshift
                - 'min_flux' column: min_flux at that redshift,
                - unit of 'min_flux': unit of the 'min_flux' column (automatic if saved using astropy.table).
                In this case, it will interpolate between redshifts (in case a redshift is out of bounds,
                the border values are used.)
            * or in hdf5 format with
                - ff["min_flux"] : min_flux mesh with the same shape as N_mesh
                - ff["min_flux"].attrs["unit"] should be the string of the astropy unit (e.g. "erg / (cm2 s)").
        Default: None. Then all galaxies are detected unless 'limit_ngal' is given.

    limit_ngal: astropy quantity, function, or string, optional
        If it is an astropy quantity, it is the desired galaxy number density (in inverse volume dimension)
        of the detected galaxies. From this, a minimum flux will be calculated for the selection function.
        If it is a function, it should take the redshift as input and 
        should output the desired galaxy number density (as an astropy quantity in inverse volume dimensions).
        If it is a string, it should be the name of the file
        containing a table in ecsv format readable by astropy.table with
            - 'redshift' column: redshift,
            - 'limit_ngal' column: limit_ngal at that redshift,
            -  unit of 'limit_ngal': unit of the 'limit_ngal' column (automatic if saved using astropy.table).
        In this case it will interpolate between redshifts (in case a redshift is out of bounds,
        the border values are used.)
        Default: None. Then all galaxies are detected unless min_flux is given.
        Will be ignored if min_flux is given.

    galaxy_selection: dict
        Dictionary containing the set of galaxies to be included in the intensity and n_gal meshes.
        The keys must contain 'n_gal' and 'intensity'.
        Options for the values are 'all', 'detected', and 'undetected'.

    sigma_noise: astropy quantity, nd-array of astropy quantities, function, or string, optional
        If it is an astropy quantity, it denotes the sigma of the Gaussian noise in intensity units.
        If it is an array of astropy quantities, it denotes the sigma of the Gaussian noise in intensity units
        at that position in the array.
        If it is a function, it should take the redshift as input and should output the 
        sigma_noise at that redshift (as an astropy quantity in intensity units).
        If it is a string, it can be either 
            1. the name of the file in an ecsv format readable by astropy.table, containing
                - 'redshift' column: redshift,
                - 'sigma_noise' column: sigma_noise at that redshift,
                - unit of 'sigma_noise' (automatic if saved using astropy.table)
                Then it will be interpolated between redshifts (in case a redshift is out of bounds,
                the border values are used.) 
            2. or the name of the hdf5 file containing the sigma_noise mesh (same shape as N_mesh)
                under the name 'sigma_noise' with the unit as a string under ff['sigma_noise'].attrs['unit'].
        Default: None. Then the noise is zero.

    lambda_restframe: astropy quantity, optional
        Rest-frame wavelength in length units.
        If lambda_restframe is given, the intensity will be specific intensity dI/dlambda.
        Optional if 'nu_restframe' is given.

    nu_restframe: astropy quantity, optional
        Rest-frame frequency in inverse time units.
        If lambda_restframe is given, the intensity will be specific intensity dI/dnu.
        Optional if 'lambda_restframe' is given.

    brightness_temperature: bool, optional
        Flag stating whether to calculate everything in temperature units
        instead of specific intensity units.        

    do_spectral_smooth: bool, optional
        Flag stating whether to smooth along the spectral dimension.
        If 'do_spectral_smooth' is True, either 'dlambda' or 'dnu' must be given,
        corresponding to whether lambda_restframe or nu_restframe are given.
        Default: False.

    dlambda: astropy quantity, optional
        Smoothing wavelength (in wavelength units) along the spectral direction.
        If 'do_spectral_tophat_smooth' is True, it is the size of the 1d top-hat.
        Otherwise, it is the Gaussian sigma for a Gaussian smoothing kernel.
        This will be evaluated at the central redshift ('redshift' parameter) of the box for smoothing.
        Must be given if 'do_spectral_smooth' is True and 'lambda_restframe' is given.
        Will be ignored otherwise.

    dnu: astropy quantity, optional
        Smoothing frequency (in frequency units) along the spectral direction.
        If 'do_spectral_tophat_smooth' is True, it is the size of the 1d top-hat.
        Otherwise, it is the Gaussian sigma for a Gaussian smoothing kernel.
        This will be evaluated at the central redshift ('redshift' parameter) of the box for smoothing.
        Must be given if 'do_spectral_smooth' is True and 'nu_restframe' is given.
        Will be ignored otherwise.

    do_spectral_tophat_smooth: bool, optional
        Flag stating whether to use a 1d top-hat smoothing kernel in the angular direction.
        If False, it will use a Gaussian kernel instead.
        Default: False.

    do_angular_smooth: bool, optional
        Flag stating whether to smooth in the directions perpendicular to the LOS.
        This will be a Gaussian smoothing kernel.
        If True, 'sigma_beam' must be given. 
        Default: False.

    sigma_beam: astropy quantity, optional
        Gaussian sigma for the Gaussian smoothing kernel perpendicular to the LOS, in angle units.
        This will be evaluated at the central redshift ('redshift' parameter) of the box for smoothing.
        Must be given if 'do_angular_smooth' is True.
        Will be ignored otherwise.

    footprint_radius: astropy quantity, optional
        Radius of a circular focal plane in angle units.
        This is necessary if you want to perform a sky subtraction and will be ignored otherwise.
        Default: None

    seed_lognormal: int, optional
        Seed for the (random) generation of the lognormal galaxy catalog and the noise.
        It it is the same twice, you should get the same results.
        If None, the code will generate a random positive integer.
        Default: None

    run_pk: dictionary, optional
        Dictionary stating which power spectra to calculate.
        Keys can include ["intensity", "n_gal", "cross", "sky_subtracted_intensity", "sky_subtracted_cross"]
        and the values are either True or False.
        Default for each key (if not given): False.

    dk: astropy quantity, optional
        Size of the linearly spaced wavenumber bins to calculate the power spectra (inverse length units).
        Default: 2 pi / minimum box size.

    kmin: astropy quantity, optional
        Minimum k of the linearly spaced wavenumber bins to calculate the power spectra (inverse length units).
        Default: 0.

    kmax: astropy quantity, optional
        Maximum k of the linearly spaced wavenumber bins to calculate the power spectra (inverse length units).
        Default: minimum Nyquist frequency.

    obs_mask: ndarray (3,) or string
        If it is an ndarray, it should contain the mask (or weights) as floats 
        and should have the same shape as given by N_mesh.
        If it is a string, it should be the name of the hdf5 file containing the
        mask array under the name 'mask'.
        Default: None (then the mask will be generated as ones).

    calc_pk_lognormal: bool, optional
        Flag stating whether to have the galaxy power spectrum calculated by lognormal_galaxies.
        In this case, it will output the galaxy power spectrum of all galaxies in a file 
        as described in https://bitbucket.org/komatsu5147/lognormal_galaxies/src/master/.
        Default: False.

    Notes
    -----
    The redshift direction is chosen to be the first axis
    such that the smallest index is closest to the observer.

    """

    def __init__(self, input_dict):
        # TODO:  change to include the default input just like in Jose's code! Much cooler.

        # initiate the input dictionary and evaluate the strings.
        if isinstance(input_dict, str):
            try:
                input_dict = yaml_file_to_dictionary(input_dict)
            except Exception as e:
                logging.error(e)
        input_dict = self._convert_input_dictionary(input_dict)
        self.input_dict = input_dict

        if "verbose" in input_dict.keys():
            self.verbose = input_dict['verbose']
            if input_dict["verbose"]:
                level = logging.INFO
            else:
                level = logging.WARNING
        else:
            level = logging.INFO

        # configure logging
        FORMAT = "%(asctime)s simple %(levelname)s: %(message)s"
        logging.basicConfig(format=FORMAT, level=level,
                            stream=sys.stdout, force=True)
        logging.info("Initializing LognormalIntensityMock instance.")

        # get info on saving directories etc.
        if "outfile_prefix" in input_dict.keys():
            self.outfile_prefix = input_dict["outfile_prefix"]
        else:
            self.outfile_prefix = "example"
        if "out_dir" in input_dict.keys():
            self.out_dir = input_dict["out_dir"]
        else:
            self.out_dir = "./data"

        # initiate cosmology
        cosmology = input_dict["cosmology"]
        if isinstance(cosmology, str):
            try:
                self.astropy_cosmo = eval(
                    "astropy_cosmology.{}".format(cosmology))
            except:
                logging.info(f"Initiating cosmology from file {cosmology}.")
                self.astropy_cosmo = Cosmology.read(
                    cosmology, format="ascii.ecsv")
        elif isinstance(cosmology, dict):
            self.astropy_cosmo = FlatwCDM(**cosmology)
            logging.debug("MNU: ", self.astropy_cosmo.m_nu)
        else:
            self.astropy_cosmo = cosmology

        self.n_s = input_dict["n_s"]
        self.lnAs = input_dict["lnAs"]

        self.Mpch = u.Mpc / self.astropy_cosmo.h
        self.Mpch = self.Mpch.to(self.Mpch)

        # initiate luminosity function
        luminosity_function = input_dict["luminosity_function"]
        if isinstance(
            luminosity_function, str
        ):  # if it is the name of a file, try to get the table from the file
            luminosity_function_table = Table.read(
                luminosity_function, format="csv")
            self.luminosity_function = log_interp1d(
                luminosity_function_table["L"],
                luminosity_function_table["dn/dL"],
                fill_value="extrapolate",
            )
            logging.warning(
                """We extrapolate the values outside of the provided tabulated values of L. 
Plot plt.loglog(Ls, lim.luminosity_function(Ls)) in a reasonable range to check the outcome!"""
            )
        elif callable(luminosity_function):  # if it is a function, directly use it
            self.luminosity_function = luminosity_function
        else:  # hope that it is a table or dictionary
            self.luminosity_function = log_interp1d(
                luminosity_function["L"],
                luminosity_function["dn/dL"],
                fill_value="extrapolate",
            )
            logging.warning(
                """We extrapolate the values outside of the provided tabulated values of L.
Plot plt.loglog(Ls, lim.luminosity_function(Ls)) in a reasonable range to check the outcome!"""
            )

        self.luminosity_unit = input_dict["luminosity_unit"]
        logging.info(f"luminosity_unit: {self.luminosity_unit}")
        self.Lmin = input_dict["Lmin"]
        if "Lmax" in input_dict.keys():
            self.Lmax = input_dict["Lmax"]
        else:
            self.Lmax = np.inf * self.luminosity_unit

        # initiate input pk filename if given, None otherwise
        if "input_pk_filename" in input_dict.keys():
            logging.info("Using input power spectrum.")
            self.input_pk_filename = input_dict["input_pk_filename"]
        else:
            self.input_pk_filename = os.path.join(
                self.out_dir, "inputs", self.outfile_prefix + "_pk.txt"
            )

        if "f_growth_filename" in input_dict.keys():
            logging.info("Using input logarithmic growth rate.")
            self.f_growth_filename = input_dict["f_growth_filename"]
        else:
            self.f_growth_filename = os.path.join(
                self.out_dir, "inputs", self.outfile_prefix + "_fnu.txt"
            )

        # transform strings back to Quantities before reading in the inputs.
        for key in input_dict.keys():
            if type(input_dict[key]) == str:
                try:
                    input_dict[key] = u.Quantity(input_dict[key])
                except:
                    pass

        # initiate general box parameters
        self.redshift = input_dict["redshift"]
        self.bias = input_dict["bias"]
        self.box_size = input_dict["box_size"]
        if isinstance(self.box_size, str):
            self.box_size = eval(self.box_size)
        self.box_size = self.box_size.to(self.Mpch)
        if "N_mesh" in input_dict.keys():
            self.N_mesh = np.array(input_dict["N_mesh"])
            self.voxel_length = self.box_size / self.N_mesh
        elif "voxel_length" in input_dict.keys():
            voxel_length = input_dict["voxel_length"]
            if isinstance(voxel_length, str):
                voxel_length = eval(voxel_length)
            self.N_mesh = (
                (self.box_size / voxel_length).to(1).value).astype(int)
            # get the exact voxel length because N_mesh has to be an integer.
            self.voxel_length = self.box_size / self.N_mesh
        else:
            raise ValueError(
                "You must give either 'N_mesh' or 'voxel_length' as input."
            )
        self.N_mesh = self.N_mesh.astype(int)
        if "single_redshift" in input_dict.keys():
            self.single_redshift = input_dict["single_redshift"]
        else:
            self.single_redshift = False
        self.LOS = np.array([1, 0, 0])
        self.RSD = input_dict["RSD"]

        # initiate selection function (either min_flux or limit_ngal)
        if "min_flux" in input_dict.keys():
            self.min_flux = input_dict["min_flux"]
        else:
            self.min_flux = None
        if isinstance(self.min_flux, str):
            self.min_flux_file = self.min_flux
            try:  # see if we can read it as a table
                min_flux_table = Table.read(
                    self.min_flux, format="ascii.ecsv")
                interp_min_flux = interp1d(
                    min_flux_table["redshift"],
                    min_flux_table["min_flux"],
                    fill_value=(min_flux_table['min_flux']
                                [0], min_flux_table['min_flux'][-1]),
                    bounds_error=False
                )
                self.min_flux = (
                    lambda z: interp_min_flux(
                        z) * min_flux_table["min_flux"].unit
                )
            except Exception as e1:
                try:  # see if we can read it as an hdf5 file
                    with h5py.File(self.min_flux, "r") as ff:
                        self.min_flux = ff['min_flux'] * \
                            u.Unit(ff['min_flux'].attrs['unit'])
                except Exception as e2:
                    logging.error(
                        "Could not read min_flux as a Table or hdf5 file:\n{}\n{}".format(e1, e2))

        if "limit_ngal" in input_dict.keys():
            self.limit_ngal = input_dict["limit_ngal"]
        else:
            self.limit_ngal = None
        logging.info("limit_ngal: {}".format(self.limit_ngal))
        if isinstance(self.limit_ngal, str):
            self.limit_ngal_file = self.limit_ngal
            limit_ngal_table = Table.read(
                self.limit_ngal, format="ascii.ecsv")
            interp_limit_ngal = interp1d(
                limit_ngal_table["redshift"],
                limit_ngal_table["min_flux"],
                fill_value=(limit_ngal_table['min_flux']
                            [0], limit_ngal_table['min_flux'][-1]),
                bounds_error=False
            )
            self.limit_ngal = (
                lambda z: interp_limit_ngal(
                    z) * limit_ngal_table["min_flux"].unit
            )
        self.galaxy_selection = input_dict["galaxy_selection"]

        # initiate line information and smoothing
        try:
            self.sigma_noise = input_dict["sigma_noise"]
        except:
            self.sigma_noise = None
        if isinstance(
            self.sigma_noise, str
        ):  # if it is the name of a file, try to get the table from the file
            self.sigma_noise_file = self.sigma_noise
            try:
                sigma_noise_table = Table.read(
                    self.sigma_noise, format="ascii.ecsv")
                interp_sigma_noise = interp1d(
                    sigma_noise_table["redshift"],
                    sigma_noise_table["sigma_noise"],
                    fill_value=(
                        sigma_noise_table['sigma_noise'][0], sigma_noise_table['sigma_noise'][-1]),
                    bounds_error=False
                )
                self.sigma_noise = (
                    lambda z: interp_sigma_noise(
                        z) * sigma_noise_table["sigma_noise"].unit
                )
            except Exception as e1:  # try reading it as an hdf5 file.
                try:
                    with h5py.File(self.sigma_noise, "r") as ff:
                        self.sigma_noise = ff['sigma_noise'] * \
                            u.Unit(ff['sigma_noise'].attrs['unit'])
                except Exception as e2:
                    logging.error(
                        "Could not read sigma_noise as Table or hdf5 file:\n{}\n{}".format(e1, e2))

        if "lambda_restframe" in input_dict.keys():
            self.lambda_restframe = input_dict["lambda_restframe"]
            self.nu_restframe = None
        elif "nu_restframe" in input_dict.keys():
            self.nu_restframe = input_dict["nu_restframe"]
            self.lambda_restframe = None
        else:
            raise ValueError(
                "You must provide either lambda_restframe or nu_restframe to calculate the specific intensity dI/dnu or dI/dlambda."
            )
        if "brightness_temperature" in input_dict.keys():
            self.brightness_temperature = input_dict["brightness_temperature"]
        else:
            self.brightness_temperature = False
        if "do_spectral_smooth" in input_dict.keys():
            self.do_spectral_smooth = input_dict["do_spectral_smooth"]
            if self.lambda_restframe is not None:
                self.dlambda = input_dict["dlambda"]
            elif self.nu_restframe is not None:
                self.dnu = input_dict["dnu"]
            else:
                raise ValueError(
                    "You need to specify either lambda_restframe and dlambda or nu_restframe and dnu!"
                )
        else:
            self.do_spectral_smooth = False
        if "do_angular_smooth" in input_dict.keys():
            self.do_angular_smooth = input_dict["do_angular_smooth"]
            self.sigma_beam = input_dict["sigma_beam"]
        else:
            self.do_angular_smooth = False
        if "do_spectral_tophat_smooth" in input_dict.keys():
            self.do_spectral_tophat_smooth = input_dict['do_spectral_tophat_smooth']
        else:
            self.do_spectral_tophat_smooth = False
        if "footprint_radius" in input_dict.keys():
            self.footprint_radius = input_dict["footprint_radius"]
            if isinstance(self.footprint_radius, str):
                self.footprint_radius = eval(self.footprint_radius)
        else:
            self.footprint_radius = None

        # initiate running parameters
        if "seed_lognormal" in input_dict.keys():
            self.seed_lognormal = input_dict["seed_lognormal"]
        else:
            self.seed_lognormal = None
        if self.seed_lognormal is None:
            self.seed_lognormal = np.random.randint(2**32 - 1)
        try:
            self.run_pk = input_dict["run_pk"]
        except:
            self.run_pk = {}
            pass
        for tracer in ["intensity", "n_gal", "cross", "sky_subtracted_intensity", "sky_subtracted_cross"]:
            if tracer not in self.run_pk.keys():
                self.run_pk[tracer] = False

        if "dk" in input_dict.keys():
            self.dk = input_dict["dk"]
        else:
            self.dk = 2 * np.pi / self.box_size.min()
        if "kmin" in input_dict.keys():
            self.kmin = input_dict["kmin"]
        else:
            self.kmin = (0 / self.Mpch).to(1 / self.Mpch)
        if "kmax" in input_dict.keys():
            self.kmax = input_dict["kmax"]
        else:
            self.kmax = (
                np.pi * self.N_mesh.min() / self.box_size.max().to(self.Mpch)
            ).to(1 / self.Mpch)
        try:
            self.nkbin = np.floor(
                (self.kmax - self.kmin) / self.dk).astype(int) - 1
        except:
            self.nkbin = np.floor(((self.kmax.to(1/self.Mpch).value - self.kmin.to(
                1/self.Mpch).value)/self.dk.to(1/self.Mpch).value).astype(int) - 1)

        # initiate obs_mask
        if "obs_mask" in input_dict.keys():
            obs_mask = input_dict["obs_mask"]
            if isinstance(obs_mask, str):
                with h5py.File(obs_mask, "r") as ff:
                    self.obs_mask = ff["mask"][:]
            else:
                self.obs_mask = obs_mask
            print((self.obs_mask.shape, self.N_mesh))
            assert (np.array(self.obs_mask.shape) == self.N_mesh).all()
        else:
            self.obs_mask = None

        if "calc_pk_lognormal" in input_dict.keys():
            self.calc_pk_lognormal = input_dict['calc_pk_lognormal']
        else:
            self.calc_pk_lognormal = False

        self.noise_mesh = None
        self.prepared_skysub_intensity_mesh_ft = None
        self.prepared_intensity_mesh_ft = None
        self.prepared_n_gal_mesh_ft = None
        self.sky_intensity_mesh = None
        self.n_gal_mesh = None
        self.intensity_mesh = None
        logging.info("Done")

    def copy_info(self):
        """
        Initiates a new LognormalIntensityMock instance with the same input parameters as the original one.

        Returns
        -------
        LognormalIntensityMock
            A new instance of LognormalIntensityMock with the same input parameters as the original.

        """

        logging.info("Copying LognormalIntensityMap input information.")
        if 'cosmology' not in self.input_dict.keys():
            self.input_dict['cosmology'] = self.astropy_cosmo
        if 'luminosity_function' not in self.input_dict.keys():
            self.input_dict['luminosity_function'] = self.luminosity_function

        return LognormalIntensityMock(self.input_dict)

    def _convert_input_dictionary(self, data):
        """
        Converts the values in the input dictionary to their respective data types.

        Parameters
        ----------
        data : dict
            The input dictionary containing the data to be evaluated (string to Quantities or numbers).

        Returns
        -------
        dict
            The modified input dictionary with the evaluated strings.

        """

        for key in data.keys():
            if isinstance(data[key], str):
                if " cosmo" in data[key]:
                    data[key] = data[key].replace(
                        " cosmo", " self.astropy_cosmo")
                try:
                    data[key] = eval(data[key])
                except Exception as e:
                    try:
                        data[key] = u.Quantity(data[key])
                    except Exception as e:
                        logging.debug(key, e)

        return data

    # method to read instance from saved file.
    @classmethod
    def from_file(
        cls,
        filename,
        catalog_filename=None,
        only_params=False,
        only_meshes=["intensity_mesh", "noise_mesh", "n_gal_mesh", "obs_mask"],
    ):
        """
        Initialize a LognormalIntensityMock instance from a saved file.

        Parameters
        ----------

        filename : str
            The filename of the saved file.

        catalog_filename : str, optional
            The filename of the galaxy catalog to be read. Default is None.

        only_params : bool, optional
            If True, only the parameters are loaded and not the meshes. Default is False.

        only_meshes : list of str, optional
            The list of specific meshes to be loaded. Default is ["intensity_mesh", "noise_mesh", "n_gal_mesh", "obs_mask"].

        Returns
        -------
        LognormalIntensityMock
            A new instance of LognormalIntensityMock initialized from the saved file.

        """

        with h5py.File(filename, "r") as ff:
            # initiate logger
            if "verbose" in ff.attrs.keys():
                if ff.attrs["verbose"]:
                    level = logging.INFO
                else:
                    level = logging.WARNING
            else:
                level = logging.INFO
            FORMAT = "%(asctime)s simple %(levelname)s: %(message)s"
            logging.basicConfig(format=FORMAT, level=level)
            stream_handler = logging.StreamHandler(stream=sys.stdout)

            logging.info("Initiating LognormalIntensityMock instance.")

            # initiate cosmology
            logging.info("Loading cosmology.")
            astropy_cosmo_table = Table()
            astropy_cosmo_table["name"] = [None]
            for cosmo_key in ff["astropy_cosmo"].keys():
                comp_key = f"astropy_cosmo/{cosmo_key}"
                if "unit" in ff[comp_key].attrs.keys():
                    if ff[comp_key].attrs["unit"] == "None":
                        astropy_cosmo_table[cosmo_key] = [ff[comp_key][()]]
                    else:
                        astropy_cosmo_table[cosmo_key] = [
                            u.Quantity(
                                ff[comp_key][()], unit=ff[comp_key].attrs["unit"]
                            )
                        ]
                        astropy_cosmo_table[cosmo_key].unit = u.Unit(
                            ff[comp_key].attrs["unit"]
                        )
                else:
                    if ff[comp_key][()] == -999:
                        astropy_cosmo_table[cosmo_key] = ff[comp_key].attrs["value"]
                    else:
                        astropy_cosmo_table[cosmo_key] = [ff[comp_key][()]]
                astropy_cosmo_table[cosmo_key].description = ff[comp_key].attrs[
                    "description"
                ]
            for meta_key in ff["astropy_cosmo"].attrs.keys():
                astropy_cosmo_table.meta[meta_key] = ff["astropy_cosmo"].attrs[meta_key]
            astropy_cosmo = Cosmology.from_format(
                astropy_cosmo_table, format="astropy.table"
            )

            # load luminosity_function
            logging.info("Loading luminosity function.")
            luminosity_function_table = Table()
            luminosity_function_table["L"] = ff["luminosity_function/luminosity"][:]
            luminosity_function_table["dn/dL"] = ff[
                "luminosity_function/luminosity_function"
            ][:]

            # load other input data
            logging.info("Loading other input data.")
            input_dict = dict(ff.attrs)
            for key in ff.keys():
                if key not in [
                    "LogNormalCatalog",
                    "astropy_cosmo",
                    "luminosity_function",
                    "intensity_mesh",
                    "n_gal_mesh",
                    "noise_mesh",
                    "obs_mask",
                    "redshift_mesh_axis",
                    "galaxy_selection",
                    "run_pk",
                    "run_corrfunc",
                    "mean_intensity",
                    "input_dict"
                ]:
                    try:
                        input_dict[key] = u.Quantity(
                            ff[key][:], unit=ff[key].attrs["unit"])
                    except Exception as e:
                        logging.error(f"{key}: {e}")

                if key in [
                    "galaxy_selection",
                    "run_pk",
                ]:  # dictionaries
                    input_dict[key] = {}
                    for small_key in ff[key].attrs.keys():
                        input_dict[key][small_key] = ff[key].attrs[small_key]

            input_dict['cosmology'] = astropy_cosmo
            input_dict['luminosity_function'] = luminosity_function_table

            logging.info("Initializing LognormalIntensityMock instance.")
            new_lim_instance = cls(input_dict)

            if not only_params:
                # load meshes
                if ("intensity_mesh" in ff.keys()) and (
                    "intensity_mesh" in only_meshes
                ):
                    logging.info("Initializing intensity_mesh.")
                    new_lim_instance.intensity_mesh = u.Quantity(
                        ff["intensity_mesh"][:], unit=ff["intensity_mesh"].attrs["unit"]
                    )
                    logging.info(
                        f"intensity_mesh.unit: {new_lim_instance.intensity_mesh.unit}"
                    )
                else:
                    new_lim_instance.intensity_mesh = None
                if ("noise_mesh" in ff.keys()) and ("noise_mesh" in only_meshes):
                    logging.info("Initializing noise_mesh.")
                    new_lim_instance.noise_mesh = u.Quantity(
                        ff["noise_mesh"][:], unit=ff["noise_mesh"].attrs["unit"]
                    )
                else:
                    new_lim_instance.noise_mesh = None
                if ("n_gal_mesh" in ff.keys()) and ("n_gal_mesh" in only_meshes):
                    logging.info("Initializing n_gal_mesh.")
                    new_lim_instance.n_gal_mesh = u.Quantity(
                        ff["n_gal_mesh"][:], unit=ff["n_gal_mesh"].attrs["unit"]
                    )
                else:
                    new_lim_instance.n_gal_mesh = None
                if ("obs_mask" in ff.keys()) and ("obs_mask" in only_meshes):
                    new_lim_instance.obs_mask = ff["obs_mask"][:]
                else:
                    new_lim_instance.obs_mask = None

            if catalog_filename is not None:
                new_lim_instance.cat = new_lim_instance.read_galaxy_catalog(
                    catalog_filename
                )
            return new_lim_instance

    def read_galaxy_catalog(self, catalog_filename):
        """
        Initializes the galaxy catalog from a specified file.

        Parameters
        ----------
        catalog_filename : str
            The filename of the galaxy catalog file.

        Returns
        -------
        dict
            A dictionary containing the galaxy catalog data.

        """

        logging.info(f"Initializing catalog from file {catalog_filename}.")
        with h5py.File(catalog_filename, "r") as ff:
            cat = {}
            for cat_key in ff.keys():
                if cat_key in ["L_box", "N_gal"]:
                    continue
                cat[cat_key] = np.array(ff[cat_key][:])
                if "unit" in ff[cat_key].attrs.keys():
                    logging.info(cat_key + " " + ff[cat_key].attrs["unit"])
                    try:
                        unit_string = "u.Unit({})".format(
                            ff[cat_key].attrs["unit"])
                        if " cosmo" in unit_string:
                            unit_string = unit_string.replace(
                                " cosmo", " self.astropy_cosmo"
                            )
                        unit = eval(unit_string)
                    except:
                        unit = u.Unit(ff[cat_key].attrs["unit"])
                    logging.info(cat_key + " " + str(unit))
                    cat[cat_key] = u.Quantity(cat[cat_key], unit=unit)
        return cat

    def save_to_file(self, filename, catalog_filename=None):
        """
        Saves the instance and its attributes to a file.

        Parameters
        ----------
        filename : str
            The filename of the file to save the instance to.
        catalog_filename : str, optional
            The filename of the galaxy catalog file to save (default is None).
            If None, it does not save the galaxy catalog.

        """

        attributes = []
        for attr in dir(self):
            if attr.startswith("_") or (attr in ['prepared_intensity_mesh', 'prepared_skysub_intensity_mesh', 'prepared_n_gal_mesh', "input_dict"]):
                continue
            # don't leave it in one line with the previous because it may cause problems.
            elif callable(getattr(self, attr)):
                continue
            else:
                attributes.append(attr)

        with h5py.File(filename, "w") as ff:
            # save mean intensity
            ff["mean_intensity"] = self.mean_intensity.value
            ff["mean_intensity"].attrs["unit"] = str(self.mean_intensity.unit)

            if callable(self.min_flux):
                ff.attrs['min_flux'] = self.min_flux_file
            if callable(self.sigma_noise):
                ff.attrs['sigma_noise'] = self.sigma_noise_file

            # save luminosity_function
            max_L_for_saving = self.Lmax.to(self.luminosity_unit).value
            if np.isfinite(max_L_for_saving):
                max_log10_L_for_saving = np.log10(max_L_for_saving)
            else:
                max_log10_L_for_saving = np.log10(
                    (1e5 * self.Lmin).to(self.luminosity_unit).value
                )
            min_log10_L_for_saving = np.log10(
                self.Lmin.to(self.luminosity_unit).value)
            N_save = 10000
            dlog_10_L = (max_log10_L_for_saving -
                         min_log10_L_for_saving) / N_save
            log_10_Ls = np.linspace(
                min_log10_L_for_saving, max_log10_L_for_saving, N_save
            )
            ff["luminosity_function/luminosity"] = 10**log_10_Ls
            ff["luminosity_function/luminosity_function"] = getattr(
                self, "luminosity_function"
            )(10**log_10_Ls)

            for key in attributes:
                # ignore Nones
                if getattr(self, key) is None:
                    pass
                # save cosmology
                elif key == "astropy_cosmo":
                    ct = Table(getattr(self, key))
                    grp = ff.create_group("astropy_cosmo")
                    for ct_key in ct.keys():
                        if ct[ct_key][0] is None:
                            continue
                        elif isinstance(ct[ct_key][0], str):
                            ff[f"astropy_cosmo/{ct_key}"] = -999
                            ff[f"astropy_cosmo/{ct_key}"].attrs["value"] = ct[ct_key][
                                0
                            ].astype("S1")
                        else:
                            ff[f"astropy_cosmo/{ct_key}"] = ct[ct_key][0].astype(
                                np.float64
                            )
                            ff[f"astropy_cosmo/{ct_key}"].attrs["unit"] = str(
                                ct[ct_key].unit
                            )
                        ff[f"astropy_cosmo/{ct_key}"].attrs["description"] = str(
                            ct[ct_key].description
                        )
                    for (
                        meta_key
                    ) in (
                        ct.meta.keys()
                    ):  # important for initializing the cosmology (FlatLambdaCDM)
                        ff["astropy_cosmo"].attrs[meta_key] = ct.meta[meta_key]

                # save meshes
                elif key in ["intensity_mesh", "noise_mesh", "n_gal_mesh"]:
                    try:
                        ff[key] = getattr(self, key).value
                        ff[key].attrs["unit"] = str(getattr(self, key).unit)
                    except AttributeError:
                        logging.info(key)
                elif key == "obs_mask":
                    ff[key] = getattr(self, key)

                # save catalog
                elif isinstance(getattr(self, key), Table):
                    continue

                # ignore easily derived attributes
                elif key in [
                    "compensation",
                    "nbody_cosmo",
                    "_input_power_spectrum",
                    "cat",
                    "comm",
                ]:
                    pass

                elif type(getattr(self, key)) == u.Quantity:
                    if getattr(self, key).size > 1:
                        ff[key] = getattr(self, key).value
                        ff[key].attrs["unit"] = str(getattr(self, key).unit)
                    else:
                        ff.attrs[key] = str(getattr(self, key))

                elif isinstance(getattr(self, key), dict):
                    tmp_dict = getattr(self, key)
                    try:
                        grp = ff.create_group(key)
                        for little_key in tmp_dict.keys():
                            ff[key].attrs[little_key] = tmp_dict[little_key]
                    except Exception as e:
                        logging.error("Error with dict {}: {}".format(key, e))

                else:
                    try:
                        ff.attrs[key] = getattr(self, key)
                    except Exception as e:
                        logging.error(
                            "Error with {} of type {}:".format(
                                key, type(getattr(self, key))
                            )
                        )
                        logging.error(e)

        if catalog_filename is not None:
            self.save_galaxy_catalog(
                keys=[
                    "Position",
                    "RSD_Position",
                    "Velocity",
                    "cosmo_redshift",
                    "RSD_redshift",
                    "luminosity",
                    "detected",
                    "flux",
                ],
                galaxy_selection="all",
                filename=catalog_filename,
            )
            logging.info(
                f"Saved instance to file {filename} and {catalog_filename}.")
            return

    def _downsample_mesh(self, mesh, new_N_mesh):
        """
        Downsamples a given mesh to a new grid resolution using NGP.

        Parameters
        ----------
        mesh : ndarray or astropy Quantity array
            The input mesh to downsample.
        new_N_mesh : list of int (len 3)
            The new grid resolution for downsampling.

        Returns
        -------
        ndarray or astropy Quantity array
            The downsampled mesh.

        """

        try:
            unit = mesh.unit
        except:
            unit = None
        map = make_map(mesh.value, Nmesh=self.N_mesh,
                       BoxSize=self.box_size.to(self.Mpch).value)
        pm_down = pmesh.pm.ParticleMesh(new_N_mesh,
                                        BoxSize=self.box_size.to(self.Mpch).value, dtype='float32', resampler="nearest")
        map = pm_down.downsample(map, keep_mean=True)
        if unit is not None:
            map = (np.array(map) * unit).to(unit)
        return map

    def downsample_all_meshes(self, new_N_mesh):
        """
        Downsamples all meshes to a new grid resolution using NGP: 
        n_gal_mesh, intensity_mesh, obs_mask, sky_intensity_mesh, noise_mesh.

        Parameters
        ----------
        new_N_mesh : list of int (len 3)
            The new grid resolution for downsampling.

        """

        assert isinstance(new_N_mesh[0], int)

        logging.info("Downsampling all meshes...")
        if self.n_gal_mesh is not None:
            self.n_gal_mesh = self._downsample_mesh(
                self.n_gal_mesh, new_N_mesh)
            logging.info("Finished n_gal_mesh.")
        if self.intensity_mesh is not None:
            self.intensity_mesh = self._downsample_mesh(
                self.intensity_mesh, new_N_mesh)
            logging.info("Finished intensity_mesh.")
        if self.obs_mask is not None:
            self.obs_mask = self._downsample_mesh(self.obs_mask, new_N_mesh)
            logging.info("Finished obs_mask.")
        if self.sky_intensity_mesh is not None:
            self.sky_intensity_mesh = self._downsample_mesh(
                self.sky_intensity_mesh, new_N_mesh)
            logging.info("Finished sky_intensity_mesh.")
        if self.noise_mesh is not None:
            self.noise_mesh = self._downsample_mesh(
                self.noise_mesh, new_N_mesh)
            logging.info("Finished noise_mesh.")

        self.N_mesh = np.array(new_N_mesh)
        self.voxel_size = (self.box_size / self.N_mesh).to(self.Mpch)
        try:
            del self.redshift_mesh_axis
        except:
            pass
        try:
            del self.voxel_volume
        except:
            pass
        try:
            del self.k_Nyquist
        except:
            pass
        try:
            del self.mean_intensity_per_redshift_mesh
        except:
            pass
        try:
            del self.mean_redshift
        except:
            pass
        self.prepared_n_gal_mesh_ft = None
        self.prepared_intensity_mesh_ft = None
        self.prepared_skysub_intensity_mesh_ft = None
        logging.info("Done.")

    @functools.cached_property
    def n_bar_gal(self):
        """
        Calculates the mean galaxy number density by
        integrating over the luminosity function from Lmin to Lmax.
        It does not take the selection function into account.
        Make sure that the luminosity function output is in units of
        luminosity_function: dn/dL
        integral: 

        .. math:: \\int \\frac{\\mathrm{d}n}{\\mathrm{d}L} \\mathrm{d}L = \\bar{n}_\\mathrm{g} [\\mathrm{Mpc}^{-3}]

        IMPORTANT: The luminosity function input has to be in units of [Mpc^-3 luminosity_unit^-1]
        so that \int dn/dl dL is in units of [Mpc^-3].

        Cached like a property self.n_bar_gal.

        Returns
        -------
        Quantity
            The expected mean galaxy number density in units of 1/Mpc**3,
            not taking into account the selection function.

        """

        logging.info("Getting mean galaxy number density.")
        integrated = quad(
            self.luminosity_function,
            self.Lmin.to(self.luminosity_unit).value,
            self.Lmax.to(self.luminosity_unit).value,
        )
        logging.info(
            "Integration result and uncertainty (should be much smaller than result!): {}".format(
                integrated
            )
        )
        n_bar_gal = (integrated[0] * u.Mpc ** (-3)).to(u.Mpc ** (-3))
        logging.info("Done.")
        return n_bar_gal

    @functools.cached_property
    def n_bar_gal_mesh(self):
        """ Calculates the mean galaxy number density from the mesh self.n_gal_mesh.
            Cached like a property self.n_bar_gal_mesh."""
        mean_ngal = np.mean(self.n_gal_mesh)
        return mean_ngal.to(u.Mpc ** (-3))

    @functools.cached_property
    def N_gal(self):
        """ Calculates the number of all galaxies (no selection function) from self.n_bar_gal.
            Cached like a property self.N_gal"""
        return int((self.n_bar_gal * self.box_volume.to(u.Mpc**3)).to(1))

    @functools.cached_property
    def N_gal_detected(self):
        """ Outputs the number of detected galaxies in the galaxy catalog (no mask).
            Cached like a property self.N_gal_detected."""
        return self.cat["detected"][self.cat["detected"]].size

    @functools.cached_property
    def n_gal_detected(self):
        """ Outputs the number density of detected galaxies in the galaxy catalog (no mask).
            Cached like a property self.n_gal_detected."""
        return (self.N_gal_detected / self.box_volume).to(1 / u.Mpc**3)

    def run_lognormal_simulation_cpp(self, output_velocity=0):
        """
        Calculates the power spectrum from input cosmology
        and runs lognormal simulation in lognormal_galaxies.
        Calculates the RSD position given the LOS.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """

        logging.info("Running lognormal simulation using lognormal_galaxies.")
        logging.info(
            "Expecting {} galaxies in the simulation.".format(self.N_gal))

        print(self.astropy_cosmo.m_nu)
        params = {
            "ofile_prefix": self.outfile_prefix,
            "inp_pk_fname": self.input_pk_filename,
            "xi_fname": "",
            "pkg_fname": "",
            "mpkg_fname": "",
            "cpkg_fname": "",
            "f_fname": self.f_growth_filename,
            "z": self.redshift,
            "mnu": np.sum(self.astropy_cosmo.m_nu.to(u.eV).value),
            "oc0h2": self.astropy_cosmo.Odm0 * self.astropy_cosmo.h**2,
            "ob0h2": self.astropy_cosmo.Ob0 * self.astropy_cosmo.h**2,
            "ns": self.n_s,  # 0.9645,
            "lnAs": self.lnAs,  # 3.094,
            "h0": self.astropy_cosmo.H0 / (100 * u.km / u.s / u.Mpc),
            "w": self.astropy_cosmo.w(self.redshift),
            "run": 0.0,
            "bias": self.bias,
            "bias_mpkG": 1.0,
            "bias_cpkG": 1.35,
            "Nrealization": 1,
            "Ngalaxies": self.N_gal,
            "Lx": self.box_size[0].to(self.Mpch).value,
            "Ly": self.box_size[1].to(self.Mpch).value,
            "Lz": self.box_size[2].to(self.Mpch).value,
            "rmax": 10000.0,
            "seed": self.seed_lognormal,
            "Pnmax": int(np.max(self.N_mesh)),
            # "losx": 1.0,
            "losy": 0.0,
            "losz": 0.0,
            "kbin": 0.01,
            "kmax": 0.0,
            "lmax": 4,
            "gen_inputs": True,
            "run_lognormal": True,
            "calc_pk": self.calc_pk_lognormal,
            "calc_cpk": False,
            "use_cpkG": 0,
            "output_matter": 0,
            "output_gal": 1,
            "output_velocity": output_velocity,
            "calc_mode_pk": 0,
            "out_dir": self.out_dir,
            "halofname_prefix": "",
            "imul_fname": "",
            "num_para": 1,
            "om0h2": self.astropy_cosmo.Om0 * self.astropy_cosmo.h**2,
            "om0": self.astropy_cosmo.Om0,
            "ob0": self.astropy_cosmo.Ob0,
            "ode0": self.astropy_cosmo.Ode0,
        }

        if self.RSD:
            params["losx"] = 1.0
        else:
            params["losx"] = 0.0

        params["As"] = np.exp(params["lnAs"]) * 1e-10
        params["aH"] = (
            100.0
            * pow(params["om0"] * pow(1.0 + params["z"], 3) + params["ode0"], 0.5)
            / (1.0 + params["z"])
        )
        # if params['inp_pk_fname'] is blanck,  use Eisenstein & Hu for input pk
        # if params["inp_pk_fname"] is None:
        #    params["inp_pk_fname"] = os.path.join(
        #        params["out_dir"], "inputs", params["ofile_prefix"] + "_pk.txt"
        #    )
        if params["xi_fname"] == "":
            params["xi_fname"] = os.path.join(
                params["out_dir"], "inputs", params["ofile_prefix"] + "_Rh_xi.txt"
            )
        if params["pkg_fname"] == "":
            params["pkg_fname"] = os.path.join(
                params["out_dir"], "inputs", params["ofile_prefix"] + "_pkG.dat"
            )
        if params["mpkg_fname"] == "":
            params["mpkg_fname"] = os.path.join(
                params["out_dir"], "inputs", params["ofile_prefix"] + "_mpkG.dat"
            )
        if params["cpkg_fname"] == "":
            if params["use_cpkG"] == 0:
                params["cpkg_fname"] = params["mpkg_fname"]  # dummy
            else:
                params["cpkg_fname"] = os.path.join(
                    params["out_dir"], "inputs", params["ofile_prefix"] + "_cpkG.dat"
                )
        # if params["f_fname"] == "":
        #    params["f_fname"] = os.path.join(
        #        params["out_dir"], "inputs", params["ofile_prefix"] + "_fnu.txt"
        #    )

        print(params)

        check_dir_im(params)

        exe = executable("exe")

        # create seeds for random
        random.seed(int(params["seed"]))
        seed1 = [random.randint(1, 100000)]
        seed2 = [random.randint(1, 100000)]
        seed3 = [random.randint(1, 100000)]

        # generate input files
        if params["gen_inputs"]:
            gen_inputs(params, exe)
        else:
            logging.info("skip: generating input files")

        # generate Poisson catalog
        if params["run_lognormal"]:
            args = (0, params, seed1, seed2, seed3, exe)
            run = wrap_gen_Poisson(args)
        else:
            logging.info("skip: generating log-normal catalog")

        # calculate Pk
        if params['calc_pk']:
            args = (0, params, exe)
            run = wrap_calc_Pk(args)
        else:
            print('skip: calculate Pk')

    @functools.cached_property
    def _input_power_spectrum(self):
        """
        Returns an interpolation function for the input power spectrum.

        Returns
        -------
        callable
            Interpolation function for the input power spectrum.

        """

        input_power_spectrum_filename = os.path.join(
            self.out_dir, "inputs", self.outfile_prefix + "_pk.txt"
        )
        pk_tab = Table.read(input_power_spectrum_filename, format="ascii")
        # col1: wavenumber [h/Mpc].
        return interp1d(pk_tab["col1"], pk_tab["col2"])
        # col2: power spectrum [Mpc^3/h^3]

    @functools.cached_property
    def lognormal_bin_filename(self):
        """ Automatically generates the filename for the binary output file of lognormal_galaxies galaxy catalog."""
        return os.path.join(
            self.out_dir,
            "lognormal",
            self.outfile_prefix + "_lognormal_rlz0.bin",
        )

    def load_lognormal_catalog_cpp(self, bin_filename, delete_bin=True):
        """
        Loads the lognormal catalog from a binary file.

        Parameters
        ----------
        bin_filename : str
            Path to the binary file containing the lognormal galaxy catalog (output from lognormal_galaxies).
        delete_bin: bool
            Set to False if you want to keep the binary file.
            Default: True.

        Returns
        -------
        None.

        Raises
        ------
        AssertionError
            If the number of galaxies in the lognormal catalog does not match the number of positions in the loaded catalog.

        """

        self.catalog_filename, N_gal_lognormal = transform_bin_to_h5(
            bin_filename)
        logging.info("Done transforming the bin to h5 file.")
        self.cat = self.read_galaxy_catalog(self.catalog_filename)
        logging.info(
            f"N_gal mismatch? {N_gal_lognormal}, {len(self.cat['Position'])}")
        assert N_gal_lognormal == len(self.cat["Position"])
        self.cat["RSD_redshift_factor"] = (
            1 + da.dot(self.cat["Velocity"], self.LOS).compute() / const.c
        ).to(1)
        if delete_bin:
            os.remove(bin_filename)

    def input_power_spectrum(self, k):
        """
        Calculates the power spectrum corresponding to the given wavenumber values.

        Parameters
        ----------
        k : array-like
            Wavenumber values in units of [h / Mpc].

        Returns
        -------
        Quantity
            Power spectrum values in units of [Mpc^3/h^3].

        """

        return (self._input_power_spectrum(k) * self.Mpch**3).to(self.Mpch**3)

    def get_comoving_distance_per_delta_z(self, mean_z, delta_z):
        """
        Calculates the comoving distance between (z + delta_z) and (z - delta_z).

        Parameters
        ----------
        mean_z : float
            Mean redshift of the box: mean_z = (z_min + z_max) / 2.
        delta_z : float
            Half of the redshift difference: delta_z = (z_max - z_min) / 2.

        Returns
        -------
        Quantity
            Comoving distance between (z + delta_z) and (z - delta_z).

        """

        return self.astropy_cosmo.comoving_distance(
            mean_z + delta_z
        ) - self.astropy_cosmo.comoving_distance(mean_z - delta_z)

    def assign_redshift_along_axis(self):
        """
        Assigns redshift along the line-of-sight (LOS) axis
        (x, y, or z) given the mean/middle redshift.
        Also assigns the redshift-space distortion 
        (RSD) redshift and inferred positions.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the LOS is not aligned with any of the axes [1,0,0], [0,1,0], or [0,0,1].

        Notes
        -----
        This function requires a lognormal galaxy catalog. If the catalog does not exist, it will generate it using lognormal simulations.

        """

        # check if there is a lognormal galaxy catalog yet. Otherwise generate it.
        try:
            self.cat["Position"][0]
        except AttributeError:
            self.run_lognormal_simulation_cpp()
            self.load_lognormal_catalog_cpp(
                bin_filename=self.lognormal_bin_filename)

        logging.info(
            "Assigning redshift along axis {}.".format(
                np.where(self.LOS == 1)[0][0])
        )
        # I feel like there should be an easier way to get from mean redshift and box length to redshift range...
        axis = np.where(self.LOS == 1)[0]
        if len(axis) != 1:
            raise ValueError("LOS must be [1,0,0] or [0,1,0] or [0,0,1].")
        axis = axis[0]
        zs = np.linspace(
            self.redshift - self.delta_redshift - 0.1,
            self.redshift + self.delta_redshift + 0.1,
            int(self.N_mesh[axis]),
        )
        distance_at_z = self.astropy_cosmo.comoving_distance(
            zs).to(u.Mpc).value
        zs_at_distance = interp1d(distance_at_z, zs)
        minimum_distance = self.minimum_distance
        # no RSD
        self.cat["cosmo_redshift"] = zs_at_distance(
            (minimum_distance + self.cat["Position"][:, axis])  # * self.Mpch
            .to(u.Mpc)
            .value
        )
        # with RSD
        self.cat["RSD_redshift"] = (
            self.cat["cosmo_redshift"] * self.cat["RSD_redshift_factor"]
        )
        self.cat["RSD_Position"] = (
            self.cat["Position"]
            + (
                (self.cat["Velocity"] * self.LOS).T
                * (1 + self.cat["cosmo_redshift"])
                / (self.astropy_cosmo.H(self.cat["cosmo_redshift"]))
            )
            .to(self.Mpch)
            .T
        )
        logging.info("Done.")
        return

    def assign_single_redshift(self):
        """
        Assigns a single redshift to the entire box.
        Adds the columns 'cosmo_redshift', 'RSD_redshift',
        'RSD_redshift_factor', 'RSD_Position' to self.cat.

        Returns
        -------
        None

        Notes
        -----
        This function requires a lognormal galaxy catalog.
        If the catalog does not exist, it will generate it using lognormal simulations.

        """

        # check if there is a lognormal galaxy catalog yet. Otherwise generate it.
        try:
            self.cat["Position"][0]
        except AttributeError:
            self.run_lognormal_simulation_cpp()
            self.load_lognormal_catalog_cpp(
                bin_filename=self.lognormal_bin_filename)

        logging.info(
            "Assigning single redshift for the whole box: {}.".format(
                self.redshift)
        )
        self.cat["cosmo_redshift"] = self.redshift * \
            np.ones(len(self.cat['RSD_redshift_factor']))
        self.delta_redshift = 0.0
        # with RSD
        self.cat["RSD_redshift"] = (
            self.cat["cosmo_redshift"] * self.cat["RSD_redshift_factor"]
        )
        self.cat["RSD_Position"] = (
            self.cat["Position"]
            + (
                (self.cat["Velocity"] * self.LOS)
                # * u.km
                # / u.s
                * ((1 + self.cat["cosmo_redshift"])
                   / (self.astropy_cosmo.H(self.cat["cosmo_redshift"])))[:, None]
            )
            .to(self.Mpch)
            # .value
        )  # in Mpc / h
        logging.info("Done.")
        return

    def log10_luminosity_function(self, log10_L):
        """ Log_10 of the luminosity function."""
        L = 10**log10_L
        # dn/dlog10_L = dn/dL dL/dlog10_L = dn/dl * ln(10) * L
        return self.luminosity_function(L) * np.log(10) * L

    def assign_luminosity(self):
        """
        Randomly assigns luminosities to the galaxies following the luminosity function.
        Adds the column 'luminosity' to self.cat.

        Returns
        -------
        None

        Notes
        -----
        This function requires a lognormal galaxy catalog. 
        If the catalog does not exist, it will generate it using lognormal simulations.

        """

        # check if there is a lognormal galaxy catalog yet. Otherwise generate it.
        try:
            self.cat["Position"][0]
        except AttributeError:
            self.run_lognormal_simulation_cpp()
            self.load_lognormal_catalog_cpp(
                bin_filename=self.lognormal_bin_filename)

        logging.info("Assigning luminosities.")

        max_L_for_probability = self.Lmax.to(self.luminosity_unit).value
        if not np.isfinite(max_L_for_probability):
            max_log10_L_for_probability = np.log10(
                (1e10 * self.Lmin).to(self.luminosity_unit).value
            )
        else:
            max_log10_L_for_probability = np.log10(
                (self.Lmax).to(self.luminosity_unit).value
            )
        min_log10_L_for_probability = np.log10(
            self.Lmin.to(self.luminosity_unit).value)
        N_probability = 10000
        dlog_10_L = (
            max_log10_L_for_probability - min_log10_L_for_probability
        ) / N_probability
        Ls = np.linspace(
            min_log10_L_for_probability, max_log10_L_for_probability, N_probability
        )
        probability = self.log10_luminosity_function(Ls)
        probability = np.max([np.zeros(len(probability)), probability], axis=0)
        probability = probability / np.sum(probability)
        logging.info(f"Minimum probability: {np.min(probability)}")

        np.random.seed(self.seed_lognormal)

        offsets = np.random.uniform(
            low=-0.5 * dlog_10_L,
            high=0.5 * dlog_10_L,
            size=self.cat["Position"].shape[0],
        )
        log10_L_choice = (
            np.random.choice(
                Ls, size=self.cat["Position"].shape[0], p=probability)
            + offsets
        )
        L_choice = 10 ** (log10_L_choice)
        logging.info("log10_L_choice: length {}".format(L_choice.shape))

        self.cat["luminosity"] = L_choice * self.luminosity_unit
        self.Lmax = self.cat["luminosity"].max()
        logging.info("Done.")
        return

    def assign_flux(self, luminosity="luminosity"):
        """
        Converts luminosity to flux in the galaxy catalog.
        Accounts for cosmological effects, not for peculiar velocities.
        Adds the column 'flux' to self.cat.

        Parameters
        ----------
        luminosity : str, optional
            Key name of the luminosity column in the catalog, by default "luminosity".

        Returns
        -------
        None

        Notes
        -----
        This function requires the galaxies to have assigned redshifts.
        If the redshifts are not assigned, they will be assigned using either a single redshift or redshift along the LOS.

        """

        # check if the galaxies are assigned a redshift.
        try:
            self.cat["cosmo_redshift"][0]
        except:  # otherwise assign it.
            if self.single_redshift:
                self.assign_single_redshift()
            else:
                self.assign_redshift_along_axis()

        logging.info("Converting luminosity to flux.")
        # check if the galaxies have luminosities. Otherwise assign them.
        try:
            self.cat[luminosity][0]
        except KeyError:
            self.assign_luminosity()

        min_z, max_z = (
            self.cat["cosmo_redshift"].min(),
            self.cat["cosmo_redshift"].max(),
        )
        zs = np.linspace(min_z, max_z, 1000)
        dl_at_z = interp1d(
            zs, self.astropy_cosmo.luminosity_distance(zs).to(u.Mpc).value
        )
        DL = (dl_at_z(self.cat["cosmo_redshift"]) * u.Mpc).to(u.cm)

        flux = self.cat[luminosity] / (4 * np.pi * DL**2)
        self.cat["flux"] = flux
        logging.info("Mean flux: {}".format(np.mean(self.cat['flux'])))
        logging.info("Done.")
        return

    def apply_selection_function(self):
        """
        Assigns whether each galaxy was detected or not given the flux limit (min_flux)
        of the observation.

        Parameters
        ----------
        min_flux : astropy quantity, nd-array of floastropy quantities, or function of redshift, optional
            Flux limit of the (mock) observation above which galaxies are detected.
            If a function, it represents the flux limit as a function of redshift.
            Units of flux (erg/s/cm**2) required.
            Either `min_flux` or `limit_ngal` must be provided.
        limit_ngal : float or function of redshift, optional
            Given number of galaxies per unit comoving volume above which galaxies are detected.
            If a function, it represents the number density limit as a function of redshift.
            Either `min_flux` or `limit_ngal` must be provided.

        Returns
        -------
        None

        Notes
        -----
        This function requires the galaxies to have assigned fluxes. If the fluxes are not assigned, they will be computed using the `assign_flux()` method.
        If neither min_flux nor limit_ngal are given, all galaxies will be detected.

        """

        # Check if the galaxies have fluxes, otherwise compute them.
        if not ("flux" in self.cat.keys()):
            self.assign_flux()

        logging.info("Applying selection function.")

        if self.min_flux is not None:  # if min_flux was given as an input
            if self.limit_ngal is not None:
                logging.warning(
                    'Only provide either "min_flux" or "limit_ngal"! We will use min_flux'
                )
            min_flux = self.min_flux
            if callable(
                min_flux
            ):  # if the flux limit is a function of redshift/wavelength.
                logging.info('min_flux is callable')
                min_flux = min_flux(self.cat["cosmo_redshift"])
                self.cat["detected"] = np.array(self.cat["flux"] > min_flux)
            elif np.size(min_flux) > 1:
                assert (np.array(np.shape(min_flux)) == self.N_mesh).all()
                self.cat["detected"] = apply_selection_function_by_position(self.cat['Position'],
                                                                            self.cat['flux'].to(
                                                                                u.erg/u.s/u.cm**2).value,
                                                                            min_flux.to(
                                                                                u.erg/u.s/u.cm**2).value,
                                                                            self.N_mesh,
                                                                            self.box_size.to(self.Mpch))
                self.cat['detected'] = np.array(
                    self.cat['detected']).astype(bool)
            else:
                self.cat["detected"] = np.array(self.cat["flux"] > min_flux)

        elif self.limit_ngal is not None:  # if instead limit_ngal was given as an input
            min_L_for_interpolation = self.Lmin.to(self.luminosity_unit).value
            max_L_for_interpolation = np.log10(
                self.Lmax.to(self.luminosity_unit).value)
            if not np.isfinite(max_L_for_interpolation):
                max_L_for_interpolation = (
                    (1000 * self.Lmin).to(self.luminosity_unit).value
                )
            Ls = np.logspace(
                np.log10(min_L_for_interpolation),
                np.log10(max_L_for_interpolation),
                1000,
            )
            n_gal_of_Lmin = [
                quad(self.luminosity_function, Lmin_tmp,
                     max_L_for_interpolation)[0]
                for Lmin_tmp in Ls
            ]
            interp_n_gal_of_Lmin = interp1d(n_gal_of_Lmin, Ls)
            if callable(self.limit_ngal):
                redshifts = np.linspace(
                    self.redshift - self.delta_redshift - 0.01,
                    self.redshift + self.delta_redshift + 0.1,
                    100,
                )
                min_flux_det = []
                flux_unit = 1e-17 * u.erg / u.s / u.cm**2
                for i in range(len(redshifts)):
                    goal_nbar = self.limit_ngal(redshifts[i])
                    min_lum_det = (
                        interp_n_gal_of_Lmin(goal_nbar.to(u.Mpc ** (-3)).value)
                        * self.luminosity_unit
                    )
                    min_flux_det.append(
                        (
                            min_lum_det
                            / (
                                4
                                * np.pi
                                * self.astropy_cosmo.luminosity_distance(redshifts[i])
                                ** 2
                            )
                        )
                        .to(flux_unit)
                        .value
                    )
                self.min_flux = interp1d(redshifts, min_flux_det)
                self.cat["detected"] = np.array(
                    self.cat["flux"].to(flux_unit).value
                    > self.min_flux(self.cat["cosmo_redshift"])
                )

            else:
                min_lum_det = (
                    interp_n_gal_of_Lmin(
                        self.limit_ngal.to(u.Mpc ** (-3)).value)
                    * self.luminosity_unit
                )
                mean_dL_sq_inv = np.mean(
                    self.astropy_cosmo.luminosity_distance(
                        self.redshift_mesh_axis)
                    ** (-2)
                )
                first_min_flux = min_lum_det / (4 * np.pi) * mean_dL_sq_inv
                first_min_flux = first_min_flux.to(
                    1e-17 * u.erg / u.s / u.cm**2)

                logging.info("Getting min_flux_of_nbar")
                min_fluxes_for_interpolation = np.linspace(
                    0.8 * first_min_flux, 1.2 * first_min_flux, 20
                )
                n_bars = []
                i = 0
                N = len(min_fluxes_for_interpolation)
                redshift_mesh_axis = self.redshift_mesh_axis
                for mf in min_fluxes_for_interpolation:
                    n_bar = np.mean(
                        [
                            quad(
                                self.luminosity_function,
                                (
                                    mf
                                    * 4
                                    * np.pi
                                    * self.astropy_cosmo.luminosity_distance(z) ** 2
                                )
                                .to(self.luminosity_unit)
                                .value,
                                max_L_for_interpolation,
                            )[0]
                            for z in redshift_mesh_axis
                        ]
                    )
                    n_bars.append(n_bar)
                    i += 1
                logging.info(
                    "Min/max n_bar: {:e} {:e}".format(min(n_bars), max(n_bars))
                )
                min_flux_of_nbar = interp1d(
                    n_bars, min_fluxes_for_interpolation)
                self.min_flux = (
                    min_flux_of_nbar(self.limit_ngal.to(1 / u.Mpc**3).value)
                    * 1e-17
                    * u.erg
                    / u.s
                    / u.cm**2
                )
                self.cat["detected"] = np.array(
                    self.cat["flux"] > self.min_flux)
            logging.info("Wanted detected n_gal: {:e}".format(self.limit_ngal))
            logging.info("Realized detected n_gal: {:e}".format(
                self.n_gal_detected))
            logging.info("Factor: {:.3f}".format(
                self.n_gal_detected / self.limit_ngal))

        else:
            self.cat['detected'] = np.ones(lim.cat['flux'].shape, dtype=bool)

        logging.info("Fraction of detected galaxies: {}".format(
            len(self.cat['detected'][self.cat['detected']])/len(self.cat['detected'])))

        logging.info("Done.")
        return

    def save_galaxy_catalog(
        self,
        cat=None,
        keys=["Position", "Velocity"],
        filename="{}_catalog.h5",
        galaxy_selection="detected",
    ):
        """
        Saves the galaxy catalog, either all galaxies or only detected or undetected galaxies,
        as an h5 file.

        Parameters
        ----------
        cat : LognormalCatalog object, optional
            Which catalog to save. Default: None. If None, use self.cat.
        keys : list of strings, optional
            Which columns to save of the catalog. Options: Position, RSD_Position, Velocity, VelocityOffset,
            cosmo_redshift, RSD_redshift, luminosity, flux, detected.
            Default: Position and Velocity.
        filename : str, optional
            Name of the output catalog file. If it contains a {}, it will be filled with the galaxy_selection argument.
            Default: '{}_galaxies.h5'.
        galaxy_selection : str, optional
            Whether to save all, detected, or undetected galaxies. Options: 'all', 'detected', 'undetected'.
            Default: 'detected'.

        Raises
        ------
        ValueError
            If galaxy_selection is not 'all', 'detected', or 'undetected'.

        Returns
        -------
        None

        Notes
        -----
        This function saves the specified columns of the galaxy catalog as an h5 file.
        The saved file can be opened using h5py and converted into an astropy Table:
        >>> import h5py
        >>> from astropy.table import Table
        >>> with h5py.File(filename, 'r') as F:
        >>>     tab = Table(F)
        """

        logging.info(
            "Saving extensions: {} for {} galaxies.".format(
                keys, galaxy_selection)
        )
        if cat is None:
            cat = self.cat

        if galaxy_selection == "all":
            mask = np.ones(cat["Position"].shape[0], dtype=bool)
        elif galaxy_selection == "detected":
            mask = np.array(cat["detected"])
        elif galaxy_selection == "undetected":
            mask = np.logical_not(cat["detected"])
        else:
            raise ValueError(
                "galaxy_selection must be either 'all' or 'detected' or 'undetected'."
            )
        filename = filename.format(galaxy_selection)

        with h5py.File(filename, "w") as F:
            for key in keys:
                try:
                    F[key] = cat[key][mask]
                    try:
                        F[key].attrs["unit"] = str(cat[key][0].unit)
                    except Exception as e:
                        logging.debug(
                            "Debug error for key {}: {}.".format(key, e))
                        try:
                            F[key].attrs["unit"] = str(
                                cat[key][0].compute().unit)
                        except Exception as ee:
                            logging.info(
                                "Key {} has no unit: {}".format(key, e))
                            pass
                        pass
                except KeyError:
                    logging.warning(f"Couldn't find key {key} in the catalog.")
        F.close()
        logging.info(f"Saved to {filename}.")
        return

    def paint_intensity_mesh(
        self,
        position="RSD_Position",
        redshift="cosmo_redshift",
        tracer="intensity",
        luminosity="luminosity",
    ):
        """
        Generates the mock intensity map obtained from Cartesian coordinates.
        It does not include noise.

        Parameters
        ----------
        position : str, optional
            The column name in the catalog representing the position of galaxies.
            Options: "RSD_Position" (redshift space), "Position" (real space).
            Default: "RSD_Position".
        redshift : str, optional
            The column name in the catalog representing the redshift of galaxies.
            This is used to calculate the intensity and irrelevant for tracer "n_gal".
            Options: "cosmo_redshift" (without RSD), "RSD_redshift" (which RSD).
            Default: "cosmo_redshift".
        tracer : str, optional
            The tracer to generate the intensity map for.
            Options: "intensity", "n_gal".
            Default: "intensity".
        luminosity : str, optional
            The column name in the catalog representing the luminosity of galaxies.
            Options: "luminosity".
            Default: "luminosity".

        Returns
        -------
        field : ndarray
            The generated intensity map.

        Raises
        ------
        ValueError
            If `tracer` argument is not one of the allowed options.

        """

        galaxy_selection = self.galaxy_selection[tracer]
        # check if the selection function was applied, otherwise apply it.
        if galaxy_selection in ["detected", "undetected"]:
            try:
                self.cat["detected"][0]
            except:
                self.apply_selection_function()
        if (position == "RSD_Position") or (redshift == "RSD_redshift"):
            try:
                self.cat["RSD_Position"][0]
            except:
                if self.single_redshift:
                    self.assign_single_redshift()
                else:
                    self.assign_redshift_along_axis()
        if tracer == 'intensity':
            try:
                self.cat[luminosity][0]
            except:
                self.assign_luminosity()

        logging.info(
            "Painting {} mesh: {} galaxies, {}.".format(
                tracer, galaxy_selection, position
            )
        )

        if galaxy_selection == "all":
            mask = np.ones(self.cat[position].shape[0], dtype=bool)
        elif galaxy_selection == "detected":
            mask = np.array(self.cat["detected"])
        elif galaxy_selection == "undetected":
            mask = np.logical_not(self.cat["detected"])
        else:
            raise ValueError(
                "galaxy_selection must be either 'all' or 'detected' or 'undetected'."
            )

        Vcell_true = (
            np.product(self.box_size.to(u.Mpc).value /
                       self.N_mesh) * (u.Mpc**3)
        ).to(u.Mpc**3)

        if tracer == "intensity":  # calculate the intensity mesh
            Zhalo = self.cat[redshift][mask]
            Hubble = self.astropy_cosmo.H(Zhalo).to(u.km / u.Mpc / u.s)

            if self.lambda_restframe is not None:
                rest_wave_or_freq = self.lambda_restframe
                unit_wave_or_freq = self.lambda_restframe.unit
            elif self.nu_restframe is not None:
                rest_wave_or_freq = self.nu_restframe
                unit_wave_or_freq = self.nu_restframe.unit
            else:
                raise ValueError(
                    "You must provide either lambda_restframe or nu_restframe to calculate the specific intensity dI/dnu or dI/dlambda."
                )

            if self.brightness_temperature:
                if self.nu_restframe is None:
                    self.nu_restframe = (
                        const.c / self.lambda_restframe).to(u.Hz)
                signal = (
                    (self.cat[luminosity][mask] / Vcell_true)
                    * const.c**3
                    * (1 + self.cat[redshift][mask]) ** 2
                    / (8 * np.pi * const.k_B * self.nu_restframe**3 * Hubble)
                )
                signal = signal.to(u.uK)
                mean_signal = np.mean(signal)
                signal = signal.to(mean_signal)
            else:
                signal = (
                    const.c
                    / (4.0 * np.pi * rest_wave_or_freq * Hubble * (1.0 * u.sr))
                    * self.cat[luminosity][mask]
                    / Vcell_true
                )
                signal = signal.to(
                    u.erg / (u.s * u.cm**2 * u.arcsec**2 * unit_wave_or_freq)
                )
                mean_signal = np.mean(signal)
                signal = signal.to(mean_signal)
        elif tracer == "n_gal":
            signal = (np.ones(self.cat[position][mask].shape[0]) / Vcell_true).to(
                u.Mpc ** (-3)
            )  # n_gal
            # n_bar_gal_masked = self.cat['luminosity'][mask].size / np.product(self.box_size.to(u.Mpc).value) * u.Mpc**(-3)
            # signal = ((signal / n_bar_gal_masked)).to(1) # divide by n_bar to get 1+delta_gal
        else:
            raise ValueError(
                f"Impermissible tracer argument: {tracer}. Possible options are 'intensity' or 'n_gal'."
            )
        logging.info("Signal : {}".format(signal))
        logging.info("Min signal: {}".format(np.min(signal)))

        field = catalog_to_mesh_cython(
            self.cat[position][mask].to(self.Mpch).value,
            signal.value.astype(float),
            self.N_mesh.astype(int),
            self.box_size.to(self.Mpch).value
        )
        field = make_map(
            field,
            Nmesh=self.N_mesh,
            BoxSize=self.box_size.to(self.Mpch).value,
        )

        logging.info(
            "Mean intensity before smoothing: {}.".format(np.mean(field)))
        logging.info(
            "Min intensity before smoothing: {}.".format(np.min(field)))

        if self.do_spectral_tophat_smooth:
            logging.warning(
                "Top-hat smoothing may result in large numerical errors. You may get negative intensity voxels. The number and value depend on the smoothing length relative to the voxel size.")

        if (self.do_angular_smooth or self.do_spectral_smooth) and (
            tracer == "intensity"
        ):  # don't smooth the galaxy field
            logging.info("Smoothing...")
            field = field.r2c()
            global sigma
            global LOS
            LOS = self.LOS
            # compute scales for the anisotropic filter (in Ztrue -> zmid)
            sigma_par = self.sigma_par().to(self.Mpch).value
            sigma_perp = self.sigma_perp().to(self.Mpch).value
            logging.info('Smoothing LOS: {}'.format(sigma_par))
            logging.info('Smoothing angular: {}'.format(sigma_perp))
            sigma = sigma_par * self.LOS + sigma_perp * (self.LOS == 0).astype(
                int
            )  # orders the sigmas in the same axes as the data.

            # raise a warning if the smoothing length is smaller than the voxel length.
            if (sigma < self.box_size.to(self.Mpch).value / self.N_mesh).any():
                logging.warning(
                    "The smoothing length along or perpendicular to the LOS is smaller than the voxel size! You should consider using a larger smoothing length."
                )
            if self.do_spectral_tophat_smooth:
                logging.info('Smoothing with a top-hat along the LOS.')
                smooth_func = aniso_filter
            else:
                logging.info('Smoothing with a Gaussian along the LOS.')
                smooth_func = aniso_filter_gaussian
            field = field.apply(smooth_func,
                                kind="wavenumber")
            field = field.c2r()

        logging.info(
            "Mean intensity after smoothing: {}.".format(np.mean(field)))
        logging.info(
            "Min intensity after smoothing: {}.".format(np.min(field)))
        logging.info("Fraction of voxels with I<0: {}.".format(np.size(
            field[field < 0.]) / np.size(field)))

        field = np.array(field) * signal.unit
        if tracer == "intensity":
            self.intensity_mesh = field
            self.intensity_mesh = self.intensity_mesh.to(self.mean_intensity)
            field = field.to(self.mean_intensity)
        elif tracer == "n_gal":
            self.n_gal_mesh = field.to(1 / u.Mpc**3)
        logging.info("Done.")
        return field

    def sigma_par(self):
        """
        Calculate the comoving distance corresponding to the
        LOS smoothing length.

        Returns
        -------
        sigma_par : Quantity
            Comoving distance corresponding to the
            LOS smoothing length.

        Notes
        -----
        This function calculates the comoving distance corresponding to the smoothing length along the line-of-sight (LOS)
        direction. The calculated value depends on the specified rest frame wavelength (lambda_restframe)
        or frequency (nu_restframe), the LOS direction, and the cosmological parameters.

        """

        zmid = self.redshift
        if (self.lambda_restframe is not None) and (self.dlambda is not None):
            sigma_par = self.do_spectral_smooth * (
                const.c
                * self.dlambda
                / (self.lambda_restframe * self.astropy_cosmo.H(zmid))
            ).to(self.Mpch)
        elif (self.nu_restframe is not None) and (self.dnu is not None):
            self.nuObs_mean = self.nu_restframe / (1 + zmid)
            sigma_par = self.do_spectral_smooth * (
                const.c
                * self.dnu
                * (1 + zmid)
                / (self.astropy_cosmo.H(zmid).to(u.km / u.Mpc / u.s) * self.nuObs_mean)
            ).to(self.Mpch)
        return sigma_par

    def sigma_perp(self):
        """
        Calculate the comoving distance corresponding to the angular smoothing length.

        Returns
        -------
        sigma_perp : Quantity
            Comoving distance corresponding to the angular smoothing length.

        Notes
        -----
        This function calculates the comoving distance corresponding to the smoothing length perpendicular to the line-of-sight (LOS)
        direction. The calculated value depends on the specified redshift, angular diameter distance, angular smoothing length,
        and the cosmological parameters.

        """

        zmid = self.redshift
        sigma_perp = self.do_angular_smooth * (
            self.astropy_cosmo.angular_diameter_distance(zmid).to(u.Mpc)
            * (1 + zmid)
            * (self.sigma_beam / (1 * u.rad))
        ).to(
            self.Mpch
        )  # (1+z) factor to make it comoving
        return sigma_perp

    @functools.cached_property
    def mean_intensity(self):
        """
        Returns the mean intensity of the mesh in units of mean intensity.
        As soon as it is calculated once, it is saved.
        """
        """
        Calculate the mean intensity of the mesh
        based on the mean along redshift axis and given the selected galaxy selection.
        The calculated value is returned in units of the mean intensity.

        Returns
        -------
        mean_intensity : Quantity
            The mean intensity of the mesh.

        """

        logging.info("Calculating mean intensity.")
        mean_intensity = np.mean(
            self.mean_intensity_per_redshift(
                self.redshift_mesh_axis,
                galaxy_selection=self.galaxy_selection["intensity"],
            )
        )
        return mean_intensity.to(mean_intensity)
        # mean_intensity = np.mean(self.intensity_mesh)
        # return mean_intensity.to(mean_intensity)

    def paint_galaxy_mesh(
        self,
        position="RSD_Position"
    ):
        """
        Calculates the galaxy number density mesh based on the specified position of galaxies.
        The parameter self.galaxy_selection_function['n_gal'] decides whether to use 
        detected, undetected, or all galaxies.

        Parameters
        ----------
        position : str, optional
            The column name in the catalog representing the position of galaxies.
            Options: "RSD_Position" (redshift space), "Position" (real space)
            Default: "RSD_Position".

        Returns
        -------
        field : ndarray
            The generated galaxy number density mesh.

        """

        return self.paint_intensity_mesh(
            position=position,
            redshift="cosmo_redshift",
            tracer="n_gal",
        )

    def get_intensity_noise_cube(self):
        """
        Generate a mock intensity noise cube following a Gaussian distribution with an input sigma.

        Returns
        -------
        noise_mesh : ndarray
            The generated intensity noise cube in units of mean intensity.

        Notes
        -----
        If sigma_noise is not provided, a zero noise cube is returned.

        """

        try:
            self.intensity_mesh.shape
        except AttributeError:
            self.paint_intensity_mesh()

        logging.info("Getting noise mesh.")

        if self.sigma_noise == None:
            self.noise_mesh = da.zeros(self.intensity_mesh.shape)
        else:
            da.random.seed(self.seed_lognormal)
            if callable(self.sigma_noise):
                los_axis = np.where(self.LOS == 1)[0][0]
                assert los_axis == 0
                redshifts = self.redshift_mesh_axis
                sigma_per_redshift = self.sigma_noise(
                    redshifts).to(self.mean_intensity).value
                sigma = sigma_per_redshift[:, None, None] * \
                    np.ones(self.intensity_mesh.shape)

            else:
                # if it is not a function, but a scalar or a numpy array we can skip the callable.
                if np.size(self.sigma_noise) > 1:
                    try:
                        assert (np.array(self.sigma_noise.shape)
                                == self.N_mesh).all()
                    except AssertionError:
                        logging.error(
                            f"{self.sigma_noise.shape=} but {self.N_mesh=}")
                self.sigma_noise = self.sigma_noise.to(self.mean_intensity)
                # in intensity or temperature units.
                sigma = self.sigma_noise.value

            noise_mesh = da.random.normal(
                # size has to be a tuple, not an array.
                loc=0, scale=sigma, size=tuple(self.N_mesh)
            )
        self.noise_mesh = (
            noise_mesh * self.mean_intensity).compute().to(self.mean_intensity)

        logging.info("Done.")
        return

    def get_sky_background(self, footprint_angle):
        """
        Compute the contribution of the intensity map to the measured sky background.
        It achieves this by smoothing the intensity mesh in the angular directions 
        using a top-hat filter of the specified angular footprint size.

        Parameters
        ----------
        footprint_angle : astropy.u.Quantity (scalar)
            Angular footprint size as an angular astropy Quantity in angle units.

        Returns
        -------
        sky_intensity_mesh : ndarray
            The computed sky background intensity mesh in units of mean intensity.

        """

        zmid = self.redshift
        global tophat_size
        global LOS
        LOS = self.LOS
        tophat_size = (
            (
                self.astropy_cosmo.angular_diameter_distance(zmid).to(u.Mpc)
                * (1 + zmid)
                * (footprint_angle / (1 * u.rad))
            )
            .to(self.Mpch)
            .value
        )
        intensity_map = self.intensity_mesh.to(self.mean_intensity).value
        if self.noise_mesh is not None:
            intensity_map = intensity_map + \
                self.noise_mesh.to(self.mean_intensity).value
        intensity_realfield = make_map(
            intensity_map,
            Nmesh=self.N_mesh,
            BoxSize=self.box_size.to(self.Mpch).value,
        )
        self.sky_intensity_mesh = (
            intensity_realfield.r2c().apply(angular_tophat_filter, kind="wavenumber")
        ).c2r()
        self.sky_intensity_mesh = (
            np.array(self.sky_intensity_mesh) * self.mean_intensity
        )
        logging.info("Fraction of negative voxels in sky_intensity_mesh: {}".format(
            np.size(self.sky_intensity_mesh[self.sky_intensity_mesh < 0.])/np.prod(self.N_mesh)))
        return self.sky_intensity_mesh.to(self.mean_intensity)

    def bin_scipy(self, pkspec, return_nmodes=False):
        """
        Perform binning of the power spectrum.

        Parameters
        ----------
        pkspec : ndarray
            The power spectrum mesh to be binned.
        return_nmodes: bool
            Set to true if you want to have the number of modes in each bin returned.
            Default: False.

        Returns
        -------
        mean_k : ndarray
            The mean wavenumber in each bin.
        monopole : ndarray
            The monopole component of the power spectrum.
        quadrupole : ndarray
            The quadrupole component of the power spectrum.

        """

        k_bins = np.linspace(self.kmin, self.kmax, self.nkbin + 1)
        kspec = np.concatenate(np.concatenate(self.kspec))
        muspec = np.concatenate(np.concatenate(self.muspec))
        pkspec = np.concatenate(np.concatenate(pkspec))

        logging.info("Calculating summary statistics.")
        # (mean_k,monopole,quadrupole,(nmodes))
        return_tuple = bin_scipy(
            pkspec, k_bins, kspec, muspec, return_nmodes=return_nmodes)
        return return_tuple

    def _get_prepared_intensity_mesh(self, sky_subtraction):
        """
        Prepare the intensity map Fourier transform delta_I for the power spectrum calculations.
        Applies normalization, adds noise, subtracting the mean intensity per redshift, 
        and applies the mask.

        Parameters
        ----------
        sky_subtraction : bool
            Flag indicating whether to perform sky subtraction.

        Returns
        -------
        prepared_intensity_mesh_ft : :class:`pyfftw.interfaces.numpy_fft.fftn.FFTWArray`
            The prepared intensity map in Fourier space after applying the necessary transformations.

        """

        intensity_map = self.intensity_mesh
        weights_im = self.mean_intensity  # _per_redshift_mesh
        intensity_map_to_use = make_map(
            (intensity_map / weights_im).to(1),
            Nmesh=self.N_mesh,
            BoxSize=self.box_size.to(self.Mpch).value,
        )

        if sky_subtraction:
            intensity_map_to_use = intensity_map_to_use - (
                self.get_sky_background(self.footprint_radius) / weights_im
            ).to(1)

        # Add noise
        if self.noise_mesh is not None:
            intensity_map_to_use = intensity_map_to_use + np.array(
                (self.noise_mesh / weights_im).to(1)
            )

        # subtract mean intensity per redshift
        intensity_map_to_use = (
            intensity_map_to_use
            - np.mean(intensity_map_to_use, axis=(1, 2))[:, None, None]
        )  # self.mean_intensity_per_redshift_mesh

        # multiply by the mask
        if self.obs_mask is not None:
            intensity_map_to_use = intensity_map_to_use * self.obs_mask

        if sky_subtraction:
            self.prepared_skysub_intensity_mesh_ft = intensity_map_to_use.r2c()
        else:
            self.prepared_intensity_mesh_ft = intensity_map_to_use.r2c()
        return

    def _get_prepared_n_gal_mesh(self):
        """
        Prepare the galaxy number density Fourier transform delta_g for the power spectrum calculations.
        Performs normalization, subtracts the mean
        galaxy number density per redshift, and applies an observation mask.

        Returns
        -------
        prepared_n_gal_mesh_ft : :class:`pyfftw.interfaces.numpy_fft.fftn.FFTWArray`
            The prepared galaxy number density map in Fourier space after applying the necessary transformations.

        Notes
        -----
        This function prepares the galaxy number density map for analysis by performing normalization, subtracting the mean
        galaxy number density per redshift, and applying an observation mask.

        """

        try:
            galaxy_map = self.n_gal_mesh
        except AttributeError:
            self.paint_galaxy_mesh()
            galaxy_map = self.n_gal_mesh

        mean_ngal_per_z = np.mean(self.n_gal_mesh, axis=(1, 2))[
            :, None, None]
        galaxy_map = (self.n_gal_mesh / mean_ngal_per_z).to(1) - 1.
        galaxy_map_to_use = make_map(galaxy_map,
                                     Nmesh=self.N_mesh,
                                     BoxSize=self.box_size.to(self.Mpch).value,
                                     )

        if self.obs_mask is not None:
            galaxy_map_to_use = galaxy_map_to_use * self.obs_mask

        self.prepared_n_gal_mesh_ft = galaxy_map_to_use.r2c()
        return

    def prepare_k_values(self):
        """ Prepares the dimensionless k values for the power spectrum calculations."""
        try:
            dk = self.dk.to(self.Mpch**-1).value
        except:
            dk = self.dk
        try:
            kmin = self.kmin.to(self.Mpch**-1).value
        except:
            kmin = self.kmin
        try:
            kmax = self.kmax.to(self.Mpch**-1).value
        except:
            kmax = self.kmax
        return (kmin, kmax, dk)

    def Pk_multipoles(self, tracer="intensity", save=False):
        """
        Compute the power spectrum monopole and quadrupole of the map.

        Parameters
        ----------
        tracer : str, optional
            Which power spectrum to compute. 
            Options: 'intensity', 'n_gal', 'cross', 'sky_subtracted_intensity', 'sky_subtracted_cross'.
            Default: 'intensity'.
        save : bool, optional
            Whether to save the computed power spectrum to a file. Default: False.

        Returns
        -------
        mean_k : array_like
            Array of mean wavenumbers.
        monopole : array_like
            Array of monopole power spectrum values.
        quadrupole : array_like
            Array of quadrupole power spectrum values.

        """

        logging.info(f"Getting power spectrum multipoles of {tracer}.")

        if tracer in [
            "intensity",
                "cross"]:
            # do it like this because otherwise the save_to_file acts weird with cached_property.
            if self.prepared_intensity_mesh_ft is None:
                self._get_prepared_intensity_mesh(sky_subtraction=False)
            intensity_map_to_use_ft = self.prepared_intensity_mesh_ft

            if tracer == "intensity":
                galaxy_map_to_use_ft = None

            logging.info("Prepared intensity map.")

        elif tracer in [
            "sky_subtracted_intensity",
            "sky_subtracted_cross",
        ]:
            if self.prepared_skysub_intensity_mesh_ft is None:
                self._get_prepared_intensity_mesh(sky_subtraction=True)
            intensity_map_to_use_ft = self.prepared_skysub_intensity_mesh_ft

            if tracer == "sky_subtracted_intensity":
                galaxy_map_to_use_ft = None

            logging.info("Prepared intensity map.")

        if tracer in ["n_gal", "cross", "sky_subtracted_cross"]:

            if self.prepared_n_gal_mesh_ft is None:
                self._get_prepared_n_gal_mesh()
            galaxy_map_to_use_ft = self.prepared_n_gal_mesh_ft
            if tracer == "n_gal":
                intensity_map_to_use_ft = None

            logging.info("Prepared galaxy map.")

        if tracer not in [
            "intensity",
            "n_gal",
            "cross",
            "sky_subtracted_intensity",
            "sky_subtracted_cross",
        ]:
            raise ValueError(
                "Invalid option in tracer: {}. Must be either 'intensity', 'n_gal', or 'cross', 'sky_subtracted_cross' or 'sky_subtracted_intensity'.".format(
                    tracer
                )
            )

        # make sure that the LOS is the x axis
        assert (self.LOS == [1, 0, 0]).all()

        (kmin, kmax, dk) = self.prepare_k_values()
        logging.info("kmin, kmax, dk = {} {} {}".format(kmin, kmax, dk))

        logging.info("Prepared k values.")

        map_1_ft = None
        if tracer in ["intensity", "sky_subtracted_intensity"]:
            map_1_ft = intensity_map_to_use_ft
            pk_unit = self.mean_intensity**2
        elif tracer == "n_gal":
            map_1_ft = galaxy_map_to_use_ft
            pk_unit = 1
        if map_1_ft is not None:
            delta_k_sq = np.real(map_1_ft) ** 2 + np.imag(map_1_ft) ** 2
            del map_1_ft
        elif tracer in ["cross", "sky_subtracted_cross"]:
            map_1_ft = intensity_map_to_use_ft
            map_2_ft = galaxy_map_to_use_ft
            delta_k_sq = np.array(map_1_ft * np.conjugate(map_2_ft))
            del map_1_ft, map_2_ft
            pk_unit = self.mean_intensity

        logging.info("Calculated delta k squared.")

        self.get_kspec()
        (
            mean_k,
            monopole,
            quadrupole,
        ) = self.bin_scipy(delta_k_sq.real)
        monopole = monopole * self.box_volume.to(self.Mpch**3).value
        quadrupole = quadrupole * self.box_volume.to(self.Mpch**3).value

        logging.info("Finished binning delta k squared.")

        if save:
            if self.RSD:
                rsd_ext = "rsd"
            else:
                rsd_ext = "realspace"
            outfilename = os.path.join(
                self.out_dir, "pk", rsd_ext, self.outfile_prefix + f"_pk.h5"
            )
            with h5py.File(outfilename, "a") as ff:
                if tracer in ff.keys():
                    del ff[tracer]
                    logging.info(
                        f"Overwriting {tracer} in file {outfilename}.")
                grp = ff.create_group(tracer)
                ff[f"{tracer}/monopole"] = monopole
                ff[f"{tracer}/quadrupole"] = quadrupole
                ff[f"{tracer}/mean_k"] = mean_k
                for key in ["monopole", "quadrupole"]:
                    ff[f"{tracer}/{key}"].attrs["unit"] = str(
                        self.Mpch**3 * pk_unit)
            logging.info(f"Saved to {outfilename}")

        return (
            mean_k,
            monopole,
            quadrupole,
        )

    def luminosity_function_times_L(self, L):
        """ dn/dL * L"""
        return self.luminosity_function(L) * L

    def mean_intensity_per_redshift(
        self, redshifts, galaxy_selection="all", tracer="intensity"
    ):
        """
        Calculates the mean intensity or the mean galaxy number density that we would expect
        at a given redshift of a given galaxy population (either all or detected/undetected galaxies).
        We use the integrals

        .. math:: \\bar{\\rho}_L(z) = \\int_{L_\\mathrm{min}}^{L_\\mathrm{max}} dL\, \\frac{\mathrm{d}n}{\mathrm{d}L} L
        .. math:: \\bar{I}_\\lambda(z) = \\frac{c \\bar{\\rho}_L(z)}{4 \\pi \\lambda_0 H(z)}
        and

        .. math:: \\bar{n}_\\mathrm{gal}(z) = \\int_{L_\\mathrm{min}}^{L_\\mathrm{max}} dL\, \\frac{\\mathrm{d}n}{\\mathrm{d}L}

        where

        ``Lmin = self.Lmin`` for all galaxies or undetected galaxies
        or ``Lmin = self.min_flux(z) * 4 * np.pi * D_L**2`` for detected galaxies
        and
        ``Lmax = self.Lmax`` for all galaxies or detected galaxies
        ``Lmax = self.min_flux(z) * 4 * np.pi * D_L**2`` for undetected galaxies.

        If min_flux is a mesh with the same size as N_mesh, 
        it will return the mean of the intensity/n_gal mesh depending on the specified tracer.

        Parameters
        ----------
        redshifts: array (n,)
            List/array of redshifts to be computed.
        galaxy_selection: str
            Which galaxies to use: 'all', 'detected', or 'undetected'. Default: 'all'.
        tracer: str
            Whether to calculate the mean intensity ('intensity') or mean galaxy number density ('n_gal').
            Default: 'intensity'.

        Returns
        -------
        mean_intensity_per_redshift: Quantity array (n,)
            List/array of the mean intensity per redshift (if tracer=='intensity') or mean galaxy number density
            per redshift (if tracer=='n_gal'). Same shape as input redshifts array.

        """

        logging.info("Getting mean {} of {} galaxies.".format(
            tracer, galaxy_selection))
        if tracer not in ["intensity", "n_gal"]:
            raise ValueError(
                "Invalid 'tracer' argument: must be either 'intensity' or 'n_gal'."
            )

        # if min_flux is a 3D array with the same shape as N_mesh,
        # return the mean intensity of the mesh as a function of LOS distance
        # because we can't make this calculation for each cell.
        if (not callable(self.min_flux)) and (np.size(self.min_flux) > 1):
            if (np.array(np.shape(self.min_flux)) == self.N_mesh).all():
                if tracer == "intensity":
                    mean_intensity_per_redshift = np.mean(
                        self.intensity_mesh, axis=(1, 2))
                elif tracer == 'n_gal':
                    mean_intensity_per_redshift = np.mean(
                        self.n_gal_mesh, axis=(1, 2))
                else:
                    raise ValueError(
                        "Invalid 'tracer' argument: must be either 'intensity' or 'n_gal'."
                    )
                return mean_intensity_per_redshift

        if galaxy_selection == "all":
            if tracer == "intensity":
                integrated = quad(
                    self.luminosity_function_times_L,
                    self.Lmin.to(self.luminosity_unit).value,
                    self.Lmax.to(self.luminosity_unit).value,
                )
                logging.info(
                    "Integration result and uncertainty (should be much smaller than result!): {}".format(
                        integrated
                    )
                )
                mean_luminosity_density = (
                    integrated[0] * u.Mpc ** (-3) * self.luminosity_unit
                ).to(u.Mpc ** (-3) * self.luminosity_unit)
            elif tracer == "n_gal":
                return (
                    np.array(
                        [self.n_bar_gal.value for i in range(len(redshifts))])
                    * self.n_bar_gal.unit
                )
            else:
                raise ValueError(
                    "Invalid 'tracer' argument: must be either 'intensity' or 'n_gal'."
                )

        elif (galaxy_selection == "detected") or (galaxy_selection == "undetected"):
            mean_luminosity_density = []
            for redshift in redshifts:
                if callable(self.min_flux):
                    F_min = self.min_flux(redshift)
                else:
                    F_min = self.min_flux
                int_L_limit = (
                    F_min
                    * 4
                    * np.pi
                    * self.astropy_cosmo.luminosity_distance(redshift) ** 2
                )
                int_L_limit = int_L_limit.to(self.luminosity_unit)
                if galaxy_selection == "detected":
                    int_L_min = int_L_limit.value
                    int_L_max = self.Lmax.to(self.luminosity_unit).value
                else:
                    int_L_min = self.Lmin.to(self.luminosity_unit).value
                    int_L_max = int_L_limit.value
                if tracer == "intensity":
                    integrated = quad(
                        self.luminosity_function_times_L, int_L_min, int_L_max
                    )
                    mean_luminosity_density.append(
                        (integrated[0] * u.Mpc ** (-3) * self.luminosity_unit)
                        .to(u.Mpc ** (-3) * self.luminosity_unit)
                        .value
                    )
                    mean_luminosity_density_unit = u.Mpc ** (-3) * \
                        self.luminosity_unit
                elif tracer == "n_gal":
                    integrated = quad(self.luminosity_function,
                                      int_L_min, int_L_max)
                    mean_luminosity_density.append(
                        (integrated[0] * u.Mpc ** (-3)).to(u.Mpc ** (-3)).value
                    )
                    mean_luminosity_density_unit = u.Mpc ** (-3)
            mean_luminosity_density = (
                np.array(mean_luminosity_density) *
                mean_luminosity_density_unit
            )
            if tracer == "n_gal":
                return mean_luminosity_density

        else:
            raise ValueError(
                "galaxy_selection must be either 'all' or 'detected' or 'undetected'."
            )

        Hubble = self.astropy_cosmo.H(redshifts).to(u.km / u.s / u.Mpc)
        if self.brightness_temperature:
            if self.nu_restframe is None:
                self.nu_restframe = (const.c / self.lambda_restframe).to(u.Hz)
            mean_intensity_per_redshift = (
                (mean_luminosity_density)
                * const.c**3
                * (1 + redshifts) ** 2
                / (8 * np.pi * const.k_B * self.nu_restframe**3 * Hubble)
            ).to(u.uK)
        else:
            if self.lambda_restframe is not None:
                rest_wave_or_freq = self.lambda_restframe
                unit_wave_or_freq = self.lambda_restframe.unit
            elif self.nu_restframe is not None:
                rest_wave_or_freq = self.nu_restframe
                unit_wave_or_freq = self.nu_restframe.unit
            else:
                raise ValueError(
                    "You must provide either lambda_restframe or nu_restframe to calculate the specific intensity dI/dnu or dI/dlambda."
                )

            mean_intensity_per_redshift = (
                const.c
                / (4.0 * np.pi * rest_wave_or_freq * Hubble * (1.0 * u.sr))
                * mean_luminosity_density
            )

        return mean_intensity_per_redshift

    @functools.cached_property
    def mean_intensity_per_redshift_mesh(self):
        """ Broadcasts the 1D mean_intensity_per_redshift array to the 3D mesh (cached)."""
        return self.mean_intensity_per_redshift(
            self.redshift_mesh_axis,
            tracer="intensity",
            galaxy_selection=self.galaxy_selection["intensity"],
        )[:, None, None]

    @functools.cached_property
    def mean_ngal_per_redshift_mesh(self):
        """ Broadcasts the 1D mean_n_gal_per_redshift array to the 3D mesh (cached)."""
        return self.mean_intensity_per_redshift(
            self.redshift_mesh_axis,
            tracer="n_gal",
            galaxy_selection=self.galaxy_selection["n_gal"],
        )[:, None, None]

    def run(self, skip_lognormal=False, save_meshes=False, save_results=True):
        """
        Runs the entire simulation and analysis pipeline.

        Parameters
        ----------
        skip_lognormal : bool, optional
            Skip the lognormal simulation and use a pre-generated catalog. Default: False.
        save_meshes : bool, optional
            Save the generated LognormalIntensityMock instance to a file and the galaxy catalog to another file. Default: False.
        save_results : bool, optional
            Save the computed power spectra to a file. Default: True.

        Notes
        -----
        This function performs all the necessary steps to run the simulation and analyze the results. 
        It includes
          * the lognormal simulation, 
          * catalog loading, 
          * redshift assignment, 
          * luminosity and flux assignment, 
          * selection function application, 
          * painting intensity and galaxy meshes, 
          * and computing power spectra.

        If `skip_lognormal` is False (default), the lognormal simulation and catalog loading will be performed. 
        Then, redshift assignment, luminosity assignment, flux assignment, and selection function application will be executed.

        The position variable used for painting the intensity and galaxy meshes depends on the RSD (Redshift-Space Distortion) setting self.RSD. 
        If RSD is enabled, "RSD_Position" is used, and if RSD is disabled, "Position" is used.

        The intensity and galaxy meshes will be painted if the corresponding power spectra are enabled in the self.run_pk setting.

        If `save_meshes` is True, the generated intensity and galaxy meshes will be saved to file.

        If `save_results` is True (default), the computed power spectra will be saved to file.

        """

        if not skip_lognormal:
            self.run_lognormal_simulation_cpp()
            self.load_lognormal_catalog_cpp(
                bin_filename=self.lognormal_bin_filename)
            if self.single_redshift:
                self.assign_single_redshift()
            else:
                self.assign_redshift_along_axis()
            self.assign_luminosity()
            self.assign_flux()
            self.apply_selection_function()

        if self.RSD:
            position = "RSD_Position"
            rsd_ext = "rsd"
        else:
            position = "Position"
            rsd_ext = "realspace"

        if self.run_pk['intensity'] or self.run_pk['cross'] or self.run_pk['sky_subtracted_intensity'] or self.run_pk['sky_subtracted_cross']:
            self.paint_intensity_mesh(position=position)
            self.get_intensity_noise_cube()
        if self.run_pk['n_gal'] or self.run_pk['cross'] or self.run_pk['sky_subtracted_cross']:
            self.paint_galaxy_mesh(position=position)

        if save_meshes:
            filename = os.path.join(
                self.out_dir,
                "lognormal",
                rsd_ext,
                self.outfile_prefix + "_lim_instance.h5",
            )
            self.catalog_filename = os.path.join(
                self.out_dir, "lognormal", self.outfile_prefix + "_lognormal_rlz0.h5"
            )
            self.save_to_file(filename=filename,
                              catalog_filename=self.catalog_filename)

        for tracer in ["intensity", "n_gal", "cross", "sky_subtracted_intensity", "sky_subtracted_cross"]:
            if self.run_pk[tracer]:
                self.Pk_multipoles(tracer=tracer, save=save_results)
