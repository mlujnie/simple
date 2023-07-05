import astropy.units as u
import numpy as np
import h5py
import psutil
import os
import pmesh
import logging
#from yaml import load
from astropy.io.misc import yaml
from scipy.stats import binned_statistic_2d, binned_statistic
from scipy.special import legendre, j1
from scipy.interpolate import interp1d

def log_interp1d(xx, yy, kind='linear',bounds_error=False,fill_value='extrapolate'):
    """
    Logarithmic interpolation accepting linear quantities as input (transformed within the function).

    Parameters
    ----------
    xx : array-like
        Array of x values.
    yy : array-like
        Array of y values.
    kind : str, optional
        Specifies the kind of interpolation. Default is 'linear'.
    bounds_error : bool, optional
        If True, raise an error when attempting to interpolate outside the bounds of the input data. Default is False.
    fill_value : str or numeric, optional
        Specifies the value to use for points outside the bounds of the input data when bounds_error is False. Default is 'extrapolate'.

    Returns
    -------
    callable
        A callable function that performs logarithmic interpolation on input values.

    """
    
    ind = np.where(yy>0)
    try:
        logx = np.log10(xx[ind].value)
    except:
        logx = np.log10(xx[ind])
    try:
        logy = np.log10(yy[ind].value)
    except:
        logy = np.log10(yy[ind])
    lin_interp = interp1d(logx, logy, kind=kind,bounds_error=bounds_error,fill_value=fill_value)
    
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))

    return log_interp

def yaml_file_to_dictionary(filename):
    """ Opens a yaml file and returns a dictionary from its contents. """
    with open(filename, "r") as stream:
        data = yaml.load(stream)
        #data = yaml_stream_to_dictionary(stream)
    return data


def yaml_stream_to_dictionary(stream):
    """ Uses an opened yaml file and returns the dicionary. """
    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper
    return load(stream, Loader=Loader)


def print_memory_usage():
    """ Prints the current memory usage in GB. """

    process = psutil.Process(os.getpid())
    print("Memory usage: ", process.memory_info().rss /
          (1024 * 1024 * 1000), " GB.")


def transform_bin_to_h5(bin_filename, h5_filename=None):
    """
    Transforms a binary file containing galaxy data from lognormal_galaxies into an HDF5 file.

    Parameters
    ----------
    bin_filename : str
        Path to the binary output file containing the galaxy catalog from lognormal_galaxies.
    h5_filename : str, optional
        Path to the output HDF5 file.
        If not provided, the output file will be created with the same name as the input file, 
        but with the extension '.h5'.

    Returns
    -------
    h5_filename
        Path to the output HDF5 file 
    N_gal
        Number of galaxies.

    """

    if h5_filename is None:
        h5_filename = ".".join(bin_filename.split(".")[:-1]) + ".h5"
    print(f"Saving to {h5_filename}")
    dtype_ls = np.dtype(np.double)
    dtype_ngal = np.dtype(np.int64)
    offset_ngal = 3 * dtype_ls.itemsize
    offset_galdata = offset_ngal + dtype_ngal.itemsize
    Lx, Ly, Lz = np.fromfile(bin_filename, dtype=np.double, count=3, offset=0)
    N_gal = np.fromfile(bin_filename, dtype=dtype_ngal,
                        count=1, offset=offset_ngal)[0]
    galdata = np.fromfile(
        bin_filename, dtype=np.float32, count=6 * N_gal, offset=offset_galdata
    )
    posvels = np.reshape(galdata, (N_gal, 6), order="C")
    del galdata
    galpos = posvels[:, :3]
    galvel = posvels[:, 3:]
    del posvels
    print_memory_usage()
    print("Edges of the galaxy coordinates:")
    print(np.min(galpos[:, 0]), np.max(galpos[:, 0]))
    print(np.min(galpos[:, 1]), np.max(galpos[:, 1]))
    print(np.min(galpos[:, 2]), np.max(galpos[:, 2]))

    with h5py.File(h5_filename, "a") as ffile:
        for key in ["Position", "Velocity", "L_box", "N_gal"]:
            if key in ffile.keys():
                print(f"Overwriting {key} in {h5_filename}.")
                del ffile[key]
        ffile["L_box"] = [Lx, Ly, Lz]
        ffile["L_box"].attrs["unit"] = "u.Mpc * cosmo.h**(-1)"
        ffile["N_gal"] = N_gal
        ffile["Position"] = galpos
        ffile["Position"].attrs["unit"] = "u.Mpc * cosmo.h**(-1)"
        ffile["Velocity"] = galvel
        ffile["Velocity"].attrs["unit"] = "u.km * u.s**(-1)"
    print(f"Saved to {h5_filename}")
    return h5_filename, N_gal


def get_cylinder_mask(N_mesh):
    """
    Generates a mask for a cylindrical region in a 3D grid.
    The cylinder is elongated in the first axis.

    Parameters
    ----------
    N_mesh : array-like
        Number of cells in each dimension (3D grid).

    Returns
    -------
    ndarray
        Cylindrical mask indicating the region of interest.

    """
    row_2d = np.array(
        [np.arange(N_mesh[1]) - N_mesh[1] / 2.0 for i in range(N_mesh[2])]
    ).T
    col_2d = np.array(
        [np.arange(N_mesh[2]) - N_mesh[1] / 2.0 for i in range(N_mesh[1])]
    )
    diff_vec = np.sqrt(row_2d**2 + col_2d**2)
    cyl_mask_2d = diff_vec < N_mesh[1] / 2.0
    obs_mask = np.array([cyl_mask_2d for i in range(N_mesh[0])])
    return obs_mask


def get_checker_mask(N_mesh, N_cells=1):
    """
    Generates a checkerboard mask for a 3D grid.
    The mask is constant along the first axis.

    Parameters
    ----------
    N_mesh : array-like
        Number of cells in each dimension (3D grid).
    N_cells : int, optional
        Number of cells per checkerboard cell (default is 1).

    Returns
    -------
    ndarray
        Checkerboard mask for the specified 3D grid.

    """

    obs_mask_2d = np.zeros(shape=(N_mesh[1], N_mesh[2]))
    row_2d = np.array([np.arange(N_mesh[1]) for i in range(N_mesh[2])]).T
    col_2d = np.array([np.arange(N_mesh[2]) for i in range(N_mesh[1])])
    obs_mask_2d[((row_2d // N_cells) % 2 == 0) &
                ((col_2d // N_cells) % 2 == 1)] = 1.0
    obs_mask = np.array([obs_mask_2d for i in range(N_mesh[0])])
    return obs_mask

def bin_scipy(pkspec, k_bins, kspec, muspec, lmax=2, return_nmodes=False):
    """
    Calculates the monopole and quadrupole (if lmax==2) power spectrum and mean k in the given k bins.

    Parameters
    ----------
    pkspec : array-like (1D)
        Array of power spectrum values.
        Flattened 3D array.
    k_bins : array-like (1D)
        Array of bin edges for the k values.
    kspec : array-like (1D)
        Array of k values corresponding to pkspec.
        Flattened 3D array.
    muspec : array-like (1D)
        Array of mu values corresponding to pkspec.
        Flattened 3D array.
    lmax : int, optional
        Maximum multipole moment to calculate. 
        Options: 1, 2
        Default: 2.

    Returns
    -------
    mean_k : ndarray
        Mean k values in each k bin.
    monopole : ndarray
        Mean monopole power spectrum in each k bin.
    quadrupole : ndarray (or None if lmax==1)
        Mean quadrupole power spectrum in each k bin.
    """

    monopole, k_edge, bin_number = binned_statistic(
        kspec, pkspec, bins=k_bins, statistic="mean"
    )
    logging.info("Calculated monopole.")
    if lmax == 2:
        quadrupole, k_edge, bin_number = binned_statistic(
            kspec,
            pkspec * legendre(2)(muspec),
            bins=k_bins,
            statistic="mean",
        )
        logging.info("Calculated quadrupole.")
    else:
        quadrupole = np.zeros(np.shape(monopole))
    mean_k, k_edge, bin_number = binned_statistic(
        kspec, kspec, bins=k_bins, statistic="mean"
    )
    logging.info("Calculated mean_k.")

    if return_nmodes:
        nmodes, k_edge, bin_number = binned_statistic(np.ones(kspec.shape), kspec, bins=k_bins, statistic='sum')
        return mean_k, monopole, 5 * quadrupole, nmodes
    else:
        return (
            mean_k,
            monopole,
            5 * quadrupole,
        )

def bin_Pk_2d(pkspec, k_bins, k_par, k_perp):
    """
    Bins the power spectrum into a grid of given k_bins in the perpendicular and parallel to the LOS directions.

    Parameters:
    -----------
    pkspec: array-like (1D)
        Power spectrum data to be binned.
        Flattened 3D array.
    k_bins: array-like
        Array specifying the bin edges for both k_par and k_perp.
    k_par: array-like (1D)
        Array of k_parallel values corresponding to pkspec.
        Flattened 3D array.
    k_perp: array-like (1D)
        Array of k_perpendicular values corresponding to pkspec.

    Returns:
    --------
    P_k_2d: array
        Binned 2D power spectrum.

    """
    P_k_2d, k_edge, mu_edge, bin_number_2d = binned_statistic_2d(
        k_par, k_perp, pkspec, bins=[k_bins, k_bins], statistic="mean"
    )
    logging.info("Calculated P_k_2d.")
    return P_k_2d

def bin_Pk_mu(pkspec, k_bins, kspec, muspec, mu_bins):
    """
    Bins the power spectrum into a grid of given k_bins and mu_bins.

    Parameters:
    -----------
    pkspec: array-like (1D)
        Power spectrum data to be binned. Flattened 3D array.
    k_bins: array-like
        Array specifying the bin edges for k values.
    kspec: array-like (1D)
        Array of k values corresponding to pkspec.
        Flattened 3D array.
    muspec: array-like (1D)
        Array of mu values corresponding to pkspec.
        Flattened 3D array.
    mu_bins: array-like
        Array specifying the bin edges for mu values.

    Returns:
    --------
    mean_k_2d: array
        Binned mean k values in the 2D grid.
    mean_mu_2d: array
        Binned mean mu values in the 2D grid.
    P_k_mu: array
        Binned power spectrum.

    """

    P_k_mu, k_edge, mu_edge, bin_number_2d = binned_statistic_2d(
        kspec, muspec, pkspec, bins=[k_bins, mu_bins], statistic="mean"
    )
    logging.info("Calculated P_k_mu.")
    mean_k_2d, k_edge, mu_edge, bin_number = binned_statistic_2d(
        kspec, muspec, kspec, bins=[k_bins, mu_bins], statistic="mean"
    )
    logging.info("Calculated mean k (2d).")
    mean_mu_2d, k_edge, mu_edge, bin_number = binned_statistic_2d(
        kspec, muspec, muspec, bins=[k_bins, mu_bins], statistic="mean"
    )
    logging.info("Calculated mean mu (2d).")
    return (mean_k_2d,
        mean_mu_2d,
        P_k_mu,)


def jinc(x):
    """ 
    Returns the Jinc function for x!=0 and the limit 0.5 at x==0.
    The Jinc function is the first Bessel function of x divided by x.

    """
    return np.where(x != 0, j1(x) / x, 0.5)

def make_map(m, Nmesh, BoxSize, type="real"):
    """
    Returns a pmesh.RealField object from an array 'm' as input.

    Parameters
    ----------
    m: array
        Numpy array to be transformed into a pmesh.RealField object.
    Nmesh: array
        Array with dimension (3,) with the number of cells in each dimension.

    """
    pm = pmesh.pm.ParticleMesh(
        Nmesh, BoxSize=BoxSize, dtype="float32")
    field = pm.create(type=type)
    field[...] = m
    return field
