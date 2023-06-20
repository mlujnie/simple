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
    '''
    Logarithmic interpolation accepting linear quantities as input (transformed
    within the function)
    '''
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
    
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1073741824  # in GB


def yaml_file_to_dictionary(filename):
    with open(filename, "r") as stream:
        data = yaml.load(stream)
        #data = yaml_stream_to_dictionary(stream)
    return data


def yaml_stream_to_dictionary(stream):
    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper
    return load(stream, Loader=Loader)


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print("Memory usage: ", process.memory_info().rss /
          (1024 * 1024 * 1000), " GB.")


def transform_bin_to_h5(bin_filename, h5_filename=None):
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
    obs_mask_2d = np.zeros(shape=(N_mesh[1], N_mesh[2]))
    row_2d = np.array([np.arange(N_mesh[1]) for i in range(N_mesh[2])]).T
    col_2d = np.array([np.arange(N_mesh[2]) for i in range(N_mesh[1])])
    obs_mask_2d[((row_2d // N_cells) % 2 == 0) &
                ((col_2d // N_cells) % 2 == 1)] = 1.0
    obs_mask = np.array([obs_mask_2d for i in range(N_mesh[0])])
    return obs_mask

def bin_scipy(pkspec, k_bins, kspec, muspec, two_d=False, mu_bins=None, lmax=2):
    if two_d:
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
    else:
        P_k_mu, mean_k_2d, mean_mu_2d = None, None, None

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

    return (
        mean_k,
        monopole,
        5 * quadrupole,
        mean_k_2d,
        mean_mu_2d,
        P_k_mu,
    )

def bin_Pk_2d(pkspec, k_bins, k_par, k_perp):
    P_k_2d, k_edge, mu_edge, bin_number_2d = binned_statistic_2d(
        k_par, k_perp, pkspec, bins=[k_bins, k_bins], statistic="mean"
    )
    logging.info("Calculated P_k_2d.")
    return P_k_2d


def jinc(x):
    return np.where(x != 0, j1(x) / x, 0.5)


def downsample_mesh(mesh, box_size, new_N_mesh=None, new_voxel_length=None, resampler='cic'):
    """
    Returns down-sampled version of a mesh 'mesh' with the new N_mesh 'new_N_mesh' or voxel_length 'new_voxel_length'.
    It assumes that the input mesh has mesh number self.N_mesh and voxel length self.voxel_length.
    """
    logging.info("Downsampling the mesh...")
    if (new_N_mesh is None) and (new_voxel_length is None):
        raise ValueError(
            "You have to provide either new_N_mesh or new_voxel_length!"
        )
    elif new_N_mesh is None:
        new_N_mesh = np.ceil(
            box_size / new_voxel_length).to(1).astype(int)

    if not isinstance(mesh, pmesh.pm.RealField):
        mesh = make_map(mesh, np.shape(mesh), box_size)
    pm_down = pmesh.pm.ParticleMesh(
        new_N_mesh,
        BoxSize=box_size,
        dtype="float32",
        resampler=resampler,
    )
    logging.info("Done.")
    return pm_down.downsample(mesh, keep_mean=True)


def make_map(m, Nmesh, BoxSize, type="real"):
    """
    Returns a pmesh.RealField object from an array 'm' as input using the cic resampler.

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
