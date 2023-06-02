import astropy.units as u
import numpy as np
import h5py
import psutil
import os
import pmesh
import logging
from yaml import load
from scipy.stats import binned_statistic_2d, binned_statistic
from scipy.special import legendre, j1

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1073741824  # in GB

def yaml_file_to_dictionary(filename):
    with open(filename, "r") as stream:
        data = yaml_stream_to_dictionary(stream)
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
    return h5_filename


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
    obs_mask_2d[((row_2d // N_cells) % 2 == 0) & ((col_2d // N_cells) % 2 == 1)] = 1.0
    obs_mask = np.array([obs_mask_2d for i in range(N_mesh[0])])
    return obs_mask


def getindep(nx, ny, nz):
    indep = np.full((nx, ny, nz // 2 + 1), False, dtype=bool)
    indep[:, :, 1: nz // 2] = True
    indep[1: nx // 2, :, 0] = True
    indep[1: nx // 2, :, nz // 2] = True
    indep[0, 1: ny // 2, 0] = True
    indep[0, 1: ny // 2, nz // 2] = True
    indep[nx // 2, 1: ny // 2, 0] = True
    indep[nx // 2, 1: ny // 2, nz // 2] = True
    indep[nx // 2, 0, 0] = True
    indep[0, ny // 2, 0] = True
    indep[nx // 2, ny // 2, 0] = True
    indep[0, 0, nz // 2] = True
    indep[nx // 2, 0, nz // 2] = True
    indep[0, ny // 2, nz // 2] = True
    indep[nx // 2, ny // 2, nz // 2] = True
    return indep


def get_kspec(nx, ny, nz, lx, ly, lz, dohalf=True, doindep=True):
    """
    docstring
    """
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    if dohalf:
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=lz / nz)[: nz // 2 + 1]
        indep = np.full((nx, ny, nz // 2 + 1), True, dtype=bool)
        if doindep:
            indep = getindep(nx, ny, nz)
    else:
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=lz / nz)
        indep = np.full((nx, ny, nz), True, dtype=bool)
    indep[0, 0, 0] = False
    kspec = np.sqrt(
        kx[:, np.newaxis, np.newaxis] ** 2
        + ky[np.newaxis, :, np.newaxis] ** 2
        + kz[np.newaxis, np.newaxis, :] ** 2
    )
    kspec[0, 0, 0] = 1.0
    muspec = np.absolute(kx[:, np.newaxis, np.newaxis]) / kspec
    kspec[0, 0, 0] = 0.0

    return kspec, muspec, indep, kx, ky, kz


def bin_scipy(pkspec, k_bins, kspec, muspec, two_d=False, mu_bins=None):
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
    quadrupole, k_edge, bin_number = binned_statistic(
        kspec,
        pkspec * legendre(2)(muspec),
        bins=k_bins,
        statistic="mean",
    )
    logging.info("Calculated quadrupole.")
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
        Nmesh, BoxSize=BoxSize, dtype="float32", resampler="cic")
    field = pm.create(type=type)
    field[...] = m
    return field