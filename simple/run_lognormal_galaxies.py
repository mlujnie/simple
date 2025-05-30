import os
import sys
import re
from string import *
import logging
import numpy as np

from simple.config import Config

config = Config()
main_lognormal_path = config.lognormal_galaxies_path

def check_dir_im(params):
    """
    Creates the necessary directories for saving output files
    if they do not already exist.

    Parameters
    -----------
    params: dict
        Dictionary containing the parameters passed to lognormal_galaxies,
        including "out_dir" specifying the output directory.

    """

    dir_names = [
        params["out_dir"],
        os.path.join(params["out_dir"], "inputs"),
        os.path.join(params["out_dir"], "lognormal"),
        os.path.join(params["out_dir"], "pk"),
        os.path.join(params["out_dir"], "coupling"),
    ]

    for dir_name in dir_names:
        print("dir_name: ", dir_name)
        for extension in ["rsd", "realspace"]:
            rsd_dir_name = os.path.join(dir_name, extension)
            print(rsd_dir_name)
            if os.path.exists(rsd_dir_name):
                logging.info(
                    "Directory "
                    + rsd_dir_name
                    + " exists already - continue to use this"
                )
            else:
                try:
                    os.makedirs(rsd_dir_name)
                    logging.info("Directory " + rsd_dir_name + " has been created")
                except:
                    pass
    try:
        os.rmdir(os.path.join(params["out_dir"], "rsd"))
        os.rmdir(os.path.join(params["out_dir"], "realspace"))
    except:
        pass


class executable:
    """Class to execute commands"""

    def __init__(self, name):
        self.name = name

    def run(self, exename, args, params):
        args = " ".join(map(str, [params[key] for key in args]))
        cmd = "time " + exename + " " + args
        print(cmd)
        os.system(cmd)


# generate input Gaussian power spectra


def gen_inputs(params, exe):
    """
    Generates the necessary input files for generate_Poisson.
    First, it runs eisensteinhubaonu/compute_pk to calculate the matter power spectrum from the cosmological parameters.
    Then it runs compute_xi/compute_xi to calculate the corresponding correlation function.
    Then it runs compute_pkG/calc_pkG to calculate the Gaussian power spectrum for the galaxy field.
    Then it runs compute_pkG/calc_pkG to calculate the Gaussian power spectrum for the matter field.
    Finally it runs compute_pkG/calc_pkG to calculate the Gaussian cross-power spectrum of the galaxy and matter fields.

    Parameters
    -----------
    params: dict
        Dictionary containing the parameters for lognormal_galaxies.
    exe: object
        An executable object used to run the external commands.

    """

    # input power spectrum
    params["ofile_eh"] = params["out_dir"] + "/inputs/" + params["ofile_prefix"]
    args = [
        "ofile_eh",
        "om0",
        "ode0",
        "ob0",
        "h0",
        "w",
        "ns",
        "run",
        "As",
        "mnu",
        "z",
    ]  # do not change the order
    exe.run(os.path.join(main_lognormal_path, "eisensteinhubaonu/compute_pk"), args, params)

    # powerspectrum to correlation function xi
    params["ofile_xi"] = os.path.join(params["out_dir"] , "inputs" , params["ofile_prefix"])
    params["len_inp_pk"] = sum(1 for line in open(params["inp_pk_fname"]))
    # do not change the order
    args = ["ofile_xi", "inp_pk_fname", "len_inp_pk"]
    exe.run(os.path.join(main_lognormal_path, "compute_xi/compute_xi"), args, params)

    # Gaussian power spectrum for galaxy field
    params["ncol"] = np.size(np.loadtxt(params["xi_fname"])[0, :])
    args = ["pkg_fname", "xi_fname", "ncol", "bias", "rmax"]  # do not change the order
    exe.run(os.path.join(main_lognormal_path,  "compute_pkG/calc_pkG"), args, params)

    # Gaussian power spectrum for matter field
    args = [
        "mpkg_fname",
        "xi_fname",
        "ncol",
        "bias_mpkG",
        "rmax",
    ]  # do not change the order
    exe.run(os.path.join(main_lognormal_path , "compute_pkG/calc_pkG"), args, params)

    # Gaussian cross power spectrum for galaxy-matter field
    if params["use_cpkG"] == 1:
        params["bias_cpkG"] = np.sqrt(params["bias_cpkG"])
        args = [
            "cpkg_fname",
            "xi_fname",
            "ncol",
            "bias_cpkG",
            "rmax",
        ]  # do not change the order
        exe.run(os.path.join(main_lognormal_path,  "compute_pkG/calc_pkG"), args, params)
        params["bias_cpkG"] = (params["bias_cpkG"]) ** 2.0


def gen_Poisson(i, params, seed1, seed2, seed3, exe):
    """
    Generate galaxy and matter density field and the Poisson-sampled lognormal galaxy catalog
    using lognormal_galaxies.

    Parameters
    -----------
    i: int
        Number of this realization.
    params: dict
        Dictionary containing the input parameters for lognormal_galaxies.
    seed1: int
        Seed.
    seed2: int
        Seed.
    seed3: int
        Seed.
    exe:
        Executable object used to run the external commands.

    """
    params_tmp = params
    params_tmp["seed1"] = seed1[i]
    params_tmp["seed2"] = seed2[i]
    params_tmp["seed3"] = seed3[i]

    # output file names
    params_tmp["Poissonfname"] = (
        params["out_dir"]
        + "/lognormal/"
        + params["ofile_prefix"]
        + "_lognormal_rlz"
        + str(i)
        + ".bin"
    )
    params_tmp["Densityfname"] = (
        params["out_dir"]
        + "/lognormal/"
        + params["ofile_prefix"]
        + "_density_lognormal_rlz"
        + str(i)
        + ".bin"
    )

    params_tmp["Velocityfname"] = (
        params["out_dir"]
        + "/lognormal/"
        + params["ofile_prefix"]
        + "_velocity_lognormal_rlz"
        + str(i)
        + ".bin"
    )

    # field generation
    args = [
        "pkg_fname",
        "mpkg_fname",
        "use_cpkG",
        "cpkg_fname",
        "Lx",
        "Ly",
        "Lz",
        "Pnmax",
        "Ngalaxies",
        "aH",
        "f_fname",
        "bias",
        "seed1",
        "seed2",
        "seed3",
        "Poissonfname",
        "Densityfname",
        "output_matter",
        "output_gal",
        "Velocityfname",
        "output_velocity"
    ]  # do not change the order
    exe.run(
        os.path.join(main_lognormal_path, "generate_Poisson/gen_Poisson_mock_LogNormal"),
        args,
        params_tmp,
    )


# wrapper of gen_Poisson


def wrap_gen_Poisson(args):
    """ Wrapper for gen_Poisson function."""
    return gen_Poisson(*args)


def calc_Pk(i, params, exe):
    """
    Calls calculate_pk/calc_pk_const_los_ngp function of lognormal_galaxies.

    Parameters
    ----------
    i: int
        Number of this realization.
    params: dict
        Dictionary containing input parameters for lognormal_galaxies.
    exe: 
        Executable object used to run the external commands.

    """
    params_tmp = params
    # input file names
    if (params_tmp["halofname_prefix"]) == "":
        params_tmp["halofname"] = (
            params["out_dir"]
            + "/lognormal/"
            + params["ofile_prefix"]
            + "_lognormal_rlz"
            + str(i)
            + ".bin"
        )
    else:
        params_tmp["halofname"] = (
            params["out_dir"]
            + "/lognormal/"
            + params["halofname_prefix"]
            + "_lognormal_rlz"
            + str(i)
            + ".bin"
        )
    if (params_tmp["imul_fname"]) == "":
        params_tmp["imul_fname"] = (
            params["out_dir"] + "/coupling/" + params["ofile_prefix"] + "_coupling.bin"
        )
    params_tmp["pk_fname"] = (
        params["out_dir"]
        + "/pk/"
        + params["ofile_prefix"]
        + "_pk_rlz"
        + str(i)
        + ".dat"
    )

    # run
    args = [
        "halofname",
        "Pnmax",
        "aH",
        "losx",
        "losy",
        "losz",
        "kbin",
        "kmax",
        "lmax",
        "imul_fname",
        "pk_fname",
        "calc_mode_pk",
    ]
    exe.run(
        os.path.join(main_lognormal_path + "calculate_pk/calc_pk_const_los_ngp"), args, params_tmp
    )


# wrapper of calc_Pk


def wrap_calc_Pk(args):
    """ Wrapper for the function calc_Pk."""
    return calc_Pk(*args)


def calc_cPk(i, params, exe):
    """
    Calls function calculate_cross/calc_cpk_const_los_v2 of lognormal_galaxies to calculate galaxy-matter power spectrum.

    Parameters
    ----------
    i: int
        Number of this realization.
    params: dict
        Input parameters for lognormal_galaxies.
    exe:
        Executable object used to run the external commands.

    """
    params_tmp = params
    # input file names
    if (params_tmp["halofname_prefix"]) == "":
        params_tmp["halofname1"] = (
            params["out_dir"]
            + "/lognormal/"
            + params["ofile_prefix"]
            + "_lognormal_rlz"
            + str(i)
            + ".bin"
        )
    else:
        params_tmp["halofname1"] = (
            params["out_dir"]
            + "/lognormal/"
            + params["halofname_prefix"]
            + "_lognormal_rlz"
            + str(i)
            + ".bin"
        )
    params_tmp["halofname2"] = (
        params["out_dir"]
        + "/lognormal/"
        + params["ofile_prefix"]
        + "_density_lognormal_rlz"
        + str(i)
        + ".bin"
    )
    if (params_tmp["imul_fname"]) == "":
        params_tmp["imul_fname"] = (
            params["out_dir"] + "/coupling/" + params["ofile_prefix"] + "_coupling.bin"
        )
    params_tmp["cpk_fname"] = (
        params["out_dir"]
        + "/pk/"
        + params["ofile_prefix"]
        + "_cpk_rlz"
        + str(i)
        + ".dat"
    )
    params_tmp["tmp1"] = 0
    params_tmp["tmp2"] = 1

    # run
    args = [
        "halofname1",
        "halofname2",
        "Pnmax",
        "aH",
        "losx",
        "losy",
        "losz",
        "kbin",
        "kmax",
        "lmax",
        "imul_fname",
        "cpk_fname",
        "calc_mode_pk",
        "tmp1",
        "tmp2",
    ]
    exe.run(
        os.path.join(main_lognormal_path, "calculate_cross/calc_cpk_const_los_v2"), args, params_tmp
    )


def wrap_calc_cPk(args):
    """ Wrapper for function calc_cPk."""
    return calc_cPk(*args)
