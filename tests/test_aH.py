import contextlib
import io
import unittest
from unittest import mock

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatwCDM, LambdaCDM, Planck18

import simple.simple
from simple.simple import LognormalIntensityMock
from simple.tools_python import yaml_file_to_dictionary

# Each cosmology with its E(z) = H(z) / H0 from the Friedmann equation,
# written out here (None: astropy's E(z), with photons and neutrinos).
# The three numbers H0 = 67.66 km/s/Mpc, Om0 = 0.31 and no radiation (astropy's
# default Tcmb0 = 0) are the same in all four.
COSMOLOGIES = {
    "Planck18": (Planck18, None),
    "flat wCDM, w = -0.8": (
        FlatwCDM(H0=67.66, Om0=0.31, w0=-0.8),
        lambda z: np.sqrt(0.31 * (1 + z)**3 + 0.69 * (1 + z)**(3 * (1 - 0.8)))),
    "flat wCDM, w = -1.2": (
        FlatwCDM(H0=67.66, Om0=0.31, w0=-1.2),
        lambda z: np.sqrt(0.31 * (1 + z)**3 + 0.69 * (1 + z)**(3 * (1 - 1.2)))),
    # Omega_k = 1 - Om0 - Ode0 = +0.05 (open) and -0.05 (closed)
    "LambdaCDM, Omega_k = +0.05": (
        LambdaCDM(H0=67.66, Om0=0.31, Ode0=0.64),
        lambda z: np.sqrt(0.31 * (1 + z)**3 + 0.05 * (1 + z)**2 + 0.64)),
    "LambdaCDM, Omega_k = -0.05": (
        LambdaCDM(H0=67.66, Om0=0.31, Ode0=0.74),
        lambda z: np.sqrt(0.31 * (1 + z)**3 - 0.05 * (1 + z)**2 + 0.74)),
}


# The same five cosmologies as SIMPLE can run them: run_lognormal_simulation_cpp
# needs neutrino masses (m_nu), which astropy has only with Tcmb0 > 0, and
# compute_pk needs Ob0. Each with (old aH) / (new aH) - 1 at the redshift of
# tests/test_lim_input.yaml, z = 2.2245, in per cent.
FULL = dict(H0=67.66, Om0=0.31, Ob0=0.049, Tcmb0=2.7255, m_nu=[0, 0, 0.06] * u.eV)
RUNNABLE = {
    "Planck18": (Planck18, -0.25),
    "flat wCDM, w = -0.8": (FlatwCDM(w0=-0.8, **FULL), -3.25),
    "flat wCDM, w = -1.2": (FlatwCDM(w0=-1.2, **FULL), +1.34),
    "LambdaCDM, Ode0 = 0.64": (LambdaCDM(Ode0=0.64, **FULL), -2.45),
    "LambdaCDM, Ode0 = 0.74": (LambdaCDM(Ode0=0.74, **FULL), +2.23),
}


class Stop(Exception):
    pass


def params_for_lognormal_galaxies(lim):
    """The parameters that lim.run_lognormal_simulation_cpp() gives
    lognormal_galaxies: the call is stopped at check_dir_im, the first step
    after the parameters are set, so nothing is run or written."""
    captured = {}

    def stop(params):
        captured.update(params)
        raise Stop

    with mock.patch.object(simple.simple, "check_dir_im", stop), contextlib.redirect_stdout(io.StringIO()):
        try:
            lim.run_lognormal_simulation_cpp()
        except Stop:
            pass
    return captured


def make_mock(cosmology):
    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
    input_dict["cosmology"] = cosmology
    return LognormalIntensityMock(input_dict)


class TestaH(unittest.TestCase):
    # aH, which lognormal_galaxies multiplies its displacements with to get
    # the velocities, must be a H(z) of the mock's cosmology in km/s per
    # Mpc/h: 100 E(z) / (1 + z), since H0 / h = 100 km/s/Mpc.
    #
    # The old formula, 100 sqrt(Om0 (1 + z)^3 + Ode0) / (1 + z), holds for
    # flat LambdaCDM without radiation only.

    def test_get_aH_is_a_H_of_the_cosmology(self):
        # get_aH against E(z) written out by hand. Four of these five
        # cosmologies have no radiation (Tcmb0 = 0), so that E(z) is short;
        # SIMPLE cannot run lognormal_galaxies for them (astropy's m_nu is
        # None without Tcmb0), which the next test does with RUNNABLE.
        for name, (cosmology, E) in COSMOLOGIES.items():
            with self.subTest(cosmology=name):
                lim = make_mock(cosmology)
                if E is None:
                    E = cosmology.efunc
                z = lim.redshift
                np.testing.assert_allclose(lim.get_aH(), 100 * E(z) / (1 + z), rtol=1e-12)

    def test_aH_given_to_lognormal_galaxies(self):
        # The aH that run_lognormal_simulation_cpp hands to lognormal_galaxies
        # is astropy's H(z) / h / (1 + z) at the redshift of the box, and the
        # old formula differs from it by the per cent in RUNNABLE.
        for name, (cosmology, percent) in RUNNABLE.items():
            with self.subTest(cosmology=name):
                params = params_for_lognormal_galaxies(make_mock(cosmology))
                z = params["z"]
                H = cosmology.H(z).to_value(u.km / u.s / u.Mpc)
                np.testing.assert_allclose(params["aH"], H / cosmology.h / (1 + z), rtol=1e-12)
                self.assertIsInstance(params["aH"], float)  # it is printed into a command line
                old = 100 * np.sqrt(cosmology.Om0 * (1 + z)**3 + cosmology.Ode0) / (1 + z)
                np.testing.assert_allclose(100 * (old / params["aH"] - 1), percent, atol=0.005)


if __name__ == "__main__":
    unittest.main()
