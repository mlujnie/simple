import unittest

import numpy as np
from astropy.cosmology import FlatwCDM, LambdaCDM, Planck18

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


class TestaH(unittest.TestCase):
    # aH, which lognormal_galaxies multiplies its displacements with to get
    # the velocities, must be a H(z) of the mock's cosmology in km/s per
    # Mpc/h: 100 E(z) / (1 + z), since H0 / h = 100 km/s/Mpc.
    #
    # The old formula, 100 sqrt(Om0 (1 + z)^3 + Ode0) / (1 + z), holds for
    # flat LambdaCDM without radiation only. At z = 2.2245 it was 0.25 % too
    # small for Planck18, 3.0 % too small for w = -0.8, 1.6 % too large for
    # w = -1.2, 2.3 % too small for Omega_k = +0.05 and 2.4 % too large for
    # Omega_k = -0.05, and the RSD displacement v (1 + z) / H(z) with it.

    def test_aH_is_a_H_of_the_cosmology(self):
        for name, (cosmology, E) in COSMOLOGIES.items():
            input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
            input_dict["cosmology"] = cosmology
            lim = LognormalIntensityMock(input_dict)
            if E is None:
                E = cosmology.efunc
            z = lim.redshift
            np.testing.assert_allclose(
                lim.get_aH(), 100 * E(z) / (1 + z), rtol=1e-12, err_msg=name)


if __name__ == "__main__":
    unittest.main()
