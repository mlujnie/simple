import os
import tempfile
import unittest

import numpy as np
import astropy.units as u
from astropy.table import QTable

from simple.simple import LognormalIntensityMock
from simple.tools_python import yaml_file_to_dictionary

# requested number density of detected galaxies, in 1/Mpc^3
TARGET = 1e-3


def detected_density(limit_ngal):
    """Apply the selection function for the given limit_ngal to galaxies placed
    uniformly in a small box (so lognormal_galaxies is not needed) and return
    the number density of detected galaxies in 1/Mpc^3."""
    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
    input_dict["box_size"] = "np.array([256, 256, 256]) * u.Mpc / cosmo.h"
    del input_dict["min_flux"]  # the flux limit comes from limit_ngal instead
    input_dict["limit_ngal"] = limit_ngal
    lim = LognormalIntensityMock(input_dict)

    # uniform catalog without velocities
    N = lim.N_gal
    box = lim.box_size.to_value(lim.Mpch)
    lim.cat = {
        "Position": np.random.default_rng(42).uniform(0, box, size=(N, 3)) * lim.Mpch,
        "Velocity": np.zeros((N, 3)) * u.km / u.s,
        "RSD_redshift_factor": np.ones(N),
    }
    lim.assign_redshift_along_axis()
    lim.assign_luminosity()
    lim.assign_flux()
    lim.apply_selection_function()
    return (np.sum(lim.cat["detected"]) / lim.box_volume).to_value(1 / u.Mpc**3)


class TestLimitNgal(unittest.TestCase):
    # The flux limit derived from limit_ngal must give the requested density of
    # detected galaxies. The Poisson scatter is about 0.4% (5e4 galaxies);
    # we allow 3% for the interpolation in apply_selection_function.

    def test_limit_ngal_number(self):
        n_detected = detected_density(f"{TARGET} / u.Mpc**3")
        np.testing.assert_allclose(n_detected, TARGET, rtol=0.03)

    def test_limit_ngal_table(self):
        # limit_ngal as a function of redshift, read from an ECSV table
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, "limit_ngal.ecsv")
            table = QTable()
            table["redshift"] = np.linspace(1.5, 3.5, 21)
            table["limit_ngal"] = np.full(21, TARGET) / u.Mpc**3
            table.write(filename, format="ascii.ecsv")
            n_detected = detected_density(filename)
        np.testing.assert_allclose(n_detected, TARGET, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
