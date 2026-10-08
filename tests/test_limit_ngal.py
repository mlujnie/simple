import os
import tempfile
import unittest

import numpy as np
import astropy.units as u
from astropy.table import QTable


def make_mock_with_uniform_catalog(limit_ngal, seed=42):
    """Set up a small mock whose galaxies are placed uniformly in the box,
    so that the selection function can be tested without running
    lognormal_galaxies."""
    from simple.simple import LognormalIntensityMock
    from simple.tools_python import yaml_file_to_dictionary

    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
    input_dict["box_size"] = "np.array([256, 256, 256]) * u.Mpc / cosmo.h"
    input_dict["seed_lognormal"] = seed
    input_dict["verbose"] = False
    del input_dict["min_flux"]
    input_dict["limit_ngal"] = limit_ngal
    lim = LognormalIntensityMock(input_dict)

    rng = np.random.default_rng(seed)
    N_gal = lim.N_gal
    box = lim.box_size.to_value(lim.Mpch)
    lim.cat = {
        "Position": rng.uniform(0, box, size=(N_gal, 3)) * lim.Mpch,
        "Velocity": np.zeros((N_gal, 3)) * u.km / u.s,
        "RSD_redshift_factor": np.ones(N_gal),
    }
    lim.assign_redshift_along_axis()
    lim.assign_luminosity()
    lim.assign_flux()
    lim.apply_selection_function()
    return lim


class TestLimitNgal(unittest.TestCase):
    """The flux limit derived from 'limit_ngal' must give the requested
    number density of detected galaxies."""

    target = 1e-3 / u.Mpc**3

    def assert_detected_density(self, lim):
        n_detected = (np.sum(lim.cat["detected"]) / lim.box_volume).to(
            1 / u.Mpc**3)
        # Poisson scatter is ~0.5% here; the rest is the interpolation
        # in apply_selection_function.
        self.assertTrue(
            u.isclose(n_detected, self.target, rtol=0.03),
            msg=f"detected n_gal = {n_detected}, wanted {self.target}")

    def test_limit_ngal_quantity(self):
        lim = make_mock_with_uniform_catalog(f"{self.target.value} / u.Mpc**3")
        self.assert_detected_density(lim)

    def test_limit_ngal_table(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, "limit_ngal.ecsv")
            table = QTable()
            table["redshift"] = np.linspace(1.5, 3.5, 21)
            table["limit_ngal"] = np.full(21, self.target.value) * self.target.unit
            table.write(filename, format="ascii.ecsv")
            lim = make_mock_with_uniform_catalog(filename)
        self.assertTrue(callable(lim.limit_ngal))
        self.assert_detected_density(lim)


if __name__ == "__main__":
    unittest.main()
