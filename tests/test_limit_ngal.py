import os
import tempfile
import unittest

import numpy as np
import astropy.units as u
from astropy.table import QTable
from scipy.integrate import quad

from simple.simple import LognormalIntensityMock
from simple.tools_python import yaml_file_to_dictionary

# requested number density of detected galaxies, in 1/Mpc^3
TARGET = 1e-3


def mock_with_selection(limit_ngal):
    """Apply the selection function for the given limit_ngal to galaxies placed
    uniformly in a small box (so lognormal_galaxies is not needed)."""
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
    return lim


def detected_density(limit_ngal):
    """Number density of detected galaxies in 1/Mpc^3."""
    lim = mock_with_selection(limit_ngal)
    return (np.sum(lim.cat["detected"]) / lim.box_volume).to_value(1 / u.Mpc**3)


def n_above(lim, L):
    """n(>L) in 1/Mpc^3: the integral of dn/dL from L to Lmax (L in luminosity_unit),
    by adaptive quadrature in ln L (accurate to ~3e-8 here), independent of
    SIMPLE's running integral."""
    L_max = lim.Lmax.to_value(lim.luminosity_unit)
    return quad(lambda lnL: lim.luminosity_function(np.exp(lnL)) * np.exp(lnL),
                np.log(L), np.log(L_max), epsrel=1e-8, epsabs=0, limit=500)[0]


def write_limit_ngal_table(tmp_dir):
    """ECSV table with limit_ngal = TARGET at all redshifts; returns its file name."""
    filename = os.path.join(tmp_dir, "limit_ngal.ecsv")
    table = QTable()
    table["redshift"] = np.linspace(1.5, 3.5, 21)
    table["limit_ngal"] = np.full(21, TARGET) / u.Mpc**3
    table.write(filename, format="ascii.ecsv")
    return filename


class TestLimitNgal(unittest.TestCase):
    # The flux limit derived from limit_ngal must give the requested density of
    # detected galaxies. The Poisson scatter is about 0.4% (5e4 galaxies);
    # we allow 3%. The flux limit itself is tested to 1e-6 below.

    def test_limit_ngal_number(self):
        n_detected = detected_density(f"{TARGET} / u.Mpc**3")
        np.testing.assert_allclose(n_detected, TARGET, rtol=0.03)

    def test_limit_ngal_table(self):
        # limit_ngal as a function of redshift, read from an ECSV table
        with tempfile.TemporaryDirectory() as tmp_dir:
            n_detected = detected_density(write_limit_ngal_table(tmp_dir))
        np.testing.assert_allclose(n_detected, TARGET, rtol=0.03)

    def test_limit_ngal_number_flux_limit_is_exact(self):
        # S_lim solves mean over the slice centres of n(>4 pi D_L^2 S_lim) = limit_ngal
        lim = mock_with_selection(f"{TARGET} / u.Mpc**3")
        L_lim = (4 * np.pi * lim.astropy_cosmo.luminosity_distance(lim.redshift_mesh_axis) ** 2
                 * lim.min_flux).to_value(lim.luminosity_unit)
        n_bar = np.mean([n_above(lim, L) for L in L_lim])
        np.testing.assert_allclose(n_bar, TARGET, rtol=1e-6)

    def test_limit_ngal_table_flux_limit_is_exact(self):
        # S_lim(z) solves n(>4 pi D_L(z)^2 S_lim(z)) = limit_ngal(z) at the table's redshifts
        with tempfile.TemporaryDirectory() as tmp_dir:
            lim = mock_with_selection(write_limit_ngal_table(tmp_dir))
        z = np.asarray(lim.min_flux_table["redshift"])[::10]
        L_lim = (4 * np.pi * lim.astropy_cosmo.luminosity_distance(z) ** 2
                 * lim.min_flux(z)).to_value(lim.luminosity_unit)
        np.testing.assert_allclose([n_above(lim, L) for L in L_lim], TARGET, rtol=1e-6)

    def test_limit_ngal_above_all_galaxies_raises(self):
        # no flux limit gives more galaxies than all of them: a ValueError, not a silent limit
        too_many = "1 / u.Mpc**3"
        with self.assertRaisesRegex(ValueError, "larger than the density"):
            mock_with_selection(too_many)
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, "limit_ngal.ecsv")
            QTable({"redshift": [1.5, 3.5], "limit_ngal": [1, 1] / u.Mpc**3}).write(filename, format="ascii.ecsv")
            with self.assertRaisesRegex(ValueError, "larger than the density"):
                mock_with_selection(filename)

    def test_save_and_load_with_limit_ngal_table(self):
        # The flux limit derived from the table must come back from the saved file,
        # because the model uses it for the detected and undetected galaxies.
        with tempfile.TemporaryDirectory() as tmp_dir:
            lim = mock_with_selection(write_limit_ngal_table(tmp_dir))
            filename = os.path.join(tmp_dir, "mock.h5")
            lim.save_to_file(filename)
            loaded = LognormalIntensityMock.from_file(filename, only_params=True)

        z = lim.cat["cosmo_redshift"]
        flux_unit = u.erg / u.s / u.cm**2
        np.testing.assert_allclose(loaded.min_flux(z).to_value(flux_unit),
                                   lim.min_flux(z).to_value(flux_unit), rtol=1e-12)
        self.assertTrue(callable(loaded.limit_ngal))


if __name__ == "__main__":
    unittest.main()
