import unittest

import astropy.units as u
import numpy as np

from simple.pk_3d_model import Power_Spectrum_Model
from simple.simple import LognormalIntensityMock
from simple.tools_python import yaml_file_to_dictionary


def model_3d(tracer, min_flux_mesh=False):
    """Return the 3D model power spectrum in real space (no mask or smoothing)
    with a shot noise P_shot, and the expected b^2 P(k) + P_shot on the same grid."""
    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
    # growth rate table, normally written by lognormal_galaxies when it runs
    input_dict["f_growth_filename"] = "./tests/input/example_fnu.txt"
    pk = Power_Spectrum_Model(input_dict, do_model_shot_noise=False)
    P_shot = 100.0  # (Mpc/h)^3
    pk.get_kspec()
    if min_flux_mesh:
        pk.min_flux = np.ones(pk.N_mesh) * 1e-17 * u.erg / u.s / u.cm**2
    model = pk.get_3d_pk_model(
        damping_function=1.0, P_shot_smoothed=P_shot * pk.Mpch**3, S_bar=0.0,
        mask_window_function_1=np.ones(pk.N_mesh),
        observed_volume=pk.box_volume, box_volume=pk.box_volume,
        Pk_unit=pk.Mpch**3, save=False, tracer=tracer, return_3d=True)[0]
    expected = pk.bias**2 * pk.Plin(np.asarray(pk.kspec)) + P_shot
    return pk, model.real, expected


class TestModelRemovedModes(unittest.TestCase):
    # The model, signal and noise, must be zero in the modes that the
    # estimators remove.

    def assert_model(self, model, expected, removed):
        """The model is zero in the removed modes and b^2 P(k) + P_shot in all others."""
        self.assertTrue(removed.any())
        # zero up to the round-off of the FFTs in the window convolution
        atol = 1e-6 * np.max(model)
        np.testing.assert_allclose(model[removed], 0.0, atol=atol)
        np.testing.assert_allclose(
            model[~removed], expected[~removed], rtol=1e-6, atol=atol)

    def test_intensity(self):
        # The intensity estimator subtracts the mean of each slice along the
        # line of sight, which removes the modes with k_perp = 0.
        pk, model, expected = model_3d("intensity")
        self.assert_model(model, expected, np.asarray(pk.k_perp) == 0)

    def test_n_gal(self):
        # The galaxy field is divided by the expected mean density, which
        # removes only k = 0.
        pk, model, expected = model_3d("n_gal")
        self.assert_model(model, expected, np.asarray(pk.kspec) == 0)

    def test_n_gal_with_min_flux_mesh(self):
        # With a min_flux mesh the galaxy field is divided by the measured mean
        # of each slice, which removes k_perp = 0 as for intensity.
        pk, model, expected = model_3d("n_gal", min_flux_mesh=True)
        self.assert_model(model, expected, np.asarray(pk.k_perp) == 0)


class TestNgalNormalization(unittest.TestCase):

    def test_line_of_sight_wave_survives(self):
        # The galaxy field is divided by the expected mean density at each
        # redshift, not by the measured mean of each slice, so a density wave
        # along the line of sight (k_perp = 0) must survive.
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        n = lim.N_mesh[0]
        wave = 0.1 * np.cos(2 * np.pi * np.arange(n) / n)[:, None, None]
        delta = wave * np.ones(lim.N_mesh)
        lim.n_gal_mesh = lim.mean_ngal_per_redshift_mesh * (1 + delta)

        lim._get_prepared_n_gal_mesh()
        prepared = np.asarray(lim.prepared_n_gal_mesh_ft.c2r())
        np.testing.assert_allclose(prepared, delta, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
