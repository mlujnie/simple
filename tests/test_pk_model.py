import os
import tempfile
import unittest
import warnings

import astropy.units as u
import numpy as np


def input_dict_with_growth_table(tmp_dir):
    """Power_Spectrum_Model reads the growth rate f(k) from a file that only a run
    of lognormal_galaxies writes, so a fresh checkout lacks it. These tests are in
    real space and never use f, so a constant table is enough."""
    from simple.tools_python import yaml_file_to_dictionary

    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
    f_growth_filename = os.path.join(tmp_dir, "fnu.txt")
    np.savetxt(f_growth_filename,
               np.column_stack([np.logspace(-4, 2, 7), np.ones(7)]))
    input_dict["f_growth_filename"] = f_growth_filename
    return input_dict


class TestModelRemovedModes(unittest.TestCase):
    """The intensity estimator subtracts the mean of each slice along the line of
    sight, so the intensity field (and the cross power spectrum) has no power in
    modes with k_perp = 0. The galaxy field is divided by the expected mean density
    and keeps those modes, except k = 0. The model must remove the same modes."""

    def model_3d(self, tracer, min_flux_mesh=False):
        from simple.pk_3d_model import Power_Spectrum_Model

        with tempfile.TemporaryDirectory() as tmp:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pk = Power_Spectrum_Model(
                    input_dict_with_growth_table(tmp), do_model_shot_noise=False,
                    out_filename=os.path.join(tmp, "model.h5"))
                pk.get_kspec()
                if min_flux_mesh:
                    pk.min_flux = np.ones(pk.N_mesh) * 1e-17 * u.erg / u.s / u.cm**2
                model = pk.get_3d_pk_model(
                    damping_function=1.0,
                    P_shot_smoothed=0.0 * pk.Mpch**3,
                    S_bar=0.0,
                    mask_window_function_1=np.ones(pk.N_mesh),
                    observed_volume=pk.box_volume,
                    box_volume=pk.box_volume,
                    Pk_unit=pk.Mpch**3,
                    save=False,
                    tracer=tracer,
                    return_3d=True,
                )[0]
        kspec = np.asarray(pk.kspec)
        expected = pk.bias**2 * pk.Plin(kspec)
        # zero up to the round-off of the FFTs in the window convolution
        atol = 1e-6 * np.max(model.real)
        return pk, model.real, expected, atol

    def test_intensity_has_no_signal_at_kperp_zero(self):
        pk, model, expected, atol = self.model_3d("intensity")
        k_perp_zero = np.asarray(pk.k_perp) == 0
        self.assertEqual(np.sum(k_perp_zero), pk.N_mesh[0])
        np.testing.assert_allclose(model[k_perp_zero], 0.0, atol=atol)
        np.testing.assert_allclose(
            model[~k_perp_zero], expected[~k_perp_zero], rtol=1e-6, atol=atol)

    def test_n_gal_keeps_kperp_zero_except_k_zero(self):
        pk, model, expected, atol = self.model_3d("n_gal")
        k_zero = np.asarray(pk.kspec) == 0
        self.assertEqual(np.sum(k_zero), 1)
        np.testing.assert_allclose(model[k_zero], 0.0, atol=atol)
        np.testing.assert_allclose(
            model[~k_zero], expected[~k_zero], rtol=1e-6, atol=atol)

    def test_n_gal_with_flux_limit_mesh_drops_kperp_zero(self):
        # With a min_flux mesh the galaxy field is divided by the measured mean of
        # each slice, so the k_perp = 0 modes are removed as for intensity.
        pk, model, expected, atol = self.model_3d("n_gal", min_flux_mesh=True)
        self.assertTrue(pk.min_flux_is_mesh)
        k_perp_zero = np.asarray(pk.k_perp) == 0
        np.testing.assert_allclose(model[k_perp_zero], 0.0, atol=atol)
        np.testing.assert_allclose(
            model[~k_perp_zero], expected[~k_perp_zero], rtol=1e-6, atol=atol)


class TestNgalNormalization(unittest.TestCase):
    """The galaxy field is divided by the expected mean density at each redshift,
    not by the measured mean of each slice, so a density wave along the line of
    sight survives."""

    def test_line_of_sight_wave_survives(self):
        from simple.simple import LognormalIntensityMock

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        self.assertIsNone(lim.obs_mask)
        n0 = lim.N_mesh[0]
        wave = 0.1 * np.cos(2 * np.pi * np.arange(n0) / n0)[:, None, None]
        delta = wave * np.ones(lim.N_mesh)
        lim.n_gal_mesh = lim.mean_ngal_per_redshift_mesh * (1 + delta)

        lim._get_prepared_n_gal_mesh()
        prepared = np.asarray(lim.prepared_n_gal_mesh_ft.c2r())

        np.testing.assert_allclose(prepared, delta, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
