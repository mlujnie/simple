import unittest
import warnings

import numpy as np
import astropy.units as u
from astropy.table import Table
from scipy.integrate import quad


class TestLmax(unittest.TestCase):

    def test_assign_luminosity_keeps_Lmax(self):
        """Lmax is an input and must not be replaced by the brightest galaxy
        of the realization (which also lost its unit for astropy Tables)."""
        from simple.simple import LognormalIntensityMock

        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        Lmax_input = lim.Lmax
        rng = np.random.default_rng(42)
        lim.cat = Table()
        lim.cat["Position"] = rng.uniform(0, 100, size=(1000, 3)) * lim.Mpch
        lim.assign_luminosity()

        self.assertEqual(lim.Lmax, Lmax_input)
        self.assertEqual(lim.Lmax.unit, Lmax_input.unit)
        self.assertTrue(np.all(lim.cat["luminosity"] <= lim.Lmax.to_value(
            lim.cat["luminosity"].unit)))


class TestIntensityShotNoise(unittest.TestCase):
    """The analytic intensity shot noise integrates the luminosity function
    times L**2 up to Lmax. With the default Lmax = 1e10 * Lmin, scipy's quad
    returned 0 for 'all' and 'detected' galaxies."""

    def setUp(self):
        from simple.pk_3d_model import Power_Spectrum_Model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pk = Power_Spectrum_Model(
                "./tests/test_lim_input.yaml", do_model_shot_noise=False)
        # the tabulated luminosity function ends at 1000 * luminosity_unit
        self.L_table_max = 1000.0

    def shot_noise(self, galaxy_selection, Lmax):
        self.pk.Lmax = Lmax
        self.pk.galaxy_selection["intensity"] = galaxy_selection
        for cached in ["mean_intensity", "mean_intensity_per_redshift_mesh"]:
            self.pk.__dict__.pop(cached, None)
        return self.pk.get_intensity_shot_noise().to_value(self.pk.Mpch**3)

    def test_integral_matches_quad_on_short_range(self):
        Lmin = self.pk.Lmin.to_value(self.pk.luminosity_unit)
        Lmax = self.pk.Lmax.to_value(self.pk.luminosity_unit)
        self.assertGreaterEqual(Lmax / Lmin, 1e10)

        integral = self.pk.integrate_luminosity_function_times_Lsq(Lmin, Lmax)
        reference, _ = quad(self.pk.luminosity_function_times_Lsq,
                            Lmin, self.L_table_max, limit=200)
        self.assertGreater(reference, 0)
        np.testing.assert_allclose(integral, reference, rtol=1e-4)

    def test_shot_noise_nonzero_and_independent_of_Lmax(self):
        for galaxy_selection in ["all", "detected", "undetected"]:
            with self.subTest(galaxy_selection=galaxy_selection):
                default_Lmax = self.shot_noise(
                    galaxy_selection, 1e10 * self.pk.Lmin)
                short_Lmax = self.shot_noise(
                    galaxy_selection, self.L_table_max * self.pk.luminosity_unit)

                self.assertGreater(default_Lmax, 0)
                np.testing.assert_allclose(default_Lmax, short_Lmax, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
