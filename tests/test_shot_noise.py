import unittest

import numpy as np
from astropy.table import Table
from scipy.integrate import quad

from simple.simple import LognormalIntensityMock
from simple.pk_3d_model import Power_Spectrum_Model
from simple.tools_python import yaml_file_to_dictionary


class TestLmax(unittest.TestCase):

    def test_assign_luminosity_keeps_Lmax(self):
        # Lmax is an input parameter. Drawing the luminosities must not replace it
        # with the luminosity of the brightest galaxy.
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        Lmax_input = lim.Lmax
        positions = np.random.default_rng(42).uniform(0, 100, size=(1000, 3))
        lim.cat = Table()
        lim.cat["Position"] = positions * lim.Mpch
        lim.assign_luminosity()
        self.assertEqual(lim.Lmax, Lmax_input)


class TestIntensityShotNoise(unittest.TestCase):

    def setUp(self):
        input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
        # growth rate table, normally written by lognormal_galaxies when it runs
        input_dict["f_growth_filename"] = "./tests/input/example_fnu.txt"
        self.pk = Power_Spectrum_Model(input_dict)

    def test_integral_matches_quad(self):
        # On a short L range scipy's quad works, so the new integral must agree
        # with it. The tabulated luminosity function ends at 1000 luminosity units.
        Lmin = self.pk.Lmin.to_value(self.pk.luminosity_unit)
        integral = self.pk.integrate_luminosity_function_times_Lsq(Lmin, 1000)
        reference = quad(self.pk.luminosity_function_times_Lsq, Lmin, 1000, limit=200)[0]
        np.testing.assert_allclose(integral, reference, rtol=1e-4)

    def test_shot_noise_does_not_depend_on_Lmax(self):
        # Lmax: inf becomes Lmax = 1e10 * Lmin, where quad used to return 0.
        # The shot noise must be > 0 and equal to the one with Lmax at the end of
        # the tabulated luminosity function.
        for selection in ["all", "detected", "undetected"]:
            self.pk.galaxy_selection["intensity"] = selection
            shot_noise = []
            for Lmax in [1e10 * self.pk.Lmin, 1000 * self.pk.luminosity_unit]:
                self.pk.Lmax = Lmax
                # forget the stored mean intensity, which depends on Lmax and the selection
                self.pk.__dict__.pop("mean_intensity", None)
                self.pk.__dict__.pop("mean_intensity_per_redshift_mesh", None)
                shot_noise.append(self.pk.get_intensity_shot_noise().value)
            self.assertGreater(shot_noise[0], 0)
            np.testing.assert_allclose(shot_noise[0], shot_noise[1], rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
