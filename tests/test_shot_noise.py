import unittest

import astropy.constants as const
import astropy.units as u
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


class TestLineUnits(unittest.TestCase):
    # The intensity can be I_lambda, I_nu or the brightness temperature T_b.
    # For the same galaxies they are related by I_nu = I_lambda lambda_obs^2 / c
    # (I_nu dnu = I_lambda dlambda) and T_b = c^2 I_nu / (2 k_B nu_obs^2)
    # (Rayleigh-Jeans, per steradian), with lambda_obs = lambda_0 (1 + z).
    # The mean intensity and the Poisson noise in physical units must follow.

    LAMBDA_0 = 1215.67 * u.angstrom

    def model(self, line, temperature):
        input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
        input_dict["f_growth_filename"] = "./tests/input/example_fnu.txt"
        input_dict["brightness_temperature"] = temperature
        if line == "nu":
            del input_dict["lambda_restframe"], input_dict["dlambda"]
            input_dict["nu_restframe"] = (const.c / self.LAMBDA_0).to(u.GHz)
            input_dict["dnu"] = 1 * u.MHz
        return Power_Spectrum_Model(input_dict)

    def test_mean_intensity_and_shot_noise(self):
        pk = self.model("lambda", False)
        z = pk.redshift_mesh_axis
        lambda_obs = self.LAMBDA_0 * (1 + z)
        to_I_nu = lambda_obs**2 / const.c
        to_T_b = lambda_obs**4 / (2 * const.k_B * const.c) * u.sr
        I = pk.mean_intensity_per_redshift(z)
        # Poisson noise per slice in physical units (it is divided by the mean intensity^2)
        P = pk.intensity_shot_noise_per_slice() * pk.mean_intensity**2
        for line, temperature, factor, unit in [
                ("nu", False, to_I_nu, u.erg / u.s / u.cm**2 / u.Hz / u.sr),
                ("lambda", True, to_T_b, u.uK), ("nu", True, to_T_b, u.uK)]:
            other = self.model(line, temperature)
            np.testing.assert_allclose(other.mean_intensity_per_redshift(z).to_value(unit),
                                       (I * factor).to_value(unit), rtol=1e-10)
            P_other = other.intensity_shot_noise_per_slice() * other.mean_intensity**2
            np.testing.assert_allclose(P_other.to_value(unit**2 * pk.Mpch**3),
                                       (P * factor**2).to_value(unit**2 * pk.Mpch**3), rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
