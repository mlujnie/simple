import unittest

import astropy.units as u
import numpy as np

from simple.pk_3d_model import Power_Spectrum_Model
from simple.tools_python import yaml_file_to_dictionary

FLUX = u.erg / u.s / u.cm**2
# Flux limits whose luminosity limit L_lim = 4 pi D_L^2 min_flux lies outside
# [Lmin, Lmax], the luminosities the mock's galaxies have. In the box of
# tests/test_lim_input.yaml (slice centres from z = 1.957 to 2.488,
# Lmin = 1e41 erg/s) a galaxy of Lmin has a flux of 1.94e-18 (far side) to
# 3.48e-18 erg/s/cm^2 (near side), and one of Lmax = 1e43 erg/s a flux 100
# times larger. Lmax = 1e43 erg/s is low enough that the luminosity function
# above it matters: without the clipping, the Poisson noise of the undetected
# galaxies with the limit ABOVE_LMAX would be 21 % too large.
BELOW_LMIN = 1e-19 * FLUX  # every galaxy is detected
ABOVE_LMAX = 1e-13 * FLUX  # no galaxy is detected (with Lmax = 1e43 erg/s)
INSIDE = 3e-17 * FLUX  # L_lim = 8.6e41 to 1.5e42 erg/s
LMAX = 1e43 * u.erg / u.s


def model(**changes):
    """A Power_Spectrum_Model of tests/test_lim_input.yaml with the given changes."""
    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
    # growth rate table, normally written by lognormal_galaxies when it runs
    input_dict["f_growth_filename"] = "./tests/input/example_fnu.txt"
    input_dict.update(changes)
    return Power_Spectrum_Model(input_dict, do_model_shot_noise=False)


def means(pk, selection):
    """The expected mean intensity and n_bar of the selection in each slice
    (the model returns the expected means for a min_flux mesh too)."""
    z = pk.redshift_mesh_axis
    return [pk.mean_intensity_per_redshift(z, selection, tracer) for tracer in ["intensity", "n_gal"]]


def shot_noise(pk, selection):
    """The Poisson noise of the intensity map of the selection in each slice."""
    pk.galaxy_selection["intensity"] = selection
    # forget the stored mean intensity, which depends on the selection
    pk.__dict__.pop("mean_intensity", None)
    pk.__dict__.pop("mean_intensity_per_redshift_mesh", None)
    return pk.intensity_shot_noise_per_slice()


class TestFluxLimitOutsideLuminosityRange(unittest.TestCase):

    def test_limit_below_Lmin_detects_all(self):
        # Before the clipping, n_bar of the detected galaxies was the integral
        # of the luminosity function from L_lim < Lmin, where the mock has no
        # galaxies, and n_bar and the mean intensity of the undetected ones
        # were negative.
        pk = model(min_flux=BELOW_LMIN)
        I_all, n_all = means(pk, "all")
        I_det, n_det = means(pk, "detected")
        # n_bar of all galaxies is integrated on 10^5 luminosities, that of the
        # detected ones on 10^4; the intensity on 10^4 in both
        np.testing.assert_allclose(n_det, n_all, rtol=1e-5)
        np.testing.assert_allclose(I_det, I_all, rtol=1e-12)
        I_undet, n_undet = means(pk, "undetected")
        self.assertTrue(np.all(I_undet.value == 0) and np.all(n_undet.value == 0))

    def test_limit_above_Lmax_detects_none(self):
        pk = model(min_flux=ABOVE_LMAX, Lmax=LMAX)
        I_all, n_all = means(pk, "all")
        I_det, n_det = means(pk, "detected")
        self.assertTrue(np.all(I_det.value == 0) and np.all(n_det.value == 0))
        I_undet, n_undet = means(pk, "undetected")
        np.testing.assert_allclose(n_undet, n_all, rtol=1e-5)
        np.testing.assert_allclose(I_undet, I_all, rtol=1e-12)

    def test_limit_crossing_Lmin_inside_the_box(self):
        # One flux limit for the box, the flux of a galaxy of Lmin at the
        # middle redshift: L_lim is below Lmin on the near side and above it
        # on the far side. n_bar of the detected galaxies in each slice must
        # be the integral of the luminosity function from max(L_lim, Lmin),
        # computed here on 10^5 luminosities.
        # The limit is given as a number and as a function of z.
        pk = model()
        z = pk.redshift_mesh_axis
        D_L = pk.astropy_cosmo.luminosity_distance
        limit = (pk.Lmin / (4 * np.pi * D_L(pk.redshift) ** 2)).to(FLUX)
        L_lim = (4 * np.pi * D_L(z) ** 2 * limit).to_value(pk.luminosity_unit)
        L_min, L_max = [L.to_value(pk.luminosity_unit) for L in (pk.Lmin, pk.Lmax)]
        self.assertTrue(np.sum(L_lim < L_min) >= 4 and np.sum(L_lim > L_min) >= 4)
        expected = []
        for L in L_lim:
            Ls = np.logspace(np.log10(max(L, L_min)), np.log10(L_max), 100000)
            expected.append(np.trapz(pk.luminosity_function(Ls), Ls))
        for min_flux in [limit, lambda z: limit]:
            pk.min_flux = min_flux
            n_det = means(pk, "detected")[1]
            np.testing.assert_allclose(n_det.to_value(u.Mpc**-3), expected, rtol=1e-5)

    def test_number_and_mesh_agree(self):
        # The same flux limit as a number and as a mesh (the same in every
        # cell) must give the same expected means and Poisson noise per slice.
        # The mesh always kept the integrals constant outside [Lmin, Lmax].
        for limit in [BELOW_LMIN, INSIDE, ABOVE_LMAX]:
            number, mesh = model(min_flux=limit, Lmax=LMAX), model(Lmax=LMAX)
            mesh.min_flux = np.full(mesh.N_mesh, limit.value) * FLUX
            I_all, n_all = means(number, "all")
            P_all = shot_noise(number, "all")
            for selection in ["detected", "undetected"]:
                I_number, n_number = means(number, selection)
                I_mesh, n_mesh = means(mesh, selection)
                # the mesh integrates on 10^5 luminosities and interpolates,
                # the number on 10^4
                np.testing.assert_allclose(I_mesh, I_number, rtol=0, atol=1e-5 * I_all.max())
                np.testing.assert_allclose(n_mesh, n_number, rtol=0, atol=1e-5 * n_all.max())
                if np.all(I_number.value == 0):
                    continue  # no galaxies in the map: its Poisson noise is 0 / 0
                np.testing.assert_allclose(shot_noise(mesh, selection), shot_noise(number, selection),
                                           rtol=1e-5, err_msg=f"{limit} {selection}")

    def test_shot_noise_with_limit_below_Lmin(self):
        # All galaxies are detected, so the Poisson noise of the intensity map
        # of the detected galaxies is that of all galaxies.
        pk = model(min_flux=BELOW_LMIN)
        np.testing.assert_allclose(shot_noise(pk, "detected"), shot_noise(pk, "all"), rtol=1e-12)

    def test_shot_noise_with_limit_above_Lmax(self):
        # No galaxy is detected, so the Poisson noise of the intensity map of
        # the undetected galaxies is that of all galaxies.
        pk = model(min_flux=ABOVE_LMAX, Lmax=LMAX)
        np.testing.assert_allclose(shot_noise(pk, "undetected"), shot_noise(pk, "all"), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
