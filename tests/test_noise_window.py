import os
import tempfile
import unittest

import astropy.units as u
import numpy as np

from simple.pk_3d_model import Power_Spectrum_Model
import simple.simple as simple_module
from simple.simple import make_map
from simple.tools_python import yaml_file_to_dictionary


def power_model(N_mesh=8, **settings):
    """A Power_Spectrum_Model of the test config (changed by settings) on a
    small grid, with a pixel noise of 0.05 times the mean intensity and the k grid."""
    input_dict = dict(yaml_file_to_dictionary("./tests/test_lim_input.yaml"), **settings)
    # growth rate table, normally written by lognormal_galaxies when it runs
    input_dict["f_growth_filename"] = "./tests/input/example_fnu.txt"
    input_dict["N_mesh"] = [N_mesh] * 3
    input_dict["galaxy_selection"] = {"intensity": "detected", "n_gal": "detected"}
    pk = Power_Spectrum_Model(input_dict, do_model_shot_noise=False)
    pk.get_kspec()
    pk.sigma_noise = 0.05 * pk.mean_intensity
    return pk


def sky_mask(pk, seed=1):
    """A mask that is the same in every slice: half of the sky observed, with
    weights from 0.5 to 1.5, so that it has structure across the slice on
    the scale of the box."""
    n = pk.N_mesh
    rng = np.random.RandomState(seed)
    W = (np.arange(n[1])[:, None] < n[1] // 2) * rng.uniform(0.5, 1.5, (n[1], n[2]))
    return np.broadcast_to(W, tuple(n)).copy()


def los_mask(pk, seed=2):
    """sky_mask times weights that change from cell to cell, with a quarter
    of the cells masked: a mask that also changes along the LOS, like one
    that masks single pixels of the smoothed map."""
    rng = np.random.RandomState(seed)
    return (sky_mask(pk) * rng.uniform(0.5, 1.5, tuple(pk.N_mesh))
            * (rng.uniform(size=tuple(pk.N_mesh)) > 0.25))


def smooth(pk, field, D):
    """field smoothed with the damping D (not squared) on the k grid."""
    return np.array(make_map(np.array(make_map(field, Nmesh=pk.N_mesh,
                                               BoxSize=pk.box_size.to_value(pk.Mpch)).r2c()) * D,
                             Nmesh=pk.N_mesh, BoxSize=pk.box_size.to_value(pk.Mpch),
                             type="complex").c2r())


def smoothing_filter(pk):
    """The filter with which SIMPLE smooths the intensity map (aniso_filter or
    aniso_filter_gaussian in simple.py, with the LOS along axis 0) on the k grid."""
    simple_module.sigma = np.array([pk.s_par, pk.s_perp, pk.s_perp])  # set as in SIMPLE
    filter_function = (simple_module.aniso_filter if pk.do_spectral_tophat_smooth
                       else simple_module.aniso_filter_gaussian)
    ones = make_map(np.ones(np.shape(pk.k_par), dtype=complex), Nmesh=pk.N_mesh,
                    BoxSize=pk.box_size.to_value(pk.Mpch), type="complex")
    return np.array(ones.apply(filter_function, kind="wavenumber")).real


def exact_noise(pk, noise_power, intensity_response, galaxy_response=None,
                sky_subtraction=False):
    """
    The expected V Re[d_1(k) d_2(k)^*] of a noise that is independent between
    cells, with the power noise_power (in (Mpc/h)^3) in each cell: the sum
    over the cells of the response of the maps to a noise in that cell alone.
    d_1 is SIMPLE's prepared intensity map (_get_prepared_intensity_mesh) with
    the intensity mesh intensity_response(noise) times the mean intensity;
    d_2 is d_1, or galaxy_response(noise) times the mask, transformed as
    SIMPLE does. The response is computed for a noise of 1 and scaled, so
    that the single-precision FFTs work with numbers of order 1. With
    sky_subtraction, d_1 is the sky-subtracted map; its sky estimate contains
    the noise, as SIMPLE's does.
    """
    V = pk.box_size.to_value(pk.Mpch).prod()
    V_cell = V / np.prod(pk.N_mesh)
    total = 0
    for i in range(np.prod(pk.N_mesh)):
        unit = np.zeros(np.prod(pk.N_mesh))
        unit[i] = 1.0
        unit = unit.reshape(pk.N_mesh)
        pk.intensity_mesh = intensity_response(unit) * pk.mean_intensity
        pk._get_prepared_intensity_mesh(sky_subtraction=sky_subtraction)
        d_1 = np.array(pk.prepared_skysub_intensity_mesh_ft if sky_subtraction
                       else pk.prepared_intensity_mesh_ft)
        if galaxy_response is None:
            d_2 = d_1
        else:
            d_2 = np.array(make_map(galaxy_response(unit) * pk.obs_mask, Nmesh=pk.N_mesh,
                                    BoxSize=pk.box_size.to_value(pk.Mpch)).r2c())
        variance = np.reshape(noise_power, -1)[i] / V_cell
        total = total + variance * V * np.real(d_1 * np.conj(d_2))
    return total


class TestNoiseThroughWindow(unittest.TestCase):
    # The estimators subtract the mean of each slice along the LOS and then
    # multiply by the mask, so the noise must lose its k_perp = 0 modes
    # before the window, not after it.

    def test_pixel_noise(self):
        # Pixel noise sigma / sqrt(mask), not smoothed, with a mask that also
        # changes along the LOS: the model is exact.
        pk = power_model()
        pk.obs_mask = los_mask(pk)
        observed = pk.obs_mask > 0
        sigma = 0.05 / np.sqrt(np.where(observed, pk.obs_mask, 1.0)) * observed
        pk.noise_mesh = None
        V_cell = pk.voxel_volume.to_value(pk.Mpch**3)
        exact = exact_noise(pk, sigma**2 * V_cell, lambda unit: unit)
        model = (pk.get_intensity_noise(sigma * pk.mean_intensity)
                 - pk.get_intensity_noise(0 * pk.mean_intensity)).to_value(pk.Mpch**3)
        np.testing.assert_allclose(model, exact, rtol=1e-5, atol=1e-6 * np.max(exact))
        # adding the noise after the window, as before, is off by more than 10%
        old = np.mean(pk.obs_mask**2 * sigma**2) * V_cell * (np.asarray(pk.k_perp) > 0)
        self.assertGreater(np.max(np.abs(old - exact)) / np.max(exact), 0.1)

    def test_poisson_noise(self):
        # Poisson noise with a power a^2(x_par) per slice, smoothed like the
        # signal and masked after the smoothing, with a mask that also changes
        # along the LOS. The model is exact if a is the same in every slice,
        # or without LOS smoothing.
        for constant_a, s_par in [(True, 40.0), (False, 0.0)]:
            with self.subTest(constant_a=constant_a, s_par=s_par):
                pk = power_model()
                pk.obs_mask = los_mask(pk)
                pk.s_par, pk.s_perp = s_par, 50.0  # Mpc/h, comparable to the cells
                a2 = pk.intensity_shot_noise_per_slice()
                if constant_a:
                    a2_constant = np.mean(a2) * np.ones(len(a2))
                    pk.intensity_shot_noise_per_slice = lambda: a2_constant
                    a2 = a2_constant
                a2 = a2.to_value(pk.Mpch**3)
                D = np.sqrt(pk.D_sq_par_of_k(pk.k_par, pk.s_par) * pk.D_sq_perp_of_k(pk.k_perp, pk.s_perp))
                exact = exact_noise(pk, a2[:, None, None] * np.ones(pk.N_mesh),
                                    lambda unit: smooth(pk, unit, D))
                model = pk.get_intensity_noise(0 * pk.mean_intensity).to_value(pk.Mpch**3)
                np.testing.assert_allclose(model, exact, rtol=1e-5, atol=1e-6 * np.max(exact))
                # adding the noise after the window, as before, is off by more than 10%
                kept = np.asarray(pk.k_perp) > 0
                old = pk.get_intensity_shot_noise().to_value(pk.Mpch**3) * D**2 * kept
                self.assertGreater(np.max(np.abs(old - exact)) / np.max(exact), 0.1)

    def test_cross_noise(self):
        # The intensity map is smoothed and loses k_perp = 0, the galaxy map
        # does neither; c(x_par) is the covariance of the two per cell. Exact
        # in the same cases as the intensity Poisson noise.
        for constant_c, s_par in [(True, 40.0), (False, 0.0)]:
            with self.subTest(constant_c=constant_c, s_par=s_par):
                pk = power_model()
                pk.obs_mask = los_mask(pk)
                pk.s_par, pk.s_perp = s_par, 50.0  # Mpc/h
                c = pk.cross_shot_noise_per_slice()
                if constant_c:
                    c_constant = np.mean(c) * np.ones(len(c))
                    pk.cross_shot_noise_per_slice = lambda: c_constant
                    c = c_constant
                c = c.to_value(pk.Mpch**3)[:, None, None]
                D = np.sqrt(pk.D_sq_par_of_k(pk.k_par, pk.s_par) * pk.D_sq_perp_of_k(pk.k_perp, pk.s_perp))
                # a noise of power c^2 in the intensity map and of power 1 in the
                # galaxy map (the same noise divided by c): their covariance is c
                exact = exact_noise(pk, c**2 * np.ones(pk.N_mesh), lambda unit: smooth(pk, unit, D),
                                    lambda unit: unit / c)
                model = pk.get_cross_noise().to_value(pk.Mpch**3)
                np.testing.assert_allclose(model, exact, rtol=1e-5, atol=1e-6 * np.max(np.abs(exact)))
                # adding the noise after the window, as before, is off by more than 10%
                kept = np.asarray(pk.k_perp) > 0
                old = pk.get_cross_shot_noise().to_value(pk.Mpch**3) * D * kept
                self.assertGreater(np.max(np.abs(old - exact)) / np.max(np.abs(exact)), 0.1)

    def test_no_mask(self):
        # Without a mask the exact noise is P_shot D^2 + S_bar V for intensity
        # and P_shot D for cross, 0 at k_perp = 0 (a(x_par) times a white noise
        # is white, with the power mean(a^2)), as before. The new model gives
        # this if the Poisson noise is the same in every slice. If it changes
        # along the LOS, the new model is off at second order in
        # s_par * dln(a)/dx_par: by 3e-4 of P_shot here.
        for constant_a, rtol in [(True, 1e-5), (False, 1e-3)]:
            with self.subTest(constant_a=constant_a):
                pk = power_model(N_mesh=16)
                if constant_a:
                    a2 = pk.intensity_shot_noise_per_slice()
                    a2_constant = np.mean(a2) * np.ones(len(a2))
                    pk.intensity_shot_noise_per_slice = lambda: a2_constant
                kept = np.asarray(pk.k_perp) > 0
                D_sq = pk.D_sq_par_of_k(pk.k_par, pk.s_par) * pk.D_sq_perp_of_k(pk.k_perp, pk.s_perp)
                S_bar_V = (pk.get_S_bar(pk.weight_mesh_im * pk.obs_mask, pk.sigma_noise)
                           * pk.box_volume).to_value(pk.Mpch**3)
                expected = (pk.get_intensity_shot_noise().to_value(pk.Mpch**3) * D_sq + S_bar_V) * kept
                model = pk.get_intensity_noise(pk.sigma_noise).to_value(pk.Mpch**3)
                np.testing.assert_allclose(model, expected, rtol=rtol, atol=rtol * np.max(expected))
                expected = pk.get_cross_shot_noise().to_value(pk.Mpch**3) * np.sqrt(D_sq) * kept
                model = pk.get_cross_noise().to_value(pk.Mpch**3)
                np.testing.assert_allclose(model, expected, rtol=1e-5, atol=1e-6 * np.max(expected))

class TestSkySubtraction(unittest.TestCase):
    # The sky subtraction subtracts the map smoothed with a top-hat of radius
    # s_f perpendicular to the LOS, so it multiplies the map (signal and noise)
    # by 1 - D_sky, D_sky = 2 J_1(k_perp s_f) / (k_perp s_f), before the mask
    # (Eqs. 22-25 of the SIMPLE paper). A footprint of 72 arcmin (s_f = 80
    # Mpc/h) makes D_sky < 0 for k_perp > 0.048 h/Mpc, inside the 8^3 grid.

    def test_factor(self):
        # Without a mask or noise, the sky-subtracted map of a random field is
        # its map times sky_subtraction_factor() in every mode, with the sign.
        pk = power_model(footprint_radius=72 * u.arcmin)
        pk.noise_mesh = None
        field = 1 + 0.1 * np.random.RandomState(3).normal(size=tuple(pk.N_mesh))
        pk.intensity_mesh = field * pk.mean_intensity
        pk._get_prepared_intensity_mesh(sky_subtraction=False)
        pk._get_prepared_intensity_mesh(sky_subtraction=True)
        d = np.array(pk.prepared_intensity_mesh_ft)
        d_sky = np.array(pk.prepared_skysub_intensity_mesh_ft)
        factor = pk.sky_subtraction_factor()
        self.assertLess(np.min(factor), 1.0 - 0.05)  # 1 - D_sky < 1 ...
        self.assertGreater(np.max(factor), 1.05)  # ... and > 1 where D_sky < 0
        np.testing.assert_allclose(d_sky, factor * d, atol=1e-5 * np.max(np.abs(d)))

    def test_noise(self):
        # Pixel noise sigma, Poisson noise a^2 and cross noise c, each the
        # same in every cell, through the sky subtraction and a mask that also
        # changes along the LOS: the model is exact.
        pk = power_model(footprint_radius=72 * u.arcmin)
        pk.obs_mask = los_mask(pk)
        pk.noise_mesh = None
        pk.s_par, pk.s_perp = 40.0, 50.0  # Mpc/h
        V_cell = pk.voxel_volume.to_value(pk.Mpch**3)
        exact = exact_noise(pk, 0.05**2 * V_cell * np.ones(pk.N_mesh), lambda unit: unit,
                            sky_subtraction=True)
        model = (pk.get_intensity_noise(0.05 * pk.mean_intensity, sky_subtraction=True)
                 - pk.get_intensity_noise(0 * pk.mean_intensity, sky_subtraction=True)
                 ).to_value(pk.Mpch**3)
        np.testing.assert_allclose(model, exact, rtol=1e-5, atol=1e-6 * np.max(exact))

        a2 = pk.intensity_shot_noise_per_slice()
        a2_constant = np.mean(a2) * np.ones(len(a2))
        pk.intensity_shot_noise_per_slice = lambda: a2_constant
        D = np.sqrt(pk.D_sq_par_of_k(pk.k_par, pk.s_par) * pk.D_sq_perp_of_k(pk.k_perp, pk.s_perp))
        exact = exact_noise(pk, a2_constant.to_value(pk.Mpch**3)[:, None, None] * np.ones(pk.N_mesh),
                            lambda unit: smooth(pk, unit, D), sky_subtraction=True)
        model = pk.get_intensity_noise(0 * pk.mean_intensity, sky_subtraction=True).to_value(pk.Mpch**3)
        np.testing.assert_allclose(model, exact, rtol=1e-5, atol=1e-6 * np.max(exact))

        c = pk.cross_shot_noise_per_slice()
        c_constant = np.mean(c) * np.ones(len(c))
        pk.cross_shot_noise_per_slice = lambda: c_constant
        c = c_constant.to_value(pk.Mpch**3)[:, None, None]
        exact = exact_noise(pk, c**2 * np.ones(pk.N_mesh), lambda unit: smooth(pk, unit, D),
                            lambda unit: unit / c, sky_subtraction=True)
        model = pk.get_cross_noise(sky_subtraction=True).to_value(pk.Mpch**3)
        np.testing.assert_allclose(model, exact, rtol=1e-5, atol=1e-6 * np.max(np.abs(exact)))

    def test_models(self):
        # Without a mask the window depends only on the LOS position, so the
        # sky-subtracted 3D models are the models without sky subtraction
        # times (1 - D_sky)^2 (intensity) or 1 - D_sky (cross) in every mode.
        # With do_model_shot_noise, the pixel noise S_bar V and the modelled
        # Poisson noise get (1 - D_sky)^2 too.
        pk = power_model(footprint_radius=72 * u.arcmin)
        pk.out_filename = os.path.join(tempfile.mkdtemp(), "model.h5")
        pk.do_model_shot_noise = True
        pk.cross_shot_noise_model = 0 * pk.Mpch**3
        kept = ~pk.removed_modes("intensity")
        factor = pk.sky_subtraction_factor()

        def model_3d(get_model, sky_subtraction, sigma=0.0, P_shot=0.0):
            pk.sigma_noise = sigma * pk.mean_intensity
            pk.im_shot_noise_model = P_shot * pk.Mpch**3
            return np.asarray(get_model(sky_subtraction=sky_subtraction, return_3d=True)[0]).real

        for get_model, power in [(pk.get_intensity_model, 2), (pk.get_cross_model, 1)]:
            P, P_sky = model_3d(get_model, False), model_3d(get_model, True)
            np.testing.assert_allclose(P_sky[kept], (factor**power * P)[kept],
                                       rtol=1e-6, atol=1e-6 * np.max(np.abs(P)))

        noise = (model_3d(pk.get_intensity_model, True, sigma=0.05, P_shot=300.0)
                 - model_3d(pk.get_intensity_model, True))
        D_sq = pk.D_sq_par_of_k(pk.k_par, pk.s_par) * pk.D_sq_perp_of_k(pk.k_perp, pk.s_perp)
        S_bar_V = 0.05**2 * pk.voxel_volume.to_value(pk.Mpch**3)  # sigma^2 V_cell, no weights
        expected = (S_bar_V + 300.0 * D_sq) * factor**2 * kept
        np.testing.assert_allclose(noise, expected, rtol=1e-6, atol=1e-6 * np.max(expected))


class TestTopHatSign(unittest.TestCase):
    # With the top-hat along the LOS, SIMPLE smooths the map with
    # D = sinc(k_par s_par / 2) exp(-k_perp^2 s_perp^2 / 2), which is negative
    # where 2 pi < k_par s_par < 4 pi. The cross power gets this D with its
    # sign, the intensity power D^2. On the 8^3 grid (cells of 64 Mpc/h,
    # k_N = 0.049 h/Mpc), s_par = 200 Mpc/h makes D < 0 for k_par >= 3/4 k_N.

    def setUp(self):
        self.pk = power_model(do_spectral_tophat_smooth=True)
        self.pk.s_par, self.pk.s_perp = 200.0, 50.0  # Mpc/h
        self.D = smoothing_filter(self.pk)
        self.assertLess(np.min(self.D), -0.05)

    def test_damping(self):
        # D_of_k is SIMPLE's filter, and D_sq_par_of_k * D_sq_perp_of_k its square.
        pk = self.pk
        np.testing.assert_allclose(pk.D_of_k(), self.D, atol=1e-6)
        np.testing.assert_allclose(pk.D_sq_par_of_k(pk.k_par, pk.s_par)
                                   * pk.D_sq_perp_of_k(pk.k_perp, pk.s_perp), self.D**2, atol=1e-6)

    def test_cross_noise(self):
        # As TestNoiseThroughWindow.test_cross_noise with a constant c, with
        # the intensity map smoothed by SIMPLE's filter: the model is exact.
        pk = self.pk
        pk.obs_mask = los_mask(pk)
        c = pk.cross_shot_noise_per_slice()
        c_constant = np.mean(c) * np.ones(len(c))
        pk.cross_shot_noise_per_slice = lambda: c_constant
        c = c_constant.to_value(pk.Mpch**3)[:, None, None]
        exact = exact_noise(pk, c**2 * np.ones(pk.N_mesh), lambda unit: smooth(pk, unit, self.D),
                            lambda unit: unit / c)
        model = pk.get_cross_noise().to_value(pk.Mpch**3)
        np.testing.assert_allclose(model, exact, rtol=1e-5, atol=1e-6 * np.max(np.abs(exact)))

    def test_cross_model(self):
        # With windows that are 1 everywhere (no mask, the same mean intensity
        # in every slice), the 3D cross model, signal and modelled Poisson
        # noise, is D times the model without smoothing.
        pk = self.pk
        pk.out_filename = os.path.join(tempfile.mkdtemp(), "model.h5")
        pk.mean_intensity_per_redshift_mesh = np.ones(pk.N_mesh) * pk.mean_intensity
        pk.do_model_shot_noise = True
        pk.cross_shot_noise_model = 100.0 * pk.Mpch**3
        model = np.asarray(pk.get_cross_model(return_3d=True)[0]).real
        pk.s_par, pk.s_perp = 0.0, 0.0
        unsmoothed = np.asarray(pk.get_cross_model(return_3d=True)[0]).real
        np.testing.assert_allclose(model, self.D * unsmoothed, rtol=1e-6,
                                   atol=1e-6 * np.max(np.abs(unsmoothed)))


if __name__ == "__main__":
    unittest.main()
