import unittest

import astropy.units as u
import numpy as np

from simple.pk_3d_model import Power_Spectrum_Model
from simple.simple import make_map
from simple.tools_python import yaml_file_to_dictionary


def power_model(N_mesh=8):
    """A Power_Spectrum_Model of the test config on a small grid, with a
    pixel noise of 0.05 times the mean intensity and the k grid."""
    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
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


def smooth(pk, field, D):
    """field smoothed with the damping D (not squared) on the k grid."""
    return np.array(make_map(np.array(make_map(field, Nmesh=pk.N_mesh,
                                               BoxSize=pk.box_size.to_value(pk.Mpch)).r2c()) * D,
                             Nmesh=pk.N_mesh, BoxSize=pk.box_size.to_value(pk.Mpch),
                             type="complex").c2r())


def exact_noise(pk, noise_power, intensity_response, galaxy_response=None):
    """
    The expected V Re[d_1(k) d_2(k)^*] of a noise that is independent between
    cells, with the power noise_power (in (Mpc/h)^3) in each cell: the sum
    over the cells of the response of the maps to a noise in that cell alone.
    d_1 is SIMPLE's prepared intensity map (_get_prepared_intensity_mesh) with
    the intensity mesh intensity_response(noise) times the mean intensity;
    d_2 is d_1, or galaxy_response(noise) times the mask, transformed as
    SIMPLE does. The response is computed for a noise of 1 and scaled, so
    that the single-precision FFTs work with numbers of order 1.
    """
    V = pk.box_size.to_value(pk.Mpch).prod()
    V_cell = V / np.prod(pk.N_mesh)
    total = 0
    for i in range(np.prod(pk.N_mesh)):
        unit = np.zeros(np.prod(pk.N_mesh))
        unit[i] = 1.0
        unit = unit.reshape(pk.N_mesh)
        pk.intensity_mesh = intensity_response(unit) * pk.mean_intensity
        pk.prepared_intensity_mesh_ft = None
        pk._get_prepared_intensity_mesh(sky_subtraction=False)
        d_1 = np.array(pk.prepared_intensity_mesh_ft)
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
        rng = np.random.RandomState(2)
        pk.obs_mask = sky_mask(pk) * rng.uniform(0.5, 1.5, (pk.N_mesh[0], 1, 1))
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
        # Poisson noise with a power a^2(x_par) per slice, smoothed along and
        # across the LOS like the signal, with a mask that is the same in
        # every slice: the model is exact.
        pk = power_model()
        pk.obs_mask = sky_mask(pk)
        pk.s_par, pk.s_perp = 40.0, 50.0  # Mpc/h, comparable to the cells
        a2 = pk.intensity_shot_noise_per_slice().to_value(pk.Mpch**3)
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
        # does neither; c(x_par) is the covariance of the two per cell.
        pk = power_model()
        pk.obs_mask = sky_mask(pk)
        pk.s_par, pk.s_perp = 40.0, 50.0  # Mpc/h
        c = pk.cross_shot_noise_per_slice().to_value(pk.Mpch**3)[:, None, None]
        D = np.sqrt(pk.D_sq_par_of_k(pk.k_par, pk.s_par) * pk.D_sq_perp_of_k(pk.k_perp, pk.s_perp))
        # a noise of power c^2 in the intensity map and of power 1 in the galaxy
        # map (the same noise divided by c): their covariance is c
        exact = exact_noise(pk, c**2 * np.ones(pk.N_mesh), lambda unit: smooth(pk, unit, D),
                            lambda unit: unit / c)
        model = pk.get_cross_noise().to_value(pk.Mpch**3)
        np.testing.assert_allclose(model, exact, rtol=1e-5, atol=1e-6 * np.max(np.abs(exact)))
        # adding the noise after the window, as before, is off by more than 10%
        kept = np.asarray(pk.k_perp) > 0
        old = pk.get_cross_shot_noise().to_value(pk.Mpch**3) * D * kept
        self.assertGreater(np.max(np.abs(old - exact)) / np.max(np.abs(exact)), 0.1)

    def test_no_mask(self):
        # Without a mask the noise is the same as before: P_shot D^2 + S_bar V
        # for intensity and P_shot D for cross, 0 at k_perp = 0.
        pk = power_model(N_mesh=16)
        kept = np.asarray(pk.k_perp) > 0
        D_sq = pk.D_sq_par_of_k(pk.k_par, pk.s_par) * pk.D_sq_perp_of_k(pk.k_perp, pk.s_perp)
        S_bar_V = (pk.get_S_bar(pk.weight_mesh_im * pk.obs_mask, pk.sigma_noise)
                   * pk.box_volume).to_value(pk.Mpch**3)
        expected = (pk.get_intensity_shot_noise().to_value(pk.Mpch**3) * D_sq + S_bar_V) * kept
        model = pk.get_intensity_noise(pk.sigma_noise).to_value(pk.Mpch**3)
        np.testing.assert_allclose(model, expected, rtol=1e-5, atol=1e-6 * np.max(expected))
        expected = pk.get_cross_shot_noise().to_value(pk.Mpch**3) * np.sqrt(D_sq) * kept
        model = pk.get_cross_noise().to_value(pk.Mpch**3)
        np.testing.assert_allclose(model, expected, rtol=1e-5, atol=1e-6 * np.max(expected))


if __name__ == "__main__":
    unittest.main()
