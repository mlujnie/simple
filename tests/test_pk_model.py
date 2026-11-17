import os
import tempfile
import unittest

import astropy.units as u
import numpy as np

from simple.pk_3d_model import Power_Spectrum_Model, rsd_cell_factor
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
        # With a min_flux mesh too the galaxy field is divided by the expected
        # mean density, so it loses only k = 0.
        pk, model, expected = model_3d("n_gal", min_flux_mesh=True)
        self.assert_model(model, expected, np.asarray(pk.kspec) == 0)


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

    def test_min_flux_mesh(self):
        # With a min_flux mesh the galaxy field is divided by the expected mean
        # density of each slice too: a mesh holding the table's flux limit at
        # each slice centre gives the table's n_bar, and a wave along the line
        # of sight survives.
        table = LognormalIntensityMock("./tests/test_lim_input.yaml")
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        z = lim.redshift_mesh_axis
        lim.min_flux = table.min_flux(z)[:, None, None] * np.ones(lim.N_mesh)
        expected = table.mean_intensity_per_redshift(z, lim.galaxy_selection["n_gal"], "n_gal")
        np.testing.assert_allclose(lim.mean_ngal_per_redshift_mesh[:, 0, 0], expected, rtol=1e-5)
        n = lim.N_mesh[0]
        delta = 0.1 * np.cos(2 * np.pi * np.arange(n) / n)[:, None, None] * np.ones(lim.N_mesh)
        lim.n_gal_mesh = lim.mean_ngal_per_redshift_mesh * (1 + delta)
        lim._get_prepared_n_gal_mesh()
        np.testing.assert_allclose(np.asarray(lim.prepared_n_gal_mesh_ft.c2r()), delta, atol=1e-6)


class TestRSDCellFactor(unittest.TestCase):
    # In redshift space the model is the Kaiser formula with the velocity
    # term f mu^2 times sinc(k_par * cell), cell the voxel length along the
    # line of sight: lognormal_galaxies gives every galaxy of a cell the
    # velocity of that cell, and the galaxies are painted in the nearest cell
    # of the same grid (see rsd_cell_factor in pk_3d_model.py).

    def inputs(self, **changes):
        input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
        input_dict["f_growth_filename"] = "./tests/input/example_fnu.txt"
        input_dict["RSD"] = True
        input_dict.update(changes)
        return input_dict

    def model(self, input_changes={}, **kwargs):
        pk = Power_Spectrum_Model(self.inputs(**input_changes), do_model_shot_noise=False, **kwargs)
        pk.get_kspec()
        return pk

    def expected(self, pk, cell):
        """(b + f mu^2 sinc(k_par cell))^2 P_lin on the model's k grid; the
        Kaiser formula itself for cell = 0."""
        k, mu, k_par = [np.asarray(x) for x in (pk.kspec, pk.muspec, pk.k_par)]
        x = k_par * cell
        sinc = np.where(x > 0, np.sin(x) / np.where(x > 0, x, 1.0), 1.0)
        return (pk.bias + pk.f_growth(k) * mu**2 * sinc)**2 * pk.Plin(k)

    def test_factor(self):
        # sinc(x) = sin(x) / x: 1 at k_par = 0 and 0 at the Nyquist frequency pi / cell
        k_par = np.array([0.0, 0.1, 0.2, np.pi / 8.0])
        expected = [1.0, np.sin(0.8) / 0.8, np.sin(1.6) / 1.6, 0.0]
        np.testing.assert_allclose(rsd_cell_factor(k_par, 8.0), expected, rtol=1e-14, atol=1e-16)

    def test_model_has_the_factor(self):
        pk = self.model()
        expected = self.expected(pk, 32.0)  # box 512 Mpc/h, 16 voxels
        np.testing.assert_allclose(pk.Pm_kspec.to(pk.Mpch**3).value, expected, rtol=1e-12)
        # the factor changes the model: by more than 10 % in some mode
        self.assertGreater(np.max(np.abs(expected / self.expected(pk, 0.0) - 1)), 0.1)

    def test_kaiser_without_the_factor(self):
        pk = self.model(rsd_cell_factor=False)
        np.testing.assert_allclose(pk.Pm_kspec.to(pk.Mpch**3).value, self.expected(pk, 0.0), rtol=1e-12)

    def test_cell_along_the_line_of_sight(self):
        # voxels of 32 x 64 x 64 Mpc/h (box 512 Mpc/h): the factor has the
        # voxel length along the line of sight (axis 0), 32 Mpc/h, and the
        # model warns that the factor is not derived for such voxels
        pk = self.model({"N_mesh": [16, 8, 8]})  # before assertLogs: the class resets the logging
        with self.assertLogs(level="WARNING") as logs:
            Pm = pk.Pm_kspec.to(pk.Mpch**3).value
        self.assertTrue(any("cubic" in message for message in logs.output))
        np.testing.assert_allclose(Pm, self.expected(pk, 32.0), rtol=1e-12)
        self.assertGreater(np.max(np.abs(Pm / self.expected(pk, 64.0) - 1)), 0.1)

    def test_from_file(self):
        # a model loaded from a saved mock has the factor by default, and
        # the Kaiser formula itself with rsd_cell_factor=False
        lim = LognormalIntensityMock(self.inputs())
        N = 1000  # save_to_file needs a catalog: a few galaxies placed uniformly
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
        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = os.path.join(tmp_dir, "mock.h5")
            lim.save_to_file(filename)
            for kwargs, cell in (({}, 32.0), ({"rsd_cell_factor": False}, 0.0)):
                pk = Power_Spectrum_Model.from_file(filename, os.path.join(tmp_dir, "model.h5"),
                                                    only_params=True, **kwargs)
                pk.get_kspec()
                # 1e-5: the file has the box size with 6 digits (the two cases differ by more than 10 %)
                np.testing.assert_allclose(pk.Pm_kspec.to(pk.Mpch**3).value, self.expected(pk, cell), rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
