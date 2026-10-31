import os
import tempfile
import unittest

import astropy.units as u
import numpy as np
from scipy.integrate import quad

from simple.pk_3d_model import Power_Spectrum_Model
from simple.tools_python import yaml_file_to_dictionary

FLUX = u.erg / u.s / u.cm**2
# p(r) of tests/input/selection_function_example.csv: a smoothstep in ln r,
# 3t^2 - 2t^3 with t = ln(r / 0.5) / ln 4, from 0 at r = 0.5 through 1/2 at
# r = 1 to 1 at r = 2 (41 rows; the table of simple-jax's realspace_probabilistic)
SELECTION_FUNCTION = "./tests/input/selection_function_example.csv"


def model(min_flux_mesh=None, selection_function=None, N_mesh=16, **changes):
    """A Power_Spectrum_Model of tests/test_lim_input.yaml with the given
    min_flux (a function of the cell indices (i, j, k), in erg/s/cm^2) and
    selection_function table."""
    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
    # growth rate table, normally written by lognormal_galaxies when it runs
    input_dict["f_growth_filename"] = "./tests/input/example_fnu.txt"
    input_dict["N_mesh"] = [N_mesh] * 3
    if selection_function is not None:
        input_dict["selection_function"] = selection_function
    input_dict.update(changes)
    pk = Power_Spectrum_Model(input_dict, do_model_shot_noise=False)
    if min_flux_mesh is not None:
        pk.min_flux = min_flux_mesh(*np.indices(pk.N_mesh)) * FLUX
    return pk


def table_mesh(i, j, k):
    """The flux limit of the test's min_flux table at each slice centre, the
    same in every cell of the slice."""
    pk = model()
    return pk.min_flux(pk.redshift_mesh_axis).to_value(FLUX)[i]


class TestHardLimitMesh(unittest.TestCase):
    # Without a selection_function, a mesh that holds the table's flux limit at
    # each slice centre selects the same galaxies as the table (at the slice
    # centres), so the model's expected means and Poisson noise per slice must
    # be the table's.

    def test_matches_table(self):
        table = model()
        mesh = model(min_flux_mesh=table_mesh)
        z = table.redshift_mesh_axis
        for selection in ["detected", "undetected"]:
            for tracer in ["intensity", "n_gal"]:
                np.testing.assert_allclose(
                    mesh.mean_intensity_per_redshift(z, selection, tracer),
                    table.mean_intensity_per_redshift(z, selection, tracer), rtol=1e-5)
            mesh.galaxy_selection["intensity"] = table.galaxy_selection["intensity"] = selection
            np.testing.assert_allclose(mesh.intensity_shot_noise_per_slice(),
                                       table.intensity_shot_noise_per_slice(), rtol=1e-5)


class TestSelectedLuminosityMoment(unittest.TestCase):
    # With a selection_function, the moment is the integral of
    # L^m dn/dL p(L / L_limit), here computed directly with quad.

    def test_matches_quad(self):
        # the smoothstep table, and one that jumps from p = 0 to 0.2 at r = 0.5
        # and from 0.8 to 1 at r = 2
        jumps = os.path.join(tempfile.mkdtemp(), "jumps.csv")
        with open(jumps, "w") as f:
            f.write("flux_ratio,p\n0.5,0.2\n1.0,0.3\n2.0,0.8\n")
        for table in [SELECTION_FUNCTION, jumps]:
            self.check_quad(table)

    def check_quad(self, selection_function):
        pk = model(selection_function=selection_function)
        p = pk.detection_probability
        L_limit = 0.5  # in luminosity_unit (1e42 erg/s); about the test table's limits
        L_min = pk.Lmin.to_value(pk.luminosity_unit)
        L_max = pk.Lmax.to_value(pk.luminosity_unit)
        for m in [0, 1, 2]:
            def integral(L_low, L_high, p_of_r):
                """quad of L^m dn/dL p(L / L_limit) over ln L, in pieces of
                0.1 dex (quad misses the function over many decades), with
                the table's nodes as pieces in the ramp of p."""
                edges = np.unique(np.r_[np.geomspace(L_low, L_high, 2 + int(10 * np.log10(L_high / L_low))),
                                        np.clip(L_limit * p.x, L_low, L_high)])
                f = lambda ln_L: np.exp((m + 1) * ln_L) * pk.luminosity_function(np.exp(ln_L)) * p_of_r(np.exp(ln_L) / L_limit)
                return sum(quad(f, a, b)[0] for a, b in zip(np.log(edges[:-1]), np.log(edges[1:])))
            # p rises from L = 0.5 L_limit to 2 L_limit and is 1 above it
            detected = integral(0.5 * L_limit, 2 * L_limit, p) + integral(2 * L_limit, L_max, lambda r: 1.0)
            all_galaxies = integral(L_min, L_max, lambda r: 1.0)
            self.assertTrue(0.05 < detected / all_galaxies < 0.95)
            np.testing.assert_allclose(
                pk.mean_selected_luminosity_moment(L_limit, "detected", m), detected, rtol=1e-5)
            np.testing.assert_allclose(
                pk.mean_selected_luminosity_moment(L_limit, "undetected", m),
                all_galaxies - detected, rtol=1e-5)

    def test_limits_outside_luminosity_range(self):
        # no galaxy is drawn outside [Lmin, Lmax]: a limit far below Lmin
        # detects all galaxies, one far above Lmax none
        pk = model(selection_function=SELECTION_FUNCTION)
        L_min = pk.Lmin.to_value(pk.luminosity_unit)
        L_max = pk.Lmax.to_value(pk.luminosity_unit)
        for m in [0, 1, 2]:
            all_galaxies = pk.mean_selected_luminosity_moment(1e-3 * L_min, "detected", m)
            # 0 up to round-off
            self.assertLess(abs(pk.mean_selected_luminosity_moment(1e-3 * L_min, "undetected", m)),
                            1e-12 * all_galaxies)
            self.assertLess(abs(pk.mean_selected_luminosity_moment(1e3 * L_max, "detected", m)),
                            1e-12 * all_galaxies)

    def test_mean_over_cells(self):
        # a quarter of the cells with one limit and the rest with another
        pk = model(selection_function=SELECTION_FUNCTION)
        L_limits = np.r_[np.ones(4), 3 * np.ones(12)]
        for selection in ["detected", "undetected"]:
            one = pk.mean_selected_luminosity_moment(1.0, selection, 1)
            three = pk.mean_selected_luminosity_moment(3.0, selection, 1)
            np.testing.assert_allclose(
                pk.mean_selected_luminosity_moment(L_limits, selection, 1),
                0.25 * one + 0.75 * three, rtol=1e-12)


class TestMockWithMesh(unittest.TestCase):
    # SIMPLE's own probabilistic selection on a uniform catalog: the number of
    # detected galaxies, the mean intensity and the intensity Poisson noise must
    # be what the model expects. The mesh changes across the sky (a quarter of
    # the cells has half the flux limit), so the cell lookup and the average
    # over the cells are checked too.

    def test_counts_mean_and_noise(self):
        def min_flux(i, j, k):
            return table_mesh(i, j, k) * np.where(j < 8, 0.5, 1.0)
        pk = model(min_flux_mesh=min_flux, selection_function=SELECTION_FUNCTION,
                   galaxy_selection={"intensity": "detected", "n_gal": "detected"},
                   do_spectral_smooth=False,  # so the noise is flat in k
                   seed_lognormal=1)  # seeds the luminosity draw
        pk.noise_mesh = None
        np.random.seed(2)  # SIMPLE's selection uses the global random state
        N = pk.N_gal
        box = pk.box_size.to_value(pk.Mpch)
        pk.cat = {
            "Position": np.random.default_rng(1).uniform(0, box, size=(N, 3)) * pk.Mpch,
            "Velocity": np.zeros((N, 3)) * u.km / u.s,
            "RSD_redshift_factor": np.ones(N),
        }
        pk.assign_redshift_along_axis()
        pk.assign_luminosity()
        pk.assign_flux()
        pk.apply_selection_function()

        # detected galaxies: N_det is Poisson-like (binomial) around the
        # expected sum over the slices of n_bar V_slice
        n_bar = pk.mean_intensity_per_redshift(
            pk.redshift_mesh_axis, "detected", "n_gal").to_value(u.Mpc**-3)
        expected = np.sum(n_bar) * (pk.box_volume / pk.N_mesh[0]).to_value(u.Mpc**3)
        N_det = np.sum(pk.cat["detected"])
        self.assertTrue(0.1 < N_det / N < 0.9)
        self.assertLess(abs(N_det - expected), 5 * np.sqrt(expected))

        # mean intensity of the painted map; its relative error is about that
        # of the sum of the detected luminosities, sqrt(sum L^2) / sum L
        pk.paint_intensity_mesh()
        measured = np.mean(pk.intensity_mesh).to_value(pk.mean_intensity.unit)
        L = pk.cat["luminosity"][pk.cat["detected"]].value
        error = np.sqrt(np.sum(L**2)) / np.sum(L)
        self.assertLess(abs(measured / pk.mean_intensity.value - 1), 5 * error)

        # intensity power at k_perp != 0, which for a uniform catalog is only
        # the Poisson noise, against the model's
        pk._get_prepared_intensity_mesh(sky_subtraction=False)
        V = pk.box_volume.to_value(pk.Mpch**3)
        power = V * np.abs(np.array(pk.prepared_intensity_mesh_ft))**2
        pk.get_kspec()
        power = power[np.asarray(pk.k_perp) > 0]
        # the map is divided by its measured box mean, the model by the expected one
        noise = pk.get_intensity_shot_noise().to_value(pk.Mpch**3) * (pk.mean_intensity.value / measured)**2
        self.assertLess(abs(np.mean(power) - noise), 5 * np.std(power) / np.sqrt(power.size))


class TestModelRuns(unittest.TestCase):
    # pk_3d_model stopped for any min_flux mesh: the slice means came from the
    # painted meshes, which the model does not have, and model_shot_noise could
    # not look up the cells of its catalog.

    def test_get_model(self):
        # A mesh holding the table's flux limits gives the table's intensity
        # and cross models (the n_gal model differs: with a mesh it also loses
        # k_perp = 0). With the selection_function, the models are finite.
        def models(**kwargs):
            pk = model(galaxy_selection={"intensity": "detected", "n_gal": "detected"}, **kwargs)
            pk.sigma_noise = 0 * pk.mean_intensity  # no pixel noise, so no noise_mesh is needed
            pk.out_filename = os.path.join(tempfile.mkdtemp(), "model.h5")
            pk.get_kspec()
            return {tracer: np.array(pk.get_model(tracer)[1:3]) for tracer in ["intensity", "n_gal", "cross"]}
        table = models()
        hard = models(min_flux_mesh=table_mesh)
        probabilistic = models(min_flux_mesh=table_mesh, selection_function=SELECTION_FUNCTION)
        for tracer in ["intensity", "n_gal", "cross"]:
            filled = np.isfinite(table[tracer])  # bins with modes
            self.assertTrue(np.all(np.isfinite(probabilistic[tracer][filled])))
            if tracer != "n_gal":
                np.testing.assert_allclose(hard[tracer][filled], table[tracer][filled],
                                           rtol=1e-5, atol=1e-5 * np.max(np.abs(table[tracer][filled])))

    def test_model_shot_noise(self):
        pk = model(min_flux_mesh=table_mesh, selection_function=SELECTION_FUNCTION,
                   galaxy_selection={"intensity": "detected", "n_gal": "detected"})
        pk.model_shot_noise(N_real=1)
        self.assertTrue(np.all(np.isfinite(pk.ngal_shot_noise_model)))


if __name__ == "__main__":
    unittest.main()
