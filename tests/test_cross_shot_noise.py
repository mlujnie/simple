import unittest

import numpy as np
import astropy.units as u

from simple.pk_3d_model import Power_Spectrum_Model
from simple.tools_python import yaml_file_to_dictionary


class TestCrossShotNoise(unittest.TestCase):

    def test_all_galaxies_detected_intensity(self):
        # The galaxy map holds all galaxies and is divided by their mean
        # density n_all; the intensity map holds only the detected galaxies.
        # The cross Poisson noise comes from the galaxies in both maps, so it is
        # I_det / (I_bar * n_all), not I_det / (I_bar * n_det).
        input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
        # growth rate table, normally written by lognormal_galaxies when it runs
        input_dict["f_growth_filename"] = "./tests/input/example_fnu.txt"
        input_dict["galaxy_selection"] = {"intensity": "detected", "n_gal": "all"}
        input_dict["do_spectral_smooth"] = False  # so the noise is flat in k
        input_dict["N_mesh"] = [32, 32, 32]
        input_dict["seed_lognormal"] = 1  # seeds the luminosity draw
        pk = Power_Spectrum_Model(input_dict, do_model_shot_noise=False)
        pk.noise_mesh = None

        # uniform catalog without velocities (so lognormal_galaxies is not
        # needed): its cross power is only the Poisson noise
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
        pk.paint_intensity_mesh()
        pk.paint_galaxy_mesh()

        # cross power of each mode, V Re[d_I d_g^*], as in Pk_multipoles, at
        # k_perp != 0 (the intensity estimator removes k_perp = 0)
        pk._get_prepared_intensity_mesh(sky_subtraction=False)
        pk._get_prepared_n_gal_mesh()
        V = pk.box_volume.to_value(pk.Mpch**3)
        cross = V * np.real(np.array(pk.prepared_intensity_mesh_ft)
                            * np.conj(np.array(pk.prepared_n_gal_mesh_ft)))
        pk.get_kspec()
        cross = cross[np.asarray(pk.k_perp) > 0]
        measured = np.mean(cross)
        error = np.std(cross) / np.sqrt(cross.size)  # about 1.4% of the noise

        # without a mask the model is flat in k: the mean over the slices of
        # cross_shot_noise_per_slice
        model = pk.get_cross_shot_noise().to_value(pk.Mpch**3)
        self.assertLess(abs(measured - model), 5 * error)
        # dividing by the density of detected galaxies, as before, is about
        # twice as large here (half of the galaxies are detected)
        pk.galaxy_selection["n_gal"] = "detected"
        old = pk.get_cross_shot_noise().to_value(pk.Mpch**3)
        self.assertGreater(abs(measured - old), 5 * error)


if __name__ == "__main__":
    unittest.main()
