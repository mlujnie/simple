import os
import tempfile
import unittest

import numpy as np
import astropy.units as u

from simple.simple import LognormalIntensityMock
from simple.tools_python import yaml_file_to_dictionary


def mock_with_k_inputs(**k_inputs):
    """Mock from the test config, with only the given k-bin inputs."""
    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
    for key in ["kmin", "kmax", "nkbin"]:
        del input_dict[key]
    input_dict.update(k_inputs)
    return LognormalIntensityMock(input_dict)


class TestKBins(unittest.TestCase):

    def test_nkbin_sets_the_bins(self):
        # 4 bins from 0 to 0.08 h/Mpc are 0.02 h/Mpc wide
        lim = mock_with_k_inputs(kmin=0.0, kmax=0.08, nkbin=4)
        self.assertEqual(lim.prepare_k_values(), (0.0, 0.08, 0.02))

        # the mean k of each bin lies inside that bin
        lim.get_kspec()
        mean_k, monopole, quadrupole = lim.bin_scipy(np.ones(np.shape(lim.kspec)))
        edges = np.linspace(0.0, 0.08, 5)
        self.assertEqual(len(mean_k), 4)
        self.assertTrue(np.all((mean_k > edges[:-1]) & (mean_k < edges[1:])))

    def test_default_bins_are_one_fundamental_mode_wide(self):
        # from 0 to the Nyquist k in steps of 2 pi / L gives N_mesh / 2 bins
        lim = mock_with_k_inputs()
        self.assertEqual(lim.nkbin, lim.N_mesh[0] // 2)

    def test_old_dk_input_keeps_the_old_bins(self):
        # earlier versions made floor((kmax - kmin) / dk) - 1 bins
        lim = mock_with_k_inputs(kmin=0.04, kmax=1.0, dk=0.04)
        self.assertEqual(lim.nkbin, 23)

    def test_nkbin_survives_save_and_load(self):
        lim = mock_with_k_inputs(kmin=0.04, kmax=1.0, nkbin=24)

        # save_to_file needs a catalog: a few galaxies placed uniformly
        N = 1000
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
            loaded = LognormalIntensityMock.from_file(filename, only_params=True)
        self.assertEqual(loaded.prepare_k_values(), (0.04, 1.0, 0.04))


if __name__ == "__main__":
    unittest.main()
