import os
import tempfile
import unittest
import warnings

import numpy as np


class TestModelKperpZero(unittest.TestCase):
    """The estimators subtract the mean of each slice along the line of sight,
    so the data have no power in modes with k_perp = 0. The model must not
    have any either."""

    def test_no_signal_in_kperp_zero_modes(self):
        from simple.pk_3d_model import Power_Spectrum_Model

        with tempfile.TemporaryDirectory() as tmp:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pk = Power_Spectrum_Model(
                    "./tests/test_lim_input.yaml", do_model_shot_noise=False,
                    out_filename=os.path.join(tmp, "model.h5"))
            pk.get_kspec()
            model = pk.get_3d_pk_model(
                damping_function=1.0,
                P_shot_smoothed=0.0 * pk.Mpch**3,
                S_bar=0.0,
                mask_window_function_1=np.ones(pk.N_mesh),
                observed_volume=pk.box_volume,
                box_volume=pk.box_volume,
                Pk_unit=pk.Mpch**3,
                tracer="n_gal",
                return_3d=True,
            )[0]

        k_perp_zero = np.asarray(pk.k_perp) == 0
        self.assertEqual(np.sum(k_perp_zero), pk.N_mesh[0])
        # zero up to the round-off of the FFTs in the window convolution
        atol = 1e-6 * np.max(model.real)
        np.testing.assert_allclose(model.real[k_perp_zero], 0.0, atol=atol)
        np.testing.assert_allclose(
            model.real[~k_perp_zero],
            pk.bias**2 * pk.Plin(np.asarray(pk.kspec)[~k_perp_zero]),
            rtol=1e-6, atol=atol)


if __name__ == "__main__":
    unittest.main()
