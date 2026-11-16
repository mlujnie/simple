import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy.integrate import simpson

import simple.simple
from simple.run_lognormal_galaxies import executable, main_lognormal_path
from simple.simple import LognormalIntensityMock
from simple.tools_python import yaml_file_to_dictionary

COMPUTE_PK = os.path.join(str(main_lognormal_path), "eisensteinhubaonu/compute_pk")


class Stop(Exception):
    pass


def params_for_lognormal_galaxies(lim):
    """The parameters that lim.run_lognormal_simulation_cpp() gives
    lognormal_galaxies: the call is stopped at check_dir_im, the first step
    after the parameters are set, so nothing is run or written."""
    captured = {}

    def stop(params):
        captured.update(params)
        raise Stop

    with mock.patch.object(simple.simple, "check_dir_im", stop), contextlib.redirect_stdout(io.StringIO()):
        try:
            lim.run_lognormal_simulation_cpp()
        except Stop:
            pass
    return captured


class TestAsPivot(unittest.TestCase):
    # The input lnAs is ln(10^10 A_s) at k = 0.05/Mpc. lognormal_galaxies'
    # compute_pk takes A_s at k = 0.002/Mpc and computes the primordial
    # spectrum A_s (k / 0.002/Mpc)^(n_s - 1).

    def test_spectrum_at_the_input_pivot(self):
        # With the lnAs that SIMPLE gives compute_pk, compute_pk's primordial
        # spectrum at k = 0.05/Mpc is the input A_s.
        input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
        lim = LognormalIntensityMock(input_dict)
        self.assertEqual((lim.lnAs, lim.n_s), (3.094, 0.9645))
        As_compute_pk = np.exp(lim.get_lnAs_for_compute_pk()) * 1e-10
        spectrum_at_pivot = As_compute_pk * (0.05 / 0.002)**(lim.n_s - 1)
        np.testing.assert_allclose(spectrum_at_pivot, np.exp(3.094) * 1e-10, rtol=1e-13)
        # for n_s < 1 the spectrum is larger at the smaller k: by 1 / 0.892
        # for n_s = 0.9645 (0.892 = 25^-0.0355)
        np.testing.assert_allclose(As_compute_pk / (np.exp(3.094) * 1e-10), 1 / 0.8920170, rtol=1e-6)

    def test_As_given_to_compute_pk(self):
        # The A_s and lnAs that run_lognormal_simulation_cpp hands to
        # lognormal_galaxies are the converted ones: A_s at 0.002/Mpc is the
        # input A_s (at 0.05/Mpc) times (0.002 / 0.05)^(n_s - 1), written out here.
        input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
        params = params_for_lognormal_galaxies(LognormalIntensityMock(input_dict))
        expected_As = np.exp(3.094) * 1e-10 * (0.002 / 0.05)**(0.9645 - 1)
        np.testing.assert_allclose(params["As"], expected_As, rtol=1e-13)
        np.testing.assert_allclose(params["lnAs"], np.log(expected_As * 1e10), rtol=1e-13)
        self.assertEqual(params["ns"], 0.9645)

    @unittest.skipUnless(os.path.exists(COMPUTE_PK), "lognormal_galaxies' compute_pk is not installed")
    def test_sigma8_of_compute_pk(self):
        # With astropy's Planck18 cosmology and n_s = 0.9665,
        # ln(10^10 A_s) = 3.047 (A_s at 0.05/Mpc), compute_pk's P(k) at z = 0
        # has sigma8 within 1 % of 0.8102. The three numbers are those of
        # Planck Collaboration 2020, A&A 641, A6 (arXiv:1807.06209), Table 2,
        # column TT,TE,EE+lowE+lensing+BAO, which astropy names as the source
        # of Planck18. With lnAs given unchanged to compute_pk, sigma8 is
        # 0.7705, 4.9 % too low (measured with compute_pk of lognormal_galaxies
        # bb7334c by the reviewer of SIMPLE #25, who also measured 0.8132
        # with the conversion).
        input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
        input_dict.update(cosmology="Planck18", n_s=0.9665, lnAs=3.047)
        params = params_for_lognormal_galaxies(LognormalIntensityMock(input_dict))
        with tempfile.TemporaryDirectory() as folder:
            # compute_pk as gen_inputs calls it, but at z = 0 and into a temporary folder
            params.update(z=0.0, ofile_eh=os.path.join(folder, "test"))
            args = ["ofile_eh", "om0", "ode0", "ob0", "h0", "w", "ns", "run", "As", "mnu", "z"]
            executable("exe").run(COMPUTE_PK, args, params)
            k, P = np.loadtxt(os.path.join(folder, "test_pk.txt"), unpack=True)  # h/Mpc, (Mpc/h)^3
        # sigma8^2 = int dlnk k^3 P(k) W(8 k)^2 / (2 pi^2), W the Fourier transform of a top hat
        x = 8 * k
        W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
        sigma8 = np.sqrt(simpson(k**3 * P * W**2 / (2 * np.pi**2), x=np.log(k)))
        np.testing.assert_allclose(sigma8, 0.8102, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
