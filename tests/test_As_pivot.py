import unittest

import numpy as np

from simple.simple import LognormalIntensityMock
from simple.tools_python import yaml_file_to_dictionary


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


if __name__ == "__main__":
    unittest.main()
