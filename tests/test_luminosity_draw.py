import unittest

import numpy as np
from astropy.table import Table

from simple.simple import LognormalIntensityMock


class TestLuminosityDraw(unittest.TestCase):

    def test_luminosities_between_Lmin_and_Lmax(self):
        # The luminosity function is only defined from Lmin to Lmax. Drawing
        # luminosities below Lmin lowers the mean luminosity (by 0.08% with
        # the luminosity function of the validation config 1).
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        positions = np.random.default_rng(42).uniform(0, 100, size=(100000, 3))
        lim.cat = Table()
        lim.cat["Position"] = positions * lim.Mpch
        lim.assign_luminosity()
        L = lim.cat["luminosity"].quantity.to_value(lim.luminosity_unit)
        self.assertGreaterEqual(L.min(), lim.Lmin.to_value(lim.luminosity_unit))
        self.assertLessEqual(L.max(), lim.Lmax.to_value(lim.luminosity_unit))


if __name__ == "__main__":
    unittest.main()
