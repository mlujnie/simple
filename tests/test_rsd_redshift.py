import unittest

import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d


def make_mock_with_moving_galaxies(single_redshift, seed=42):
    """Small mock with uniformly placed galaxies that move along the line of
    sight, without running lognormal_galaxies."""
    from simple.simple import LognormalIntensityMock
    from simple.tools_python import yaml_file_to_dictionary

    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
    input_dict["single_redshift"] = single_redshift
    input_dict["verbose"] = False
    lim = LognormalIntensityMock(input_dict)

    rng = np.random.default_rng(seed)
    N_gal = 10000
    box = lim.box_size.to_value(lim.Mpch)
    velocity = rng.normal(0, 100, size=(N_gal, 3)) * u.km / u.s
    lim.cat = {
        "Position": rng.uniform(0, box, size=(N_gal, 3)) * lim.Mpch,
        "Velocity": velocity,
        # as in load_lognormal_catalog_cpp
        "RSD_redshift_factor": (1 + velocity.dot(lim.LOS) / const.c).to(1),
    }
    return lim


class TestRSDRedshift(unittest.TestCase):
    """The observed redshift of a moving galaxy is
    1 + z_obs = (1 + z_cosmo) * (1 + v_LOS / c)."""

    def test_along_axis_matches_RSD_Position(self):
        """RSD_redshift must be the redshift at the comoving distance of
        RSD_Position, which is shifted by (1 + z) v_LOS / H(z)."""
        lim = make_mock_with_moving_galaxies(single_redshift=False)
        lim.assign_redshift_along_axis()

        axis = np.where(lim.LOS == 1)[0][0]
        zs = np.linspace(lim.redshift - 0.5, lim.redshift + 0.5, 10000)
        z_at_distance = interp1d(
            lim.astropy_cosmo.comoving_distance(zs).to_value(u.Mpc), zs)

        def z_of(position):
            return z_at_distance(
                (lim.minimum_distance + position[:, axis]).to_value(u.Mpc))

        shift_from_position = (z_of(lim.cat["RSD_Position"])
                               - z_of(lim.cat["Position"]))
        shift = (np.asarray(lim.cat["RSD_redshift"])
                 - np.asarray(lim.cat["cosmo_redshift"]))
        # the velocities shift the redshifts by ~(1 + z) 100 km/s / c ~ 1e-3
        self.assertGreater(np.std(shift), 5e-4)
        # equal to first order in v / c; z * v / c instead of (1 + z) * v / c
        # would be off by v / c ~ 3e-4
        np.testing.assert_allclose(shift, shift_from_position, atol=2e-5)

    def test_single_redshift(self):
        lim = make_mock_with_moving_galaxies(single_redshift=True)
        lim.assign_single_redshift()

        v_LOS = lim.cat["Velocity"].dot(lim.LOS)
        expected = (1 + lim.redshift) * (1 + (v_LOS / const.c).to_value(1)) - 1
        np.testing.assert_allclose(
            np.asarray(lim.cat["RSD_redshift"]), expected, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
