import unittest

import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d

from simple.simple import LognormalIntensityMock
from simple.tools_python import yaml_file_to_dictionary


def mock_with_moving_galaxies(single_redshift):
    """Small mock with 10000 galaxies placed uniformly in the box and moving
    with random velocities of 100 km/s per axis (so lognormal_galaxies is not
    needed)."""
    input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
    input_dict["single_redshift"] = single_redshift
    lim = LognormalIntensityMock(input_dict)

    rng = np.random.default_rng(42)
    N = 10000
    box = lim.box_size.to_value(lim.Mpch)
    velocity = rng.normal(0, 100, size=(N, 3)) * u.km / u.s
    lim.cat = {
        "Position": rng.uniform(0, box, size=(N, 3)) * lim.Mpch,
        "Velocity": velocity,
        # 1 + v_LOS / c, as in load_lognormal_catalog_cpp
        "RSD_redshift_factor": (1 + velocity.dot(lim.LOS) / const.c).to(1),
    }
    return lim


class TestRSDRedshift(unittest.TestCase):
    # The observed redshift of a moving galaxy is
    # 1 + z_obs = (1 + z_cosmo) * (1 + v_LOS / c).

    def test_along_axis_matches_RSD_Position(self):
        # RSD_redshift must be the redshift at the distance of RSD_Position.
        lim = mock_with_moving_galaxies(single_redshift=False)
        lim.assign_redshift_along_axis()

        # redshift as a function of the position along the line of sight
        zs = np.linspace(lim.redshift - 0.5, lim.redshift + 0.5, 10000)
        distances = lim.astropy_cosmo.comoving_distance(zs).to_value(u.Mpc)
        z_at_distance = interp1d(distances, zs)
        axis = np.argmax(lim.LOS)

        def z_of(position):
            return z_at_distance(
                (lim.minimum_distance + position[:, axis]).to_value(u.Mpc))

        shift = np.asarray(lim.cat["RSD_redshift"] - lim.cat["cosmo_redshift"])
        shift_from_position = (z_of(lim.cat["RSD_Position"])
                               - z_of(lim.cat["Position"]))
        # the velocities shift the redshifts by about (1 + z) 100 km/s / c = 1e-3
        self.assertGreater(np.std(shift), 5e-4)
        # Equal to first order in v / c. The old formula, z v / c, is off by
        # v / c, which is about 3e-4 here.
        np.testing.assert_allclose(shift, shift_from_position, atol=2e-5)

    def test_single_redshift(self):
        lim = mock_with_moving_galaxies(single_redshift=True)
        lim.assign_single_redshift()

        v_LOS = lim.cat["Velocity"].dot(lim.LOS)
        expected = (1 + lim.redshift) * (1 + (v_LOS / const.c).to_value(1)) - 1
        np.testing.assert_allclose(
            np.asarray(lim.cat["RSD_redshift"]), expected, rtol=1e-12)

    def test_single_redshift_slices(self):
        # With single_redshift every galaxy is at the box's redshift, so every
        # slice along the line of sight must be too, also after
        # assign_single_redshift has run, and the mean intensity is the one at
        # that redshift.
        lim = mock_with_moving_galaxies(single_redshift=True)
        lim.galaxy_selection = {"intensity": "all", "n_gal": "all"}
        lim.assign_single_redshift()
        np.testing.assert_array_equal(lim.redshift_mesh_axis, lim.redshift)
        I_z = lim.mean_intensity_per_redshift(np.array([lim.redshift]))[0]
        self.assertAlmostEqual((lim.mean_intensity / I_z).to_value(1), 1, places=12)


if __name__ == "__main__":
    unittest.main()
