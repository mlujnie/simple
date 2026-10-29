"""The k binning averages over the full FFT grid, although SIMPLE stores only
the rfft half grid (k_z >= 0)."""
import unittest

import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d

from simple.simple import LognormalIntensityMock
from simple.tools import get_kspec_cython
from simple.tools_python import (bin_scipy, bin_Pk_2d, bin_Pk_mu, half_grid_weights,
                                 yaml_file_to_dictionary)


def full_grid_power(nx, ny, nz, seed=1):
    """|FFT|^2 of a random real field on the full grid, so P(-k) = P(k)."""
    field = np.random.default_rng(seed).standard_normal((nx, ny, nz))
    return np.abs(np.fft.fftn(field)) ** 2


class TestHalfGridBinning(unittest.TestCase):

    def test_weights_count_every_full_grid_mode_once(self):
        # the weights of the half grid add up to the number of full-grid modes
        for N_mesh in [(8, 8, 8), (6, 5, 7), (4, 6, 9)]:
            self.assertEqual(np.sum(half_grid_weights(N_mesh)), np.prod(N_mesh))

    def test_half_grid_bins_equal_full_grid_bins(self):
        # even and odd numbers of cells, and a box that is not a cube
        for N_mesh, box in [((16, 16, 16), (512.0, 512.0, 512.0)),
                            ((12, 10, 9), (300.0, 250.0, 225.0))]:
            nx, ny, nz = N_mesh
            k_full, mu_full = get_kspec_cython(nx, ny, nz, *box, dohalf=False)[:2]
            k_par_full, k_perp_full = get_kspec_cython(nx, ny, nz, *box, dohalf=False)[6:]
            k_half, mu_half = get_kspec_cython(nx, ny, nz, *box, dohalf=True)[:2]
            k_par_half, k_perp_half = get_kspec_cython(nx, ny, nz, *box, dohalf=True)[6:]
            P_full = full_grid_power(nx, ny, nz)
            P_half = P_full[:, :, : nz // 2 + 1]
            k_bins = np.linspace(0.0, np.max(k_half), 9)

            ours = bin_scipy(P_half.ravel(), k_bins, k_half.ravel(), mu_half.ravel(),
                             half_grid_weights(N_mesh).ravel(), return_nmodes=True)

            # plain means over the full grid
            def full_mean(values):
                return binned_statistic(k_full.ravel(), values.ravel(), bins=k_bins,
                                        statistic="mean")[0]
            L2 = (3 * mu_full**2 - 1) / 2
            expected = [full_mean(k_full), full_mean(P_full), 5 * full_mean(P_full * L2)]
            for a, b in zip(ours[:3], expected):
                np.testing.assert_allclose(a, b, rtol=1e-12)

            # independent modes: half the number of full-grid modes
            n_full = binned_statistic(k_full.ravel(), k_full.ravel(), bins=k_bins,
                                      statistic="count")[0]
            np.testing.assert_allclose(ours[3], n_full / 2, rtol=1e-12)

            # the same for the 2D binnings: P(k_par, k_perp), and P, k and mu in (k, mu) bins
            w = half_grid_weights(N_mesh).ravel()
            ours_2d = bin_Pk_2d(P_half.ravel(), k_bins, k_par_half.ravel(), k_perp_half.ravel(), w)
            expected_2d = binned_statistic_2d(k_par_full.ravel(), k_perp_full.ravel(), P_full.ravel(),
                                              bins=[k_bins, k_bins], statistic="mean")[0]
            np.testing.assert_allclose(ours_2d, expected_2d, rtol=1e-12)
            mu_bins = np.linspace(0.0, 1.0, 5)
            ours_mu = bin_Pk_mu(P_half.ravel(), k_bins, k_half.ravel(), mu_half.ravel(), mu_bins, w)
            for a, values in zip(ours_mu, [k_full, mu_full, P_full]):
                b = binned_statistic_2d(k_full.ravel(), mu_full.ravel(), values.ravel(),
                                        bins=[k_bins, mu_bins], statistic="mean")[0]
                np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_isotropic_field_has_no_quadrupole(self):
        # In a cubic box the x, y and z axes are equivalent, so the mean of
        # mu^2 = k_x^2 / k^2 over all modes in a bin is exactly 1/3, and the
        # quadrupole of an isotropic P(k) vanishes. Plain means over the half
        # grid gave P2 / P0 = 0.41 in the first of these bins.
        input_dict = yaml_file_to_dictionary("./tests/test_lim_input.yaml")
        # 8 bins up to the Nyquist k of the 16^3 grid in the 512 Mpc/h box
        input_dict.update(kmin=0.01, kmax=0.098, nkbin=8)
        lim = LognormalIntensityMock(input_dict)
        lim.get_kspec()
        k = np.asarray(lim.kspec)
        P = np.where(k > 0, k, 1.0) ** -1.5
        mean_k, monopole, quadrupole = lim.bin_scipy(P)
        np.testing.assert_allclose(quadrupole / monopole, 0.0, atol=1e-12)

        # the full grid (dohalf=False) gives the same bins
        lim.get_kspec(dohalf=False)
        k = np.asarray(lim.kspec)
        full = lim.bin_scipy(np.where(k > 0, k, 1.0) ** -1.5)
        for a, b in zip(full, (mean_k, monopole, quadrupole)):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * np.max(monopole))


if __name__ == "__main__":
    unittest.main()
