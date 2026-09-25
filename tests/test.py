import unittest
import numpy as np
import astropy.units as u

class TestStringMethods(unittest.TestCase):

    def test_import_simple(self):
        from simple.simple import LognormalIntensityMock
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        print(lim.sigma_beam)
        lim.run()
        self.assertEqual(lim.N_mesh[0], 16)
        self.assertTrue(len(lim.cat['Position']) > 0)

        lim.run(skip_lognormal=True)
        lim.downsample_all_meshes([8,8,8])
        lim.run(skip_lognormal=True)

    def test_import_pk3dmodel(self):
        from simple.pk_3d_model import Power_Spectrum_Model
        from simple.simple import LognormalIntensityMock
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        pk3d = Power_Spectrum_Model(
            "./tests/test_lim_input.yaml", do_model_shot_noise=False)
        self.assertEqual(lim.N_mesh[0], pk3d.N_mesh[0])
        print(pk3d.do_model_shot_noise)

    def test_run_simple(self):
        from simple.simple import LognormalIntensityMock
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        lim.run()

    def test_cython_functions_reject_quantities(self):
        """Arrays with units must be rejected at the call site, rather than
        failing deep inside the loop with a confusing message."""
        from simple.tools import catalog_to_mesh_cython

        positions = np.array([[1.0, 2.0, 3.0]]) * u.Mpc
        weights = np.ones(1)
        N_mesh = np.array([4, 4, 4])
        box_size = np.array([10.0, 10.0, 10.0])

        with self.assertRaises(TypeError):
            catalog_to_mesh_cython(positions, weights, N_mesh, box_size)

        # the same call without units still works
        mesh = catalog_to_mesh_cython(
            positions.to(u.Mpc).value, weights, N_mesh, box_size)
        self.assertEqual(np.shape(mesh), (4, 4, 4))
        self.assertAlmostEqual(np.sum(mesh), 1.0)

    def test_index_and_position_branches_agree(self):
        """Both ways of painting a mesh must put every galaxy in the same
        voxel, including galaxies outside the (periodic) box."""
        from simple.tools import (catalog_to_mesh_cython,
                                  catalog_to_mesh_cython_use_indices,
                                  get_galaxy_indices_cython)

        N_mesh = np.array([4, 4, 4])
        box_size = np.array([10.0, 10.0, 10.0])
        positions = np.array([[1.0, 2.0, 3.0],
                              [9.9, 0.0, 5.0],
                              [10.5, -0.1, 9.9]])  # last one outside the box
        weights = np.array([1.0, 2.0, 3.0])

        indices = np.asarray(
            get_galaxy_indices_cython(positions, N_mesh, box_size))
        from_positions = np.asarray(
            catalog_to_mesh_cython(positions, weights, N_mesh, box_size))
        from_indices = np.asarray(
            catalog_to_mesh_cython_use_indices(
                indices, weights, N_mesh, box_size))

        np.testing.assert_allclose(from_indices, from_positions)
        self.assertAlmostEqual(np.sum(from_positions), np.sum(weights))

    def test_paint_intensity_mesh_without_indices(self):
        """paint_intensity_mesh must also work when the voxel indices have not
        been precomputed, i.e. when it paints straight from the positions,
        and give the same mesh as the branch that uses the indices."""
        from simple.simple import LognormalIntensityMock
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        lim.run()

        position = "RSD_Position" if lim.RSD else "Position"
        indices = "RSD_indices" if lim.RSD else "realspace_indices"
        self.assertIn(indices, lim.cat.keys())

        mesh_from_indices = lim.paint_intensity_mesh(position=position)

        del lim.cat[indices]  # force the position-based branch
        mesh_from_positions = lim.paint_intensity_mesh(position=position)

        self.assertEqual(mesh_from_positions.shape, mesh_from_indices.shape)
        self.assertTrue(np.all(np.isfinite(mesh_from_positions.value)))
        self.assertTrue(
            u.allclose(mesh_from_positions, mesh_from_indices, rtol=1e-5))

if __name__ == '__main__':
    unittest.main()
