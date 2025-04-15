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

if __name__ == '__main__':
    unittest.main()
