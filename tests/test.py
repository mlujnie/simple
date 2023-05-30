import unittest

class TestStringMethods(unittest.TestCase):

    def test_import_simple(self):
        from simple.simple import LognormalIntensityMock
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        lim.run()
        self.assertEqual(lim.N_mesh[0], 16)
        self.assertTrue(len(lim.cat['Position'])>0)

    def test_import_pk3dmodel(self):
        from simple.pk_3d_model import Power_Spectrum_Model
        from simple.simple import LognormalIntensityMock
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        pk3d = Power_Spectrum_Model("./tests/test_lim_input.yaml", do_model_shot_noise=False)
        self.assertEqual(lim.N_mesh[0], pk3d.N_mesh[0])


if __name__ == '__main__':
    unittest.main()
