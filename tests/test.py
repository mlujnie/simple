import unittest

class TestStringMethods(unittest.TestCase):

    def test_import(self):
        from simple.simple import LognormalIntensityMock
        lim = LognormalIntensityMock("./tests/test_lim_input.yaml")
        lim.run()
        self.assertEqual(lim.N_mesh[0], 16)
        self.assertTrue(len(lim.cat['Position'])>0)


if __name__ == '__main__':
    unittest.main()
