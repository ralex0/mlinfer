import unittest

from holo.model import HolopyAlphaModel

class TestHolopyModels(unittest.TestCase):
    def test_init_HolopyAlphaModel(self):
        params = {'x': 1., 'y': 1., 'z': 1., 'n': 1., 'r': 1., 'alpha': 1.}
        model = HolopyAlphaModel(params)
        self.assertTrue(len(model.param_names) == 6)

if __name__ == '__main__':
    unittest.main()