import unittest

import numpy as np

import torch
from torch import tensor

from holo import infer
from holo.model import _SimpleGaussianModel

class TestInfernce(unittest.TestCase):
    def test_ADVI_SimpleModel(self):
        true_params = {'a': 2.5, 'b': 2.1, 'c': 2.81}
        guess_params = {'a': (2.45, .2), 'b': (1.95, .2), 'c': (2.87, .1)}
        data = _simple_data(true_params, noise_sd=0.1)
        model = _SimpleGaussianModel(**guess_params)
        result = infer.advi_linear(model, data, steps=20000)
        param_ok = [np.allclose(v[0], true_params[k], atol=v[1]) 
                    for k, v in result.items()]
        self.assertTrue(all(param_ok))


def _simple_data(params, noise_sd):
    x = tensor(np.linspace(0, 10, 1000), dtype=torch.float32)
    np.random.seed(101)
    noise = noise_sd * np.random.randn(1000)
    y = _SimpleGaussianModel(**params).forward(x, params) + tensor(noise, dtype=torch.float32)
    return {'x': x, 'y': y}

if __name__ == '__main__':
    unittest.main()
