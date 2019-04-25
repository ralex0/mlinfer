import unittest

import numpy as np

import torch
from torch import tensor

from holo import infer
from holo.model import NoisyNormalModel

class TestInference(unittest.TestCase):
    def test_ADVI_SimpleModel(self):
        true_params = {'a': 2.5, 'b': 2.1, 'c': 2.81, 'noise_sd': 0.1}
        guess_params = {'a': (2.45, .2**2), 'b': (1.95, .2**2), 'c': (2.87, .1**2),
                        'noise_sd': (.13, 0.05**2)}
        data = _simple_data(true_params, noise_sd=0.1)
        model = _SimpleModel(**guess_params)
        svi = infer.ADVI(mode='diagonal')
        result = svi.run(model, data, steps=20000)
        param_ok = [np.allclose(v[0], true_params[k], atol=v[1]**.5) 
                    for k, v in result.items()]
        self.assertTrue(all(param_ok))


class _SimpleModel(NoisyNormalModel):
    """A  simple generative model for testing inference

    model(x) = a * sin(b * x + c) + N(0, noise_sd ** 2)
    """
    def forward(self, x, params):
        a = params['a']
        b = params['b']
        c = params['c']
        return a * torch.sin(b * x + c)


def _simple_data(params, noise_sd):
    x = torch.linspace(0, 10, 1000)
    torch.manual_seed(101)
    noise = noise_sd * torch.randn(1000)
    y = _SimpleModel(**params).forward(x, params) + noise
    return {'x': x, 'y': y}


if __name__ == '__main__':
    unittest.main()
