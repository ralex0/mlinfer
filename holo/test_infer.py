from datetime import timedelta
import time
import unittest

from holo import infer
from holo.model import HolopyAlphaModel, NoisyNormalModel

import holopy as hp

from nose.plugins.attrib import attr

import numpy as np

from pyro.optim import Adam

import torch
from torch import tensor

TARGET_SIMPLE = {'a': 2.5, 'b': 2.1, 'c': 2.81, 'noise_sd': 0.1}
GUESS_SIMPLE = {'a': (2.45, .2**2), 'b': (1.95, .2**2), 
                'c': (2.87, .1**2), 'noise_sd': (.13, 0.05**2)}

TARGET_FIT = {'x': 4.128, 'y': 4.256, 'z': 5.024,
              'n': 1.5875, 'r': .5123, 
              'alpha': .8784, 'noise_sd': 0.0213}

INITIAL_GUESS = {'x': (4.12, .1**2), 'y': (4.25, .1**2), 'z': (5, 1**2),
                 'n': (1.59, .1**2), 'r': (.5, .05**2), 
                 'alpha': (0.8, .1**2), 'noise_sd': (0.02, 0.01**2)}

RANDOM_SEED = 101

DEFAULT_ADAM_PARAMS = {"lr": 0.0001, "betas": (0.90, 0.999)}
DEFAULT_OPTIMIZER = Adam(DEFAULT_ADAM_PARAMS)

__TIMER_CLICK__ = time.time()
def tick_tock():
    global __TIMER_CLICK__
    last_time =  __TIMER_CLICK__
    current_time  = __TIMER_CLICK__ = time.time()
    return timedelta(seconds=(current_time - last_time))

class TestInference(unittest.TestCase):
    @attr('slow')
    def test_accuracy_ADVI_SimpleModel(self):
        data = _simulate_simple_data(TARGET_SIMPLE)
        model = _SimpleModel(GUESS_SIMPLE)
        svi = infer.ADVI(mode='diagonal')
        result = svi.run(model, data, steps=5000)
        param_ok = [np.allclose(v[0], TARGET_SIMPLE[k], atol=v[1]**.5) 
                    for k, v in result.items()]
        self.assertTrue(all(param_ok))

    def test_run_ADVI_HolopyAlphaModel(self):
        true_params = TARGET_FIT
        guess_params = INITIAL_GUESS
        model = HolopyAlphaModel(guess_params)
        data = _simulate_AlphaModel_data(TARGET_FIT)
        svi = infer.ADVI(mode='multivariate', optimizer=DEFAULT_OPTIMIZER)
        result = svi.run(model, data, steps=10)
        self.assertTrue(True)


class _SimpleModel(NoisyNormalModel):
    """A  simple  model for testing inference
    model(x) = a * sin(b * x + c) + N(0, noise_sd ** 2)
    """
    def forward(self, x, params):
        a = params['a']
        b = params['b']
        c = params['c']
        return a * torch.sin(b * x + c)


def _simulate_simple_data(params):
    x = torch.linspace(0, 10, 1000)
    torch.manual_seed(RANDOM_SEED)
    noise = params['noise_sd'] * torch.randn(1000)
    y = _SimpleModel(params).forward(x, params) + noise
    return {'x': x, 'y': y}


def _simulate_AlphaModel_data(params):
    detector = hp.detector_grid(shape=250, spacing=.1)
    metadata = {'medium_index': 1.33, 'illum_wavelen': 0.660, 
                'illum_polarization': (1, 0)}
    sphere = hp.scattering.Sphere(n = params['n'], r = params['r'], 
                                  center = (params['x'], params['y'], params['z']))
    holo = hp.scattering.calc_holo(detector, sphere, scaling=params['alpha'], 
                                   **metadata)
    np.random.seed(RANDOM_SEED)
    noise = params['noise_sd'] * np.random.randn(*holo.shape)
    noisy_holo = hp.core.metadata.copy_metadata(holo, holo + noise)
    torch_holo = tensor(noisy_holo.values.squeeze(), dtype=torch.float32)
    return {'x': noisy_holo, 'y': torch_holo}


if __name__ == '__main__':
    unittest.main()
