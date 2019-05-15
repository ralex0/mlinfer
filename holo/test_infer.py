from datetime import timedelta
import pickle
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

TARGET_SIMPLE = {'a': 2.5, 'b': 2.1, 'c': 2.81, 
                 'noise_sd': 0.1}

GUESS_SIMPLE = {'a': (2.45, .2**2), 'b': (1.95, .2**2), 'c': (2.87, .1**2),
                'noise_sd': (.13, 0.05**2)}

TARGET_SPHERE = {'x': 4.128, 'y': 4.256, 'z': 5.024,
                 'n': 1.5875, 'r': .5123, 
                 'alpha': .8784,# 'lens_angle': .9, 
                 'noise_sd': 0.0213}

GUESS_SPHERE = {'x': (4.12, 2*.1**2), 'y': (4.25, 2*.1**2), 'z': (5, 2*1**2),
                'n': (1.59, 2*.1**2), 'r': (.5, 2*.05**2), 
                'alpha': (0.8, 2*.1**2),# 'lens_angle': (.91, .02**2), 
                'noise_sd': (0.02, 2*0.01**2)}

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
    def test_run_ADVI_SimpleModel(self):
        result = _run_SimpleModel_advi_for(steps=2)
        self.assertTrue(True)

    @attr('slow')
    def test_convergence_ADVI_SimpleModel(self):
        result = _run_SimpleModel_advi_for(steps=5000)
        params_ok = _check_convergence(result, TARGET_SIMPLE)
        self.assertTrue(all(params_ok.values()))

    def test_run_MCMC_SimpleModel(self):
        result = _run_SimpleModel_mcmc_for(steps=2)
        self.assertTrue(True)

    @attr('slow')
    def test_convergence_MCMC_SimpleModel(self):
        result = _run_SimpleModel_mcmc_for(steps=512)
        params_ok = _check_convergence(result, TARGET_SIMPLE)
        self.assertTrue(all(params_ok.values()))

    def test_run_ADVI_AlphaModel(self):
        result = _run_AlphaModel_advi_for(steps=2)
        self.assertTrue(True)

    def test_run_MCMC_AlphaModel(self):
        _run_AlphaModel_mcmc_for(steps=2)
        self.assertTrue(True)

    # @attr('slow')
    # def test_convergence_MCMC_AlphaModel(self):
    #     result = _run_AlphaModel_mcmc_for(steps=1000)
    #     param_ok = _check_convergence(result, TARGET_SPHERE)
    #     self.assertTrue(all(param_ok.values()))


class _SimpleModel(NoisyNormalModel):
    """A  simple  model for testing inference
    model(x) = a * sin(b * x + c) + N(0, noise_sd ** 2)
    """
    def forward(self, x, params):
        a = params['a']
        b = params['b']
        c = params['c']
        return a * torch.sin(b * x + c)


def _run_SimpleModel_advi_for(steps):
    data = _simulate_simple_data(TARGET_SIMPLE)
    model = _SimpleModel(GUESS_SIMPLE)
    svi = infer.ADVI(mode='diagonal', optimizer=DEFAULT_OPTIMIZER)
    result = svi.run(model, data, steps=steps)
    return result

def _simulate_simple_data(params):
    x = torch.linspace(0, 10, 1000)
    torch.manual_seed(RANDOM_SEED)
    noise = params['noise_sd'] * torch.randn(1000)
    y = _SimpleModel(params).forward(x, params) + noise
    return {'x': x, 'y': y}

def _check_convergence(result, target):
    # result is {param: (mean, var)}
    params = target.keys()
    converged = [np.allclose(v[0], target[k], atol=v[1]**.5) 
                 for k, v in result.items()]
    return {k: v  for k, v in zip(params, converged)}

def _run_SimpleModel_mcmc_for(steps):
    data = _simulate_simple_data(TARGET_SIMPLE)
    model = _SimpleModel(GUESS_SIMPLE)
    mcmc = infer.MCMC(mode='NUTS')
    result = mcmc.run(model, data, steps=steps)
    return result

def _run_AlphaModel_advi_for(steps):
    model = HolopyAlphaModel(GUESS_SPHERE)
    data = _simulate_AlphaModel_data(TARGET_SPHERE)
    svi = infer.ADVI(mode='multivariate', optimizer=DEFAULT_OPTIMIZER)
    result = svi.run(model, data, steps=steps)
    return result

def _simulate_AlphaModel_data(params):
    return _simulate_sphere_data(params, theory='mieonly')

def _simulate_PerfectLensModel_data(params):
    return _simulate_sphere_data(params, theory='mielens')

def _simulate_sphere_data(params, theory):
    detector = hp.detector_grid(shape=100, spacing=.1)
    metadata = {'medium_index': 1.33, 'illum_wavelen': 0.660, 
                'illum_polarization': (1, 0)}
    sphere = hp.scattering.Sphere(n = params['n'], r = params['r'], 
                                  center = (params['x'], params['y'], params['z']))
    if theory == 'mieonly':
        alpha = params['alpha']
        metadata['scaling'] = alpha
    elif theory == 'mielens': 
        lens_angle = params['lens_angle']
        metadata['theory'] = hp.scattering.theory.MieLens(lens_angle=lens_angle)

    holo = hp.scattering.calc_holo(detector, sphere, **metadata)
    np.random.seed(RANDOM_SEED)
    noise = params['noise_sd'] * np.random.randn(*holo.shape)
    noisy_holo = hp.core.metadata.copy_metadata(holo, holo + noise)
    torch_holo = tensor(noisy_holo.values.squeeze(), dtype=torch.float32)
    return {'x': noisy_holo, 'y': torch_holo}

def _run_AlphaModel_mcmc_for(steps):
    data = _simulate_AlphaModel_data(TARGET_SPHERE)
    model = HolopyAlphaModel(GUESS_SPHERE)
    mcmc = infer.MCMC(mode='NUTS')
    result = mcmc.run(model, data, steps=steps)
    return result

if __name__ == '__main__':
    #unittest.main()
    data = _simulate_AlphaModel_data(TARGET_SPHERE)
    model = HolopyAlphaModel(GUESS_SPHERE)
    mcmc = infer.MCMC(mode='NUTS', step_size=2.5e-2, adapt_step_size=False)
    np.random.seed(RANDOM_SEED)
    tick_tock()
    result = mcmc.run(model, data, steps=2048, chains=2, burn=512)
    np.save('result_chain.npy', mcmc.chain_numpy)
    with open('result.pkl', 'wb') as f:
        pickle.dump(mcmc.chain, f)
    print(f"mcmc took {tick_tock()}")