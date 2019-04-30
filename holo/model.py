"""
model.py

Pyro models from bayesian inference on data
"""
import pyro
import pyro.distributions as dist

import torch

import numpy as np

from scipy.ndimage.filters import gaussian_filter

from holopy.core.metadata import update_metadata
from holopy.scattering import Sphere, calc_holo

class BaseModel(object):
    """Base Model for bayesian inference
    """
    def __init__(self, params):
        self.params = params
        self.param_names = list(self.params.keys())

    def __call__(self, data):
        return self.model(data)

    def model(self, data):
        x = data['x']
        y = data['y']
        return pyro.condition(self.likelihood, data={'likelihood': y})(x)

    def likelihood(self, x):
        raise NotImplementedError("Implement in subclass")

    def forward(self, x, params):
        raise NotImplementedError("Implement in subclass")


class NormalModel(BaseModel):
    """Model where all parameters have Gaussian priors. The initial params dict 
    specifies gaussian priors as {(loc, scale) for loc, scale in params.items()}
    """
    def likelihood(self, x):
        """
        """
        params = {k: pyro.sample(k, dist.Normal(*v)) 
                  for k, v in self.params.items()}
        noise_sd = 0.1
        expected = self.forward(x, params)
        return pyro.sample('likelihood', dist.Normal(expected, noise_sd**2))


class NoisyNormalModel(BaseModel):
    """Model where all parameters have Gaussian priors, except the noise_sd, 
    which log-normal distributed. This allows us to estimate noise in the data
    The initial params dict specifies the priors as 
    {(loc, scale) for loc, scale in params.items()}
    """
    def likelihood(self, x):
        """Since noise is included as parameter, sample ln_noise from Normal
        """
        params = {}
        for k, v in self.params.items():
            if k == 'noise_sd':
                ln_sigma = torch.log(torch.tensor(v[0]))
                # FIXME: Use estimate of variance from initial guess.
                ln_sigma_var = torch.abs(ln_sigma) / 10
                param = pyro.sample(k, dist.Normal(ln_sigma, ln_sigma_var))
            else:
                param = pyro.sample(k, dist.Normal(*v))
            params[k] = param
        
        expected = self.forward(x, params)
        try:
            noise = torch.exp(params['noise_sd']) ** 2
        except KeyError:
            try: 
                noise = torch.tensor(x.noise_sd) ** 2
            except:
                noise = torch.tensor(1.)
        return pyro.sample('likelihood', dist.Normal(expected, noise))


class HolopyAlphaModel(NoisyNormalModel):
    def forward(self, metadata, params):
        x = float(params['x'])
        y = float(params['y'])
        z = float(params['z'])
        n = float(params['n'])
        r = float(params['r'])
        alpha = float(params['alpha'])

        sph = Sphere(center = (x, y, z), n=n, r=r)
        mod = calc_holo(metadata, sph, scaling=alpha).values.squeeze()
        return torch.tensor(mod, dtype=torch.float32)

    def convert_holopy(self, data):
        x = data.copy()
        if x.noise_sd is None:
            x = update_metadata(x, noise_sd=self.estimate_noise_from(data))
        y = torch.tensor(data.values.squeeze(), dtype=torch.float32)
        return {'x': x, 'y': y}

    def estimate_noise_from(self, data):
        data = data.values.squeeze()
        smoothed_data = gaussian_filter(data, sigma=1)
        noise = np.std(data - smoothed_data)
        return noise