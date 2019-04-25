"""
model.py

Pyro models from bayesian inference on data
"""
import pyro
import pyro.distributions as dist

import torch
from torch import tensor

class _SimpleGaussianModel:
    """A  simple generative model for testing inference

    model(x) = a * sin(b * x + c)
    """
    def __init__(self, a, b, c, noise_sd=None):
        self.guess = {'a': a, 'b': b, 'c': c}
        self.param_names = list(self.guess.keys())

    def __call__(self, data):
        return self.model(data)

    def model(self, data):
        x = data['x']
        y = data['y']
        return pyro.condition(self.likelihood, data={'likelihood': y})(x)

    def likelihood(self, x):
        a = pyro.sample('a', dist.Normal(*self.guess['a']))
        b = pyro.sample('b', dist.Normal(*self.guess['b']))
        c = pyro.sample('c', dist.Normal(*self.guess['c']))
        noise_sd = 0.1
        params = {'a': a, 'b': b, 'c': c}
        expected = self.forward(x, params)
        return pyro.sample('likelihood', dist.Normal(expected, noise_sd**2))

    def forward(self, x, params):
        a = params['a']
        b = params['b']
        c = params['c']
        return a * torch.sin(b * x + c)


