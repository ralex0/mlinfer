"""
infer.py

Do bayesian inference on models
"""
import pyro
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import torch
from torch import tensor
from torch.distributions import constraints

import numpy as np


class ADVI:
    """Class for doing Automatic Differentiation Variational Inference on a 
    model. 
    """
    def __init__(self, mode='diagonal', optimizer=Adam({'lr': 1e-4})):
        self.mode = mode
        self.optimizer = optimizer

    def run(self, model, data, steps=20000):
        self._init_run(model, steps)
        guide = self._guide(model)
        svi = SVI(model, guide, self.optimizer, loss=Trace_ELBO())
        self._run(svi, data, steps)
        return self._parse_svi_result(model)

    def _init_run(self, model, steps):
        pyro.clear_param_store()
        self._register(model.params)
        self.chain = torch.empty(len(model.params), steps)
        self.losses = []

    def _register(self, params):
        loc = tensor([v[0] for v in params.values()])
        pyro.param('auto_loc', loc)

        if self.mode == 'diagonal':
            scale = tensor([v[1] for v in params.values()])
            pyro.param("auto_scale", scale, constraint=constraints.positive)
        elif self.mode == 'multivariate':
            scale = torch.diag(tensor([v[1] for v in params.values()]))
            pyro.param("auto_scale_tril", scale, constraint=constraints.positive)

    def _guide(self, model):
        if self.mode == 'diagonal':
            return AutoDiagonalNormal(model)
        elif self.mode == 'multivariate':
            return AutoMultivariateNormal(model)

    def _run(self, svi, data, steps):
        for t in range(steps):
            self.losses.append(svi.step(data))
            # FIXME:  ADVI crashes when I try to update chain this way  
            # self.chain[:, t] = pyro.param('auto_loc')

    def _parse_svi_result(self, model):
        mean = pyro.param('auto_loc').detach().numpy()
        if self.mode == 'diagonal':
            var = pyro.param('auto_scale').detach().numpy()
        elif self.mode == 'multivariate':
            var = torch.diag(pyro.param('auto_scale_tril')).detach().numpy()
        vals = [(v, s) for v, s in zip(mean, var)]
        params = {k: v for k, v in zip(model.param_names, vals)}
        if 'noise_sd' in params:
            params['noise_sd'] = (np.exp(params['noise_sd'][0]), 
                                  np.exp(params['noise_sd'][1]))
        return params
