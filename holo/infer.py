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
    def __init__(self, mode='diagonal', optimizer=Adam({'lr': 1e-4}), trace_chisq=True):
        self.mode = mode
        self.optimizer = optimizer
        self.trace_chisq = trace_chisq

    def run(self, model, data, steps=20000):
        self._init_run(model, steps)
        guide = self._guide(model)
        svi = SVI(model, guide, self.optimizer, loss=Trace_ELBO())
        self._run(svi, data, steps)
        return self._parse_svi_result(model)

    def _init_run(self, model, steps):
        pyro.clear_param_store()
        self._register(model.params)
        self.chain = np.empty((len(model.params), steps))
        self.losses = []
        self.chisq = []

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
            if t % (steps // 10) == 0:
                print('.', end='')
            params = pyro.param('auto_loc').detach().numpy()
            self.chain[:, t] = params
            if self.trace_chisq:
                params_dict = {k: v for k, v in zip(svi.model.param_names, params)}
                residuals = svi.model.forward(data['x'], params_dict) - data['y']
                self.chisq.append(float(torch.sum(residuals ** 2)))

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
        # FIXME: this is a hack beacause i am sampling log of r to keep it positive
        # if 'r' in params:
        #     params['r'] = (np.exp(params['r'][0]), 
        #                           np.exp(params['r'][1]))
        return params
