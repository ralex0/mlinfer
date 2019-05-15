"""
infer.py

Do bayesian inference on models
"""
import pyro
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.abstract_infer import EmpiricalMarginal
from pyro.infer.mcmc import NUTS

import torch
from torch import tensor
from torch.distributions import constraints

import numpy as np


class ADVI:
    """Class for doing Automatic Differentiation Variational Inference on a 
    model. 

    mode : type of Gaussian guide function - 'diagonal' or 'multivariate'
    optimizer : pyro.optim.Optimizer object
    trace_chisq : if True, keep record of mean-squared error at every iteration
    """
    def __init__(self, mode='diagonal', optimizer=None, trace_chisq=False, quiet=False):
        self.mode = mode
        self.optimizer = optimizer
        self.trace_chisq = trace_chisq
        self.quiet = quiet

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
            if (t % max(int(steps / 10), 1) == 0) and (not self.quiet):
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
        return params

class MCMC:
    """Class for doing Markov Chain Monte Carlo Inference on a model. 
    """
    def __init__(self, mode='NUTS', step_size='adapt', adapt_step_size=False):
        self.mode = mode
        self.step_size = step_size
        self.adapt_step_size = adapt_step_size

    def run(self, model, data, steps, burn=0, chains=1):
        kernel = self._setup_kernel(model)
        self._init_mcmc(model, kernel, data)
        sampler = pyro.infer.mcmc.MCMC(kernel, num_samples=steps, warmup_steps=burn, num_chains=chains)
        self.result = sampler.run(data)
        self._finalize_chain(model)
        return self._summarize_mcmc_result()

    def _setup_kernel(self, model):
        if self.mode == 'NUTS':
            if self.step_size == 'adapt':
                return NUTS(model, adapt_step_size=True)
            else:
                return NUTS(model, step_size=self.step_size, 
                            adapt_step_size=self.adapt_step_size)

    def _init_mcmc(self, model, kernel, data):
        initial_trace = pyro.poutine.trace(model).get_trace(data)
        for param in model.param_names:
            initial_trace.nodes[param]['value'] = tensor(model.params[param][0],
                                                         dtype=torch.float32)
        kernel.initial_trace = initial_trace

    def _finalize_chain(self, model):
        # Why is this method of the marginal private?
        chain_torch = EmpiricalMarginal(self.result, model.param_names)._samples
        self.chain_numpy = _numpy_from(chain_torch)
        self.chain = {k: v for k, v in zip(model.param_names, self.chain_numpy.T)}

    def _summarize_mcmc_result(self):
        loc = [v.mean() for v in self.chain.values()]
        scale = [v.var() for v in self.chain.values()]
        vals = [(l, s) for l, s in zip(loc, scale)]
        params = {k: v for k, v in zip(self.chain.keys(), vals)}
        if 'noise_sd' in params:
            params['noise_sd'] = (np.exp(params['noise_sd'][0]), 
                                  np.exp(params['noise_sd'][1]))
        return params

def _numpy_from(tensor):
    return tensor.cpu().detach().numpy()