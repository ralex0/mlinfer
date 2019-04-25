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

def advi_linear(model, data, steps=20000):
    pyro.clear_param_store()
    optimizer = Adam({'lr': 1e-3})#Adam({"lr": 0.0005, "betas": (0.90, 0.999)})
    register_params_linear(model)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    #chain = np.empty((len(model.param_names), steps))
    for stp in range(steps):
        svi.step(data)
        #chain[:, stp] = pyro.param('auto_loc').detach().numpy()

    val = pyro.param('auto_loc').detach().numpy()
    std = pyro.param('auto_scale').detach().numpy() ** .5
    vals = [(v, s) for v, s in zip(val, std)]
    params = {k: v for k, v in zip(model.param_names, vals)}
    return params#, chain

def advi_nonlinear(model, data, steps=50):
    pyro.clear_param_store()
    optimizer = Adam({"lr": 0.0005, "betas": (0.90, 0.999)})
    register_params_nonlinear(model)
    guide = AutoMultivariateNormal(model)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for stp in range(steps):
        svi.step(data)
        
    val = pyro.param('auto_loc').detach().numpy()
    std = pyro.param('auto_scale_tril').detach().numpy()
    vals = [(v, s) for v, s in zip(val, std)]
    params = {k: v for k, v in zip(model.param_names, vals)}
    return params


def register_params_linear(model):
    loc = tensor([v[0] for v in model.guess.values()])
    scale = tensor([v[1]**2 for v in model.guess.values()])
    pyro.param('auto_loc', loc)
    pyro.param("auto_scale", scale, constraint=constraints.positive)


def register_params_nonlinear(model):
    loc = tensor([v[0] for v in model.guess.values()])
    scale = torch.diag(tensor([v[1] for v in model.guess.values()]))
    pyro.param('auto_loc', loc)
    pyro.param("auto_scale_tril", scale, constraint=constraints.positive)