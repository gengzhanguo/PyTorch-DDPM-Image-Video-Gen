import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List

# ----------------------------------------------------------- General Utils -----------------------------------------------------------
def exists(x):
    return x is not None

def cast_tuple(t, length = 1):
    '''
    t -> (t, t, ..., t) of length `length`
    '''
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def default(val, d):
    '''
    Provides default values for handling variables that may be None or undefined, 
    while supporting cases where the default is callable.
    '''
    if exists(val):
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(x, *args, **kwargs):
    return x    

# ----------------------------------------------------------- Normalization -----------------------------------------------------------
def normalize_data(data):
    '''
    [0-1] -> [-1, 1]
    '''
    return data * 2 - 1

def normalize_label(label):
    return 2* (label - min(label))/ (label.max() - label.min()) - 1

def denormalize_data(data):
    '''
    [-1, 1] -> [0, 1]
    '''
    return (data + 1) / 2

def denormalize_label(label):
    return (label + 1) / 2 * (label.max() - label.min()) + min(label)


# ----------------------------------------------------------- Probabilistic Inference -----------------------------------------------------------
def log_catgorical(x: torch.Tensor, p: torch.Tensor, num_classes: int, reduction: int=0, dim: List[int]=[-1]) -> torch.Tensor:
    PI = torch.tensor(math.pi)
    EPS = 1.e-5

    if dim is None:
        dim = []
    x_one_hot = F.one_hot(x, num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1.0 - EPS))
    if reduction == 0:
        return -torch.sum(log_p, dim=dim)
    elif reduction == 1:
        return -torch.mean(log_p, dim=dim)
    else:
        return log_p
    
def log_bernoulli(x: torch.Tensor, p: torch.Tensor, num_classes: int, reduction: int=0, dim: List[int]=[-1]) -> torch.Tensor:
    PI = torch.tensor(math.pi)
    EPS = 1.e-5

    if dim is None:
        dim = []

    log_p = x * torch.log(torch.clamp(p, EPS, 1.0 - EPS)) + (1. - x) * torch.log(1. - torch.clamp(p, EPS, 1.0 - EPS))
    if reduction == 0:
        return -torch.sum(log_p, dim=dim)
    elif reduction == 1:
        return -torch.mean(log_p, dim=dim)
    else:
        return log_p

def log_normal_diag(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, reduction: int=0, dim: List[int]=[-1]) -> torch.Tensor:
    PI = torch.tensor(math.pi)
    EPS = 1.e-5

    if dim is None:
        dim = []

    log_p = -0.5 * (logvar + (x - mu) ** 2. / torch.exp(logvar) + torch.log(2. * PI))
    if reduction == 0:
        return torch.sum(log_p, dim=dim)
    elif reduction == 1:
        return torch.mean(log_p, dim=dim)
    else:
        return log_p
    
def log_standard_normal(x: torch.Tensor, reduction: int=0, dim: List[int]=[-1]) -> torch.Tensor:
    PI = torch.tensor(math.pi)
    EPS = 1.e-5

    if dim is None:
        dim = []

    log_p = -0.5 * (x ** 2. + torch.log(2. * PI))
    if reduction == 0:
        return torch.sum(log_p, dim=dim)
    elif reduction == 1:
        return torch.mean(log_p, dim=dim)
    else:
        return log_p
    
def log_min_exp(a: torch.Tensor, b: torch.Tensor, epsilon: float=1e-8) -> torch.Tensor:
    """
    Source: https://github.com/jornpeters/integer_discrete_flows
    Computes the log of exp(a) - exp(b) in a (more) numerically stable fashion.
    Using:
    log(exp(a) - exp(b))
    c + log(exp(a-c) - exp(b-c))
    a + log(1 - exp(b-a))
    And note that we assume b < a always.
    """

    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y

def log_integer_probability(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    var = torch.exp(logvar)
    sigma = torch.sqrt(var)
    logp = log_min_exp(
        F.logsigmoid((x + 0.5 - mu) / sigma),
        F.logsigmoid((x - 0.5 - mu) / sigma)
    )
    return logp

def log_interger_probability_standard(x: torch.Tensor) -> torch.Tensor:
    logp = log_min_exp(
        F.logsigmoid(x + 0.5),
        F.logsigmoid(x - 0.5)
    )
    return logp

def get_trainable_parameter(model):
    for p in model.parameters():
        if p.requires_grad:
            print(p.name, p.data)
