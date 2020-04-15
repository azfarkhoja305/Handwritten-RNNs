import math
import random
import numpy
import numpy as np
from matplotlib import pyplot
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal
import pdb


def plot_stroke(stroke, save_name=None):
    # Plot a single example.
    f, ax = pyplot.subplots()

    x = numpy.cumsum(stroke[:, 1])
    y = numpy.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = numpy.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        pyplot.show()
    else:
        try:
            pyplot.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print("Error building image!: " + save_name)

    pyplot.close()

# START = [[2,0,0]]
# STOP = [[3,0,0]]
# PAD = [[4,0,0]]


def check_cuda():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

device = check_cuda()

def calc_prob(target,rho,sigma,mu):
    sigma = sigma + 1e-8
    target = target.unsqueeze(-2).expand_as(mu)
    norm1 = math.pi*torch.prod(sigma,-1)*torch.sqrt(1-rho**2+1e-12)
    z = (((target-mu)**2)/sigma**2).sum(-1) - 2*rho*torch.prod(target-mu,-1)/torch.prod(sigma,-1)
    norm2 = 2*(1-rho**2)
    ans = torch.exp(-z/norm2)/norm1
    if torch.isnan(ans).sum() > 0:
        pdb.set_trace()
    return ans

def ce_sample(ce,temp):
    """Sample 0,1,<start>,<end>"""
    ces = Categorical(logits=ce/temp).sample().data
    return ces.float()

def bivariate_sample(pi,rho,sigma,mu):
    """Sample from a mixture of bivariate gaussians"""
    pi,rho,sigma,mu = pi.squeeze(),rho.squeeze(),sigma.squeeze(),mu.squeeze()
    pis = Categorical(probs=pi).sample().data
    bs = torch.arange(sigma.size(0))
    sample_mu = mu[bs,pis]
    covar = torch.empty((sigma.size(0),2,2)).to(device)
    covar[bs,0,0] = sigma[bs,pis,0] ** 2
    covar[bs,0,1] = rho[bs,pis] * sigma[bs,pis,0] * sigma[bs,pis,1] 
    covar[bs,1,0] = rho[bs,pis] * sigma[bs,pis,0] * sigma[bs,pis,1] 
    covar[bs,1,1] = sigma[bs,pis,1] ** 2
    sample = MultivariateNormal(sample_mu, covar).sample()
    return sample





