import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class VBSBM(nn.Module):
    def __init__(self, c, mu, sigma, use_vi=True, use_degree_correction=True):
        super().__init__()
        # Normally, to do Bayesian stuff we need nice conjugate distributions. But we're using
        # variational inference, so we can use whatever distrbutions we want, so long as we can
        # reparameterise the sampling with respect to the parameters. Otherwise, we can't run
        # gradient based optimisation. This does mean we have to get clever and pull some tricks
        # such as using a softmax method of selecting blocks during training

        self.prior_c = nn.Parameter(c.clone(), requires_grad=False)
        self.prior_mu = nn.Parameter(mu.clone(), requires_grad=False)
        self.prior_sigma = nn.Parameter(sigma.clone(), requires_grad=False)

        # Note, c has shape (n, K) where K is the number of blocks
        # mu has shape (K, K)
        self.c = nn.Parameter(c)
        self.mu = nn.Parameter(mu)
        self.sigma = nn.Parameter(sigma)

        # Degree correction parameter
        self.d = nn.Parameter(torch.ones(c.shape[0], 1))

        # Whether to use variational inference
        self.use_vi = use_vi
        self.use_degree_correction = use_degree_correction


    def forward(self, A): # This computes the ELBO
        # Reparameterised sampling
        posterior_mu = torch.distributions.Normal(self.mu, torch.clamp(self.sigma, min=1e-6))
        c = F.softmax(self.c, dim=1) # This makes it differentiable :)

        theta = torch.clamp(posterior_mu.rsample(), 1e-6, 1-1e-6)
        log_likelihood = 0

        theta_c = c @ theta @ c.T

        if self.use_degree_correction:
            d = torch.clamp(self.d @ self.d.T, 1e-6)
            log_likelihood = torch.sum(A * torch.log(d * theta_c) - d * theta_c)
        else:
            log_likelihood = torch.sum(A * torch.log(theta_c) - (1 - A) * torch.log(1 - theta_c))

        # Compute evidence lower bound
        elbo = log_likelihood
        if self.use_vi:
            elbo -= self.kl()

        return elbo

    def kl(self):
        posterior_c = torch.distributions.Categorical(logits=self.c)
        posterior_mu = torch.distributions.Normal(self.mu, torch.clamp(self.sigma, min=1e-6))

        prior_c = torch.distributions.Categorical(logits=self.prior_c)
        prior_mu = torch.distributions.Normal(self.prior_mu, self.prior_sigma)

        kl_c = torch.sum(torch.distributions.kl_divergence(posterior_c, prior_c))
        kl_mu = torch.sum(torch.distributions.kl_divergence(posterior_mu, prior_mu))

        # Note, since we assume all parameters to be independent, we can add their KL divergences
        return kl_c + kl_mu
