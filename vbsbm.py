import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class VBSBM(nn.Module):
    def __init__(self, c, tau, sigma, use_vi=True):
        super().__init__()
        # Normally, to do Bayesian stuff we need nice conjugate distributions. But we're using
        # variational inference, so we can use whatever distrbutions we want, so long as we can
        # reparameterise the sampling with respect to the parameters. Otherwise, we can't run
        # gradient based optimisation. This does mean we have to get clever and pull some tricks
        # such as using a softmax method of selecting blocks during training

        self.prior_c = nn.Parameter(c.clone(), requires_grad=False)
        self.prior_tau = nn.Parameter(tau.clone(), requires_grad=False)
        self.prior_sigma = nn.Parameter(sigma.clone(), requires_grad=False)

        # Note, c has shape (n, K) where K is the number of blocks
        # tau has shape (K, K)
        self.c = nn.Parameter(c)
        self.tau = nn.Parameter(tau)
        self.sigma = nn.Parameter(sigma)

        # Degree correction parameter
        self.d = nn.Parameter(torch.ones(c.shape[0], 1))

        # Whether to use variational inference
        self.use_vi = use_vi


    def forward(self, A): # This computes the ELBO
        # Reparameterised sampling
        posterior_tau = torch.distributions.Normal(self.tau, torch.clamp(self.sigma, min=1e-6))
        c = F.softmax(self.c, dim=1) # This makes it differentiable :)

        theta = torch.clamp(posterior_tau.rsample(), 1e-6, 1-1e-6)
        log_likelihood = 0

        # This is not fully vectorised, and kind of inefficient
        #theta_c = torch.zeros_like(A)
        #for i in range(A.shape[0]):
            #for j in range(A.shape[1]):
                #theta_c[i, j] = theta @ c[j] @ c[i] # This is like indexing, but differentiable :)

        # Fully vectorised version
        # This is equivalent to the above, but literally 50x-100x faster
        # And took me way too damn long to figure out
        theta_c = c @ theta @ c.T

        d = torch.clamp(self.d @ self.d.T, 1e-6)

        log_likelihood = torch.sum(A * torch.log(d * theta_c) - d * theta_c)

        # Compute evidence lower bound
        elbo = log_likelihood
        if self.use_vi:
            elbo -= self.kl()

        return elbo

    def kl(self):
        posterior_c = torch.distributions.Categorical(logits=self.c)
        posterior_tau = torch.distributions.Normal(self.tau, torch.clamp(self.sigma, min=1e-6))

        prior_c = torch.distributions.Categorical(logits=self.prior_c)
        prior_tau = torch.distributions.Normal(self.prior_tau, self.prior_sigma)

        kl_c = torch.sum(torch.distributions.kl_divergence(posterior_c, prior_c))
        kl_tau = torch.sum(torch.distributions.kl_divergence(posterior_tau, prior_tau))

        # Note, since we assume all parameters to be independent, we can add their KL divergences
        return kl_c + kl_tau

    def sample(self):
        c = torch.distributions.Categorical(logits=self.c).sample()
        A = torch.zeros(self.c.shape[0], self.c.shape[0])
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                theta_c = self.tau[c[i], c[j]]
                A[i, j] = (torch.rand(1).to(device=theta_c.device) < theta_c).float()
        return c, A
