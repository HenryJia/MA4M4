import copy
import torch
from torch import nn
import torch.nn.functional as F

class WSBM(nn.Module):
    def __init__(self, mu, tau, sigma, alpha):
        super().__init__()
        # Normally, to do Bayesian stuff we need nice conjugate distributions. But we're using
        # variational inference, so we can use whatever distrbutions we want, so long as we can
        # reparameterise the sampling with respect to the parameters. Otherwise, we can't run
        # gradient based optimisation. This does mean we have to get clever and pull some tricks
        # such as using a softmax method of selecting blocks during training

        self.prior_mu = nn.Parameter(mu.clone(), requires_grad=False)
        self.prior_tau = nn.Parameter(tau.clone(), requires_grad=False)
        self.prior_sigma = nn.Parameter(sigma.clone(), requires_grad=False)

        # Note, mu has shape (n, K) where K is the number of blocks
        # tau has shape (K, K)
        self.mu = nn.Parameter(mu)
        self.tau = nn.Parameter(tau)
        self.sigma = nn.Parameter(sigma)

        self.alpha = alpha

    def forward(self, A): # This computes the ELBO
        # Reparameterised sampling
        posterior_tau = torch.distributions.Normal(self.tau, torch.clamp(self.sigma, min=1e-6))
        z = F.softmax(self.mu, dim=1) # This makes it differentiable :)

        theta = torch.clamp(posterior_tau.rsample(), 1e-6, 1-1e-6)
        log_likelihood = 0

        # This is not fully vectorised, and kind of inefficient, but hey ho, good enough
        theta_z = torch.zeros_like(A)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                theta_z[i, j] = theta @ z[j] @ z[i] # This is like indexing, but differentiable :)

        log_likelihood = torch.sum(A * torch.log(theta_z) + (1 - A) * torch.log(1 - theta_z))

        # Compute evidence lower bound
        elbo = log_likelihood - self.kl()

        return elbo

    def kl(self):
        posterior_mu = torch.distributions.Categorical(logits=self.mu)
        posterior_tau = torch.distributions.Normal(self.tau, torch.clamp(self.sigma, min=1e-6))

        prior_mu = torch.distributions.Categorical(logits=self.prior_mu)
        prior_tau = torch.distributions.Normal(self.prior_tau, self.prior_sigma)

        kl_mu = torch.sum(torch.distributions.kl_divergence(posterior_mu, prior_mu))
        kl_tau = torch.sum(torch.distributions.kl_divergence(posterior_tau, prior_tau))

        # Note, since we assume all parameters to be independent, we can add their KL divergences
        return kl_mu + kl_tau

    def sample(self):
        z = torch.distributions.Categorical(logits=self.mu).sample()
        A = torch.zeros(self.mu.shape[0], self.mu.shape[0])
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                theta_z = self.tau[z[i], z[j]]
                A[i, j] = (torch.rand(1) < theta_z).float()
        return z, A


def visualise_undirected_clusters(z, A):
    A_clusters = np.zeros((np.max(z) + 1, np.max(z) + 1))

    for i in range(A.shape[0]):
        for j in range(i):
            A_clusters[z[i], z[j]] += A[i, j]
            A_clusters[z[j], z[i]] += A[i, j]

    return A_clusters


if __name__ == "__main__":
    import numpy as np
    import networkx as nx
    from torch import optim
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    plt.ion()

    G_karate = nx.read_edgelist('ucidata-zachary/out.ucidata-zachary', comments='%', nodetype=int)
    #sizes = np.array([4, 8, 16])
    #probs = [[0.25, 0.05, 0.02], [0.05, 0.35, 0.07], [0.02, 0.07, 0.40]]
    #G_karate = nx.stochastic_block_model(sizes, probs, seed=0)

    plt.figure()
    nx.draw_circular(G_karate, with_labels=True)

    A_karate = torch.from_numpy(nx.to_numpy_array(G_karate))

    K = 8
    mu = torch.ones(A_karate.shape[0], K)
    tau = torch.eye(K) * torch.mean(A_karate) # Use this as our prior
    sigma = torch.tensor(0.1)
    alpha = 0.5

    model = WSBM(mu, tau, sigma, alpha)
    nx.draw_circular(G_karate, with_labels=True)
    model.sample()

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    pb = tqdm()
    for i in range(200):
        optimizer.zero_grad()
        elbo = -model(A_karate)
        elbo.backward()
        optimizer.step()
        pb.set_postfix({'elbo': elbo.item(), 'kl': model.kl().item()})
        pb.update(1)
    print(model.tau[:5, :5])
    print(model.sigma)
    print(model.mu[:5, :5])

    colors = (np.linspace(0, 1, K) * 20 + 10)
    z, A = model.sample()
    z = z.numpy()
    A = A.numpy()

    A = A * (1 - np.eye(A.shape[0])) # Remove self loops, we know this graph has no self loops
    for i in range(A.shape[0]): # And make symmetric
        for j in range(A.shape[1]):
            A[i, j] = float(A[i, j] > 0 or A[j, i] > 0)

    plt.figure()
    nx.draw_circular(nx.Graph(A), with_labels=True)

    A_karate = A_karate.numpy()
    A_karate_clusters = visualise_undirected_clusters(z, A_karate)
    G_karate_clusters = nx.Graph(A_karate_clusters)

    print(A_karate_clusters)

    plt.figure()
    pos = nx.circular_layout(G_karate_clusters)
    edge_labels = nx.get_edge_attributes(G_karate_clusters,'weight')
    nx.draw_networkx(G_karate_clusters, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G_karate_clusters, pos, edge_labels=edge_labels)

    print('IOU:', np.sum(A_karate * A) / np.sum((A_karate + A) > 0))
