import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class FiLM(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super(FiLM, self).__init__()
        # Two small linear layers map conditions → gamma, beta
        self.gamma_fc = nn.Linear(cond_dim, hidden_dim)
        self.beta_fc = nn.Linear(cond_dim, hidden_dim)

    def forward(self, h, cond):
        gamma = self.gamma_fc(cond)  # shape [batch, hidden_dim]
        beta = self.beta_fc(cond)    # shape [batch, hidden_dim]
        return gamma * h + beta      # feature-wise modulation

class encoder(nn.Module):
    def __init__(self, n_input, n_latent, d_alpha=None, d_eta=None):
        super(encoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(self.n_latent, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_latent).normal_(mean=0.0, std=0.1))

        # FiLM module (condition on alpha + eta)
        self.use_film = d_alpha is not None and d_eta is not None and d_alpha > 0 and d_eta > 0
        if self.use_film:
            cond_dim = d_alpha + d_eta
            self.film = FiLM(cond_dim, n_hidden)

    def forward(self, x, alpha=None, eta=None):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        if self.use_film and alpha is not None and eta is not None:
            cond = torch.cat([
            F.layer_norm(alpha, alpha.shape[1:]),
            F.layer_norm(eta, eta.shape[1:])
        ], dim=-1)
            h = self.film(h, cond)
        z = F.linear(h, self.W_2, self.b_2)
        return z

class generator(nn.Module):
    def __init__(self, n_input, n_latent):
        super(generator, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_latent).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(self.n_input, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_input).normal_(mean=0.0, std=0.1))

    def forward(self, z):
        h = F.relu(F.linear(z, self.W_1, self.b_1))
        x = F.linear(h, self.W_2, self.b_2)
        return x

class discriminator(nn.Module):
    def __init__(self, n_input):
        super(discriminator, self).__init__()
        self.n_input = n_input
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(n_hidden, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_3 = nn.Parameter(torch.Tensor(1, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_3 = nn.Parameter(torch.Tensor(1).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        h = F.relu(F.linear(h, self.W_2, self.b_2))
        score = F.linear(h, self.W_3, self.b_3)
        return torch.clamp(score, min=-50.0, max=50.0)