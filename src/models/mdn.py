import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical


class MDN(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size, n_components=2):
        """Mixture Density Networks -- fits a K-Mixture of Gaussians, where k = n_components..."""
        super().__init__()
        self.obs_dim, self.act_dim, self.hidden_size, self.n_components = obs_dim, act_dim, hidden_size, n_components

        # Pi --> Predicts probability over components...
        self.pi_network = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, n_components))

        # Gaussian --> Predict means, diagonalized variances for *each* component
        self.gaussian_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.GELU(), nn.Linear(hidden_size, 2 * act_dim * n_components)
        )

    def forward(self, obs):
        mean, std = torch.split(self.gaussian_network(obs), self.act_dim * self.n_components, dim=1)
        mean, std = torch.stack(mean.split(self.act_dim, 1)), torch.exp(torch.stack(std.split(self.act_dim, 1)))
        print(std)
        return OneHotCategorical(logits=self.pi_network(obs)), Normal(mean.transpose(0, 1), std.transpose(0, 1))

    def get_action(self, obs, deterministic=False):
        # TODO - Hacky...
        assert len(obs.shape) < 2
        obs = obs.unsqueeze(0)

        # Actual Sampling Code...
        pi_dist, gaussian_dist = self.forward(obs)
        if deterministic:
            component = pi_dist.probs.argmax()
            mean = gaussian_dist.mean
        else:
            component = pi_dist.sample().argmax()
            mean = gaussian_dist.sample()

        return mean[0][component]
