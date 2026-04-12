"""
=============================================================================
 F1TENTH PPO Racing Agent — Actor-Critic Network Architecture
 PyTorch implementation with LayerNorm, Orthogonal Init, and continuous
 action distribution (diagonal Gaussian).
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Shared-encoder Actor-Critic for continuous control PPO.

    Architecture:
      ┌────────────┐
      │  Obs (111)  │
      └─────┬──────┘
            │
      ┌─────▼──────┐
      │ Encoder MLP │  (256 → LN → ReLU → 256 → LN → ReLU)
      └──┬─────┬───┘
         │     │
    ┌────▼──┐ ┌▼────────┐
    │ Actor │ │  Critic  │
    │ μ, σ  │ │  V(s)    │
    └───────┘ └──────────┘

    Parameters
    ----------
    obs_dim : int
        Dimensionality of the observation vector (default: 111).
    action_dim : int
        Dimensionality of the action vector (default: 2).
    hidden_sizes : list of int
        Sizes of hidden layers in the shared encoder.
    use_layer_norm : bool
        Whether to apply LayerNorm after each hidden layer.
    ortho_gain : float
        Gain for orthogonal weight initialization (sqrt(2) ≈ 1.414).
    log_std_init : float
        Initial value of the learnable log standard deviation.
    """

    def __init__(
        self,
        obs_dim: int = 110,
        action_dim: int = 2,
        hidden_sizes: list = None,
        use_layer_norm: bool = True,
        ortho_gain: float = np.sqrt(2),
        log_std_init: float = -0.5,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ── Shared Encoder ────────────────────────────────────────────────
        encoder_layers = []
        prev_dim = obs_dim
        for h_size in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_dim, h_size))
            if use_layer_norm:
                encoder_layers.append(nn.LayerNorm(h_size))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_size

        self.encoder = nn.Sequential(*encoder_layers)

        # ── Actor Head (mean of Gaussian) ─────────────────────────────────
        self.actor_mean = nn.Linear(prev_dim, action_dim)

        # Learnable log standard deviation (state-independent)
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), log_std_init, dtype=torch.float32)
        )

        # ── Critic Head (state-value V(s)) ────────────────────────────────
        self.critic = nn.Linear(prev_dim, 1)

        # ── Weight Initialization ─────────────────────────────────────────
        self._initialize_weights(ortho_gain)

    def _initialize_weights(self, gain: float):
        """
        Apply Orthogonal Initialization to all linear layers.

        - Encoder & Actor mean: gain = sqrt(2) for ReLU networks
        - Critic output: gain = 1.0 (value predictions have different scale)
        - Actor mean output: gain = 0.01 (small initial actions for stable
          exploration)
        """
        for module in self.encoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.constant_(module.bias, 0.0)

        # Actor mean — small gain for conservative initial policy
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

        # Critic — unit gain
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        Forward pass through the full network.

        Parameters
        ----------
        obs : torch.Tensor, shape (batch, obs_dim)

        Returns
        -------
        action_mean : torch.Tensor, shape (batch, action_dim)
        action_std  : torch.Tensor, shape (action_dim,)
        value       : torch.Tensor, shape (batch, 1)
        """
        features = self.encoder(obs)
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp()
        value = self.critic(features)
        return action_mean, action_std, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor = None
    ):
        """
        Sample an action from the policy and compute value + log_prob.

        Parameters
        ----------
        obs : torch.Tensor
            Observation batch.
        action : torch.Tensor or None
            If provided, compute log_prob for this action (used in PPO update).
            If None, sample a new action from the policy distribution.

        Returns
        -------
        action      : torch.Tensor, shape (batch, action_dim)
        log_prob    : torch.Tensor, shape (batch,)
        entropy     : torch.Tensor, scalar
        value       : torch.Tensor, shape (batch, 1)
        """
        action_mean, action_std, value = self.forward(obs)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.rsample()  # reparameterized sample

        # Sum log_prob across action dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        return action, log_prob, entropy, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute state value V(s) only — no action sampling."""
        features = self.encoder(obs)
        return self.critic(features)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the mean action (no exploration noise) for evaluation."""
        features = self.encoder(obs)
        return self.actor_mean(features)

    def get_action_distribution(self, obs: torch.Tensor) -> Normal:
        """Return the full Normal distribution for analysis/logging."""
        action_mean, action_std, _ = self.forward(obs)
        return Normal(action_mean, action_std)


def build_model(config: dict, device: torch.device = None) -> ActorCritic:
    """
    Factory function to build the ActorCritic model from configuration.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    device : torch.device
        Target device (cpu or cuda).

    Returns
    -------
    ActorCritic model on the target device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config.get("model", {})
    wrapper_cfg = config.get("wrapper", {})
    env_cfg = config.get("env", {})

    # Compute observation dimension
    raw_beams = env_cfg.get("num_beams", 1080)
    downsample = wrapper_cfg.get("downsample_factor", 10)
    num_lidar = raw_beams // downsample
    obs_dim = num_lidar + 2  # lidar + vx + steer (vy removed — always 0)

    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=2,  # steer + speed
        hidden_sizes=model_cfg.get("hidden_sizes", [256, 256]),
        use_layer_norm=model_cfg.get("use_layer_norm", True),
        ortho_gain=model_cfg.get("ortho_init_gain", np.sqrt(2)),
        log_std_init=model_cfg.get("action_log_std_init", -0.5),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] ActorCritic created — {total_params:,} params ({trainable:,} trainable)")
    print(f"[Model] Obs dim: {obs_dim} | Action dim: 2 | Device: {device}")

    return model
