"""
=============================================================================
 F1TENTH PPO Racing Agent — Training Engine
 Professional RL training loop with GAE, gradient clipping, and TensorBoard
 telemetry.
=============================================================================
"""

import os
import sys
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from model import ActorCritic, build_model
from racing_wrappers import make_env, load_centerline


# =========================================================================
#  Rollout Buffer — Stores trajectories for PPO update
# =========================================================================
class RolloutBuffer:
    """
    Fixed-size buffer storing (obs, action, log_prob, reward, done, value)
    tuples collected during environment interaction.
    """

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, device: torch.device):
        self.size = buffer_size
        self.device = device
        self.ptr = 0

        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

    def store(self, obs, action, log_prob, reward, done, value):
        """Store a single transition."""
        idx = self.ptr
        self.observations[idx] = obs
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.dones[idx] = float(done)
        self.values[idx] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Compute Generalized Advantage Estimation (GAE-λ).

        GAE(γ, λ):
            δ_t     = r_t + γ · V(s_{t+1}) · (1 - done) − V(s_t)
            A_t^GAE = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}

        This provides a bias-variance tradeoff controlled by λ:
          λ = 0 → pure TD (high bias, low variance)
          λ = 1 → pure MC (low bias, high variance)
          λ = 0.95 → sweet spot for most tasks
        """
        n = self.ptr
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_nonterminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_nonterminal
                - self.values[t]
            )
            self.advantages[t] = last_gae = (
                delta + gamma * gae_lambda * next_nonterminal * last_gae
            )

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, mini_batch_size: int):
        """
        Yield shuffled mini-batches as torch tensors on the target device.
        """
        n = self.ptr
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, mini_batch_size):
            end = min(start + mini_batch_size, n)
            batch_idx = indices[start:end]

            yield (
                torch.FloatTensor(self.observations[batch_idx]).to(self.device),
                torch.FloatTensor(self.actions[batch_idx]).to(self.device),
                torch.FloatTensor(self.log_probs[batch_idx]).to(self.device),
                torch.FloatTensor(self.returns[batch_idx]).to(self.device),
                torch.FloatTensor(self.advantages[batch_idx]).to(self.device),
            )

    def reset(self):
        """Reset pointer for next rollout collection."""
        self.ptr = 0


# =========================================================================
#  PPO Update Step
# =========================================================================
def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    config: dict,
) -> dict:
    """
    Perform multiple epochs of PPO clipped objective optimization.

    Returns a dict of loss metrics for logging.
    """
    ppo_cfg = config["ppo"]
    clip_eps = ppo_cfg["clip_epsilon"]
    ent_coeff = ppo_cfg["entropy_coeff"]
    vf_coeff = ppo_cfg["value_loss_coeff"]
    max_grad = ppo_cfg["max_grad_norm"]
    epochs = ppo_cfg["ppo_epochs"]
    batch_size = ppo_cfg["mini_batch_size"]

    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy_loss": 0.0,
        "total_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    num_updates = 0

    for _ in range(epochs):
        for obs_b, act_b, old_logp_b, ret_b, adv_b in buffer.get_batches(batch_size):

            # Normalize advantages (per mini-batch)
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            # Forward pass
            _, new_logp, entropy, new_val = model.get_action_and_value(obs_b, act_b)

            # ── Policy Loss (clipped surrogate objective) ─────────────
            log_ratio = new_logp - old_logp_b
            ratio = log_ratio.exp()

            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # ── Value Loss (clipped value function) ───────────────────
            new_val = new_val.squeeze(-1)
            value_loss = 0.5 * ((new_val - ret_b) ** 2).mean()

            # ── Entropy Bonus ─────────────────────────────────────────
            entropy_loss = -entropy  # negative because we maximize entropy

            # ── Total Loss ────────────────────────────────────────────
            total_loss = policy_loss + vf_coeff * value_loss + ent_coeff * entropy_loss

            # ── Backpropagation with gradient clipping ────────────────
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            optimizer.step()

            # ── Metrics ───────────────────────────────────────────────
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
                clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            metrics["policy_loss"] += policy_loss.item()
            metrics["value_loss"] += value_loss.item()
            metrics["entropy_loss"] += entropy_loss.item()
            metrics["total_loss"] += total_loss.item()
            metrics["approx_kl"] += approx_kl
            metrics["clip_fraction"] += clip_frac
            num_updates += 1

    # Average metrics
    for k in metrics:
        metrics[k] /= max(num_updates, 1)

    return metrics


# =========================================================================
#  Main Training Loop
# =========================================================================
def train(config_path: str = "config.yaml"):
    """
    Full PPO training pipeline for the F1TENTH racing environment.
    """
    # ── Load configuration ────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    ppo_cfg = config["ppo"]
    train_cfg = config["training"]
    env_cfg = config["env"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    # ── Setup directories ─────────────────────────────────────────────────
    os.makedirs(train_cfg["log_dir"], exist_ok=True)
    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)

    # ── TensorBoard writer ────────────────────────────────────────────────
    run_name = f"f1tenth_ppo_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join(train_cfg["log_dir"], run_name))
    print(f"[Train] TensorBoard logs: {train_cfg['log_dir']}{run_name}")

    # ── Environment ───────────────────────────────────────────────────────
    centerline = load_centerline(env_cfg["map_path"])
    env = make_env(config, render_mode=None, centerline=centerline)
    print(f"[Train] Environment created — obs_dim: {env.obs_dim}, action_dim: 2")

    # ── Model & Optimizer ─────────────────────────────────────────────────
    model = build_model(config, device)
    optimizer = optim.Adam(model.parameters(), lr=ppo_cfg["learning_rate"], eps=1e-5)

    # ── Rollout Buffer ────────────────────────────────────────────────────
    buffer = RolloutBuffer(
        buffer_size=ppo_cfg["rollout_length"],
        obs_dim=env.obs_dim,
        action_dim=2,
        device=device,
    )

    # ── Training State ────────────────────────────────────────────────────
    total_timesteps = train_cfg["total_timesteps"]
    global_step = 0
    episode_num = 0
    best_mean_reward = -float("inf")

    # Rolling metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_speeds = deque(maxlen=100)
    episode_lap_times = deque(maxlen=100)

    print(f"\n{'='*70}")
    print(f"  F1TENTH PPO Training — Spielberg (Red Bull Ring)")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Rollout length:  {ppo_cfg['rollout_length']}")
    print(f"  Mini-batch size: {ppo_cfg['mini_batch_size']}")
    print(f"  PPO epochs:      {ppo_cfg['ppo_epochs']}")
    print(f"{'='*70}\n")

    # ── Initial reset ─────────────────────────────────────────────────────
    obs, info = env.reset(
        start_x=env_cfg["start_x"],
        start_y=env_cfg["start_y"],
        start_theta=env_cfg["start_theta"],
    )

    ep_reward = 0.0
    ep_length = 0
    ep_speeds = []
    start_time = time.time()

    # ── Main Loop ─────────────────────────────────────────────────────────
    while global_step < total_timesteps:

        # ── Collect Rollout ───────────────────────────────────────────
        buffer.reset()
        model.eval()

        for step in range(ppo_cfg["rollout_length"]):
            global_step += 1

            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                action, log_prob, _, value = model.get_action_and_value(obs_t)

                action_np = action.cpu().numpy().flatten()
                log_prob_np = log_prob.cpu().item()
                value_np = value.cpu().item()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Store transition
            buffer.store(obs, action_np, log_prob_np, reward, done, value_np)

            ep_reward += reward
            ep_length += 1
            ep_speeds.append(info.get("speed", 0.0))

            if done:
                episode_num += 1
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                episode_speeds.append(np.mean(ep_speeds) if ep_speeds else 0.0)

                lap_time = info.get("lap_time", 0.0)
                if isinstance(lap_time, (list, np.ndarray)):
                    lap_time = float(lap_time[0]) if len(lap_time) > 0 else 0.0
                episode_lap_times.append(lap_time)

                # ── TensorBoard Logging ──────────────────────────────
                writer.add_scalar("Episode/Reward", ep_reward, episode_num)
                writer.add_scalar("Episode/Length", ep_length, episode_num)
                writer.add_scalar("Episode/Avg_Velocity", episode_speeds[-1], episode_num)
                writer.add_scalar("Episode/Lap_Time", lap_time, episode_num)

                # ── Console logging ──────────────────────────────────
                if episode_num % 5 == 0:
                    elapsed = time.time() - start_time
                    fps = global_step / max(elapsed, 1)
                    mean_r = np.mean(episode_rewards)
                    mean_v = np.mean(episode_speeds)
                    print(
                        f"[Ep {episode_num:5d}] "
                        f"Step: {global_step:8d}/{total_timesteps} | "
                        f"R: {ep_reward:8.2f} (μ={mean_r:7.2f}) | "
                        f"V: {mean_v:5.2f} m/s | "
                        f"Len: {ep_length:5d} | "
                        f"FPS: {fps:6.0f}"
                    )

                # ── Render every N episodes ──────────────────────────
                if episode_num % train_cfg["render_every"] == 0:
                    try:
                        env.render(mode="human")
                    except Exception:
                        pass  # headless fallback

                # ── Save checkpoint ──────────────────────────────────
                if episode_num % train_cfg["save_every"] == 0:
                    ckpt_path = os.path.join(
                        train_cfg["checkpoint_dir"],
                        f"model_ep{episode_num}.pt",
                    )
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "episode": episode_num,
                        "global_step": global_step,
                        "mean_reward": float(np.mean(episode_rewards)),
                    }, ckpt_path)
                    print(f"[Save] Checkpoint → {ckpt_path}")

                # ── Save best model ──────────────────────────────────
                mean_rew = np.mean(episode_rewards)
                if mean_rew > best_mean_reward and len(episode_rewards) >= 10:
                    best_mean_reward = mean_rew
                    best_path = train_cfg["best_model_path"]
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "episode": episode_num,
                        "global_step": global_step,
                        "mean_reward": float(mean_rew),
                    }, best_path)
                    print(f"[Best] New best model saved (μR={mean_rew:.2f}) → {best_path}")

                # Reset episode
                ep_reward = 0.0
                ep_length = 0
                ep_speeds = []
                obs, info = env.reset(
                    start_x=env_cfg["start_x"],
                    start_y=env_cfg["start_y"],
                    start_theta=env_cfg["start_theta"],
                )
            else:
                obs = next_obs

            if global_step >= total_timesteps:
                break

        # ── Compute GAE ──────────────────────────────────────────────
        with torch.no_grad():
            last_obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            last_value = model.get_value(last_obs_t).cpu().item()

        buffer.compute_gae(last_value, ppo_cfg["gamma"], ppo_cfg["gae_lambda"])

        # ── PPO Update ───────────────────────────────────────────────
        model.train()
        metrics = ppo_update(model, optimizer, buffer, config)

        # Log PPO metrics
        writer.add_scalar("Loss/Policy", metrics["policy_loss"], global_step)
        writer.add_scalar("Loss/Value", metrics["value_loss"], global_step)
        writer.add_scalar("Loss/Entropy", metrics["entropy_loss"], global_step)
        writer.add_scalar("Loss/Total", metrics["total_loss"], global_step)
        writer.add_scalar("PPO/Approx_KL", metrics["approx_kl"], global_step)
        writer.add_scalar("PPO/Clip_Fraction", metrics["clip_fraction"], global_step)

        if len(episode_rewards) > 0:
            writer.add_scalar("Rollout/Mean_Reward", np.mean(episode_rewards), global_step)
            writer.add_scalar("Rollout/Mean_Length", np.mean(episode_lengths), global_step)

    # ── Training Complete ─────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Training Complete!")
    print(f"  Total time:      {elapsed/3600:.2f} hours")
    print(f"  Total episodes:  {episode_num}")
    print(f"  Total steps:     {global_step:,}")
    print(f"  Best mean reward:{best_mean_reward:.2f}")
    print(f"{'='*70}\n")

    # Final save
    final_path = os.path.join(train_cfg["checkpoint_dir"], "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode_num,
        "global_step": global_step,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
    }, final_path)
    print(f"[Save] Final model → {final_path}")

    writer.close()
    env.close()


# =========================================================================
#  Entry Point
# =========================================================================
if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    train(config_file)
