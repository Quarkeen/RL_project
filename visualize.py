"""
=============================================================================
 F1TENTH PPO Racing Agent — Real-Time Visualization
 Loads a trained model and runs continuous rendering at ~60 FPS.
=============================================================================
Usage:
    python visualize.py                          # uses best_model.pt
    python visualize.py checkpoints/model_ep50.pt  # specific checkpoint
"""

import sys
import time
import yaml
import numpy as np
import torch

from model import build_model
from racing_wrappers import make_env, load_centerline


def visualize(config_path: str = "config.yaml", model_path: str = None):
    # ── Load config ───────────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config["env"]
    eval_cfg = config["eval"]

    if model_path is None:
        model_path = eval_cfg.get("model_path", "checkpoints/best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ────────────────────────────────────────────────────────
    model = build_model(config, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Viz] Loaded: {model_path}")

    # ── Create environment ────────────────────────────────────────────────
    centerline = load_centerline(env_cfg["map_path"])
    env = make_env(config, render_mode="human", centerline=centerline)

    # ── Render settings ───────────────────────────────────────────────────
    TARGET_FPS = 60
    FRAME_TIME = 1.0 / TARGET_FPS
    NUM_EPISODES = 10

    print(f"[Viz] Running {NUM_EPISODES} episodes at {TARGET_FPS} FPS")
    print(f"[Viz] Press Ctrl+C to stop\n")

    try:
        for ep in range(NUM_EPISODES):
            obs, info = env.reset(
                start_x=env_cfg["start_x"],
                start_y=env_cfg["start_y"],
                start_theta=env_cfg["start_theta"],
            )

            done = False
            ep_reward = 0.0
            ep_steps = 0
            speeds = []

            while not done:
                frame_start = time.time()

                # Get deterministic action
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action = model.get_deterministic_action(obs_t)
                    action_np = action.cpu().numpy().flatten()

                # Clip actions
                action_np[0] = np.clip(action_np[0], -env_cfg["max_steer"], env_cfg["max_steer"])
                action_np[1] = np.clip(action_np[1], -env_cfg["max_accel"], env_cfg["max_accel"])

                # Step
                obs, reward, terminated, truncated, info = env.step(action_np)
                done = terminated or truncated

                ep_reward += reward
                ep_steps += 1
                speeds.append(info.get("speed", 0.0))

                # Render every frame
                env.render(mode="human")

                # Frame rate limiter — sleep to maintain ~60 FPS
                elapsed = time.time() - frame_start
                if elapsed < FRAME_TIME:
                    time.sleep(FRAME_TIME - elapsed)

            avg_speed = np.mean(speeds) if speeds else 0.0
            status = "💥 CRASH" if info.get("collision", False) else "✅ CLEAN"
            print(
                f"  Episode {ep+1}/{NUM_EPISODES} | "
                f"Reward: {ep_reward:8.1f} | "
                f"Speed: {avg_speed:5.2f} m/s | "
                f"Steps: {ep_steps:5d} | "
                f"{status}"
            )

    except KeyboardInterrupt:
        print("\n[Viz] Stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    config_file = "config.yaml"
    model_file = sys.argv[1] if len(sys.argv) > 1 else None
    visualize(config_file, model_file)
