"""
=============================================================================
 F1TENTH PPO Racing Agent — Validation / Evaluation Script
 Loads a trained model, runs deterministic evaluation laps, and exports
 trajectory data for post-run analysis.
=============================================================================
"""

import os
import sys
import csv
import time
import yaml
import numpy as np
import torch

from model import ActorCritic, build_model
from racing_wrappers import make_env, load_centerline


def evaluate(config_path: str = "config.yaml", model_path: str = None):
    """
    Run deterministic evaluation of a trained PPO agent.

    Outputs:
      - Console: Fastest lap time, average speed per lap, total time
      - CSV:     Trajectory (x, y) per lap in trajectories/ directory
    """
    # ── Load configuration ────────────────────────────────────────────────
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    eval_cfg = config["eval"]
    env_cfg = config["env"]

    if model_path is None:
        model_path = eval_cfg["model_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(eval_cfg["trajectory_dir"], exist_ok=True)

    # ── Load Model ────────────────────────────────────────────────────────
    if not os.path.isfile(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("        Train the agent first with: python train.py")
        sys.exit(1)

    model = build_model(config, device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_episode = checkpoint.get("episode", "?")
    train_reward = checkpoint.get("mean_reward", "?")
    print(f"[Eval] Loaded model from {model_path}")
    print(f"       Trained for {train_episode} episodes | Mean reward: {train_reward}")

    # ── Environment ───────────────────────────────────────────────────────
    centerline = load_centerline(env_cfg["map_path"])
    env = make_env(config, render_mode="human", centerline=centerline)

    # ── Evaluation Parameters ─────────────────────────────────────────────
    num_laps = eval_cfg.get("num_laps", 5)
    deterministic = eval_cfg.get("deterministic", True)

    print(f"\n{'='*60}")
    print(f"  F1TENTH Agent Evaluation — Spielberg")
    print(f"  Laps to run: {num_laps}")
    print(f"  Mode: {'Deterministic' if deterministic else 'Stochastic'}")
    print(f"{'='*60}\n")

    # ── Run Evaluation Laps ───────────────────────────────────────────────
    lap_results = []
    all_trajectories = []

    for lap_idx in range(num_laps):
        obs, info = env.reset(
            start_x=env_cfg["start_x"],
            start_y=env_cfg["start_y"],
            start_theta=env_cfg["start_theta"],
        )

        trajectory = []
        lap_reward = 0.0
        lap_steps = 0
        speeds = []
        lap_start = time.time()

        done = False
        while not done:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

                if deterministic:
                    action = model.get_deterministic_action(obs_t)
                else:
                    action, _, _, _ = model.get_action_and_value(obs_t)

                action_np = action.cpu().numpy().flatten()

            # Clip to normalized [-1, 1] — wrapper rescales to physical units
            action_np[0] = np.clip(action_np[0], -1.0, 1.0)
            action_np[1] = np.clip(action_np[1], -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            # Record trajectory
            trajectory.append({
                "x": info.get("pose_x", 0.0),
                "y": info.get("pose_y", 0.0),
                "theta": info.get("pose_theta", 0.0),
                "speed": info.get("speed", 0.0),
                "steer": float(action_np[0]),
            })

            lap_reward += reward
            lap_steps += 1
            speeds.append(info.get("speed", 0.0))

            # Render
            try:
                env.render(mode="human")
            except Exception:
                pass

        lap_time = time.time() - lap_start
        sim_lap_time = info.get("lap_time", lap_time)
        if isinstance(sim_lap_time, (list, np.ndarray)):
            sim_lap_time = float(sim_lap_time[0]) if len(sim_lap_time) > 0 else lap_time

        avg_speed = np.mean(speeds) if speeds else 0.0
        collision = info.get("collision", False)

        lap_results.append({
            "lap": lap_idx + 1,
            "lap_time": sim_lap_time,
            "wall_time": lap_time,
            "reward": lap_reward,
            "steps": lap_steps,
            "avg_speed": avg_speed,
            "collision": collision,
        })
        all_trajectories.append(trajectory)

        status = "💥 CRASH" if collision else "✅ CLEAN"
        print(
            f"  Lap {lap_idx+1}/{num_laps} | "
            f"Time: {sim_lap_time:7.3f}s | "
            f"Avg Speed: {avg_speed:5.2f} m/s | "
            f"Reward: {lap_reward:8.2f} | "
            f"Steps: {lap_steps:5d} | "
            f"{status}"
        )

        # ── Export trajectory CSV ─────────────────────────────────────
        csv_path = os.path.join(
            eval_cfg["trajectory_dir"],
            f"trajectory_lap{lap_idx+1}.csv",
        )
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["x", "y", "theta", "speed", "steer"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for point in trajectory:
                writer.writerow(point)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*60}")

    clean_laps = [r for r in lap_results if not r["collision"]]
    if clean_laps:
        fastest = min(clean_laps, key=lambda r: r["lap_time"])
        slowest = max(clean_laps, key=lambda r: r["lap_time"])
        avg_time = np.mean([r["lap_time"] for r in clean_laps])
        avg_speed_all = np.mean([r["avg_speed"] for r in clean_laps])

        print(f"  🏆 Fastest Lap:  {fastest['lap_time']:.3f}s  (Lap {fastest['lap']})")
        print(f"  🐢 Slowest Lap:  {slowest['lap_time']:.3f}s  (Lap {slowest['lap']})")
        print(f"  📊 Average Time: {avg_time:.3f}s")
        print(f"  ⚡ Avg Speed:    {avg_speed_all:.2f} m/s")
    else:
        print("  ⚠️  No clean laps completed.")

    crashes = sum(1 for r in lap_results if r["collision"])
    print(f"  💥 Crashes:      {crashes}/{num_laps}")
    print(f"\n  📁 Trajectories saved to: {eval_cfg['trajectory_dir']}")
    print(f"{'='*60}\n")

    env.close()
    return lap_results


# =========================================================================
#  Entry Point
# =========================================================================
if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    model_file = sys.argv[2] if len(sys.argv) > 2 else None
    evaluate(config_file, model_file)
