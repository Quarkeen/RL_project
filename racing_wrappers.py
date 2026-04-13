"""
=============================================================================
 F1TENTH PPO Racing Agent — Perception & Logic Wrapper
 Transforms raw f1tenth_gym observations into a compact, normalized state
 vector suitable for policy learning.
=============================================================================

Observation Vector Layout (110-dimensional):
  [0:108]  → Downsampled & normalized LiDAR  (108 beams)
  [108]    → Normalized longitudinal velocity  (vx / v_max)
  [109]    → Normalized steering angle          (δ  / δ_max)

Action Space (2-dimensional, continuous):
  [0]      → Desired steering angle  [-max_steer, +max_steer]
  [1]      → Desired speed            [0, max_speed]

  The speed is passed directly to f1tenth_gym's internal PID controller
  which computes the appropriate acceleration/braking to reach the
  requested velocity. This means the agent can command full speed
  instantly on any timestep.

Reward Function:
  R_t = speed_weight · (v / v_max) · cos(Δθ)  — normalized speed along heading
      + progress_weight · progress             — centerline arc-length gain
      + collision_penalty · collision           — hard crash penalty
      + steer_change_penalty · |δ_t − δ_{t-1}| — steering smoothness
"""

import numpy as np
import gym
from gym import spaces


class F1TENTH_Wrapper:
    """
    Wrapper for the F1TENTH racing environment (gym 0.19 API).

    Transforms the raw Dict observation into a compact float32 vector and
    provides a shaped reward signal tuned for high-speed racing.

    Does NOT subclass gymnasium.Wrapper since f1tenth_gym uses the legacy
    OpenAI Gym API. Instead, it delegates to the underlying env directly.

    Parameters
    ----------
    env : gym.Env
        The base f1tenth_gym environment instance.
    config : dict
        Configuration dictionary (loaded from config.yaml).
    centerline : np.ndarray or None
        (N, 2) array of track centerline waypoints for progress calculation.
        If None, progress reward term is disabled.
    """

    def __init__(self, env, config: dict, centerline: np.ndarray = None):
        self.env = env

        # ── Configuration ─────────────────────────────────────────────────
        wrapper_cfg = config.get("wrapper", {})
        reward_cfg = config.get("reward", {})
        env_cfg = config.get("env", {})

        self.downsample_factor = wrapper_cfg.get("downsample_factor", 10)
        self.downsample_mode = wrapper_cfg.get("downsample_mode", "min")
        self.normalize_lidar = wrapper_cfg.get("normalize_lidar", True)
        self.velocity_scale = wrapper_cfg.get("velocity_scale", 20.0)
        self.steer_scale = wrapper_cfg.get("steer_scale", 0.4189)
        self.lidar_max_range = env_cfg.get("lidar_max_range", 30.0)

        # Reward weights
        self.speed_weight = reward_cfg.get("speed_weight", 1.0)
        self.progress_weight = reward_cfg.get("progress_weight", 0.1)
        self.collision_penalty = reward_cfg.get("collision_penalty", -10.0)
        self.steer_change_penalty = reward_cfg.get("steer_change_penalty", -0.01)

        # Action limits — agent outputs [steer, speed] directly
        self.max_steer = env_cfg.get("max_steer", 0.4189)
        self.max_speed = env_cfg.get("max_speed", 20.0)

        # Episode truncation
        self.max_episode_steps = env_cfg.get("max_episode_steps", 2000)
        self.episode_step_count = 0

        # ── Derived constants ─────────────────────────────────────────────
        raw_beams = env_cfg.get("num_beams", 1080)
        self.num_downsampled = raw_beams // self.downsample_factor  # 108
        self.obs_dim = self.num_downsampled + 2  # lidar + vx + steer (no vy)

        # ── Observation & Action spaces ───────────────────────────────────
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # Normalized action space: both dimensions in [-1, 1]
        # The wrapper rescales to physical units:
        #   steer = action[0] * max_steer
        #   speed = (action[1] + 1) / 2 * max_speed  →  mean=0 gives max_speed/2
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ── State tracking ────────────────────────────────────────────────
        self.prev_steer = 0.0
        self.prev_theta = 0.0
        self.episode_sim_time = 0.0

        # ── Centerline for progress computation ──────────────────────────
        self.centerline = centerline
        self.centerline_cumlen = None
        self.prev_s = 0.0
        if centerline is not None:
            self._precompute_centerline()

    # =====================================================================
    #  Centerline utilities
    # =====================================================================
    def _precompute_centerline(self):
        """Pre-compute cumulative arc-length along centerline waypoints."""
        diffs = np.diff(self.centerline, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        self.centerline_cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
        self.track_length = self.centerline_cumlen[-1]

    def _project_to_centerline(self, x: float, y: float) -> float:
        """
        Project (x, y) onto the nearest centerline segment and return
        the Frenet s-coordinate (cumulative arc-length).
        """
        if self.centerline is None:
            return 0.0

        pos = np.array([x, y])
        # Vector from each waypoint to the query point
        vecs = pos - self.centerline[:-1]
        segs = np.diff(self.centerline, axis=0)
        seg_lens_sq = np.sum(segs ** 2, axis=1) + 1e-8

        # Parameterized projection t ∈ [0, 1] onto each segment
        t = np.clip(np.sum(vecs * segs, axis=1) / seg_lens_sq, 0.0, 1.0)
        projections = self.centerline[:-1] + t[:, None] * segs
        dists = np.linalg.norm(pos - projections, axis=1)

        nearest_idx = np.argmin(dists)
        s = self.centerline_cumlen[nearest_idx] + t[nearest_idx] * np.sqrt(seg_lens_sq[nearest_idx])
        return s

    # =====================================================================
    #  LiDAR processing pipeline
    # =====================================================================
    def _process_lidar(self, raw_scan: np.ndarray) -> np.ndarray:
        """
        Downsample raw 1080-point scan to 108 beams via min-pooling,
        then normalize to [0, 1].

        Min-pooling ensures we retain the *closest* obstacle in each
        angular window — a conservative (safe) strategy.
        """
        n = len(raw_scan)
        k = self.downsample_factor

        # Trim if not perfectly divisible
        trimmed = raw_scan[: n - (n % k)]
        reshaped = trimmed.reshape(-1, k)

        if self.downsample_mode == "min":
            downsampled = reshaped.min(axis=1)
        elif self.downsample_mode == "mean":
            downsampled = reshaped.mean(axis=1)
        else:
            downsampled = reshaped.min(axis=1)

        if self.normalize_lidar:
            downsampled = np.clip(downsampled / self.lidar_max_range, 0.0, 1.0)

        return downsampled.astype(np.float32)

    # =====================================================================
    #  Observation builder
    # =====================================================================
    def _build_obs(self, raw_obs: dict) -> np.ndarray:
        """Fuse LiDAR + proprioception into a flat observation vector."""
        # LiDAR
        scan = raw_obs["scans"][0]
        lidar_features = self._process_lidar(scan)

        # Proprioception (normalized)
        # Note: f1tenth_gym hardcodes linear_vels_y to 0, so we omit it
        vx = raw_obs["linear_vels_x"][0] / self.velocity_scale
        steer = self.prev_steer / self.steer_scale  # most recent command

        proprio = np.array([vx, steer], dtype=np.float32)
        obs = np.concatenate([lidar_features, proprio])

        return obs

    # =====================================================================
    #  Reward shaping
    # =====================================================================
    def _compute_reward(self, raw_obs: dict, collision: bool) -> float:
        """
        Weighted reward:
          R_t = speed_weight · (v · cos(Δθ))
              + progress_weight · Δs
              + collision_penalty · collision
              + steer_change_penalty · |δ_t − δ_{t-1}|
        """
        # ── Speed component (velocity aligned with heading change) ──────
        vx = raw_obs["linear_vels_x"][0]
        # Use vx directly (NOT abs) — backwards driving gives NEGATIVE reward
        v_normalized = vx / self.max_speed  # in [-1, 1]

        theta = raw_obs["poses_theta"][0]
        delta_theta = self._angle_diff(theta, self.prev_theta)
        speed_reward = self.speed_weight * v_normalized * np.cos(delta_theta)

        # ── Backward penalty — explicit discouragement of reverse motion ─
        backward_penalty = -0.5 if vx < -0.1 else 0.0

        # ── Progress component (Frenet arc-length gain) ─────────────────
        progress_reward = 0.0
        if self.centerline is not None:
            x = raw_obs["poses_x"][0]
            y = raw_obs["poses_y"][0]
            current_s = self._project_to_centerline(x, y)
            delta_s = current_s - self.prev_s

            # Handle wrap-around at start/finish line
            if delta_s < -self.track_length * 0.5:
                delta_s += self.track_length
            elif delta_s > self.track_length * 0.5:
                delta_s -= self.track_length

            progress_reward = self.progress_weight * delta_s
            self.prev_s = current_s

        # ── Collision penalty ───────────────────────────────────────────
        collision_reward = self.collision_penalty if collision else 0.0

        # ── Steering smoothness ─────────────────────────────────────────
        steer_penalty = self.steer_change_penalty * abs(
            self.prev_steer - self._last_applied_steer
        )

        reward = speed_reward + progress_reward + collision_reward + steer_penalty + backward_penalty
        return float(reward)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Smallest signed angular difference, wrapped to [-π, π]."""
        diff = a - b
        return (diff + np.pi) % (2 * np.pi) - np.pi

    # =====================================================================
    #  Gymnasium interface
    # =====================================================================
    def reset(self, **kwargs):
        """Reset the environment and return processed observation."""
        start_x = kwargs.pop("start_x", None)
        start_y = kwargs.pop("start_y", None)
        start_theta = kwargs.pop("start_theta", None)

        # f1tenth_gym reset expects poses as np.array([[x, y, θ]])
        if start_x is not None:
            poses = np.array([[start_x, start_y, start_theta]])
        else:
            # Use defaults from config or env
            poses = np.array([[0.0, 0.0, 0.0]])

        raw_obs, reward, done, info = self.env.reset(poses)

        # Reset internal state
        self.prev_steer = 0.0
        self._last_applied_steer = 0.0
        self.prev_theta = raw_obs["poses_theta"][0]
        self.episode_step_count = 0
        self.episode_sim_time = 0.0

        if self.centerline is not None:
            x = raw_obs["poses_x"][0]
            y = raw_obs["poses_y"][0]
            self.prev_s = self._project_to_centerline(x, y)

        obs = self._build_obs(raw_obs)
        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute one step in the environment.

        Parameters
        ----------
        action : np.ndarray, shape (2,)
            [steering_angle, desired_speed]
            The speed is passed directly to f1tenth_gym's PID controller.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        # Rescale normalized actions to physical units
        steer = float(np.clip(action[0], -1.0, 1.0)) * self.max_steer
        speed = float(np.clip(action[1], -1.0, 1.0))
        speed = (speed + 1.0) / 2.0 * self.max_speed  # [-1,1] → [0, max_speed]

        # f1tenth_gym expects [[steer, speed]] — PID handles acceleration
        gym_action = np.array([[steer, speed]])
        raw_obs, step_time, done, info = self.env.step(gym_action)

        # Track episode length and cumulative simulation time
        self.episode_step_count += 1
        self.episode_sim_time += step_time

        # Detect collision from obs
        collision = bool(raw_obs.get("collisions", [0.0])[0])

        # Build reward
        self._last_applied_steer = self.prev_steer
        self.prev_steer = steer
        reward = self._compute_reward(raw_obs, collision)

        # Update heading tracker
        self.prev_theta = raw_obs["poses_theta"][0]

        # Build processed observation
        obs = self._build_obs(raw_obs)

        # Termination: collision or env done
        terminated = done or collision
        # Truncation: max episode length reached
        truncated = (self.episode_step_count >= self.max_episode_steps) and not terminated

        # Enrich info dict
        info["collision"] = collision
        info["speed"] = abs(raw_obs["linear_vels_x"][0])
        info["pose_x"] = raw_obs["poses_x"][0]
        info["pose_y"] = raw_obs["poses_y"][0]
        info["pose_theta"] = raw_obs["poses_theta"][0]
        info["lap_time"] = self.episode_sim_time
        info["lap_count"] = raw_obs.get("lap_counts", [0])[0]

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Delegate rendering to the underlying environment."""
        return self.env.render(mode=mode)

    def close(self):
        """Delegate close to the underlying environment."""
        return self.env.close()


def make_env(config: dict, render_mode: str = None, centerline: np.ndarray = None):
    """
    Factory function to create a wrapped F1TENTH environment.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    render_mode : str or None
        Pass 'human' to enable rendering, None for headless.
    centerline : np.ndarray
        (N, 2) centerline waypoints for progress reward.

    Returns
    -------
    F1TENTH_Wrapper
    """
    try:
        from f110_gym.envs.base_classes import Integrator
    except ImportError:
        Integrator = None

    env_cfg = config["env"]
    integrator_map = {"RK4": Integrator.RK4} if Integrator else {}
    integrator = integrator_map.get(env_cfg.get("integrator", "RK4"), None)

    # Import and register the f1tenth environment
    try:
        import f110_gym  # noqa: F401 — registers the environment
    except ImportError:
        raise ImportError(
            "f1tenth_gym not installed. Run setup_env.sh first."
        )

    env_kwargs = dict(
        map=env_cfg["map_path"],
        map_ext=env_cfg["map_ext"],
        num_agents=env_cfg.get("num_agents", 1),
        timestep=env_cfg.get("timestep", 0.01),
    )
    if integrator is not None:
        env_kwargs["integrator"] = integrator

    env = gym.make("f110_gym:f110-v0", **env_kwargs)

    # ── Register a render callback so the camera follows the ego car ─────
    def render_callback(env_renderer):
        """Update pyglet camera to track the ego vehicle."""
        e = env_renderer
        # Car vertices are stored as flat [x0,y0, x1,y1, ...] in screen coords
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom = max(y), min(y)
        left, right = min(x), max(x)
        # Reposition label and viewport
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

    try:
        env.add_render_callback(render_callback)
    except AttributeError:
        # Fallback: some versions may not support this
        pass

    wrapped = F1TENTH_Wrapper(env, config, centerline=centerline)

    return wrapped


def load_centerline(map_path: str) -> np.ndarray:
    """
    Attempt to load a centerline CSV for progress calculation.
    Expects a file named '{map_name}_centerline.csv' with columns (x, y).
    Falls back to raceline.csv or returns None.
    """
    import os
    import glob

    search_dir = os.path.dirname(map_path) if os.path.isfile(map_path) else map_path
    # map_path might be a base name like ".../maps/berlin" — try parent dir
    if not os.path.isdir(search_dir):
        search_dir = os.path.dirname(map_path)
    patterns = [
        os.path.join(search_dir, "*centerline*"),
        os.path.join(search_dir, "*raceline*"),
        os.path.join(search_dir, "*waypoints*"),
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        for match in matches:
            if match.endswith(".csv"):
                try:
                    data = np.loadtxt(match, delimiter=",", skiprows=1)
                    if data.ndim == 2 and data.shape[1] >= 2:
                        print(f"[Wrapper] Loaded centerline from {match} ({len(data)} pts)")
                        return data[:, :2]
                    elif data.ndim == 2:
                        data = np.loadtxt(match, delimiter=";", skiprows=1)
                        if data.shape[1] >= 2:
                            print(f"[Wrapper] Loaded centerline from {match} ({len(data)} pts)")
                            return data[:, :2]
                except Exception:
                    continue

    print("[Wrapper] No centerline found — progress reward disabled.")
    return None
