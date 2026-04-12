"""
Generate a centerline CSV for the Berlin F1TENTH map.
Uses skeletonization of the free-space to find the track center,
then orders the points into a single closed loop.
"""

import numpy as np
from PIL import Image
import yaml
from scipy import ndimage
from skimage.morphology import skeletonize
import os

def generate_centerline(map_dir):
    # ── Load map metadata ─────────────────────────────────────────────────
    yaml_path = os.path.join(map_dir, "berlin.yaml")
    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)

    resolution = meta['resolution']  # meters per pixel
    origin = meta['origin']  # [x, y, theta]
    origin_x, origin_y = origin[0], origin[1]

    # ── Load and binarize map ─────────────────────────────────────────────
    img_path = os.path.join(map_dir, "berlin.png")
    img = np.array(Image.open(img_path).convert('L'))

    # Free space = white (>200), walls = dark (<50)
    free_space = (img > 200).astype(np.uint8)

    print(f"Map size: {img.shape}, resolution: {resolution} m/px")
    print(f"Origin: ({origin_x}, {origin_y})")
    print(f"Free pixels: {free_space.sum()}")

    # ── Erode to get rough center ─────────────────────────────────────────
    # Erode the free space to pull away from walls, then skeletonize
    eroded = ndimage.binary_erosion(free_space, iterations=5).astype(np.uint8)

    # ── Skeletonize ───────────────────────────────────────────────────────
    skeleton = skeletonize(eroded).astype(np.uint8)
    skel_points = np.argwhere(skeleton)  # (row, col) format

    print(f"Skeleton points: {len(skel_points)}")

    if len(skel_points) == 0:
        raise ValueError("Skeletonization produced no points!")

    # ── Convert pixel coords to world coords ──────────────────────────────
    # Image row 0 = top, but map origin is bottom-left
    # PIL loads top-down, f1tenth_gym flips vertically:
    #   world_x = col * resolution + origin_x
    #   world_y = (height - row) * resolution + origin_y
    height = img.shape[0]
    world_x = skel_points[:, 1] * resolution + origin_x
    world_y = (height - skel_points[:, 0]) * resolution + origin_y

    points = np.column_stack([world_x, world_y])

    # ── Order points into a connected loop (nearest-neighbor) ─────────────
    ordered = _order_points_nn(points)

    # ── Smooth the centerline ─────────────────────────────────────────────
    # Apply a moving average to remove jaggedness
    window = 15
    smoothed = np.zeros_like(ordered)
    n = len(ordered)
    for i in range(n):
        indices = [(i + j) % n for j in range(-window//2, window//2 + 1)]
        smoothed[i] = ordered[indices].mean(axis=0)

    # ── Subsample to reasonable density ───────────────────────────────────
    # Keep every Nth point to get ~500 waypoints
    step = max(1, len(smoothed) // 500)
    final = smoothed[::step]

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(map_dir, "berlin_centerline.csv")
    np.savetxt(out_path, final, delimiter=',', header='x,y', comments='')
    print(f"Saved {len(final)} centerline points to {out_path}")

    return final


def _order_points_nn(points):
    """
    Order an unordered point cloud into a closed loop using
    nearest-neighbor traversal.
    """
    from scipy.spatial import KDTree

    tree = KDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    ordered = []

    # Start from the point closest to the known start position
    start_pos = np.array([-0.8, 0.03])  # Berlin start pose
    _, start_idx = tree.query(start_pos)

    current = start_idx
    for _ in range(len(points)):
        ordered.append(points[current])
        visited[current] = True

        # Find nearest unvisited neighbor
        distances, indices = tree.query(points[current], k=min(20, len(points)))
        found_next = False
        for d, idx in zip(distances, indices):
            if not visited[idx]:
                current = idx
                found_next = True
                break

        if not found_next:
            break

    return np.array(ordered)


if __name__ == "__main__":
    map_dir = "/home/quark/coding/RL_project/f1tenth_gym/gym/f110_gym/envs/maps"
    centerline = generate_centerline(map_dir)
    print(f"\nCenterline shape: {centerline.shape}")
    print(f"X range: [{centerline[:,0].min():.2f}, {centerline[:,0].max():.2f}]")
    print(f"Y range: [{centerline[:,1].min():.2f}, {centerline[:,1].max():.2f}]")
