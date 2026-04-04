"""Proximity detection using depth estimation.

Replaces brightness-based danger detection with depth-based analysis.
Close tissue = low depth values = dangerous.
"""

import dataclasses
import cv2
import numpy as np
from .config import Config


@dataclasses.dataclass
class ProximityResult:
    danger_score: float  # 0-1
    danger_direction: np.ndarray  # (2,) unit vector toward closest region
    min_depth: float  # closest point depth (0-1)
    mean_depth: float  # average depth
    depth_gradient: float  # rate of change (negative = approaching)
    is_dangerous: bool
    grid_depths: np.ndarray  # (3,3) mean depth per cell


class ProximityDetector:
    def __init__(self, config: Config):
        self.config = config
        self.g = 3  # grid size
        self.border_mask: np.ndarray | None = None
        self._prev_min_depth: float | None = None

    def process(
        self,
        depth_map: np.ndarray,
        frame_bgr: np.ndarray | None = None,
    ) -> ProximityResult:
        """Compute proximity danger from depth map.

        Args:
            depth_map: (H, W) float32, normalized [0,1]. Higher = farther.
        """
        h, w = depth_map.shape
        g = self.g
        cell_h, cell_w = h // g, w // g

        # Grid depths
        grid = np.zeros((g, g), dtype=np.float64)
        for r in range(g):
            for c in range(g):
                cell = depth_map[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
                if self.border_mask is not None:
                    cell_mask = self.border_mask[
                        r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w
                    ]
                    valid = cell[cell_mask > 0]
                    grid[r, c] = float(valid.mean()) if len(valid) > 0 else 1.0
                else:
                    grid[r, c] = float(cell.mean())

        # Min and mean depth (within valid region)
        if self.border_mask is not None:
            valid_depth = depth_map[self.border_mask > 0]
        else:
            valid_depth = depth_map.flatten()

        if len(valid_depth) == 0:
            min_depth, mean_depth = 1.0, 1.0
        else:
            # Use 5th percentile as "min" to avoid noise
            min_depth = float(np.percentile(valid_depth, 5))
            mean_depth = float(valid_depth.mean())

        # Danger score: how close is the closest tissue
        safety = self.config.safety_min_depth
        if min_depth < safety:
            proximity_danger = 1.0 - (min_depth / safety)
        else:
            proximity_danger = 0.0
        proximity_danger = min(max(proximity_danger, 0.0), 1.0)

        # Depth gradient (approach speed)
        if self._prev_min_depth is not None:
            depth_gradient = min_depth - self._prev_min_depth  # negative = approaching
        else:
            depth_gradient = 0.0
        self._prev_min_depth = min_depth

        # Approach speed contribution to danger
        approach_danger = max(0.0, -depth_gradient * 10)  # scale up small gradients
        approach_danger = min(approach_danger, 1.0)

        # Combined danger score
        w_approach = self.config.approach_speed_weight
        danger_score = (1 - w_approach) * proximity_danger + w_approach * approach_danger
        danger_score = min(max(danger_score, 0.0), 1.0)

        is_dangerous = danger_score > self.config.danger_threshold

        # Danger direction: toward shallowest (closest) grid cell
        min_cell = np.unravel_index(grid.argmin(), grid.shape)
        cy = (2 * min_cell[0] + 1) / g - 1.0
        cx = (2 * min_cell[1] + 1) / g - 1.0
        direction = np.array([cx, cy])
        norm = np.linalg.norm(direction)
        if norm > 1e-6:
            direction /= norm
        else:
            direction = np.array([0.0, 0.0])

        return ProximityResult(
            danger_score=danger_score,
            danger_direction=direction,
            min_depth=min_depth,
            mean_depth=mean_depth,
            depth_gradient=depth_gradient,
            is_dangerous=is_dangerous,
            grid_depths=grid,
        )
