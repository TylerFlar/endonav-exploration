"""Monocular depth estimation using Depth Anything v3.

Uses the DA3 model from ByteDance-Seed/Depth-Anything-3 cloned into third_party/.
"""

import sys
import os
import cv2
import numpy as np
import torch
from .config import Config


class DepthEstimator:
    """Depth Anything v3 wrapper for per-frame depth estimation."""

    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device
        self._model = None
        self._load_model()

    def _load_model(self):
        # Add DA3 source to path
        da3_src = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "third_party", "depth_anything_3", "src"
        )
        if da3_src not in sys.path:
            sys.path.insert(0, da3_src)

        import logging
        logging.getLogger("depth_anything_3").setLevel(logging.WARNING)

        from depth_anything_3.api import DepthAnything3

        model_id = self.config.depth_model
        print(f"  Loading depth model ({model_id}) on {self.device}...")
        self._model = DepthAnything3.from_pretrained(model_id).to(self.device)

    @torch.no_grad()
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Estimate depth from a BGR frame.

        Returns:
            depth_map: (H, W) float32 array, normalized to [0, 1].
                       Higher values = farther from camera.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        pred = self._model.inference(image=[rgb], process_res=504)
        depth_np = pred.depth[0]

        # Resize to frame size if needed
        if depth_np.shape[0] != h or depth_np.shape[1] != w:
            depth_np = cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.zeros_like(depth_np)

        return depth_np.astype(np.float32)
