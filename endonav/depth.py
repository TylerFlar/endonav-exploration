"""Monocular depth estimation using Depth Anything v2."""

import cv2
import numpy as np
import torch
from PIL import Image
from .config import Config


class DepthEstimator:
    """Wrapper around Depth Anything v2 for per-frame depth estimation."""

    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_name = self.config.depth_model
        print(f"  Loading depth model ({model_name}) on {self.device}...")
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self._model = self._model.to(self.device).eval()

    @torch.no_grad()
    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Estimate depth from a BGR frame.

        Returns:
            depth_map: (H, W) float32 array. Higher values = farther from camera.
                       Values are relative (not metric), normalized to [0, 1].
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)

        # Interpolate to original frame size
        depth = outputs.predicted_depth[0]
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        depth_np = depth.cpu().numpy()

        # Normalize to [0, 1]
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.zeros_like(depth_np)

        return depth_np.astype(np.float32)
