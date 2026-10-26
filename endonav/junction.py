"""Junction detection using depth-gated brightness analysis.

Depth estimation identifies open areas (large deep regions).
Within those areas, brightness analysis counts distinct branch openings
by finding dark sub-regions separated by brighter tissue ridges.
"""

import dataclasses
import enum
from collections import deque
import cv2
import numpy as np
from .config import Config


class LumenType(enum.Enum):
    SINGLE_LUMEN = "SINGLE_LUMEN"
    JUNCTION = "JUNCTION"
    DEAD_END = "DEAD_END"


@dataclasses.dataclass
class BlobInfo:
    centroid: tuple[float, float]  # (x, y)
    area: int
    mean_intensity: float
    contour: np.ndarray


@dataclasses.dataclass
class JunctionResult:
    raw_type: LumenType
    confirmed_type: LumenType
    blob_count: int
    blobs: list[BlobInfo]
    dark_mask: np.ndarray
    depth_open_fraction: float  # fraction of image that is "deep" (open area)
    max_depth: float  # maximum depth in valid region


class JunctionDetector:
    def __init__(self, config: Config):
        self.config = config
        self._border_mask: np.ndarray | None = None
        self._bright_threshold: float | None = None
        self._history: deque[LumenType] = deque(
            maxlen=max(config.junction_confirm_frames, config.dead_end_confirm_frames)
        )

    def calibrate(self, frames_gray: list[np.ndarray]) -> None:
        """Build border mask and calibrate brightness threshold."""
        if not frames_gray:
            return

        # Border mask: pixels consistently near-black (<10) in >80% of frames
        h, w = frames_gray[0].shape
        dark_count = np.zeros((h, w), dtype=np.float32)
        for f in frames_gray:
            dark_count += (f < 10).astype(np.float32)
        border_ratio = dark_count / len(frames_gray)
        self._border_mask = (border_ratio < 0.8).astype(np.uint8) * 255
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._border_mask = cv2.erode(self._border_mask, erode_kernel, iterations=1)

        # Brightness threshold: mean of valid pixels across calibration frames
        valid_means = []
        for f in frames_gray:
            valid = f[self._border_mask > 0]
            if len(valid) > 0:
                valid_means.append(float(valid.mean()))
        if valid_means:
            base = float(np.percentile(valid_means, 40))
            self._bright_threshold = base - self.config.junction_brightness_offset
        else:
            self._bright_threshold = 60.0

    def process(
        self, gray: np.ndarray, depth_map: np.ndarray | None = None
    ) -> JunctionResult:
        if self._bright_threshold is None:
            self._bright_threshold = float(gray.mean()) * 0.4

        # Compute depth statistics within valid region
        if depth_map is not None and self._border_mask is not None:
            valid_depth = depth_map[self._border_mask > 0]
        elif depth_map is not None:
            valid_depth = depth_map.flatten()
        else:
            valid_depth = np.array([0.5])

        max_depth = float(valid_depth.max()) if len(valid_depth) > 0 else 0.0

        # Depth gate: check if there's a large open area
        if depth_map is not None:
            deep_thresh = np.percentile(valid_depth, self.config.depth_deep_percentile)
            deep_mask = (depth_map > deep_thresh).astype(np.uint8) * 255
            if self._border_mask is not None:
                deep_mask = cv2.bitwise_and(deep_mask, self._border_mask)
            valid_area = self._border_mask.sum() / 255 if self._border_mask is not None else gray.size
            deep_fraction = float(deep_mask.sum() / 255) / max(valid_area, 1)
        else:
            deep_fraction = 0.5  # assume open if no depth

        # Always run branch detection (needed for dead end detection too)
        blob_count, blobs, dark_mask = self._find_branches(gray, depth_map)

        # Classify
        min_branches = self.config.expected_graph.min_branch_count
        if (
            blob_count >= min_branches
            and deep_fraction >= self.config.depth_open_area_threshold
        ):
            # Junction: 3+ blobs AND depth gate is open
            areas = sorted([b.area for b in blobs], reverse=True)
            ratio = areas[0] / max(areas[-1], 1)
            if ratio <= self.config.max_blob_ratio:
                raw_type = LumenType.JUNCTION
            else:
                raw_type = LumenType.SINGLE_LUMEN
        elif blob_count == 0:
            # Dead end: no branch openings found at the tight threshold.
            # Temporal filter (8 consecutive frames) prevents false triggers.
            raw_type = LumenType.DEAD_END
        else:
            raw_type = LumenType.SINGLE_LUMEN

        confirmed_type = self._temporal_filter(raw_type)

        return JunctionResult(
            raw_type=raw_type,
            confirmed_type=confirmed_type,
            blob_count=blob_count,
            blobs=blobs,
            dark_mask=dark_mask,
            depth_open_fraction=deep_fraction,
            max_depth=max_depth,
        )

    def _find_branches(
        self, gray: np.ndarray, depth_map: np.ndarray | None
    ) -> tuple[int, list[BlobInfo], np.ndarray]:
        """Find distinct branch openings using brightness within the deep region."""
        cfg = self.config

        # Use tight threshold to separate branch openings
        blurred = cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), cfg.blur_sigma)
        _, mask = cv2.threshold(
            blurred, self._bright_threshold, 255, cv2.THRESH_BINARY_INV
        )

        # Apply border mask
        if self._border_mask is not None:
            mask = cv2.bitwise_and(mask, self._border_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, cfg.morph_open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_open_iter)

        # Find and filter blobs
        image_area = gray.shape[0] * gray.shape[1]
        min_area = int(image_area * cfg.min_blob_fraction)
        max_area = int(image_area * cfg.max_blob_fraction)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        blobs = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue
            cx, cy = centroids[i]
            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = contours[0] if contours else np.array([])
            blob_pixels = gray[labels == i]
            mean_int = float(blob_pixels.mean()) if len(blob_pixels) > 0 else 0.0
            blobs.append(BlobInfo(
                centroid=(float(cx), float(cy)),
                area=int(area),
                mean_intensity=mean_int,
                contour=contour,
            ))

        blobs.sort(key=lambda b: b.area, reverse=True)
        return len(blobs), blobs, mask

    def _temporal_filter(self, raw_type: LumenType) -> LumenType:
        self._history.append(raw_type)
        cfg = self.config
        if len(self._history) >= cfg.junction_confirm_frames:
            recent = list(self._history)[-cfg.junction_confirm_frames:]
            if all(t == LumenType.JUNCTION for t in recent):
                return LumenType.JUNCTION
        if len(self._history) >= cfg.dead_end_confirm_frames:
            recent = list(self._history)[-cfg.dead_end_confirm_frames:]
            if all(t == LumenType.DEAD_END for t in recent):
                return LumenType.DEAD_END
        return LumenType.SINGLE_LUMEN
