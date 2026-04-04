from typing import Iterator
import cv2
import numpy as np
from .config import Config


class VideoLoader:
    def __init__(self, video_path: str, config: Config):
        self.video_path = video_path
        self.config = config
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._orig_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._orig_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def original_resolution(self) -> tuple[int, int]:
        return (self._orig_w, self._orig_h)

    @property
    def target_resolution(self) -> tuple[int, int]:
        return (self.config.target_width, self.config.target_height)

    def frames(self, skip: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame_index, bgr_frame) resized to target resolution."""
        self.reset()
        tw, th = self.config.target_width, self.config.target_height
        idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            if idx % skip == 0:
                if frame.shape[1] != tw or frame.shape[0] != th:
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
                yield idx, frame
            idx += 1

    def reset(self):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __del__(self):
        if self._cap.isOpened():
            self._cap.release()
