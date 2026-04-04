import dataclasses
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from .config import Config
from .vlad import DinoV2ExtractFeatures, VLAD


@dataclasses.dataclass
class PlaceRecognitionResult:
    descriptor: np.ndarray  # VLAD or HSV descriptor
    max_similarity: float
    matched_keyframe_idx: int | None
    is_revisit: bool
    is_new_keyframe: bool
    method: str  # "dino_vlad" or "hsv_histogram"


class PlaceRecognizer:
    def __init__(self, config: Config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.method = "dino_vlad" if config.use_dino else "hsv_histogram"

        # Keyframe buffer: list of (frame_idx, descriptor)
        self.keyframes: list[tuple[int, np.ndarray]] = []

        # Cached descriptor for frame-skip optimization
        self._last_descriptor: np.ndarray | None = None
        self._last_method: str = self.method

        # DINOv2 + VLAD components
        self._dino: DinoV2ExtractFeatures | None = None
        self._vlad: VLAD | None = None
        self._transform: T.Compose | None = None

        if config.use_dino:
            self._init_dino()

    def _init_dino(self):
        cfg = self.config
        print(f"  Loading DINOv2 ({cfg.dino_model}, layer={cfg.dino_layer}, "
              f"facet={cfg.dino_facet}) on {self.device}...")
        self._dino = DinoV2ExtractFeatures(
            dino_model=cfg.dino_model,
            layer=cfg.dino_layer,
            facet=cfg.dino_facet,
            use_cls=False,
            norm_descs=True,
            device=self.device,
        )
        self._vlad = VLAD(
            num_clusters=cfg.vlad_num_clusters,
            intra_norm=True,
            norm_descs=True,
        )
        # DINOv2 expects ImageNet-normalized input, size must be multiple of 14
        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((cfg.dino_input_size, cfg.dino_input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def fit_vocabulary(self, frames_bgr: list[np.ndarray]) -> None:
        """Pass 1: Extract DINOv2 patch features from sampled frames, fit VLAD clusters."""
        if not self.config.use_dino:
            return

        print(f"  Extracting DINOv2 features from {len(frames_bgr)} sampled frames...")
        all_patch_descs = []
        for i, frame in enumerate(frames_bgr):
            patch_feats = self._extract_dino_patches(frame)
            all_patch_descs.append(patch_feats)
            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(frames_bgr)} frames processed")

        print(f"  Fitting VLAD vocabulary ({self.config.vlad_num_clusters} clusters)...")
        self._vlad.fit(all_patch_descs)
        print(f"  VLAD vocabulary ready. Descriptor dim = {self._vlad.vlad_dim}")

    def process(self, frame_bgr: np.ndarray, frame_idx: int) -> PlaceRecognitionResult:
        """Generate descriptor, compare to keyframe buffer, maybe store.

        DINOv2 is only run every `place_recognition_interval` frames.
        On skipped frames, reuse the last descriptor.
        """
        interval = self.config.place_recognition_interval
        should_compute = (frame_idx % interval == 0) or self._last_descriptor is None

        if should_compute:
            if self.config.use_dino and self._vlad is not None and self._vlad.cluster_centers is not None:
                descriptor = self._extract_dino_descriptor(frame_bgr)
                method = "dino_vlad"
            else:
                descriptor = self._extract_hsv_descriptor(frame_bgr)
                method = "hsv_histogram"
            self._last_descriptor = descriptor
            self._last_method = method
        else:
            descriptor = self._last_descriptor
            method = self._last_method

        max_sim, matched_idx = self._compare_to_keyframes(descriptor)
        is_revisit = max_sim >= self.config.revisit_threshold
        is_new_keyframe = max_sim < self.config.keyframe_novelty_threshold

        if is_new_keyframe:
            if len(self.keyframes) >= self.config.max_keyframes:
                # Remove oldest keyframe
                self.keyframes.pop(0)
            self.keyframes.append((frame_idx, descriptor))

        return PlaceRecognitionResult(
            descriptor=descriptor,
            max_similarity=max_sim,
            matched_keyframe_idx=matched_idx,
            is_revisit=is_revisit,
            is_new_keyframe=is_new_keyframe,
            method=method,
        )

    def _extract_dino_patches(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR frame -> DINOv2 patch features as numpy array."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0)  # (1, 3, H, W)
        features = self._dino(tensor)  # (1, num_patches, desc_dim)
        return features[0].cpu().numpy()  # (num_patches, desc_dim)

    def _extract_dino_descriptor(self, frame_bgr: np.ndarray) -> np.ndarray:
        """BGR frame -> DINOv2 patches -> VLAD vector."""
        patches = self._extract_dino_patches(frame_bgr)
        return self._vlad.generate(patches)

    def _extract_hsv_descriptor(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Fallback: concatenated HSV histogram, L2-normalized."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256]).flatten()
        descriptor = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        norm = np.linalg.norm(descriptor)
        if norm > 1e-6:
            descriptor /= norm
        return descriptor

    def _compare_to_keyframes(
        self, descriptor: np.ndarray
    ) -> tuple[float, int | None]:
        """Cosine similarity against keyframe buffer."""
        if not self.keyframes:
            return 0.0, None

        max_sim = -1.0
        matched_idx = None
        for kf_idx, (frame_idx, kf_desc) in enumerate(self.keyframes):
            sim = float(np.dot(descriptor, kf_desc))
            # Both descriptors are L2-normalized, so dot = cosine similarity
            if sim > max_sim:
                max_sim = sim
                matched_idx = kf_idx

        return max_sim, matched_idx
