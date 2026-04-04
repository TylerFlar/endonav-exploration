"""
DINOv2 feature extraction and VLAD aggregation.

Ported from AnyLoc (https://github.com/AnyLoc/AnyLoc) demo/utilities.py
Original authors: Keetha et al., "AnyLoc: Towards Universal Visual Place Recognition"
License: BSD-3-Clause

Adapted for kidney endoscope place recognition with ViT-S/14.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class DinoV2ExtractFeatures:
    """Extract per-patch features from a specific DINOv2 layer and facet."""

    def __init__(
        self,
        dino_model: str = "dinov2_vits14",
        layer: int = 9,
        facet: str = "value",
        use_cls: bool = False,
        norm_descs: bool = True,
        device: str = "cpu",
    ):
        self.device = device
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        self.layer = layer
        self.facet = facet

        # Load model
        self.model = torch.hub.load(
            "facebookresearch/dinov2", dino_model, verbose=False
        )
        self.model = self.model.eval().to(device)

        # Feature dimension from the model
        self.desc_dim = self.model.embed_dim

        # Number of attention heads
        self.num_heads = self.model.num_heads

        # Hook storage
        self._features = None
        self._hook = None
        self._register_hook()

    def _register_hook(self):
        """Register a forward hook on the target layer to extract features."""
        target_block = self.model.blocks[self.layer]

        if self.facet == "token":
            # Hook on the full block output
            def hook_fn(module, input, output):
                self._features = output

            self._hook = target_block.register_forward_hook(hook_fn)

        elif self.facet in ("query", "key", "value"):
            # Hook on the attention module
            attn_module = target_block.attn

            def hook_fn(module, input, output):
                # input[0] is the input tensor to attention
                B, N, C = input[0].shape
                qkv = module.qkv(input[0])
                qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
                facet_idx = {"query": 0, "key": 1, "value": 2}[self.facet]
                feat = qkv[facet_idx]  # (B, heads, N, head_dim)
                feat = feat.permute(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
                self._features = feat

            self._hook = attn_module.register_forward_hook(hook_fn)
        else:
            raise ValueError(f"Unknown facet: {self.facet}")

    @torch.no_grad()
    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features.

        Args:
            img_tensor: (B, 3, H, W) normalized tensor. H, W must be multiples of 14.

        Returns:
            (B, num_patches, desc_dim) features
        """
        img_tensor = img_tensor.to(self.device)
        _ = self.model(img_tensor)

        features = self._features
        if not self.use_cls:
            # Remove CLS token (first token)
            features = features[:, 1:, :]

        if self.norm_descs:
            features = F.normalize(features, p=2, dim=-1)

        return features


class VLAD:
    """
    Vector of Locally Aggregated Descriptors.

    Builds a visual vocabulary via KMeans, then aggregates patch descriptors
    into a single compact global descriptor per image.
    """

    def __init__(
        self,
        num_clusters: int = 32,
        desc_dim: int | None = None,
        intra_norm: bool = True,
        norm_descs: bool = True,
    ):
        self.num_clusters = num_clusters
        self.desc_dim = desc_dim
        self.intra_norm = intra_norm
        self.norm_descs = norm_descs
        self.cluster_centers: np.ndarray | None = None

    @property
    def vlad_dim(self) -> int:
        """Dimension of the output VLAD descriptor."""
        assert self.desc_dim is not None
        return self.num_clusters * self.desc_dim

    def fit(self, descriptors: list[np.ndarray]) -> None:
        """
        Fit KMeans vocabulary on a collection of patch descriptors.

        Args:
            descriptors: list of (num_patches, desc_dim) arrays
        """
        all_descs = np.concatenate(descriptors, axis=0)
        self.desc_dim = all_descs.shape[1]

        # Subsample if too many descriptors (>500k)
        if all_descs.shape[0] > 500_000:
            indices = np.random.choice(
                all_descs.shape[0], 500_000, replace=False
            )
            all_descs = all_descs[indices]

        kmeans = KMeans(
            n_clusters=self.num_clusters,
            random_state=42,
            n_init=3,
            max_iter=100,
        )
        kmeans.fit(all_descs)
        self.cluster_centers = kmeans.cluster_centers_  # (num_clusters, desc_dim)

    def generate(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Generate a VLAD descriptor from patch descriptors of a single image.

        Args:
            descriptors: (num_patches, desc_dim) array

        Returns:
            (vlad_dim,) L2-normalized VLAD vector
        """
        assert self.cluster_centers is not None, "Must call fit() first"

        # Assign each descriptor to nearest cluster
        # (num_patches, num_clusters) distance matrix
        dists = np.linalg.norm(
            descriptors[:, np.newaxis, :] - self.cluster_centers[np.newaxis, :, :],
            axis=2,
        )
        assignments = dists.argmin(axis=1)  # (num_patches,)

        # Compute residuals per cluster
        vlad = np.zeros(
            (self.num_clusters, self.desc_dim), dtype=np.float32
        )
        for k in range(self.num_clusters):
            mask = assignments == k
            if mask.sum() > 0:
                residuals = descriptors[mask] - self.cluster_centers[k]
                vlad[k] = residuals.sum(axis=0)

        # Intra-normalization: L2-normalize each cluster's residual
        if self.intra_norm:
            for k in range(self.num_clusters):
                norm = np.linalg.norm(vlad[k])
                if norm > 1e-6:
                    vlad[k] /= norm

        # Flatten and L2-normalize
        vlad_flat = vlad.flatten()
        if self.norm_descs:
            norm = np.linalg.norm(vlad_flat)
            if norm > 1e-6:
                vlad_flat /= norm

        return vlad_flat
