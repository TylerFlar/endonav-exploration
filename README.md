# endonav-exploration

Reactive kidney endoscope exploration system that navigates a kidney's collecting system (tree-like structure: renal pelvis -> major calyces -> minor calyces) to find and zap kidney stones. No global SLAM or 3D reconstruction -- purely reactive with topological navigation.

## Architecture

The pipeline runs 4 modules per-frame on endoscope video:

1. **Depth Estimation** (`depth.py`) -- Depth Anything v2 produces per-frame depth maps. Higher values = farther from camera.

2. **Proximity Detection** (`proximity.py`) -- Depth-based collision avoidance. Computes danger score from minimum depth (closest tissue), approach speed (depth gradient), and danger direction.

3. **Junction Detection** (`junction.py`) -- Depth-gated brightness analysis. Uses depth maps to identify open areas, then brightness thresholding within those regions to count distinct branch openings (lumen segmentation via dark regions separated by tissue ridges).

4. **Place Recognition** (`place_recognition.py`) -- AnyLoc-style VLAD descriptors built on DINOv2 ViT-S/14 patch features (value facet, layer 9). Detects revisited locations for backtracking decisions. Two-pass: vocabulary fitting on sampled frames, then per-frame descriptor matching.

5. **Exploration State Machine** (`state_machine.py`) -- Consumes all module outputs to build a topological graph. Nodes are junctions, dead ends, and the start point. Edges represent traversal cost (frame count).

## Setup

Requires Python 3.11+ and a CUDA GPU.

```bash
# Install dependencies
uv sync
uv pip install --no-verify-hashes torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Run on a video
uv run python run.py "path/to/video.mp4" -o output/run_name

# Options
uv run python run.py video.mp4 -o output/name --no-dino    # HSV histogram fallback (no GPU needed for place rec)
uv run python run.py video.mp4 -o output/name --frame-skip 3  # 3x faster, process every 3rd frame
uv run python run.py video.mp4 -o output/name --device cpu    # CPU-only (slow)
```

## Outputs

| File | Description |
|------|-------------|
| `dashboard.png` | 5-row time series: proximity danger, depth, junction blob count, place recognition similarity, state timeline |
| `topological_graph.png` | Exploration graph with frame thumbnails at each node |
| `exploration_log.json` | Per-frame record of all signals and state machine decisions |
| `frames/` | Annotated frames (original + depth map + junction mask) every 5 frames |

## Dataset

Designed for phantom kidney endoscope footage organized as:
```
artificial_kidney_path_data/
  lower pole/   (3 trials: video + tracking CSVs)
  mid pole/
  pelvis/
  upper pole/
```

Each trial has an MP4 video (60fps, 1024x768) with TrakStar position tracking and Zamenix instrument parameter CSVs.

## Dependencies

- `torch`, `torchvision` -- DINOv2, Depth Anything v2
- `transformers` -- Depth Anything v2 model loading
- `opencv-python` -- video I/O, image processing
- `scikit-learn` -- KMeans for VLAD vocabulary
- `matplotlib`, `networkx` -- visualization
- `numpy` -- everything

## Key Design Decisions

- **Depth + brightness hybrid** for junction detection: depth maps (DA2 Small) are too smooth to resolve thin tissue ridges between branch openings, so depth gates the analysis region and brightness thresholding finds the actual branches within.
- **3+ blob requirement** for junctions: 2-blob detections in tube interiors are unreliable due to shadows and specular reflections. Requiring 3+ similarly-sized blobs eliminates false positives.
- **DINOv2 VLAD** for place recognition: per-patch value facet features aggregated via VLAD (32 clusters) produce compact, discriminative descriptors. Run every 3 frames for performance.
- **One-way kidney entry detection**: depth range threshold detects entry into the kidney. Once inside, the system stays inside (the scope doesn't exit and re-enter during a procedure).

## License

MIT
