import dataclasses
from dataclasses import field


@dataclasses.dataclass
class ExpectedGraph:
    """Describes the expected topological structure of the anatomy being explored.

    The exploration system uses this to:
    - Set min_branch_count for junction detection
    - Validate coverage completeness
    - Report how many branches were found vs expected
    """

    # How many branches at the first major junction?
    # Kidney: 3 (upper pole, mid pole, lower pole)
    # Bronchial tree: 2 (left/right main bronchus)
    first_junction_branches: int = 3

    # Total expected terminal branches (dead ends)
    # Kidney: 3-12 (3 poles, each may have sub-branches)
    min_terminal_branches: int = 3
    max_terminal_branches: int = 12

    # Minimum blobs to classify as a junction.
    # Derived from anatomy: if first junction has 3 branches, need 3+ blobs.
    # For 2-branch bifurcations, set to 2 (but increases false positives).
    min_branch_count: int = 3

    # Description for logging
    description: str = "Kidney collecting system (3 poles)"


# Presets for common anatomies
KIDNEY_GRAPH = ExpectedGraph(
    first_junction_branches=3,
    min_terminal_branches=3,
    max_terminal_branches=12,
    min_branch_count=3,
    description="Kidney collecting system (3 poles)",
)

BRONCHIAL_GRAPH = ExpectedGraph(
    first_junction_branches=2,
    min_terminal_branches=6,
    max_terminal_branches=20,
    min_branch_count=2,
    description="Bronchial tree (binary branching)",
)


@dataclasses.dataclass
class Config:
    # Paths
    dataset_root: str = r"G:\01_Active\Code\arclab\_datasets\artificial_kidney_path_data"
    output_root: str = "./output"

    # --- Expected anatomy ---
    expected_graph: ExpectedGraph = field(default_factory=lambda: ExpectedGraph())

    # Video processing
    target_width: int = 512
    target_height: int = 384
    frame_skip: int = 1  # process every Nth frame

    # --- Depth Estimation ---
    # Supports any HuggingFace depth-estimation model.
    # DA2 Small: "depth-anything/Depth-Anything-V2-Small-hf" (Apache 2.0, 24.8M params)
    # DA2 Base:  "depth-anything/Depth-Anything-V2-Base-hf" (CC-BY-NC, 97.5M params)
    # DA3: not yet available via HF transformers (numpy<2 conflict with pip package)
    # DA3 model IDs: "depth-anything/DA3-SMALL", "depth-anything/DA3-BASE",
    #                "depth-anything/DA3-LARGE-1.1", "depth-anything/DA3-GIANT-1.1"
    depth_model: str = "depth-anything/DA3-SMALL"
    depth_interval: int = 2  # run depth every Nth frame, reuse between

    # --- Module 1: Proximity (depth-based) ---
    danger_threshold: float = 0.7
    safety_min_depth: float = 0.15  # danger if closest valid point depth < this (0-1 scale)
    approach_speed_weight: float = 0.3  # weight of approach speed in danger score

    # --- Module 2: Junction Detector (depth-gated brightness) ---
    calibration_frames: int = 20
    # Depth gate: only look for junctions when a large deep region exists
    depth_open_area_threshold: float = 0.20  # deep region must be >20% of valid area
    depth_deep_percentile: float = 70.0  # percentile for "deep" region in depth map
    # Brightness-based branch counting (within depth-gated deep region)
    blur_kernel: int = 9
    blur_sigma: float = 3.0
    junction_brightness_offset: float = 15.0  # threshold = calibrated - this
    morph_open_kernel: tuple[int, int] = (5, 5)
    morph_open_iter: int = 2
    min_blob_fraction: float = 0.015  # blob must be >1.5% of image area
    max_blob_fraction: float = 0.5
    # min_branch_count is in expected_graph.min_branch_count
    max_blob_ratio: float = 4.0  # largest/smallest blob ratio must be < this
    # Temporal confirmation
    junction_confirm_frames: int = 3
    dead_end_confirm_frames: int = 8

    # --- Module 3: Place Recognition ---
    dino_model: str = "dinov2_vits14"
    dino_layer: int = 9
    dino_facet: str = "value"
    vlad_num_clusters: int = 32
    vlad_vocab_sample_interval: int = 30
    revisit_threshold: float = 0.92
    keyframe_novelty_threshold: float = 0.95
    max_keyframes: int = 200
    use_dino: bool = True
    dino_input_size: int = 518
    place_recognition_interval: int = 3

    # --- Module 4: State Machine ---
    min_junction_gap_frames: int = 180  # ~3 sec between node creation
    # min_expected_branches and max_expected_branches are now in expected_graph

    # --- Kidney Entry Detection (depth-based) ---
    inside_kidney_depth_range_threshold: float = 0.7  # depth range > this = inside
    inside_kidney_confirm_frames: int = 60

    # --- Visualization ---
    annotated_frame_interval: int = 5
    dashboard_dpi: int = 150
