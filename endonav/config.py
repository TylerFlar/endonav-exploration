import dataclasses


@dataclasses.dataclass
class Config:
    # Paths
    dataset_root: str = r"G:\01_Active\Code\arclab\_datasets\artificial_kidney_path_data"
    output_root: str = "./output"

    # Video processing
    target_width: int = 512
    target_height: int = 384
    frame_skip: int = 1  # process every Nth frame

    # --- Depth Estimation ---
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    depth_interval: int = 2  # run depth every Nth frame, interpolate between

    # --- Module 1: Proximity (depth-based) ---
    danger_threshold: float = 0.7
    safety_min_depth: float = 0.15  # danger if closest valid point depth < this (0-1 scale)
    approach_speed_weight: float = 0.3  # weight of approach speed in danger score

    # --- Module 2: Junction Detector (depth-gated brightness) ---
    # Border mask & calibration
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
    min_branch_count: int = 3  # minimum blobs for junction (3+ for reliable detection)
    max_blob_ratio: float = 4.0  # largest/smallest blob ratio must be < this
    # Temporal confirmation
    junction_confirm_frames: int = 3
    dead_end_confirm_frames: int = 8
    # Dead end: all tissue is close (no deep regions)
    dead_end_shallow_threshold: float = 0.5  # if depth range < this AND 0 blobs, it's a dead end

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
    min_expected_branches: int = 6
    max_expected_branches: int = 12

    # --- Kidney Entry Detection (depth-based) ---
    inside_kidney_depth_range_threshold: float = 0.7  # depth range > this = inside
    inside_kidney_confirm_frames: int = 60

    # --- Visualization ---
    annotated_frame_interval: int = 5
    dashboard_dpi: int = 150
