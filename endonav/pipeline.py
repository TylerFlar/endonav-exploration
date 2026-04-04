import json
import os
import time
import cv2
import numpy as np
from .config import Config
from .video_loader import VideoLoader
from .depth import DepthEstimator
from .proximity import ProximityDetector
from .junction import JunctionDetector
from .place_recognition import PlaceRecognizer
from .state_machine import ExplorationStateMachine
from .visualization import Visualization


class Pipeline:
    def __init__(self, video_path: str, config: Config, device: str = "cuda"):
        self.video_path = video_path
        self.config = config
        self.device = device

        self.loader = VideoLoader(video_path, config)
        self.depth = DepthEstimator(config, device=device)
        self.proximity = ProximityDetector(config)
        self.junction = JunctionDetector(config)
        self.place = PlaceRecognizer(config, device=device)
        self.state_machine = ExplorationStateMachine(config)
        self.viz = Visualization(config)

    def run(self) -> dict:
        os.makedirs(self.config.output_root, exist_ok=True)
        frames_dir = os.path.join(self.config.output_root, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        print(f"Video: {self.video_path}")
        print(f"  Resolution: {self.loader.original_resolution} -> {self.loader.target_resolution}")
        print(f"  Frames: {self.loader.frame_count}, FPS: {self.loader.fps:.1f}")
        print()

        # Pass 1: VLAD vocabulary + junction calibration
        print("=== Pass 1: Building VLAD vocabulary ===")
        t0 = time.time()
        self._pass1_vocabulary()
        print(f"  Pass 1 complete in {time.time() - t0:.1f}s\n")

        # Pass 2: Process all frames
        print("=== Pass 2: Processing all frames ===")
        t0 = time.time()
        log = self._pass2_process(frames_dir)
        elapsed = time.time() - t0
        fps_proc = len(log) / elapsed if elapsed > 0 else 0
        print(f"  Pass 2 complete: {len(log)} frames in {elapsed:.1f}s ({fps_proc:.1f} fps)\n")

        # Generate outputs
        print("=== Generating outputs ===")
        dashboard_path = os.path.join(self.config.output_root, "dashboard.png")
        self.viz.save_dashboard(log, dashboard_path)
        print(f"  Dashboard: {dashboard_path}")

        graph_path = os.path.join(self.config.output_root, "topological_graph.png")
        self.viz.save_topological_graph(
            self.state_machine.nodes, self.state_machine.edges,
            graph_path, fps=self.loader.fps,
        )
        print(f"  Graph: {graph_path}")

        log_path = os.path.join(self.config.output_root, "exploration_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=1)
        print(f"  Log: {log_path}")

        report = self.state_machine.get_coverage_report()
        print(f"\n=== Coverage Report ===")
        for k, v in report.items():
            print(f"  {k}: {v}")

        return {"log": log, "coverage": report}

    def _pass1_vocabulary(self):
        interval = self.config.vlad_vocab_sample_interval
        sampled_frames = []
        calibration_grays = []

        for idx, frame in self.loader.frames(skip=interval):
            sampled_frames.append(frame)
            if len(calibration_grays) < self.config.calibration_frames:
                calibration_grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        print(f"  Sampled {len(sampled_frames)} frames (every {interval}th)")

        if calibration_grays:
            self.junction.calibrate(calibration_grays)
            print(f"  Junction brightness threshold: {self.junction._bright_threshold:.1f}")
            if self.junction._border_mask is not None:
                valid_pct = self.junction._border_mask.sum() / 255 / self.junction._border_mask.size * 100
                print(f"  Border mask: {valid_pct:.1f}% valid pixels")
                self.proximity.border_mask = self.junction._border_mask

        if self.config.use_dino and sampled_frames:
            self.place.fit_vocabulary(sampled_frames)

    def _detect_inside_kidney(self, depth_map: np.ndarray) -> bool:
        """Detect if scope is inside kidney using depth range.
        Inside the kidney, the depth map has a wide range (close tissue + far lumen).
        Outside, the depth is more uniform (lab environment)."""
        if self.junction._border_mask is not None:
            valid = depth_map[self.junction._border_mask > 0]
        else:
            valid = depth_map.flatten()
        if len(valid) == 0:
            return False
        depth_range = float(valid.max() - valid.min())
        return depth_range > self.config.inside_kidney_depth_range_threshold

    def _pass2_process(self, frames_dir: str) -> list[dict]:
        log = []
        total = self.loader.frame_count // self.config.frame_skip
        initialized = False
        inside_kidney = False
        inside_counter = 0
        confirm_frames = self.config.inside_kidney_confirm_frames
        last_depth_map = None
        depth_interval = self.config.depth_interval

        for idx, frame in self.loader.frames(skip=self.config.frame_skip):
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Depth estimation (every N frames for performance)
            if idx % depth_interval == 0 or last_depth_map is None:
                depth_map = self.depth.process(frame)
                last_depth_map = depth_map
            else:
                depth_map = last_depth_map

            # Kidney entry detection (one-way)
            if not inside_kidney:
                if self._detect_inside_kidney(depth_map):
                    inside_counter += 1
                    if inside_counter >= confirm_frames:
                        inside_kidney = True
                        print(f"    Kidney entry detected at frame {idx}")
                else:
                    inside_counter = 0

            # Module 1: Proximity (depth-based)
            prox_result = self.proximity.process(depth_map, frame)

            # Module 2: Junction (depth-gated brightness)
            junc_result = self.junction.process(curr_gray, depth_map)

            # Module 3: Place Recognition
            place_result = self.place.process(frame, idx)

            # Module 4: State Machine
            if not initialized and inside_kidney:
                self.state_machine.initialize(idx, frame)
                initialized = True
            if initialized and inside_kidney:
                state = self.state_machine.update(
                    idx, prox_result, junc_result, place_result, frame_bgr=frame
                )
            else:
                state = self.state_machine.state if initialized else None

            # Log record
            record = {
                "frame_idx": idx,
                "timestamp_sec": round(idx / self.loader.fps, 3) if self.loader.fps > 0 else 0,
                "proximity": {
                    "danger_score": round(float(prox_result.danger_score), 4),
                    "danger_direction": prox_result.danger_direction.tolist(),
                    "min_depth": round(float(prox_result.min_depth), 4),
                    "mean_depth": round(float(prox_result.mean_depth), 4),
                    "depth_gradient": round(float(prox_result.depth_gradient), 4),
                    "is_dangerous": bool(prox_result.is_dangerous),
                },
                "junction": {
                    "blob_count": junc_result.blob_count,
                    "raw_type": junc_result.raw_type.value,
                    "confirmed_type": junc_result.confirmed_type.value,
                    "depth_open_fraction": round(float(junc_result.depth_open_fraction), 4),
                    "max_depth": round(float(junc_result.max_depth), 4),
                },
                "place_recognition": {
                    "max_similarity": round(float(place_result.max_similarity), 4),
                    "is_revisit": bool(place_result.is_revisit),
                    "matched_keyframe_idx": place_result.matched_keyframe_idx,
                    "is_new_keyframe": bool(place_result.is_new_keyframe),
                    "method": place_result.method,
                    "num_keyframes": len(self.place.keyframes),
                },
                "state": state.value if state is not None else "OUTSIDE",
                "inside_kidney": inside_kidney,
            }
            log.append(record)

            # Save annotated frame
            if idx % self.config.annotated_frame_interval == 0:
                self.viz.save_annotated_frame(
                    frame, prox_result, junc_result, depth_map, idx, frames_dir
                )

            # Progress
            if len(log) % 500 == 0:
                print(f"    Frame {idx} ({len(log)}/{total} processed)")

        return log
