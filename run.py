"""CLI entry point for the endoscope exploration system."""

import argparse
import torch
from endonav.config import Config, ExpectedGraph, KIDNEY_GRAPH, BRONCHIAL_GRAPH
from endonav.pipeline import Pipeline


GRAPH_PRESETS = {
    "kidney": KIDNEY_GRAPH,
    "bronchial": BRONCHIAL_GRAPH,
}


def main():
    parser = argparse.ArgumentParser(
        description="Reactive endoscope exploration analysis"
    )
    parser.add_argument("video", help="Path to MP4 video file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument(
        "--no-dino", action="store_true",
        help="Use HSV histogram fallback instead of DINOv2 VLAD"
    )
    parser.add_argument(
        "--frame-skip", type=int, default=1,
        help="Process every Nth frame (default: 1 = all frames)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (default: cuda if available, else cpu)"
    )

    # Expected graph structure
    graph_group = parser.add_argument_group("Expected anatomy")
    graph_group.add_argument(
        "--anatomy", choices=list(GRAPH_PRESETS.keys()), default="kidney",
        help="Anatomy preset (default: kidney)"
    )
    graph_group.add_argument(
        "--min-branches", type=int, default=None,
        help="Min blobs to detect a junction (overrides preset)"
    )
    graph_group.add_argument(
        "--expected-terminals", type=str, default=None,
        help="Expected terminal branch range, e.g. '3-12' (overrides preset)"
    )

    args = parser.parse_args()

    config = Config()
    config.output_root = args.output
    config.use_dino = not args.no_dino
    config.frame_skip = args.frame_skip

    # Apply anatomy preset
    config.expected_graph = GRAPH_PRESETS[args.anatomy]

    # Override specific values if provided
    if args.min_branches is not None:
        config.expected_graph.min_branch_count = args.min_branches
        config.expected_graph.first_junction_branches = args.min_branches
    if args.expected_terminals is not None:
        parts = args.expected_terminals.split("-")
        config.expected_graph.min_terminal_branches = int(parts[0])
        if len(parts) > 1:
            config.expected_graph.max_terminal_branches = int(parts[1])

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Anatomy: {config.expected_graph.description}")
    print(f"  Min branches for junction: {config.expected_graph.min_branch_count}")
    print(f"  Expected terminals: {config.expected_graph.min_terminal_branches}-{config.expected_graph.max_terminal_branches}")
    print(f"Depth model: {config.depth_model}")
    print(f"DINOv2 VLAD: {'enabled' if config.use_dino else 'disabled (HSV fallback)'}")
    print(f"Frame skip: {config.frame_skip}")
    print()

    pipeline = Pipeline(args.video, config, device=device)
    pipeline.run()


if __name__ == "__main__":
    main()
