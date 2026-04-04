"""CLI entry point for the kidney endoscope exploration system."""

import argparse
import torch
from endonav.config import Config
from endonav.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Reactive kidney endoscope exploration analysis"
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
    args = parser.parse_args()

    config = Config()
    config.output_root = args.output
    config.use_dino = not args.no_dino
    config.frame_skip = args.frame_skip

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"DINOv2 VLAD: {'enabled' if config.use_dino else 'disabled (HSV fallback)'}")
    print(f"Frame skip: {config.frame_skip}")
    print()

    pipeline = Pipeline(args.video, config, device=device)
    pipeline.run()


if __name__ == "__main__":
    main()
