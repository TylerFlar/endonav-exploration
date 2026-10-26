"""Run the exploration pipeline on all 12 videos and summarize results."""

import os
import json
import time
from pathlib import Path
from endonav.config import Config, KIDNEY_GRAPH
from endonav.pipeline import Pipeline
import torch


DATASET_ROOT = r"G:\01_Active\Code\arclab\_datasets\artificial_kidney_path_data"
OUTPUT_ROOT = "./output"

VIDEOS = [
    ("lower pole", "lower_pole_trial1_video.mp4"),
    ("lower pole", "lower_pole_trial2_video.mp4"),
    ("lower pole", "lower_pole_trial3_video.mp4"),
    ("mid pole", "mid_pole_trial1_video.mp4"),
    ("mid pole", "mid_pole_trial2_video.mp4"),
    ("mid pole", "mid_pole_trial3_video.mp4"),
    ("pelvis", "pelvis_trial1_video.mp4"),
    ("pelvis", "pelvis_trial2_video.mp4"),
    ("pelvis", "pelvis_trial3_video.mp4"),
    ("upper pole", "upper_pole_trial1_video.mp4"),
    ("upper pole", "upper_pole_trial2_video.mp4"),
    ("upper pole", "upper_pole_trial3_video.mp4"),
]


def run_video(region: str, filename: str, config: Config, device: str) -> dict:
    video_path = os.path.join(DATASET_ROOT, region, filename)
    run_name = filename.replace("_video.mp4", "")
    config.output_root = os.path.join(OUTPUT_ROOT, run_name)

    print(f"\n{'='*60}")
    print(f"  {run_name} ({region})")
    print(f"{'='*60}")

    t0 = time.time()
    pipeline = Pipeline(video_path, config, device=device)
    result = pipeline.run()
    elapsed = time.time() - t0

    coverage = result["coverage"]
    return {
        "name": run_name,
        "region": region,
        "elapsed_sec": round(elapsed, 1),
        "total_nodes": coverage["total_nodes"],
        "junction_nodes": coverage["junction_nodes"],
        "dead_end_nodes": coverage["dead_end_nodes"],
        "total_edges": coverage["total_edges"],
        "unexplored_branches": coverage["unexplored_branches"],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config()
    config.expected_graph = KIDNEY_GRAPH

    results = []
    for region, filename in VIDEOS:
        video_path = os.path.join(DATASET_ROOT, region, filename)
        if not os.path.exists(video_path):
            print(f"SKIP: {video_path} not found")
            continue
        try:
            r = run_video(region, filename, config, device)
            results.append(r)
        except Exception as e:
            print(f"ERROR on {filename}: {e}")
            results.append({"name": filename, "region": region, "error": str(e)})

    # Summary table
    print(f"\n\n{'='*80}")
    print("BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"{'Video':<30} {'Region':<15} {'Nodes':>6} {'Junc':>5} {'Dead':>5} {'Edges':>6} {'Time':>7}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30} {r['region']:<15} ERROR: {r['error']}")
        else:
            print(
                f"{r['name']:<30} {r['region']:<15} "
                f"{r['total_nodes']:>6} {r['junction_nodes']:>5} "
                f"{r['dead_end_nodes']:>5} {r['total_edges']:>6} "
                f"{r['elapsed_sec']:>6.0f}s"
            )

    # Expected: each video should have START + 1 junction (3-way) + 1 dead end minimum
    print(f"\n{'='*80}")
    print("VALIDATION")
    print(f"{'='*80}")
    for r in results:
        if "error" in r:
            continue
        issues = []
        if r["junction_nodes"] < 1:
            issues.append("NO JUNCTION (expected 3-way pelvis junction)")
        if r["dead_end_nodes"] < 1:
            issues.append("NO DEAD END (expected at pole terminus)")
        if r["total_nodes"] < 3:
            issues.append(f"TOO FEW NODES ({r['total_nodes']}, expected >= 3)")
        status = "OK" if not issues else "ISSUES"
        print(f"  {r['name']:<30} [{status}] {'; '.join(issues) if issues else ''}")

    # Save summary
    summary_path = os.path.join(OUTPUT_ROOT, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
