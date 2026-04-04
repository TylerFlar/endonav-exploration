import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import networkx as nx
from .config import Config
from .proximity import ProximityResult
from .junction import JunctionResult
from .state_machine import GraphNode, GraphEdge


# State colors
STATE_COLORS = {
    "OUTSIDE": "#BDBDBD",
    "EXPLORE": "#4CAF50",
    "JUNCTION_DETECTED": "#FF9800",
    "ENTERING_BRANCH": "#2196F3",
    "DEAD_END": "#F44336",
    "BACKTRACKING": "#9C27B0",
    "SAFETY_OVERRIDE": "#FF0000",
}

NODE_COLORS = {
    "start": "#4CAF50",
    "junction": "#FF9800",
    "dead_end": "#F44336",
}


class Visualization:
    def __init__(self, config: Config):
        self.config = config

    def save_dashboard(self, log: list[dict], output_path: str) -> None:
        """5-row time series dashboard."""
        if not log:
            return

        frames = [r["frame_idx"] for r in log]
        times = [r["timestamp_sec"] for r in log]

        danger_scores = [r["proximity"]["danger_score"] for r in log]
        min_depths = [r["proximity"]["min_depth"] for r in log]
        mean_depths = [r["proximity"]["mean_depth"] for r in log]
        blob_counts = [r["junction"]["blob_count"] for r in log]
        open_fracs = [r["junction"]["depth_open_fraction"] for r in log]
        place_sims = [r["place_recognition"]["max_similarity"] for r in log]
        states = [r["state"] for r in log]

        fig, axes = plt.subplots(5, 1, figsize=(20, 13), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1, 0.8, 1, 0.5]})
        fig.patch.set_facecolor("white")
        fig.suptitle("Endoscope Exploration Dashboard", fontsize=16, fontweight="bold", y=0.995)

        x = times

        # Find event times for vertical markers
        junction_times = [r["timestamp_sec"] for r in log if r["state"] == "JUNCTION_DETECTED"
                         and (not log[log.index(r)-1]["state"] == "JUNCTION_DETECTED"
                              if log.index(r) > 0 else True)]
        # Simplify: just get first frame of each junction/dead-end event
        jt, dt = [], []
        prev_state = None
        for r in log:
            if r["state"] == "JUNCTION_DETECTED" and prev_state != "JUNCTION_DETECTED":
                jt.append(r["timestamp_sec"])
            if r["state"] == "DEAD_END" and prev_state != "DEAD_END":
                dt.append(r["timestamp_sec"])
            prev_state = r["state"]

        def add_event_markers(ax):
            for t in jt:
                ax.axvline(x=t, color="#FF9800", alpha=0.15, linewidth=1, zorder=0)
            for t in dt:
                ax.axvline(x=t, color="#E53935", alpha=0.2, linewidth=1.5, zorder=0)

        # Row 1: Proximity danger score
        ax = axes[0]
        ax.plot(x, danger_scores, color="#E53935", linewidth=0.6, alpha=0.85)
        ax.axhline(y=0.7, color="#B71C1C", linestyle="--", alpha=0.6, linewidth=1,
                   label="Danger threshold (0.7)")
        ax.fill_between(x, danger_scores, alpha=0.15, color="#E53935")
        ax.set_ylabel("Danger\nScore", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax.set_title("Proximity", fontsize=10, loc="left", fontweight="bold")
        add_event_markers(ax)

        # Row 2: Depth
        ax = axes[1]
        ax.plot(x, min_depths, color="#1565C0", linewidth=0.5, alpha=0.8, label="Min depth (closest)")
        ax.plot(x, mean_depths, color="#FF6F00", linewidth=0.5, alpha=0.6, label="Mean depth")
        ax.fill_between(x, min_depths, alpha=0.15, color="#1565C0")
        ax.set_ylabel("Depth\n(0-1)", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax.set_title("Depth Estimation", fontsize=10, loc="left", fontweight="bold")
        add_event_markers(ax)

        # Row 3: Junction blob count
        ax = axes[2]
        ax.fill_between(x, blob_counts, alpha=0.3, color="#2E7D32", step="mid")
        ax.step(x, blob_counts, color="#2E7D32", linewidth=0.8, alpha=0.9, where="mid")
        ax.axhline(y=2, color="#E65100", linestyle="--", alpha=0.6, linewidth=1,
                   label="Junction threshold ($\\geq$2)")
        ax.set_ylabel("Blob\nCount", fontsize=9)
        ax.set_ylim(-0.1, max(max(blob_counts) + 0.5, 3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax.set_title("Junction Openings", fontsize=10, loc="left", fontweight="bold")
        add_event_markers(ax)

        # Row 4: Place recognition similarity
        ax = axes[3]
        ax.plot(x, place_sims, color="#6A1B9A", linewidth=0.4, alpha=0.7)
        ax.axhline(y=0.85, color="#C62828", linestyle="--", alpha=0.5, linewidth=1,
                   label="Revisit threshold (0.85)")
        ax.axhline(y=0.90, color="#1565C0", linestyle=":", alpha=0.4, linewidth=1,
                   label="Novelty threshold (0.90)")
        ax.fill_between(x, place_sims, 0.85, where=[s >= 0.85 for s in place_sims],
                        alpha=0.1, color="#6A1B9A")
        ax.set_ylabel("Max\nSimilarity", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
        ax.set_title("Place Recognition (VLAD)", fontsize=10, loc="left", fontweight="bold")
        add_event_markers(ax)

        # Row 5: State timeline
        ax = axes[4]
        self._draw_state_timeline(ax, x, states)
        ax.set_ylabel("State")
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Exploration State", fontsize=10)

        plt.tight_layout()
        fig.savefig(output_path, dpi=self.config.dashboard_dpi, bbox_inches="tight")
        plt.close(fig)

    def _draw_state_timeline(self, ax, times, states):
        """Draw color-coded state timeline using broken_barh."""
        if not states:
            return

        unique_states = list(STATE_COLORS.keys())
        state_to_y = {s: i for i, s in enumerate(unique_states)}

        # Build segments: consecutive spans of the same state
        segments = []
        start_t = times[0]
        current_state = states[0]

        for i in range(1, len(states)):
            if states[i] != current_state:
                segments.append((start_t, times[i] - start_t, current_state))
                start_t = times[i]
                current_state = states[i]
        # Last segment
        end_t = times[-1] + (times[-1] - times[-2] if len(times) > 1 else 0.1)
        segments.append((start_t, end_t - start_t, current_state))

        for start, width, state in segments:
            color = STATE_COLORS.get(state, "#999999")
            ax.barh(0, width, left=start, height=0.8, color=color, edgecolor="none")

        ax.set_yticks([])
        # Legend
        patches = [
            mpatches.Patch(color=c, label=s) for s, c in STATE_COLORS.items()
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=3)

    def save_annotated_frame(
        self,
        frame: np.ndarray,
        proximity: ProximityResult,
        junction: JunctionResult,
        depth_map: np.ndarray,
        frame_idx: int,
        output_dir: str,
    ) -> None:
        """3-panel annotated frame: original+contours, depth map, dark mask."""
        h, w = frame.shape[:2]

        # Panel 1: Original with junction contours
        panel1 = frame.copy()
        for blob in junction.blobs:
            if len(blob.contour) > 0:
                cv2.drawContours(panel1, [blob.contour], -1, (0, 255, 0), 2)
                cx, cy = int(blob.centroid[0]), int(blob.centroid[1])
                cv2.circle(panel1, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(
                    panel1, f"A={blob.area}",
                    (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 255), 1,
                )

        # Panel 2: Depth map (colorized)
        d_norm = (depth_map * 255).astype(np.uint8)
        panel2 = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)

        # Panel 3: Dark mask (junction detection mask)
        panel3 = cv2.cvtColor(junction.dark_mask, cv2.COLOR_GRAY2BGR)

        # Text overlays
        cv2.putText(
            panel1,
            f"Frame {frame_idx} | Danger: {proximity.danger_score:.2f} | Blobs: {junction.blob_count}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
        cv2.putText(
            panel2,
            f"Depth (min={proximity.min_depth:.2f}, mean={proximity.mean_depth:.2f})",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
        cv2.putText(
            panel3,
            f"Junction Mask (open={junction.depth_open_fraction:.2f})",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

        combined = np.hstack([panel1, panel2, panel3])
        out_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def save_topological_graph(
        self,
        nodes: list[GraphNode],
        edges: list[GraphEdge],
        output_path: str,
        fps: float = 60.0,
    ) -> None:
        """Render topological exploration graph with hierarchical layout."""
        if not nodes:
            return

        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node.node_id, node_type=node.node_type,
                       frame_idx=node.frame_idx,
                       branches_total=node.branches_total,
                       branches_explored=node.branches_explored)
        for edge in edges:
            G.add_edge(edge.source, edge.target, frame_cost=edge.frame_cost)

        # --- Hierarchical layout using BFS depth from start node ---
        pos = self._hierarchical_layout(G, nodes, edges)

        fig, ax = plt.subplots(1, 1, figsize=(18, 12))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")

        # --- Draw edges ---
        # Scale edge width by frame cost (longer traversal = thicker)
        max_cost = max((e.frame_cost for e in edges), default=1)
        for edge in edges:
            src_pos = pos[edge.source]
            tgt_pos = pos[edge.target]
            width = 0.8 + 2.5 * (edge.frame_cost / max_cost)
            alpha = 0.4 + 0.5 * (edge.frame_cost / max_cost)
            ax.annotate("",
                xy=tgt_pos, xytext=src_pos,
                arrowprops=dict(
                    arrowstyle="-|>", color="#888888",
                    lw=width, alpha=alpha,
                    connectionstyle="arc3,rad=0.1",
                    shrinkA=12, shrinkB=12,
                ))
            # Edge label: show time in seconds
            mid_x = (src_pos[0] + tgt_pos[0]) / 2
            mid_y = (src_pos[1] + tgt_pos[1]) / 2
            time_sec = edge.frame_cost / fps
            if time_sec >= 1.0:
                label = f"{time_sec:.1f}s"
            else:
                label = f"{edge.frame_cost}f"
            ax.text(mid_x, mid_y, label,
                    fontsize=6, color="#666666", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#FAFAFA",
                              edgecolor="none", alpha=0.8))

        # --- Draw nodes with thumbnails ---
        border_colors = {
            "start":    "#2E7D32",
            "junction": "#E65100",
            "dead_end": "#B71C1C",
        }
        for node in nodes:
            x, y = pos[node.node_id]
            border_color = border_colors[node.node_type]

            if node.thumbnail is not None:
                # Convert BGR thumbnail to RGB for matplotlib
                thumb_rgb = cv2.cvtColor(node.thumbnail, cv2.COLOR_BGR2RGB)
                imagebox = OffsetImage(thumb_rgb, zoom=0.9)
                imagebox.image.axes = ax
                ab = AnnotationBbox(
                    imagebox, (x, y), frameon=True,
                    bboxprops=dict(edgecolor=border_color, linewidth=2.5,
                                   boxstyle="round,pad=0.05"),
                    zorder=5,
                )
                ax.add_artist(ab)
            else:
                # Fallback: colored marker if no thumbnail
                shape = "s" if node.node_type == "dead_end" else "o"
                ax.plot(x, y, marker=shape,
                       markersize=18, color=border_color,
                       markeredgecolor=border_color,
                       markeredgewidth=1.5, zorder=5)

            # Node label above thumbnail
            if node.node_type == "start":
                label = "START"
            elif node.node_type == "junction":
                label = f"J{node.node_id}"
            else:
                label = f"D{node.node_id}"

            ax.text(x, y + 0.07, label, fontsize=7, fontweight="bold",
                    ha="center", va="bottom", color=border_color, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=border_color, alpha=0.9, linewidth=1))

            # Annotation below thumbnail
            time_str = f"{node.frame_idx / fps:.1f}s"
            if node.node_type == "junction":
                annot = f"{time_str} | {node.branches_explored}/{node.branches_total}"
            else:
                annot = time_str
            ax.text(x, y - 0.08, annot, fontsize=5.5, ha="center", va="top",
                    color="#555555", zorder=6)

        # --- Legend & title ---
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E7D32",
                       markersize=12, label="Start", markeredgecolor="#1B5E20", markeredgewidth=1.5),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FB8C00",
                       markersize=10, label="Junction (explored/total)",
                       markeredgecolor="#E65100", markeredgewidth=1.5),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#E53935",
                       markersize=10, label="Dead End",
                       markeredgecolor="#B71C1C", markeredgewidth=1.5),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
                  framealpha=0.9, edgecolor="#CCCCCC")

        # Summary stats
        n_junctions = sum(1 for n in nodes if n.node_type == "junction")
        n_dead_ends = sum(1 for n in nodes if n.node_type == "dead_end")
        unexplored = sum(max(0, n.branches_total - n.branches_explored)
                        for n in nodes if n.node_type == "junction")
        summary = (f"{len(nodes)} nodes  |  {n_junctions} junctions  |  "
                   f"{n_dead_ends} dead ends  |  {len(edges)} edges  |  "
                   f"{unexplored} unexplored branches")
        fig.suptitle("Topological Exploration Graph", fontsize=15, fontweight="bold", y=0.98)
        ax.set_title(summary, fontsize=9, color="#666666", pad=10)

        ax.set_axis_off()

        # Set axis limits tightly around the node positions
        if pos:
            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]
            pad = 0.15
            x_range = max(xs) - min(xs) if len(xs) > 1 else 1.0
            y_range = max(ys) - min(ys) if len(ys) > 1 else 1.0
            ax.set_xlim(min(xs) - pad * x_range - 0.2, max(xs) + pad * x_range + 0.2)
            ax.set_ylim(min(ys) - pad * y_range - 0.2, max(ys) + pad * y_range + 0.2)

        fig.savefig(output_path, dpi=120,
                    bbox_inches="tight", facecolor="#FAFAFA")
        plt.close(fig)

    def _hierarchical_layout(
        self, G: nx.DiGraph, nodes: list[GraphNode], edges: list[GraphEdge]
    ) -> dict[int, tuple[float, float]]:
        """Compute a hierarchical tree layout using BFS from the start node."""
        if len(nodes) <= 1:
            return {nodes[0].node_id: (0, 0)} if nodes else {}

        # Build undirected version for BFS
        G_undir = G.to_undirected()

        # BFS from start node to assign depths
        start_id = nodes[0].node_id
        depths: dict[int, int] = {}
        visited = {start_id}
        queue = [(start_id, 0)]
        order = []
        while queue:
            nid, depth = queue.pop(0)
            depths[nid] = depth
            order.append(nid)
            for neighbor in G_undir.neighbors(nid):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        # Handle disconnected nodes
        for node in nodes:
            if node.node_id not in depths:
                depths[node.node_id] = max(depths.values(), default=0) + 1
                order.append(node.node_id)

        # Assign x positions: spread nodes at each depth level
        max_depth = max(depths.values(), default=0)
        level_nodes: dict[int, list[int]] = {}
        for nid, d in depths.items():
            level_nodes.setdefault(d, []).append(nid)

        pos = {}
        for depth, nids in level_nodes.items():
            n = len(nids)
            for i, nid in enumerate(sorted(nids)):
                x = (i - (n - 1) / 2) * 1.0  # spread horizontally
                y = -depth * 1.0  # depth goes downward
                pos[nid] = (x, y)

        # Refine with spring layout using hierarchical positions as initial
        pos = nx.spring_layout(G_undir, pos=pos, k=1.5, iterations=30,
                               seed=42, weight=None)

        return pos
