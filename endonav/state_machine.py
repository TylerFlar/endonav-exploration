import dataclasses
import enum
import numpy as np
from .config import Config
from .proximity import ProximityResult
from .junction import JunctionResult, LumenType
from .place_recognition import PlaceRecognitionResult


class ExplorationState(enum.Enum):
    EXPLORE = "EXPLORE"
    JUNCTION_DETECTED = "JUNCTION_DETECTED"
    ENTERING_BRANCH = "ENTERING_BRANCH"
    DEAD_END = "DEAD_END"
    BACKTRACKING = "BACKTRACKING"
    SAFETY_OVERRIDE = "SAFETY_OVERRIDE"


@dataclasses.dataclass
class GraphNode:
    node_id: int
    frame_idx: int
    node_type: str  # "start", "junction", "dead_end"
    branches_total: int = 0
    branches_explored: int = 0
    keyframe_buf_idx: int | None = None  # index in place recognizer's buffer
    thumbnail: np.ndarray | None = None  # BGR frame snapshot for visualization


@dataclasses.dataclass
class GraphEdge:
    source: int
    target: int
    frame_cost: int


class ExplorationStateMachine:
    def __init__(self, config: Config):
        self.config = config
        self._state = ExplorationState.EXPLORE
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []

        # Tracking
        self._last_node_frame: int = 0  # frame of last graph node
        self._last_node_id: int | None = None
        self._frames_in_state: int = 0
        self._was_dangerous: bool = False

    @property
    def state(self) -> ExplorationState:
        return self._state

    def initialize(self, frame_idx: int, frame_bgr: np.ndarray | None = None):
        """Create start node."""
        thumb = self._make_thumbnail(frame_bgr) if frame_bgr is not None else None
        node = GraphNode(
            node_id=0,
            frame_idx=frame_idx,
            node_type="start",
            thumbnail=thumb,
        )
        self.nodes.append(node)
        self._last_node_frame = frame_idx
        self._last_node_id = 0

    def update(
        self,
        frame_idx: int,
        proximity: ProximityResult,
        junction: JunctionResult,
        place: PlaceRecognitionResult,
        frame_bgr: np.ndarray | None = None,
    ) -> ExplorationState:
        prev_state = self._state
        self._frames_in_state += 1

        # SAFETY_OVERRIDE takes priority
        if proximity.is_dangerous:
            if self._state != ExplorationState.SAFETY_OVERRIDE:
                self._transition(ExplorationState.SAFETY_OVERRIDE)
            self._was_dangerous = True
            return self._state

        # Exiting safety override
        if self._state == ExplorationState.SAFETY_OVERRIDE and not proximity.is_dangerous:
            self._transition(ExplorationState.EXPLORE)
            self._was_dangerous = False

        # Minimum gap between node creation to prevent rapid re-triggering.
        # Junctions use a longer gap (measured from last junction only).
        # Dead ends use a shorter gap (measured from any node).
        min_gap = self.config.min_junction_gap_frames
        last_junction_frame = max(
            (n.frame_idx for n in self.nodes if n.node_type == "junction"),
            default=0,
        )
        junction_allowed = (frame_idx - last_junction_frame) >= min_gap
        dead_end_allowed = (frame_idx - self._last_node_frame) >= min_gap

        # Dead end detection is checked in all navigating states
        dead_end_detected = (
            junction.confirmed_type == LumenType.DEAD_END
            and dead_end_allowed
        )

        # Junction detection checked in multiple states
        junction_detected = (
            junction.confirmed_type == LumenType.JUNCTION
            and junction_allowed
        )

        # Check if place recognition matches a specific known junction node
        matched_junction_node = None
        if place.is_revisit and place.matched_keyframe_idx is not None:
            for node in self.nodes:
                if (
                    node.node_type == "junction"
                    and node.keyframe_buf_idx == place.matched_keyframe_idx
                ):
                    matched_junction_node = node
                    break

        # State transitions
        if self._state == ExplorationState.EXPLORE:
            if junction_detected:
                self._transition(ExplorationState.JUNCTION_DETECTED)
                self._on_junction(frame_idx, junction, place, frame_bgr)
            elif dead_end_detected:
                self._transition(ExplorationState.DEAD_END)
                self._on_dead_end(frame_idx, place, frame_bgr)
            elif (matched_junction_node is not None
                  and self._frames_in_state > 30):
                self._transition(ExplorationState.BACKTRACKING)

        elif self._state == ExplorationState.JUNCTION_DETECTED:
            self._transition(ExplorationState.ENTERING_BRANCH)

        elif self._state == ExplorationState.ENTERING_BRANCH:
            self._transition(ExplorationState.EXPLORE)

        elif self._state == ExplorationState.DEAD_END:
            self._transition(ExplorationState.BACKTRACKING)

        elif self._state == ExplorationState.BACKTRACKING:
            if junction_detected:
                self._transition(ExplorationState.JUNCTION_DETECTED)
                self._on_junction(frame_idx, junction, place, frame_bgr)
            elif dead_end_detected:
                self._transition(ExplorationState.DEAD_END)
                self._on_dead_end(frame_idx, place, frame_bgr)
            elif (matched_junction_node is not None
                  and matched_junction_node.branches_explored
                  < matched_junction_node.branches_total):
                # Back at a junction with unexplored branches
                self._transition(ExplorationState.JUNCTION_DETECTED)
            elif not place.is_revisit and self._frames_in_state > 30:
                # If we stop matching known places, we've entered new territory
                self._transition(ExplorationState.EXPLORE)

        return self._state

    def _transition(self, new_state: ExplorationState):
        self._state = new_state
        self._frames_in_state = 0

    @staticmethod
    def _make_thumbnail(frame_bgr: np.ndarray | None, size: int = 64) -> np.ndarray | None:
        if frame_bgr is None:
            return None
        import cv2
        h, w = frame_bgr.shape[:2]
        scale = size / max(h, w)
        return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def _on_junction(
        self, frame_idx: int, junction: JunctionResult,
        place: PlaceRecognitionResult, frame_bgr: np.ndarray | None = None
    ):
        """Handle arriving at a junction."""
        if place.is_revisit and place.matched_keyframe_idx is not None:
            for node in self.nodes:
                if node.keyframe_buf_idx == place.matched_keyframe_idx:
                    node.branches_explored = min(
                        node.branches_explored + 1, node.branches_total
                    )
                    self._add_edge_from_last(node.node_id, frame_idx)
                    self._last_node_id = node.node_id
                    self._last_node_frame = frame_idx
                    return

        node_id = len(self.nodes)
        node = GraphNode(
            node_id=node_id,
            frame_idx=frame_idx,
            node_type="junction",
            branches_total=junction.blob_count,
            branches_explored=1,
            keyframe_buf_idx=place.matched_keyframe_idx,
            thumbnail=self._make_thumbnail(frame_bgr),
        )
        self.nodes.append(node)
        self._add_edge_from_last(node_id, frame_idx)
        self._last_node_id = node_id
        self._last_node_frame = frame_idx

    def _on_dead_end(
        self, frame_idx: int, place: PlaceRecognitionResult,
        frame_bgr: np.ndarray | None = None
    ):
        """Handle arriving at a dead end."""
        node_id = len(self.nodes)
        node = GraphNode(
            node_id=node_id,
            frame_idx=frame_idx,
            node_type="dead_end",
            keyframe_buf_idx=place.matched_keyframe_idx,
            thumbnail=self._make_thumbnail(frame_bgr),
        )
        self.nodes.append(node)
        self._add_edge_from_last(node_id, frame_idx)
        self._last_node_id = node_id
        self._last_node_frame = frame_idx

    def _add_edge_from_last(self, target_id: int, current_frame: int):
        if self._last_node_id is not None:
            cost = current_frame - self._last_node_frame
            # Avoid duplicate edges
            for edge in self.edges:
                if edge.source == self._last_node_id and edge.target == target_id:
                    return
            self.edges.append(
                GraphEdge(
                    source=self._last_node_id,
                    target=target_id,
                    frame_cost=max(cost, 1),
                )
            )

    def get_coverage_report(self) -> dict:
        """Analyze topological graph for coverage completeness."""
        junction_nodes = [n for n in self.nodes if n.node_type == "junction"]
        dead_end_nodes = [n for n in self.nodes if n.node_type == "dead_end"]
        terminal_count = len(dead_end_nodes)

        # Unexplored branches
        unexplored = 0
        for jn in junction_nodes:
            unexplored += max(0, jn.branches_total - jn.branches_explored)

        coverage_complete = (
            terminal_count >= self.config.min_expected_branches
            and unexplored == 0
        )

        return {
            "total_nodes": len(self.nodes),
            "junction_nodes": len(junction_nodes),
            "dead_end_nodes": terminal_count,
            "total_edges": len(self.edges),
            "unexplored_branches": unexplored,
            "min_expected_branches": self.config.min_expected_branches,
            "max_expected_branches": self.config.max_expected_branches,
            "coverage_complete": coverage_complete,
            "coverage_note": (
                "Coverage appears complete"
                if coverage_complete
                else f"Incomplete: found {terminal_count} terminal branches "
                f"(expected {self.config.min_expected_branches}-"
                f"{self.config.max_expected_branches}), "
                f"{unexplored} unexplored branches remain"
            ),
        }
