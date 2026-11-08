"""Deterministic benchmark generation and validation for BorgesBench.

The benchmark is deliberately structural: no LLM is needed to generate or
score an instance.  Each instance is a directed acyclic graph (DAG) whose
nodes carry narrative metadata and whose edges identify controlled coherent
and incoherent continuations.  The public generator creates the frozen
factorial design used by the experiments::

    3 families x 2 sizes x 3 difficulties x 5 replicates = 90 instances

All randomness is derived from a stable hash of the root seed and the design
cell, so output is independent of iteration order and Python hash seeding.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = "1.0"
DEFAULT_SEED = 20260803
FAMILIES = ("labyrinth", "double", "circular_time")
SIZES = (30, 60)
DIFFICULTIES = ("low", "medium", "high")
REPLICATES = 5
MIN_PATH_LENGTH = 6
MAX_PATH_LENGTH = 10
OBJECTIVE_NAMES = ("coherence", "thematic_coverage", "structural_novelty")


FAMILY_SPEC: dict[str, dict[str, Any]] = {
    "labyrinth": {
        "label": "The Labyrinth",
        "motifs": ("labyrinth", "choice", "false_exit"),
        "characters": ("seeker", "cartographer", "gatekeeper"),
        "locations": ("entrance", "forking_gallery", "blind_chamber", "archive"),
        "violation": "spatial_discontinuity",
    },
    "double": {
        "label": "The Double",
        "motifs": ("double", "identity", "mirror"),
        "characters": ("original", "counterpart", "witness"),
        "locations": ("study", "mirror_hall", "station", "court"),
        "violation": "identity_conflation",
    },
    "circular_time": {
        "label": "The Circular Time",
        "motifs": ("recurrence", "memory", "causal_loop"),
        "characters": ("traveler", "archivist", "future_self"),
        "locations": ("observatory", "archive", "threshold", "return_room"),
        "violation": "temporal_contradiction",
    },
}


DIFFICULTY_SPEC: dict[str, dict[str, float | int]] = {
    "low": {
        "target_invalid_ratio": 0.10,
        "target_mean_out_degree": 2.0,
        "mandatory_count": 1,
        "long_range_count": 1,
        "incompatible_count": 1,
    },
    "medium": {
        "target_invalid_ratio": 0.20,
        "target_mean_out_degree": 2.8,
        "mandatory_count": 2,
        "long_range_count": 2,
        "incompatible_count": 3,
    },
    "high": {
        "target_invalid_ratio": 0.30,
        "target_mean_out_degree": 3.6,
        "mandatory_count": 3,
        "long_range_count": 3,
        "incompatible_count": 5,
    },
}


NODE_REQUIRED_FIELDS = {
    "id",
    "order",
    "event_type",
    "text",
    "characters",
    "introduces",
    "location",
    "time_index",
    "motifs",
    "preconditions",
    "thematic_weight",
    "structural_rarity",
}
EDGE_REQUIRED_FIELDS = {
    "source",
    "target",
    "valid",
    "coherence",
    "violation",
    "structural_rarity",
}


def _stable_seed(root_seed: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in (root_seed, *parts)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _normalise_family(value: str) -> str:
    key = value.strip().lower().replace("the ", "").replace("-", "_").replace(" ", "_")
    aliases = {"circular": "circular_time", "circularity": "circular_time"}
    key = aliases.get(key, key)
    if key not in FAMILIES:
        raise ValueError(f"Unknown family {value!r}; expected one of {FAMILIES}")
    return key


def _normalise_difficulty(value: str) -> str:
    key = value.strip().lower()
    if key not in DIFFICULTIES:
        raise ValueError(f"Unknown difficulty {value!r}; expected one of {DIFFICULTIES}")
    return key


@dataclass
class NarrativeInstance:
    """A JSON-serialisable branching-narrative search instance."""

    instance_id: str
    family: str
    size: int
    difficulty: str
    replicate: int
    seed: int
    nodes: dict[str, dict[str, Any]]
    edges: list[dict[str, Any]]
    start_node: str
    end_nodes: list[str]
    min_path_length: int
    max_path_length: int
    mandatory_events: list[str]
    incompatible_pairs: list[list[str]]
    required_motifs: list[str]
    long_range_constraints: list[dict[str, str]]
    witness_path: list[str]
    soft_objectives: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def end_node(self) -> str:
        """The canonical end node (the generated design has exactly one)."""
        return self.end_nodes[0]

    @property
    def hard_constraints(self) -> dict[str, Any]:
        """Return the hard-constraint block expected by external consumers."""
        return {
            "start_node": self.start_node,
            "end_nodes": list(self.end_nodes),
            "min_path_length": self.min_path_length,
            "max_path_length": self.max_path_length,
            "mandatory_events": list(self.mandatory_events),
            "incompatible_pairs": [list(pair) for pair in self.incompatible_pairs],
            "long_range_constraints": [dict(item) for item in self.long_range_constraints],
            "characters_must_be_introduced": True,
            "transitions_must_exist_and_be_valid": True,
            "nodes_must_not_repeat": True,
        }

    def successors(self, node_id: str, valid_only: bool = False) -> list[str]:
        """Return successors in stable topological order."""
        values = [
            edge["target"]
            for edge in self.edges
            if edge["source"] == node_id and (edge["valid"] or not valid_only)
        ]
        return sorted(values, key=lambda item: self.nodes[item]["order"])

    def predecessors(self, node_id: str, valid_only: bool = False) -> list[str]:
        values = [
            edge["source"]
            for edge in self.edges
            if edge["target"] == node_id and (edge["valid"] or not valid_only)
        ]
        return sorted(values, key=lambda item: self.nodes[item]["order"])

    def edge(self, source: str, target: str) -> dict[str, Any] | None:
        for edge in self.edges:
            if edge["source"] == source and edge["target"] == target:
                return edge
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "instance_id": self.instance_id,
            "family": self.family,
            "size": self.size,
            "difficulty": self.difficulty,
            "replicate": self.replicate,
            "seed": self.seed,
            "nodes": self.nodes,
            "edges": self.edges,
            "start_node": self.start_node,
            "end_nodes": self.end_nodes,
            "min_path_length": self.min_path_length,
            "max_path_length": self.max_path_length,
            "mandatory_events": self.mandatory_events,
            "incompatible_pairs": self.incompatible_pairs,
            "required_motifs": self.required_motifs,
            "long_range_constraints": self.long_range_constraints,
            "witness_path": self.witness_path,
            "hard_constraints": self.hard_constraints,
            "soft_objectives": self.soft_objectives,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NarrativeInstance":
        hard = data.get("hard_constraints", {})
        fields = {
            "instance_id": str(data["instance_id"]),
            "family": str(data["family"]),
            "size": int(data["size"]),
            "difficulty": str(data["difficulty"]),
            "replicate": int(data["replicate"]),
            "seed": int(data["seed"]),
            "nodes": {str(key): dict(value) for key, value in data["nodes"].items()},
            "edges": [dict(edge) for edge in data["edges"]],
            "start_node": str(data.get("start_node", hard.get("start_node"))),
            "end_nodes": list(data.get("end_nodes", hard.get("end_nodes", []))),
            "min_path_length": int(data.get("min_path_length", hard.get("min_path_length", 6))),
            "max_path_length": int(data.get("max_path_length", hard.get("max_path_length", 10))),
            "mandatory_events": list(data.get("mandatory_events", hard.get("mandatory_events", []))),
            "incompatible_pairs": [
                list(pair) for pair in data.get("incompatible_pairs", hard.get("incompatible_pairs", []))
            ],
            "required_motifs": list(data.get("required_motifs", [])),
            "long_range_constraints": [
                dict(item)
                for item in data.get(
                    "long_range_constraints", hard.get("long_range_constraints", [])
                )
            ],
            "witness_path": list(data["witness_path"]),
            "soft_objectives": dict(data.get("soft_objectives", {})),
            "metadata": dict(data.get("metadata", {})),
        }
        return cls(**fields)

    def to_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination

    @classmethod
    def from_json(cls, path: str | Path) -> "NarrativeInstance":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self, *, raise_on_error: bool = False) -> "ValidationReport":
        return validate_instance(self, raise_on_error=raise_on_error)


@dataclass(frozen=True)
class ValidationReport:
    """Structured result of exhaustive instance validation."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, Any]

    def __bool__(self) -> bool:
        return self.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "metrics": self.metrics,
        }


def _anchor_indices(size: int, path_length: int) -> list[int]:
    # Integer interpolation includes both endpoints and is unique because
    # ``size`` is much greater than the maximum path length.
    return [round(index * (size - 1) / (path_length - 1)) for index in range(path_length)]


def _event_type(family: str, order: int, size: int) -> str:
    if order == 0:
        return "opening"
    if order == size - 1:
        return "resolution"
    family_events = {
        "labyrinth": ("choice", "passage", "false_exit", "clue"),
        "double": ("encounter", "testimony", "reflection", "disclosure"),
        "circular_time": ("departure", "memory", "premonition", "return"),
    }
    return family_events[family][order % len(family_events[family])]


def _make_nodes(
    instance_id: str,
    family: str,
    size: int,
    anchors: Sequence[int],
    rng: random.Random,
) -> dict[str, dict[str, Any]]:
    spec = FAMILY_SPEC[family]
    characters = spec["characters"]
    introduction_order = {
        characters[0]: 0,
        characters[1]: anchors[1],
        characters[2]: anchors[2],
    }
    nodes: dict[str, dict[str, Any]] = {}
    anchor_position = {order: position for position, order in enumerate(anchors)}
    for order in range(size):
        node_id = f"{instance_id}_n{order:03d}"
        introduced = [name for name, intro_order in introduction_order.items() if intro_order == order]
        available = [name for name, intro_order in introduction_order.items() if intro_order <= order]
        if order in anchor_position:
            # The witness makes all canonical introductions and carries at
            # least one required motif per internal segment.
            present = available[-2:] if len(available) > 1 else available
            motif = spec["motifs"][anchor_position[order] % len(spec["motifs"])]
            motifs = [motif]
        else:
            present = [rng.choice(available)]
            if len(available) > 1 and rng.random() < 0.30:
                present.append(rng.choice([item for item in available if item not in present]))
            motifs = [rng.choice(spec["motifs"])] if rng.random() < 0.58 else []
        # A node featuring a non-primary character requires that character's
        # unique introduction event.  Skipping it creates a controlled hard
        # violation even though the graph transition itself may exist.
        preconditions = []
        for character in present:
            intro_order = introduction_order[character]
            if intro_order not in (0, order):
                preconditions.append(f"{instance_id}_n{intro_order:03d}")
        location = spec["locations"][order % len(spec["locations"])]
        event = _event_type(family, order, size)
        nodes[node_id] = {
            "id": node_id,
            "order": order,
            "event_type": event,
            "text": f"{spec['label']} event {order}: {event} at the {location}.",
            "characters": sorted(set(present)),
            "introduces": introduced,
            "location": location,
            "time_index": order,
            "motifs": motifs,
            "preconditions": sorted(set(preconditions)),
            "thematic_weight": round(0.55 + 0.45 * rng.random(), 6),
            "structural_rarity": 0.0,  # filled after edge generation
        }
    return nodes


def _candidate_edges(
    family: str,
    size: int,
    desired_count: int,
    anchors: Sequence[int],
    rng: random.Random,
) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set(zip(anchors, anchors[1:]))
    # The chain keeps every node structurally reachable from the opening and
    # able to reach the resolution (validity is labelled separately).
    pairs.update((index, index + 1) for index in range(size - 1))
    candidates = [
        (left, right) for left in range(size - 1) for right in range(left + 2, size)
    ]
    # Families have distinct structural signatures while preserving the same
    # edge budget.  A random tie-breaker keeps replications heterogeneous.
    tie_break = {pair: rng.random() for pair in candidates}
    if family == "labyrinth":
        # Dense local forks and later reconvergence create many alternative
        # corridors rather than a few very long shortcuts.
        target_span = max(3, round(size / 8))
        candidates.sort(
            key=lambda pair: (
                abs((pair[1] - pair[0]) - target_span),
                tie_break[pair],
            )
        )
    elif family == "double":
        # Prefer paired, near-parallel corridors that split and merge over a
        # small window, reflecting the identity-double construction.
        def double_priority(pair: tuple[int, int]) -> tuple[float, float]:
            left, right = pair
            span = right - left
            paired_window = min(abs(span - 2), abs(span - 3))
            parity_bonus = 0.0 if left % 2 == right % 2 else 0.35
            return paired_window + parity_bonus, tie_break[pair]

        candidates.sort(key=double_priority)
    else:
        # Long forward dependencies approximate circular-time recurrences in a
        # DAG: late events can depend on much earlier information, while
        # invalid versions are labelled temporal contradictions.
        candidates.sort(key=lambda pair: (-(pair[1] - pair[0]), tie_break[pair]))
    for pair in candidates:
        if len(pairs) >= desired_count:
            break
        pairs.add(pair)
    return pairs


def _make_edges(
    instance_id: str,
    family: str,
    size: int,
    anchors: Sequence[int],
    difficulty: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    parameters = DIFFICULTY_SPEC[difficulty]
    desired_count = max(size - 1, round(size * float(parameters["target_mean_out_degree"])))
    pairs = _candidate_edges(family, size, desired_count, anchors, rng)
    witness_pairs = set(zip(anchors, anchors[1:]))
    eligible_invalid = sorted(pairs - witness_pairs)
    rng.shuffle(eligible_invalid)
    invalid_count = min(
        len(eligible_invalid), round(len(pairs) * float(parameters["target_invalid_ratio"]))
    )
    invalid_pairs = set(eligible_invalid[:invalid_count])
    edges: list[dict[str, Any]] = []
    for source_order, target_order in sorted(pairs):
        valid = (source_order, target_order) not in invalid_pairs
        if valid:
            coherence = round(rng.uniform(0.67, 1.0), 6)
            violation = None
        else:
            coherence = round(rng.uniform(0.05, 0.42), 6)
            violation = FAMILY_SPEC[family]["violation"]
        edges.append(
            {
                "source": f"{instance_id}_n{source_order:03d}",
                "target": f"{instance_id}_n{target_order:03d}",
                "valid": valid,
                "coherence": coherence,
                "violation": violation,
                "structural_rarity": 0.0,
            }
        )
    return edges


def _add_static_rarity(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    """Add population-independent rarity scores derived from local topology."""
    indegree = {node_id: 0 for node_id in nodes}
    outdegree = {node_id: 0 for node_id in nodes}
    for edge in edges:
        outdegree[edge["source"]] += 1
        indegree[edge["target"]] += 1
    for node_id, node in nodes.items():
        in_rarity = 1.0 / max(1, indegree[node_id])
        out_rarity = 1.0 / max(1, outdegree[node_id])
        node["structural_rarity"] = round((in_rarity + out_rarity) / 2.0, 6)
    for edge in edges:
        edge["structural_rarity"] = round(
            (
                1.0 / max(1, outdegree[edge["source"]])
                + 1.0 / max(1, indegree[edge["target"]])
            )
            / 2.0,
            6,
        )


def generate_instance(
    family: str,
    size: int,
    difficulty: str,
    replicate: int = 0,
    *,
    seed: int = DEFAULT_SEED,
) -> NarrativeInstance:
    """Generate one deterministic, fully validated instance.

    ``replicate`` is zero-based.  A stable cell seed is stored in the instance;
    passing the same arguments always yields byte-equivalent ``to_dict`` data.
    """
    family = _normalise_family(family)
    difficulty = _normalise_difficulty(difficulty)
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}")
    if replicate not in range(REPLICATES):
        raise ValueError(f"replicate must be in [0, {REPLICATES - 1}]")
    cell_seed = _stable_seed(seed, family, size, difficulty, replicate)
    rng = random.Random(cell_seed)
    family_code = {"labyrinth": "lab", "double": "dbl", "circular_time": "cir"}[family]
    difficulty_code = {"low": "lo", "medium": "md", "high": "hi"}[difficulty]
    instance_id = f"bb_{family_code}_{size}_{difficulty_code}_{replicate + 1:02d}"
    path_length = MIN_PATH_LENGTH + (cell_seed % (MAX_PATH_LENGTH - MIN_PATH_LENGTH + 1))
    anchors = _anchor_indices(size, path_length)
    nodes = _make_nodes(instance_id, family, size, anchors, rng)
    edges = _make_edges(instance_id, family, size, anchors, difficulty, rng)
    _add_static_rarity(nodes, edges)
    witness = [f"{instance_id}_n{order:03d}" for order in anchors]
    parameters = DIFFICULTY_SPEC[difficulty]
    internal_witness = witness[1:-1]
    mandatory_count = int(parameters["mandatory_count"])
    mandatory = internal_witness[:mandatory_count]

    constraint_count = int(parameters["long_range_count"])
    long_range: list[dict[str, str]] = []
    candidate_positions = list(range(2, len(witness) - 1))
    for offset in range(constraint_count):
        after_position = candidate_positions[offset % len(candidate_positions)]
        before_position = max(1, after_position - 2)
        long_range.append(
            {
                "before": witness[before_position],
                "after": witness[after_position],
                "type": "causal_dependency",
            }
        )
        preconditions = nodes[witness[after_position]]["preconditions"]
        if witness[before_position] not in preconditions:
            preconditions.append(witness[before_position])
            preconditions.sort()

    distractors = [node_id for node_id in nodes if node_id not in witness]
    rng.shuffle(distractors)
    incompatible: list[list[str]] = []
    for left, right in zip(distractors[::2], distractors[1::2]):
        if len(incompatible) >= int(parameters["incompatible_count"]):
            break
        incompatible.append(sorted([left, right], key=lambda item: nodes[item]["order"]))

    actual_invalid_ratio = sum(not edge["valid"] for edge in edges) / len(edges)
    actual_mean_out = len(edges) / size
    instance = NarrativeInstance(
        instance_id=instance_id,
        family=family,
        size=size,
        difficulty=difficulty,
        replicate=replicate,
        seed=cell_seed,
        nodes=nodes,
        edges=edges,
        start_node=witness[0],
        end_nodes=[witness[-1]],
        min_path_length=MIN_PATH_LENGTH,
        max_path_length=MAX_PATH_LENGTH,
        mandatory_events=mandatory,
        incompatible_pairs=incompatible,
        required_motifs=list(FAMILY_SPEC[family]["motifs"]),
        long_range_constraints=long_range,
        witness_path=witness,
        soft_objectives={
            "names": list(OBJECTIVE_NAMES),
            "direction": "maximize",
            "range": [0.0, 1.0],
            "coherence": "65% mean edge coherence + 35% hard-constraint satisfaction",
            "thematic_coverage": "required motifs observed / required motifs",
            "structural_novelty": "mean static node/edge rarity from graph degree",
        },
        metadata={
            "family_label": FAMILY_SPEC[family]["label"],
            "generator": "borgesbench.structural.v1",
            "root_seed": int(seed),
            "frozen": True,
            "generation_parameters": dict(parameters),
            "observed_properties": {
                "edge_count": len(edges),
                "valid_edge_count": sum(edge["valid"] for edge in edges),
                "invalid_edge_count": sum(not edge["valid"] for edge in edges),
                "invalid_edge_ratio": round(actual_invalid_ratio, 6),
                "mean_out_degree": round(actual_mean_out, 6),
                "mandatory_count": len(mandatory),
                "long_range_count": len(long_range),
                "incompatible_count": len(incompatible),
                "witness_length": len(witness),
            },
        },
    )
    validate_instance(instance, raise_on_error=True)
    return instance


def generate_benchmark(*, seed: int = DEFAULT_SEED, validate: bool = True) -> list[NarrativeInstance]:
    """Generate the complete 90-instance benchmark in canonical order."""
    instances = [
        generate_instance(family, size, difficulty, replicate, seed=seed)
        for family in FAMILIES
        for size in SIZES
        for difficulty in DIFFICULTIES
        for replicate in range(REPLICATES)
    ]
    if validate:
        validate_benchmark(instances, raise_on_error=True)
    return instances


def _coerce_instance(instance: NarrativeInstance | Mapping[str, Any]) -> NarrativeInstance:
    return instance if isinstance(instance, NarrativeInstance) else NarrativeInstance.from_dict(instance)


def _path_violations(instance: NarrativeInstance, path: Sequence[str]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if not path:
        return [{"type": "empty_path", "detail": "A path must contain nodes."}]
    unknown = [node_id for node_id in path if node_id not in instance.nodes]
    for node_id in unknown:
        violations.append({"type": "unknown_node", "node": node_id})
    if unknown:
        return violations
    if path[0] != instance.start_node:
        violations.append(
            {"type": "invalid_start", "expected": instance.start_node, "observed": path[0]}
        )
    if path[-1] not in instance.end_nodes:
        violations.append(
            {"type": "invalid_end", "expected": instance.end_nodes, "observed": path[-1]}
        )
    if len(path) < instance.min_path_length or len(path) > instance.max_path_length:
        violations.append(
            {
                "type": "invalid_length",
                "minimum": instance.min_path_length,
                "maximum": instance.max_path_length,
                "observed": len(path),
            }
        )
    repeated = sorted({node_id for node_id in path if path.count(node_id) > 1})
    for node_id in repeated:
        violations.append({"type": "repeated_node", "node": node_id})
    for source, target in zip(path, path[1:]):
        edge = instance.edge(source, target)
        if edge is None:
            violations.append({"type": "missing_transition", "source": source, "target": target})
        elif not edge["valid"]:
            violations.append(
                {
                    "type": "invalid_transition",
                    "source": source,
                    "target": target,
                    "cause": edge["violation"],
                }
            )
    positions = {node_id: index for index, node_id in enumerate(path)}
    for node_id in instance.mandatory_events:
        if node_id not in positions:
            violations.append({"type": "missing_mandatory_event", "node": node_id})
    for left, right in instance.incompatible_pairs:
        if left in positions and right in positions:
            violations.append({"type": "incompatible_events", "nodes": [left, right]})
    introduced: set[str] = set()
    for node_id in path:
        node = instance.nodes[node_id]
        introduced.update(node["introduces"])
        for character in node["characters"]:
            if character not in introduced:
                violations.append(
                    {"type": "character_before_introduction", "node": node_id, "character": character}
                )
        for prerequisite in node["preconditions"]:
            if prerequisite not in positions or positions[prerequisite] >= positions[node_id]:
                violations.append(
                    {"type": "unsatisfied_precondition", "node": node_id, "requires": prerequisite}
                )
    for constraint in instance.long_range_constraints:
        before, after = constraint["before"], constraint["after"]
        if after in positions and (before not in positions or positions[before] >= positions[after]):
            marker = {"type": "long_range_constraint", "before": before, "after": after}
            if marker not in violations:
                violations.append(marker)
    return violations


def evaluate_path(
    instance: NarrativeInstance | Mapping[str, Any] | Sequence[str],
    path: Sequence[str] | NarrativeInstance | Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one path using static, algorithm-independent objectives.

    The canonical signature is ``evaluate_path(instance, path)``.  For
    convenience when using older notebooks, ``evaluate_path(path, instance)``
    is also accepted and normalised here.
    """
    if not isinstance(instance, (NarrativeInstance, Mapping)) and isinstance(
        path, (NarrativeInstance, Mapping)
    ):
        instance, path = path, instance
    graph = _coerce_instance(instance)  # type: ignore[arg-type]
    candidate = list(path)  # type: ignore[arg-type]
    violations = _path_violations(graph, candidate)
    known_nodes = [node_id for node_id in candidate if node_id in graph.nodes]
    traversed_edges = [
        edge
        for source, target in zip(candidate, candidate[1:])
        if (edge := graph.edge(source, target)) is not None
    ]
    mean_edge_coherence = (
        mean(float(edge["coherence"]) for edge in traversed_edges) if traversed_edges else 0.0
    )
    # Missing/invalid hard conditions lower coherence while remaining distinct
    # from feasibility, which the optimizers handle constraint-first.
    hard_denominator = max(
        1,
        5
        + max(0, len(candidate) - 1)
        + len(graph.mandatory_events)
        + len(graph.long_range_constraints),
    )
    hard_satisfaction = max(0.0, 1.0 - len(violations) / hard_denominator)
    coherence = 0.65 * mean_edge_coherence + 0.35 * hard_satisfaction
    observed_motifs = {
        motif for node_id in known_nodes for motif in graph.nodes[node_id].get("motifs", [])
    }
    motif_denominator = max(1, len(graph.required_motifs))
    thematic_coverage = len(observed_motifs.intersection(graph.required_motifs)) / motif_denominator
    rarity_values = [float(graph.nodes[node_id]["structural_rarity"]) for node_id in known_nodes]
    rarity_values.extend(float(edge["structural_rarity"]) for edge in traversed_edges)
    structural_novelty = mean(rarity_values) if rarity_values else 0.0
    objectives = {
        "coherence": round(min(1.0, max(0.0, coherence)), 10),
        "thematic_coverage": round(min(1.0, max(0.0, thematic_coverage)), 10),
        "structural_novelty": round(min(1.0, max(0.0, structural_novelty)), 10),
    }
    return {
        **objectives,
        "objectives": objectives,
        "feasible": not violations,
        "violations": violations,
        "violation_count": len(violations),
    }


def set_diversity(paths: Iterable[Sequence[str]]) -> float:
    """Mean pairwise Jaccard distance over directed edge sets."""
    edge_sets = [set(zip(path, path[1:])) for path in paths]
    if len(edge_sets) < 2:
        return 0.0
    distances = []
    for left, right in combinations(edge_sets, 2):
        union = left | right
        distances.append(1.0 - len(left & right) / len(union) if union else 0.0)
    return float(mean(distances))


def _shortest_valid_segment(instance: NarrativeInstance, source: str, target: str) -> list[str] | None:
    if source == target:
        return [source]
    queue: deque[list[str]] = deque([[source]])
    visited = {source}
    while queue:
        prefix = queue.popleft()
        for successor in instance.successors(prefix[-1], valid_only=True):
            if instance.nodes[successor]["order"] > instance.nodes[target]["order"]:
                continue
            candidate = prefix + [successor]
            if successor == target:
                return candidate
            if successor not in visited:
                visited.add(successor)
                queue.append(candidate)
    return None


def repair_path(
    instance: NarrativeInstance | Mapping[str, Any],
    path: Sequence[str],
    rng: Any | None = None,
) -> list[str]:
    """Repair an arbitrary genotype to a feasible path when possible.

    Existing known nodes are retained in topological order, incompatible events
    are removed, mandatory events are inserted, and gaps are bridged by shortest
    valid segments.  If hard constraints still fail, the certified witness is
    returned.  ``rng`` is accepted for a uniform optimizer API; tie-breaking is
    intentionally deterministic so repair itself does not consume evaluations.
    """
    del rng  # deterministic repair makes ablation and seed accounting cleaner
    graph = _coerce_instance(instance)
    selected = {node_id for node_id in path if node_id in graph.nodes}
    selected.update(graph.mandatory_events)
    selected.add(graph.start_node)
    selected.add(graph.end_node)
    for left, right in graph.incompatible_pairs:
        if left in selected and right in selected:
            # Preserve a mandatory endpoint if present, otherwise preserve the
            # earlier event to respect the DAG's temporal progression.
            if right in graph.mandatory_events:
                selected.discard(left)
            else:
                selected.discard(right)
    checkpoints = sorted(selected, key=lambda item: graph.nodes[item]["order"])
    assembled = [checkpoints[0]]
    for checkpoint in checkpoints[1:]:
        segment = _shortest_valid_segment(graph, assembled[-1], checkpoint)
        if segment is None:
            return list(graph.witness_path)
        assembled.extend(segment[1:])
        if len(assembled) > graph.max_path_length:
            return list(graph.witness_path)
    if evaluate_path(graph, assembled)["feasible"]:
        return assembled
    return list(graph.witness_path)


def _reachable(instance: NarrativeInstance, start: str, *, reverse: bool = False) -> set[str]:
    seen = {start}
    queue = deque([start])
    while queue:
        node_id = queue.popleft()
        adjacent = instance.predecessors(node_id) if reverse else instance.successors(node_id)
        for neighbor in adjacent:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def validate_instance(
    instance: NarrativeInstance | Mapping[str, Any], *, raise_on_error: bool = False
) -> ValidationReport:
    """Exhaustively validate schema, DAG, constraints, labels and witness."""
    graph = _coerce_instance(instance)
    errors: list[str] = []
    warnings: list[str] = []
    if graph.family not in FAMILIES:
        errors.append(f"unknown family: {graph.family}")
    if graph.difficulty not in DIFFICULTIES:
        errors.append(f"unknown difficulty: {graph.difficulty}")
    if graph.size not in SIZES:
        errors.append(f"unsupported size: {graph.size}")
    if len(graph.nodes) != graph.size:
        errors.append(f"declared size {graph.size} != {len(graph.nodes)} nodes")
    if graph.start_node not in graph.nodes:
        errors.append("start node does not exist")
    if not graph.end_nodes or any(node_id not in graph.nodes for node_id in graph.end_nodes):
        errors.append("one or more end nodes do not exist")
    if not (MIN_PATH_LENGTH <= graph.min_path_length <= graph.max_path_length <= MAX_PATH_LENGTH):
        errors.append("path bounds must remain within [6, 10]")

    orders: list[int] = []
    for node_id, node in graph.nodes.items():
        missing = NODE_REQUIRED_FIELDS - set(node)
        if missing:
            errors.append(f"node {node_id} misses fields {sorted(missing)}")
            continue
        if node["id"] != node_id:
            errors.append(f"node key/id mismatch for {node_id}")
        orders.append(node["order"])
        if not 0.0 <= float(node["thematic_weight"]) <= 1.0:
            errors.append(f"node {node_id} thematic_weight outside [0,1]")
        if not 0.0 <= float(node["structural_rarity"]) <= 1.0:
            errors.append(f"node {node_id} structural_rarity outside [0,1]")
        for prerequisite in node["preconditions"]:
            if prerequisite not in graph.nodes:
                errors.append(f"node {node_id} has unknown precondition {prerequisite}")
            elif graph.nodes[prerequisite]["order"] >= node["order"]:
                errors.append(f"node {node_id} has non-prior precondition {prerequisite}")
    if len(orders) != len(set(orders)) or sorted(orders) != list(range(len(graph.nodes))):
        errors.append("node orders must be a permutation of 0..size-1")

    edge_keys: set[tuple[str, str]] = set()
    for index, edge in enumerate(graph.edges):
        missing = EDGE_REQUIRED_FIELDS - set(edge)
        if missing:
            errors.append(f"edge {index} misses fields {sorted(missing)}")
            continue
        source, target = edge["source"], edge["target"]
        if source not in graph.nodes or target not in graph.nodes:
            errors.append(f"edge {source}->{target} references an unknown node")
            continue
        if (source, target) in edge_keys:
            errors.append(f"duplicate edge {source}->{target}")
        edge_keys.add((source, target))
        if graph.nodes[source]["order"] >= graph.nodes[target]["order"]:
            errors.append(f"edge {source}->{target} violates DAG order")
        if not isinstance(edge["valid"], bool):
            errors.append(f"edge {source}->{target} valid flag is not boolean")
        if not 0.0 <= float(edge["coherence"]) <= 1.0:
            errors.append(f"edge {source}->{target} coherence outside [0,1]")
        if not 0.0 <= float(edge["structural_rarity"]) <= 1.0:
            errors.append(f"edge {source}->{target} rarity outside [0,1]")
        if edge["valid"] and edge["violation"] is not None:
            errors.append(f"valid edge {source}->{target} has a violation label")
        if not edge["valid"] and not edge["violation"]:
            errors.append(f"invalid edge {source}->{target} lacks a violation label")

    if graph.edges and not any(edge["valid"] for edge in graph.edges):
        errors.append("graph has no valid edges")
    if graph.edges and not any(not edge["valid"] for edge in graph.edges):
        errors.append("graph has no controlled invalid edges")
    if graph.start_node in graph.nodes:
        reachable = _reachable(graph, graph.start_node)
        unreachable = sorted(set(graph.nodes) - reachable)
        if unreachable:
            errors.append(f"{len(unreachable)} nodes are unreachable from start")
    if graph.end_nodes and all(end in graph.nodes for end in graph.end_nodes):
        can_reach_end = set().union(*(_reachable(graph, end, reverse=True) for end in graph.end_nodes))
        dead = sorted(set(graph.nodes) - can_reach_end)
        if dead:
            errors.append(f"{len(dead)} nodes cannot reach an end")

    for node_id in graph.mandatory_events:
        if node_id not in graph.nodes:
            errors.append(f"unknown mandatory event {node_id}")
    seen_pairs: set[tuple[str, str]] = set()
    for pair in graph.incompatible_pairs:
        if len(pair) != 2 or pair[0] == pair[1]:
            errors.append(f"malformed incompatible pair {pair}")
            continue
        if any(item not in graph.nodes for item in pair):
            errors.append(f"incompatible pair references unknown node: {pair}")
        key = tuple(sorted(pair))
        if key in seen_pairs:
            errors.append(f"duplicate incompatible pair: {pair}")
        seen_pairs.add(key)
    graph_motifs = {motif for node in graph.nodes.values() for motif in node["motifs"]}
    missing_motifs = set(graph.required_motifs) - graph_motifs
    if missing_motifs:
        errors.append(f"required motifs absent from graph: {sorted(missing_motifs)}")

    witness_result = evaluate_path(graph, graph.witness_path)
    if not witness_result["feasible"]:
        errors.append(f"witness path is infeasible: {witness_result['violations']}")
    parameters = DIFFICULTY_SPEC.get(graph.difficulty)
    invalid_ratio = (
        sum(not edge["valid"] for edge in graph.edges) / len(graph.edges) if graph.edges else 0.0
    )
    mean_out_degree = len(graph.edges) / len(graph.nodes) if graph.nodes else 0.0
    if parameters:
        target_ratio = float(parameters["target_invalid_ratio"])
        if abs(invalid_ratio - target_ratio) > max(0.02, 1.0 / max(1, len(graph.edges))):
            errors.append(
                f"invalid-edge ratio {invalid_ratio:.3f} does not match {graph.difficulty} target {target_ratio:.3f}"
            )
        expected_counts = {
            "mandatory events": (len(graph.mandatory_events), int(parameters["mandatory_count"])),
            "long-range constraints": (
                len(graph.long_range_constraints),
                int(parameters["long_range_count"]),
            ),
            "incompatible pairs": (
                len(graph.incompatible_pairs),
                int(parameters["incompatible_count"]),
            ),
        }
        for name, (observed, expected) in expected_counts.items():
            if observed != expected:
                errors.append(f"{name}: observed {observed}, expected {expected} for {graph.difficulty}")
    metrics = {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "invalid_edge_ratio": round(invalid_ratio, 6),
        "mean_out_degree": round(mean_out_degree, 6),
        "mandatory_count": len(graph.mandatory_events),
        "long_range_count": len(graph.long_range_constraints),
        "incompatible_count": len(graph.incompatible_pairs),
        "witness_length": len(graph.witness_path),
        "witness_objectives": witness_result["objectives"],
    }
    report = ValidationReport(not errors, tuple(errors), tuple(warnings), metrics)
    if raise_on_error and errors:
        raise ValueError(f"Invalid instance {graph.instance_id}: " + "; ".join(errors))
    return report


def validate_benchmark(
    instances: Iterable[NarrativeInstance | Mapping[str, Any]], *, raise_on_error: bool = False
) -> dict[str, Any]:
    """Validate all instances and the exact factorial benchmark design."""
    graphs = [_coerce_instance(instance) for instance in instances]
    errors: list[str] = []
    reports: dict[str, dict[str, Any]] = {}
    ids = [graph.instance_id for graph in graphs]
    if len(ids) != len(set(ids)):
        errors.append("instance identifiers are not unique")
    expected_cells = {
        (family, size, difficulty, replicate)
        for family in FAMILIES
        for size in SIZES
        for difficulty in DIFFICULTIES
        for replicate in range(REPLICATES)
    }
    observed_cells = {
        (graph.family, graph.size, graph.difficulty, graph.replicate) for graph in graphs
    }
    missing = expected_cells - observed_cells
    extra = observed_cells - expected_cells
    if len(graphs) != 90:
        errors.append(f"benchmark contains {len(graphs)} instances, expected 90")
    if missing:
        errors.append(f"benchmark misses {len(missing)} design cells")
    if extra:
        errors.append(f"benchmark contains {len(extra)} unexpected design cells")
    for graph in graphs:
        report = validate_instance(graph)
        reports[graph.instance_id] = report.to_dict()
        if not report.valid:
            errors.extend(f"{graph.instance_id}: {error}" for error in report.errors)
    summary = difficulty_summary(graphs)
    # The construction claims controlled increasing difficulty.  Verify the
    # aggregate monotonicity rather than trusting labels in metadata.
    if len(summary) == len(DIFFICULTIES):
        for metric in ("mean_invalid_edge_ratio", "mean_out_degree", "mean_mandatory_count", "mean_long_range_count"):
            values = [summary[difficulty][metric] for difficulty in DIFFICULTIES]
            if not all(left < right for left, right in zip(values, values[1:])):
                errors.append(f"difficulty summary is not strictly increasing for {metric}: {values}")
    result = {
        "valid": not errors,
        "errors": errors,
        "instance_count": len(graphs),
        "difficulty_summary": summary,
        "family_summary": family_summary(graphs),
        "instance_reports": reports,
    }
    if raise_on_error and errors:
        raise ValueError("Invalid benchmark: " + "; ".join(errors))
    return result


def difficulty_summary(
    instances: Iterable[NarrativeInstance | Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate auditable structural properties by difficulty label."""
    grouped: dict[str, list[NarrativeInstance]] = {key: [] for key in DIFFICULTIES}
    for instance in instances:
        graph = _coerce_instance(instance)
        grouped.setdefault(graph.difficulty, []).append(graph)
    result: dict[str, dict[str, float | int]] = {}
    for difficulty in DIFFICULTIES:
        graphs = grouped.get(difficulty, [])
        if not graphs:
            continue
        result[difficulty] = {
            "instance_count": len(graphs),
            "mean_node_count": round(mean(len(graph.nodes) for graph in graphs), 6),
            "mean_edge_count": round(mean(len(graph.edges) for graph in graphs), 6),
            "mean_out_degree": round(
                mean(len(graph.edges) / len(graph.nodes) for graph in graphs), 6
            ),
            "mean_invalid_edge_ratio": round(
                mean(
                    sum(not edge["valid"] for edge in graph.edges) / len(graph.edges)
                    for graph in graphs
                ),
                6,
            ),
            "mean_mandatory_count": round(mean(len(graph.mandatory_events) for graph in graphs), 6),
            "mean_long_range_count": round(
                mean(len(graph.long_range_constraints) for graph in graphs), 6
            ),
            "mean_incompatible_count": round(
                mean(len(graph.incompatible_pairs) for graph in graphs), 6
            ),
        }
    return result


def family_summary(
    instances: Iterable[NarrativeInstance | Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate the distinct structural signature of each narrative family."""
    grouped: dict[str, list[NarrativeInstance]] = {key: [] for key in FAMILIES}
    for instance in instances:
        graph = _coerce_instance(instance)
        grouped.setdefault(graph.family, []).append(graph)
    result: dict[str, dict[str, float | int]] = {}
    for family in FAMILIES:
        graphs = grouped.get(family, [])
        if not graphs:
            continue
        graph_spans: list[float] = []
        branch_ratios: list[float] = []
        reconvergence_ratios: list[float] = []
        for graph in graphs:
            outdegree = {node_id: 0 for node_id in graph.nodes}
            indegree = {node_id: 0 for node_id in graph.nodes}
            spans = []
            for edge in graph.edges:
                outdegree[edge["source"]] += 1
                indegree[edge["target"]] += 1
                spans.append(
                    graph.nodes[edge["target"]]["order"]
                    - graph.nodes[edge["source"]]["order"]
                )
            graph_spans.append(mean(spans))
            branch_ratios.append(
                sum(value > 1 for value in outdegree.values()) / len(outdegree)
            )
            reconvergence_ratios.append(
                sum(value > 1 for value in indegree.values()) / len(indegree)
            )
        result[family] = {
            "instance_count": len(graphs),
            "mean_edge_span": round(mean(graph_spans), 6),
            "mean_branch_node_ratio": round(mean(branch_ratios), 6),
            "mean_reconvergence_node_ratio": round(mean(reconvergence_ratios), 6),
        }
    return result


def save_benchmark(
    instances: Iterable[NarrativeInstance | Mapping[str, Any]],
    path: str | Path,
    *,
    root_seed: int | None = None,
) -> Path:
    """Serialise a benchmark collection as one stable, human-readable JSON file."""
    graphs = [_coerce_instance(instance) for instance in instances]
    validation = validate_benchmark(graphs)
    if not validation["valid"]:
        raise ValueError("Refusing to save invalid benchmark: " + "; ".join(validation["errors"]))
    if root_seed is None and graphs:
        root_seed = int(graphs[0].metadata.get("root_seed", DEFAULT_SEED))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "BorgesBench",
        "root_seed": root_seed,
        "instance_count": len(graphs),
        "design": {
            "families": list(FAMILIES),
            "sizes": list(SIZES),
            "difficulties": list(DIFFICULTIES),
            "replicates": REPLICATES,
        },
        "difficulty_summary": difficulty_summary(graphs),
        "family_summary": family_summary(graphs),
        "instances": [graph.to_dict() for graph in graphs],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def load_benchmark(path: str | Path, *, validate: bool = True) -> list[NarrativeInstance]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    else:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version {payload.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        records = payload.get("instances", [])
    instances = [NarrativeInstance.from_dict(record) for record in records]
    if validate:
        validate_benchmark(instances, raise_on_error=True)
    return instances


# Readable aliases for notebooks and external packages.
build_benchmark = generate_benchmark
benchmark_summary = difficulty_summary


__all__ = [
    "DEFAULT_SEED",
    "DIFFICULTIES",
    "FAMILIES",
    "MAX_PATH_LENGTH",
    "MIN_PATH_LENGTH",
    "NarrativeInstance",
    "OBJECTIVE_NAMES",
    "REPLICATES",
    "SCHEMA_VERSION",
    "SIZES",
    "ValidationReport",
    "benchmark_summary",
    "build_benchmark",
    "difficulty_summary",
    "evaluate_path",
    "family_summary",
    "generate_benchmark",
    "generate_instance",
    "load_benchmark",
    "repair_path",
    "save_benchmark",
    "set_diversity",
    "validate_benchmark",
    "validate_instance",
]
