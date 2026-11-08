"""Multi-objective and path-set metrics used by BorgesBench."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np


def nondominated_mask(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the maximization non-dominated mask, retaining duplicate points."""
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    if values.ndim != 2:
        raise ValueError("points must be a two-dimensional array")
    mask = np.ones(len(values), dtype=bool)
    for index, point in enumerate(values):
        dominators = np.all(values >= point, axis=1) & np.any(values > point, axis=1)
        if np.any(dominators):
            mask[index] = False
    return mask


def nondominated(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return unique maximization non-dominated points."""
    values = np.asarray(points, dtype=float)
    if values.size == 0:
        width = values.shape[1] if values.ndim == 2 else 3
        return np.empty((0, width), dtype=float)
    front = values[nondominated_mask(values)]
    return np.unique(front, axis=0)


def _union_area_origin(rectangles: np.ndarray) -> float:
    """Area of rectangles spanning the 2-D origin and each supplied corner."""
    if rectangles.size == 0:
        return 0.0
    y_levels = np.unique(rectangles[:, 0])
    area = 0.0
    previous_y = 0.0
    for y_level in y_levels:
        active = rectangles[rectangles[:, 0] >= y_level - 1e-15]
        max_z = float(np.max(active[:, 1])) if len(active) else 0.0
        area += float(y_level - previous_y) * max_z
        previous_y = float(y_level)
    return area


def hypervolume_3d(
    points: Sequence[Sequence[float]],
    reference: Sequence[float] = (0.0, 0.0, 0.0),
) -> float:
    """Compute exact 3-D hypervolume for maximization and an axis-aligned reference."""
    values = np.asarray(points, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if ref.shape != (3,):
        raise ValueError("reference must contain three values")
    if values.size == 0:
        return 0.0
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("hypervolume_3d expects an n x 3 array")
    shifted = values - ref
    shifted = shifted[np.all(shifted > 0.0, axis=1)]
    if not len(shifted):
        return 0.0
    shifted = nondominated(shifted)
    x_levels = np.unique(shifted[:, 0])
    volume = 0.0
    previous_x = 0.0
    for x_level in x_levels:
        active = shifted[shifted[:, 0] >= x_level - 1e-15, 1:]
        volume += float(x_level - previous_x) * _union_area_origin(active)
        previous_x = float(x_level)
    return float(volume)


def igd_plus(
    approximation: Sequence[Sequence[float]],
    reference_front: Sequence[Sequence[float]],
) -> float:
    """IGD+ for maximization; lower values indicate better reference-front coverage."""
    approx = np.asarray(approximation, dtype=float)
    reference = np.asarray(reference_front, dtype=float)
    if reference.size == 0:
        return float("nan")
    if approx.size == 0:
        return float("inf")
    if approx.ndim != 2 or reference.ndim != 2 or approx.shape[1] != reference.shape[1]:
        raise ValueError("approximation and reference_front must have equal widths")
    shortfalls = np.maximum(reference[:, None, :] - approx[None, :, :], 0.0)
    distances = np.sqrt(np.sum(shortfalls**2, axis=2))
    return float(np.mean(np.min(distances, axis=1)))


def path_set_diversity(paths: Iterable[Sequence[str]]) -> float:
    """Mean pairwise Jaccard distance over directed edge sets."""
    edge_sets = []
    for path in paths:
        edge_sets.append(set(zip(path[:-1], path[1:])))
    if len(edge_sets) < 2:
        return 0.0
    distances = []
    for left, right in combinations(edge_sets, 2):
        union = left | right
        distances.append(1.0 - len(left & right) / len(union) if union else 0.0)
    return float(np.mean(distances))

