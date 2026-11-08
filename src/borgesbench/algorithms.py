"""Search algorithms for BorgesBench.

The module deliberately depends only on NumPy and the standard library.  All
algorithms use the same :class:`_Evaluator` wrapper, which enforces an exact
upper bound on calls to ``benchmark.evaluate_path`` and maintains a
constraint-aware, non-dominated archive.

Public entry points return JSON-friendly dictionaries.  In particular, every
result contains ``archive``, ``evaluations``, ``runtime_sec``, and ``trace``.
Objective values are maximised in the order coherence, thematic coverage, and
structural novelty.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


OBJECTIVE_NAMES = ("coherence", "thematic_coverage", "structural_novelty")


class EvaluationBudgetExhausted(RuntimeError):
    """Raised internally when an algorithm attempts to exceed its budget."""


@dataclass(frozen=True)
class _Candidate:
    path: tuple[Any, ...]
    objectives: tuple[float, float, float]
    feasible: bool
    violations: int
    evaluation: int

    def to_record(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "objectives": dict(zip(OBJECTIVE_NAMES, self.objectives)),
            "feasible": bool(self.feasible),
            "violations": int(self.violations),
            "evaluation": int(self.evaluation),
        }


def _field(instance: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(instance, Mapping) and name in instance:
            return instance[name]
        if hasattr(instance, name):
            return getattr(instance, name)
    return default


def _nodes(instance: Any) -> list[Any]:
    value = _field(instance, "nodes", default=[])
    if isinstance(value, Mapping):
        return list(value.keys())
    result: list[Any] = []
    for node in value:
        if isinstance(node, Mapping):
            result.append(node.get("id", node.get("node_id")))
        else:
            result.append(node)
    return [node for node in result if node is not None]


def _start(instance: Any) -> Any:
    value = _field(instance, "start_node", "start", default=None)
    if value is None:
        values = _nodes(instance)
        if not values:
            raise ValueError("Narrative instance has no nodes or start node")
        value = values[0]
    return value


def _ends(instance: Any) -> set[Any]:
    value = _field(instance, "end_nodes", "ends", "end", default=())
    if value is None:
        return set()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return {value}
    return set(value)


def _length_bounds(instance: Any) -> tuple[int, int]:
    minimum = int(_field(instance, "min_path_length", "min_len", default=1))
    maximum = int(
        _field(instance, "max_path_length", "max_len", default=max(minimum, 10))
    )
    if minimum < 1 or maximum < minimum:
        raise ValueError(f"Invalid path-length bounds: {minimum}, {maximum}")
    return minimum, maximum


def _normalise_successor(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("target", value.get("to", value.get("target_id")))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        # ``successors`` normally returns node ids, but tolerate (node, data).
        if isinstance(value[1], Mapping):
            return value[0]
    return value


def _successors(instance: Any, node: Any) -> list[Any]:
    method = getattr(instance, "successors", None)
    if callable(method):
        try:
            values = method(node, valid_only=False)
        except TypeError:
            values = method(node)
        return [
            successor
            for successor in (_normalise_successor(value) for value in values)
            if successor is not None
        ]

    edges = _field(instance, "edges", default=[])
    if isinstance(edges, Mapping):
        values = edges.get(node, [])
        if isinstance(values, Mapping):
            values = values.keys()
        return [
            successor
            for successor in (_normalise_successor(value) for value in values)
            if successor is not None
        ]

    result: list[Any] = []
    for edge in edges:
        if isinstance(edge, Mapping):
            source = edge.get("source", edge.get("from", edge.get("source_id")))
            target = edge.get("target", edge.get("to", edge.get("target_id")))
        else:
            source, target = edge[0], edge[1]
        if source == node and target is not None:
            result.append(target)
    return result


def _domain_evaluator() -> Callable[[Any, Sequence[Any]], Any]:
    # Lazy import prevents a circular import while benchmark instances are built.
    from .benchmark import evaluate_path

    return evaluate_path


def _domain_repairer() -> Callable[..., Sequence[Any]]:
    from .benchmark import repair_path

    return repair_path


def _call_repair(instance: Any, path: Sequence[Any], rng: np.random.Generator) -> tuple[Any, ...]:
    repair = _domain_repairer()
    signature = inspect.signature(repair)
    if "rng" in signature.parameters:
        repaired = repair(instance, list(path), rng=rng)
    elif len(signature.parameters) >= 3:
        repaired = repair(instance, list(path), rng)
    else:
        repaired = repair(instance, list(path))
    return tuple(repaired)


def _violation_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.number)):
        return max(0, int(value))
    try:
        return len(value)
    except TypeError:
        return int(bool(value))


def _normalise_evaluation(raw: Any) -> tuple[tuple[float, float, float], bool, int]:
    if isinstance(raw, Mapping):
        objective_source = raw.get("objectives", raw)
        if isinstance(objective_source, Mapping):
            coherence = objective_source.get(
                "coherence", objective_source.get("coherence_score", 0.0)
            )
            coverage = objective_source.get(
                "thematic_coverage",
                objective_source.get("coverage", objective_source.get("theme_coverage", 0.0)),
            )
            novelty = objective_source.get(
                "structural_novelty",
                objective_source.get("novelty", objective_source.get("diversity", 0.0)),
            )
            objectives = (float(coherence), float(coverage), float(novelty))
        else:
            objectives = tuple(float(value) for value in objective_source[:3])  # type: ignore[index]
        violations = _violation_count(
            raw.get("violations", raw.get("violation_count", raw.get("n_violations", 0)))
        )
        feasible = bool(raw.get("feasible", violations == 0))
    elif hasattr(raw, "objectives"):
        objectives = tuple(float(value) for value in raw.objectives[:3])
        violations = _violation_count(getattr(raw, "violations", 0))
        feasible = bool(getattr(raw, "feasible", violations == 0))
    elif isinstance(raw, Sequence) and len(raw) >= 3:
        objectives = tuple(float(value) for value in raw[:3])
        violations = 0
        feasible = True
    else:
        raise TypeError("evaluate_path must return a mapping or a 3-objective sequence")

    if len(objectives) != 3 or not all(math.isfinite(value) for value in objectives):
        raise ValueError(f"Invalid objective vector returned by evaluate_path: {objectives}")
    return (objectives[0], objectives[1], objectives[2]), feasible, violations


def _dominates(left: _Candidate, right: _Candidate, constraint_aware: bool = True) -> bool:
    if constraint_aware:
        if left.feasible != right.feasible:
            return left.feasible
        if not left.feasible and left.violations != right.violations:
            return left.violations < right.violations
    weakly_better = all(a >= b for a, b in zip(left.objectives, right.objectives))
    strictly_better = any(a > b for a, b in zip(left.objectives, right.objectives))
    return weakly_better and strictly_better


def _archive_insert(archive: list[_Candidate], candidate: _Candidate) -> None:
    for existing in archive:
        if existing.path == candidate.path:
            return
        if _dominates(existing, candidate):
            return
    archive[:] = [existing for existing in archive if not _dominates(candidate, existing)]
    archive.append(candidate)


class _Evaluator:
    def __init__(self, instance: Any, budget: int):
        if budget <= 0:
            raise ValueError("budget must be positive")
        self.instance = instance
        self.budget = int(budget)
        self.evaluations = 0
        self.feasible_evaluations = 0
        self.archive: list[_Candidate] = []
        self.trace: list[dict[str, Any]] = []
        self._evaluate_path = _domain_evaluator()

    @property
    def remaining(self) -> int:
        return self.budget - self.evaluations

    def evaluate(self, path: Sequence[Any]) -> _Candidate:
        if self.remaining <= 0:
            raise EvaluationBudgetExhausted
        # Attempted calls consume budget even if the domain evaluator raises.
        self.evaluations += 1
        clean_path = tuple(path)
        raw = self._evaluate_path(self.instance, list(clean_path))
        objectives, feasible, violations = _normalise_evaluation(raw)
        candidate = _Candidate(
            path=clean_path,
            objectives=objectives,
            feasible=feasible,
            violations=violations,
            evaluation=self.evaluations,
        )
        self.feasible_evaluations += int(feasible)
        _archive_insert(self.archive, candidate)
        record = candidate.to_record()
        record["archive_size"] = len(self.archive)
        self.trace.append(record)
        return candidate


def _random_path(instance: Any, rng: np.random.Generator) -> tuple[Any, ...]:
    minimum, maximum = _length_bounds(instance)
    start = _start(instance)
    ends = _ends(instance)
    best = (start,)
    # Restarts help when a random branch reaches a dead end before min length.
    for _ in range(8):
        path = [start]
        while len(path) < maximum:
            if len(path) >= minimum and path[-1] in ends:
                # Prefer a legal termination, but sometimes explore beyond it.
                if rng.random() < 0.85:
                    break
            successors = _successors(instance, path[-1])
            if not successors:
                break
            unseen = [node for node in successors if node not in path]
            choices = unseen or successors
            path.append(choices[int(rng.integers(len(choices)))])
        if len(path) > len(best):
            best = tuple(path)
        if len(path) >= minimum and (not ends or path[-1] in ends):
            return tuple(path)
    return best


def _prepare_path(
    instance: Any,
    path: Sequence[Any],
    rng: np.random.Generator,
    use_repair: bool,
) -> tuple[Any, ...]:
    if use_repair:
        return _call_repair(instance, path, rng)
    return tuple(path)


def _complete_partial(
    instance: Any, partial: Sequence[Any], rng: np.random.Generator
) -> tuple[Any, ...]:
    """Roll a partial beam state forward to a complete evaluation candidate.

    Completion and beam expansion are intentionally separate: the completed
    copy is scored and archived, while the original partial path remains the
    search state.  ``repair_path`` performs the final hard-constraint repair.
    """

    path = list(partial)
    _, maximum = _length_bounds(instance)
    ends = _ends(instance)
    while path and len(path) < maximum and (not ends or path[-1] not in ends):
        successors = _successors(instance, path[-1])
        if not successors:
            break
        unseen = [node for node in successors if node not in path]
        choices = unseen or successors
        path.append(choices[int(rng.integers(len(choices)))])
    return tuple(path)


def _mutate(
    instance: Any,
    path: Sequence[Any],
    rng: np.random.Generator,
    rate: float = 0.35,
) -> tuple[Any, ...]:
    result = list(path)
    if rng.random() >= rate:
        return tuple(result)
    nodes = _nodes(instance)
    _, maximum = _length_bounds(instance)
    operation = int(rng.integers(4))

    if operation == 0 and len(result) > 2:  # delete an internal event
        del result[int(rng.integers(1, len(result) - 1))]
    elif operation == 1 and len(result) < maximum and nodes:  # insert
        position = int(rng.integers(1, len(result) + 1))
        result.insert(position, nodes[int(rng.integers(len(nodes)))])
    elif operation == 2 and len(result) > 1 and nodes:  # replace
        position = int(rng.integers(1, len(result)))
        result[position] = nodes[int(rng.integers(len(nodes)))]
    else:  # reroute a suffix along actual graph edges
        cut = int(rng.integers(1, max(2, len(result))))
        result = result[:cut]
        while len(result) < maximum:
            successors = _successors(instance, result[-1])
            if not successors:
                break
            result.append(successors[int(rng.integers(len(successors)))])
            if result[-1] in _ends(instance) and rng.random() < 0.85:
                break
    return tuple(result[:maximum])


def _crossover(
    first: Sequence[Any], second: Sequence[Any], rng: np.random.Generator
) -> tuple[Any, ...]:
    if len(first) < 2:
        return tuple(second)
    if len(second) < 2:
        return tuple(first)
    # Sorting removes dependence on process-level hash randomisation.
    common = sorted(set(first[1:]) & set(second[1:]), key=str)
    if common:
        pivot = common[int(rng.integers(len(common)))]
        left = list(first[: first.index(pivot) + 1])
        right = list(second[second.index(pivot) + 1 :])
        return tuple(left + right)
    cut_first = int(rng.integers(1, len(first)))
    cut_second = int(rng.integers(1, len(second)))
    return tuple(first[:cut_first]) + tuple(second[cut_second:])


def _non_dominated_sort(
    population: Sequence[_Candidate], constraint_aware: bool = True
) -> list[list[int]]:
    dominates: list[list[int]] = [[] for _ in population]
    domination_count = [0] * len(population)
    first_front: list[int] = []
    for i, left in enumerate(population):
        for j in range(i + 1, len(population)):
            right = population[j]
            if _dominates(left, right, constraint_aware):
                dominates[i].append(j)
                domination_count[j] += 1
            elif _dominates(right, left, constraint_aware):
                dominates[j].append(i)
                domination_count[i] += 1
        if domination_count[i] == 0:
            first_front.append(i)

    fronts: list[list[int]] = []
    current = first_front
    while current:
        fronts.append(current)
        next_front: list[int] = []
        for i in current:
            for j in dominates[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current = next_front
    return fronts


def _crowding(population: Sequence[_Candidate], indices: Sequence[int]) -> dict[int, float]:
    distances = {index: 0.0 for index in indices}
    if len(indices) <= 2:
        return {index: math.inf for index in indices}
    for objective in range(3):
        ordered = sorted(indices, key=lambda index: population[index].objectives[objective])
        distances[ordered[0]] = math.inf
        distances[ordered[-1]] = math.inf
        low = population[ordered[0]].objectives[objective]
        high = population[ordered[-1]].objectives[objective]
        if high <= low:
            continue
        for position in range(1, len(ordered) - 1):
            if math.isinf(distances[ordered[position]]):
                continue
            previous = population[ordered[position - 1]].objectives[objective]
            following = population[ordered[position + 1]].objectives[objective]
            distances[ordered[position]] += (following - previous) / (high - low)
    return distances


def _rank_and_crowding(population: Sequence[_Candidate]) -> tuple[list[int], dict[int, float]]:
    fronts = _non_dominated_sort(population)
    rank = [0] * len(population)
    crowding: dict[int, float] = {}
    for level, front in enumerate(fronts):
        for index in front:
            rank[index] = level
        crowding.update(_crowding(population, front))
    return rank, crowding


def _environmental_selection(
    population: Sequence[_Candidate], size: int
) -> list[_Candidate]:
    selected: list[_Candidate] = []
    for front in _non_dominated_sort(population):
        if len(selected) + len(front) <= size:
            selected.extend(population[index] for index in front)
        else:
            distances = _crowding(population, front)
            ordered = sorted(front, key=lambda index: distances[index], reverse=True)
            selected.extend(population[index] for index in ordered[: size - len(selected)])
            break
    return selected


def _tournament(
    population: Sequence[_Candidate],
    rank: Sequence[int],
    crowding: Mapping[int, float],
    rng: np.random.Generator,
) -> _Candidate:
    first, second = rng.integers(0, len(population), size=2)
    i, j = int(first), int(second)
    if rank[i] != rank[j]:
        return population[i if rank[i] < rank[j] else j]
    if crowding[i] != crowding[j]:
        return population[i if crowding[i] > crowding[j] else j]
    return population[i if rng.random() < 0.5 else j]


def _result(
    algorithm: str,
    evaluator: _Evaluator,
    seed: int,
    budget: int,
    started: float,
    **parameters: Any,
) -> dict[str, Any]:
    archive = sorted(
        evaluator.archive,
        key=lambda candidate: (*candidate.objectives, tuple(map(str, candidate.path))),
        reverse=True,
    )
    return {
        "algorithm": algorithm,
        "seed": int(seed),
        "budget": int(budget),
        "parameters": parameters,
        "archive": [candidate.to_record() for candidate in archive],
        "evaluations": int(evaluator.evaluations),
        "feasible_evaluations": int(evaluator.feasible_evaluations),
        "feasible_rate": evaluator.feasible_evaluations / evaluator.evaluations,
        "runtime_sec": float(time.perf_counter() - started),
        "trace": evaluator.trace,
    }


def random_search(
    instance: Any,
    seed: int,
    budget: int,
    pop_size: int = 40,
    repair: bool = True,
) -> dict[str, Any]:
    """Independent random-path sampling baseline."""

    del pop_size  # Accepted to keep the common public API.
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    evaluator = _Evaluator(instance, budget)
    while evaluator.remaining:
        path = _prepare_path(instance, _random_path(instance, rng), rng, repair)
        evaluator.evaluate(path)
    return _result("random", evaluator, seed, budget, started, repair=repair)


def multiobjective_beam_search(
    instance: Any,
    seed: int,
    budget: int,
    pop_size: int = 40,
    repair: bool = True,
) -> dict[str, Any]:
    """Pareto beam search; ``pop_size`` is used as the beam width.

    Every partial child is scored through a completed/repaired copy.  Thus the
    archive, trace, and feasible rate contain only comparable start-to-end
    candidate solutions, while beam expansion still operates on partial paths.
    """

    if pop_size <= 0:
        raise ValueError("pop_size must be positive")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    evaluator = _Evaluator(instance, budget)
    _, maximum = _length_bounds(instance)
    beam: list[tuple[Any, ...]] = [(_start(instance),)]
    scored: dict[tuple[Any, ...], _Candidate] = {}
    expanded: set[tuple[Any, ...]] = set()

    while evaluator.remaining:
        children: list[tuple[Any, ...]] = []
        for path in beam:
            if path in expanded or len(path) >= maximum:
                continue
            expanded.add(path)
            for successor in _successors(instance, path[-1]):
                child = tuple(path) + (successor,)
                if child in scored:
                    continue
                children.append(child)

        if not children:
            # Start a fresh Pareto-beam pass.  Completion rollouts and shuffled
            # ties make successive passes useful while retaining beam-search
            # semantics and an exact common evaluation budget.
            beam = [(_start(instance),)]
            expanded.clear()
            scored.clear()
            if not _successors(instance, _start(instance)):
                # Degenerate instances are rejected by benchmark validation,
                # but this guard prevents an infinite loop for external input.
                path = _prepare_path(instance, _random_path(instance, rng), rng, repair)
                evaluator.evaluate(path)
            continue

        rng.shuffle(children)
        scored_children: list[tuple[tuple[Any, ...], _Candidate]] = []
        for partial in children:
            if not evaluator.remaining:
                break
            evaluation_path = _complete_partial(instance, partial, rng)
            evaluation_path = _prepare_path(instance, evaluation_path, rng, repair)
            candidate = evaluator.evaluate(evaluation_path)
            scored[partial] = candidate
            scored_children.append((partial, candidate))

        if not scored_children:
            break
        candidates = [candidate for _, candidate in scored_children]
        selected_indices: list[int] = []
        for front in _non_dominated_sort(candidates, constraint_aware=True):
            if len(selected_indices) + len(front) <= pop_size:
                selected_indices.extend(front)
            else:
                distances = _crowding(candidates, front)
                ordered = sorted(front, key=lambda index: distances[index], reverse=True)
                selected_indices.extend(ordered[: pop_size - len(selected_indices)])
                break
        beam = [scored_children[index][0] for index in selected_indices]

    return _result(
        "beam", evaluator, seed, budget, started, beam_width=pop_size, repair=repair
    )


beam_search = multiobjective_beam_search


def nsga2(
    instance: Any,
    seed: int,
    budget: int,
    pop_size: int = 40,
    repair: bool = True,
) -> dict[str, Any]:
    """NSGA-II with constraint-domination and an optional repair ablation."""

    if pop_size < 2:
        raise ValueError("pop_size must be at least 2")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    evaluator = _Evaluator(instance, budget)
    population: list[_Candidate] = []
    for _ in range(min(pop_size, evaluator.remaining)):
        path = _prepare_path(instance, _random_path(instance, rng), rng, repair)
        population.append(evaluator.evaluate(path))

    while evaluator.remaining and population:
        rank, crowding = _rank_and_crowding(population)
        offspring: list[_Candidate] = []
        target = min(pop_size, evaluator.remaining)
        for _ in range(target):
            first = _tournament(population, rank, crowding, rng)
            second = _tournament(population, rank, crowding, rng)
            child_path = _crossover(first.path, second.path, rng)
            child_path = _mutate(instance, child_path, rng)
            child_path = _prepare_path(instance, child_path, rng, repair)
            offspring.append(evaluator.evaluate(child_path))
        population = _environmental_selection(population + offspring, pop_size)

    return _result(
        "nsga2", evaluator, seed, budget, started, pop_size=pop_size, repair=repair
    )


def _weight_vectors(size: int, rng: np.random.Generator) -> np.ndarray:
    if size <= 3:
        return np.eye(3, dtype=float)[:size]
    weights = np.empty((size, 3), dtype=float)
    weights[:3] = np.eye(3, dtype=float)
    weights[3:] = rng.dirichlet(np.ones(3), size=size - 3)
    return weights


def _scalar_fitness(candidate: _Candidate, weight: np.ndarray, ideal: np.ndarray) -> float:
    if not candidate.feasible:
        # Any feasible solution is preferred to any infeasible one.  Violation
        # count provides a useful gradient before feasibility is reached.
        return 1e6 + candidate.violations
    objective = np.asarray(candidate.objectives, dtype=float)
    return float(np.max(np.maximum(weight, 1e-6) * np.abs(ideal - objective)))


def moead(
    instance: Any,
    seed: int,
    budget: int,
    pop_size: int = 40,
    repair: bool = True,
) -> dict[str, Any]:
    """MOEA/D using Tchebycheff decomposition and neighbourhood mating."""

    if pop_size < 3:
        raise ValueError("pop_size must be at least 3")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    evaluator = _Evaluator(instance, budget)
    actual_size = min(pop_size, budget)
    weights = _weight_vectors(actual_size, rng)
    neighbour_count = min(max(2, int(round(math.sqrt(actual_size)))), actual_size)
    distances = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    neighbourhoods = np.argsort(distances, axis=1)[:, :neighbour_count]

    population: list[_Candidate] = []
    for _ in range(actual_size):
        path = _prepare_path(instance, _random_path(instance, rng), rng, repair)
        population.append(evaluator.evaluate(path))
    ideal = np.max(np.asarray([candidate.objectives for candidate in population]), axis=0)

    cursor = 0
    while evaluator.remaining:
        subproblem = cursor % actual_size
        neighbours = neighbourhoods[subproblem]
        first, second = rng.choice(neighbours, size=2, replace=len(neighbours) < 2)
        path = _crossover(population[int(first)].path, population[int(second)].path, rng)
        path = _mutate(instance, path, rng)
        path = _prepare_path(instance, path, rng, repair)
        child = evaluator.evaluate(path)
        ideal = np.maximum(ideal, np.asarray(child.objectives))

        for index_value in neighbours:
            index = int(index_value)
            incumbent = population[index]
            replace = False
            if child.feasible != incumbent.feasible:
                replace = child.feasible
            elif not child.feasible and child.violations != incumbent.violations:
                replace = child.violations < incumbent.violations
            else:
                replace = _scalar_fitness(child, weights[index], ideal) <= _scalar_fitness(
                    incumbent, weights[index], ideal
                )
            if replace:
                population[index] = child
        cursor += 1

    return _result(
        "moead",
        evaluator,
        seed,
        budget,
        started,
        pop_size=pop_size,
        neighbourhood_size=neighbour_count,
        repair=repair,
    )


def _choose_archive_leader(
    archive: Sequence[_Candidate], rng: np.random.Generator
) -> _Candidate:
    if len(archive) == 1:
        return archive[0]
    indices = list(range(len(archive)))
    distances = _crowding(archive, indices)
    boundary = [index for index, value in distances.items() if math.isinf(value)]
    if boundary and rng.random() < 0.5:
        return archive[boundary[int(rng.integers(len(boundary)))]]
    finite = np.asarray(
        [0.0 if math.isinf(distances[index]) else distances[index] for index in indices],
        dtype=float,
    )
    probabilities = finite + 1e-6
    probabilities /= probabilities.sum()
    return archive[int(rng.choice(indices, p=probabilities))]


def _prefer_personal_best(
    candidate: _Candidate, incumbent: _Candidate, rng: np.random.Generator
) -> bool:
    if _dominates(candidate, incumbent):
        return True
    if _dominates(incumbent, candidate):
        return False
    # For mutually non-dominated points, a fresh random scalarisation avoids a
    # permanent bias towards any one objective.
    weights = rng.dirichlet(np.ones(3))
    left = float(np.dot(weights, candidate.objectives))
    right = float(np.dot(weights, incumbent.objectives))
    if candidate.feasible != incumbent.feasible:
        return candidate.feasible
    if not candidate.feasible and candidate.violations != incumbent.violations:
        return candidate.violations < incumbent.violations
    return left >= right


def discrete_pso(
    instance: Any,
    seed: int,
    budget: int,
    pop_size: int = 40,
    repair: bool = True,
) -> dict[str, Any]:
    """Path-based discrete multi-objective particle swarm optimisation.

    Velocity is represented by stochastic edit operations: recombination with
    the personal best (cognitive term), recombination with an archive leader
    (social term), and mutation (inertial exploration).
    """

    if pop_size < 2:
        raise ValueError("pop_size must be at least 2")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    evaluator = _Evaluator(instance, budget)
    actual_size = min(pop_size, budget)
    particles: list[_Candidate] = []
    for _ in range(actual_size):
        path = _prepare_path(instance, _random_path(instance, rng), rng, repair)
        particles.append(evaluator.evaluate(path))
    personal_best = list(particles)

    iteration = 0
    expected_iterations = max(1, math.ceil(max(0, budget - actual_size) / actual_size))
    while evaluator.remaining:
        progress = min(1.0, iteration / expected_iterations)
        inertia = 0.9 - 0.5 * progress
        cognitive = 0.65
        social = 0.75
        for index in range(actual_size):
            if not evaluator.remaining:
                break
            path = particles[index].path
            if rng.random() < cognitive:
                path = _crossover(path, personal_best[index].path, rng)
            if evaluator.archive and rng.random() < social:
                leader = _choose_archive_leader(evaluator.archive, rng)
                path = _crossover(path, leader.path, rng)
            path = _mutate(instance, path, rng, rate=max(0.15, inertia))
            path = _prepare_path(instance, path, rng, repair)
            candidate = evaluator.evaluate(path)
            particles[index] = candidate
            if _prefer_personal_best(candidate, personal_best[index], rng):
                personal_best[index] = candidate
        iteration += 1

    return _result(
        "dpso",
        evaluator,
        seed,
        budget,
        started,
        pop_size=pop_size,
        repair=repair,
        representation="path_edit_velocity",
    )


pso = discrete_pso


def run_algorithm(
    name: str,
    instance: Any,
    seed: int,
    budget: int,
    pop_size: int = 40,
    repair: bool = True,
) -> dict[str, Any]:
    """Run a named algorithm through the common reproducible interface.

    ``name='nsga2_no_repair'`` is a convenience alias for the planned repair
    ablation and overrides the supplied ``repair`` value.
    """

    normalised = name.lower().strip().replace("-", "_").replace("/", "_")
    if normalised in {"nsga2_no_repair", "nsga_ii_no_repair"}:
        normalised = "nsga2"
        repair = False
    algorithms: dict[str, Callable[..., dict[str, Any]]] = {
        "random": random_search,
        "random_search": random_search,
        "beam": multiobjective_beam_search,
        "beam_search": multiobjective_beam_search,
        "multiobjective_beam_search": multiobjective_beam_search,
        "nsga2": nsga2,
        "nsga_ii": nsga2,
        "moead": moead,
        "moea_d": moead,
        "pso": discrete_pso,
        "dpso": discrete_pso,
        "discrete_pso": discrete_pso,
    }
    if normalised not in algorithms:
        available = ", ".join(sorted(set(algorithms)))
        raise ValueError(f"Unknown algorithm {name!r}. Available names: {available}")
    return algorithms[normalised](
        instance=instance,
        seed=seed,
        budget=budget,
        pop_size=pop_size,
        repair=repair,
    )


__all__ = [
    "OBJECTIVE_NAMES",
    "EvaluationBudgetExhausted",
    "random_search",
    "multiobjective_beam_search",
    "beam_search",
    "nsga2",
    "moead",
    "discrete_pso",
    "pso",
    "run_algorithm",
]
