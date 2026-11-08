"""Execute the frozen BorgesBench experiment and repair ablation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .algorithms import OBJECTIVE_NAMES, run_algorithm
from .benchmark import (
    NarrativeInstance,
    generate_benchmark,
    save_benchmark,
    validate_benchmark,
)
from .metrics import hypervolume_3d, igd_plus, nondominated, path_set_diversity


BASELINE_ALGORITHMS = {"random", "beam"}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialise {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _objectives(record: dict[str, Any]) -> list[float]:
    values = record["objectives"]
    return [float(values[name]) for name in OBJECTIVE_NAMES]


def _single_run(payload: tuple[NarrativeInstance, str, int, int, int, bool]) -> dict[str, Any]:
    instance, algorithm, seed, budget, population_size, repair = payload
    result = run_algorithm(
        algorithm,
        instance,
        seed,
        budget,
        pop_size=population_size,
        repair=repair,
    )
    if int(result["evaluations"]) != budget:
        raise RuntimeError(
            f"{instance.instance_id}/{algorithm}/{seed} used {result['evaluations']} != {budget} evaluations"
        )
    archive = result["archive"]
    feasible = [record for record in archive if record["feasible"]]
    return {
        "instance_id": instance.instance_id,
        "family": instance.family,
        "size": instance.size,
        "difficulty": instance.difficulty,
        "replicate": instance.replicate,
        "algorithm": algorithm,
        "seed": int(seed),
        "budget": int(budget),
        "repair": bool(repair),
        "evaluations": int(result["evaluations"]),
        "feasible_evaluations": int(result["feasible_evaluations"]),
        "feasible_rate": float(result["feasible_rate"]),
        "best_coherence": max(
            (float(record["objectives"]["coherence"]) for record in feasible),
            default=0.0,
        ),
        "runtime_sec": float(result["runtime_sec"]),
        "archive_size": len(feasible),
        "path_set_diversity": path_set_diversity(record["path"] for record in feasible),
        "archive": feasible,
    }


def _execute_tasks(
    tasks: Sequence[tuple[NarrativeInstance, str, int, int, int, bool]],
    *,
    workers: int,
    progress_every: int = 100,
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    if workers <= 1:
        for index, task in enumerate(tasks, 1):
            completed.append(_single_run(task))
            if index % progress_every == 0 or index == len(tasks):
                print(f"completed {index}/{len(tasks)} runs", flush=True)
        return completed
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_single_run, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            completed.append(future.result())
            if index % progress_every == 0 or index == len(tasks):
                print(f"completed {index}/{len(tasks)} runs", flush=True)
    return completed


def _reference_fronts(runs: Iterable[dict[str, Any]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[list[float]]] = {}
    for run in runs:
        grouped.setdefault(run["instance_id"], []).extend(
            _objectives(record) for record in run["archive"]
        )
    fronts: dict[str, np.ndarray] = {}
    for instance_id, points in grouped.items():
        if not points:
            raise RuntimeError(f"No feasible solution was found for {instance_id}")
        fronts[instance_id] = nondominated(points)
    return fronts


def _run_rows(
    runs: Iterable[dict[str, Any]], reference_fronts: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    worst_igd = math.sqrt(len(OBJECTIVE_NAMES))
    for run in runs:
        points = [_objectives(record) for record in run["archive"]]
        row = {key: value for key, value in run.items() if key != "archive"}
        row["hypervolume"] = hypervolume_3d(points)
        row["igd_plus"] = (
            igd_plus(points, reference_fronts[run["instance_id"]]) if points else worst_igd
        )
        rows.append(row)
    return rows


def _write_archives(path: Path, runs: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for run in sorted(
            runs, key=lambda item: (item["instance_id"], item["algorithm"], item["seed"], item["repair"])
        ):
            handle.write(json.dumps(run, sort_keys=True, default=_json_default) + "\n")


def _write_reference_fronts(path: Path, fronts: dict[str, np.ndarray]) -> None:
    payload = {
        instance_id: [dict(zip(OBJECTIVE_NAMES, point)) for point in values.tolist()]
        for instance_id, values in sorted(fronts.items())
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    config_path: str | Path,
    output_root: str | Path,
    *,
    workers: int | None = None,
    quick: bool = False,
) -> dict[str, Path]:
    """Generate the benchmark, run all methods, and write raw and derived artifacts."""
    config_path = Path(config_path)
    output_root = Path(output_root)
    data_dir = output_root / "data"
    results_dir = output_root / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    started_at = datetime.now(timezone.utc)

    instances = generate_benchmark(seed=int(config["benchmark_seed"]))
    validation = validate_benchmark(instances, raise_on_error=True)
    if quick:
        instances = [instance for instance in instances if instance.replicate == 0][:6]
    benchmark_path = save_benchmark(
        generate_benchmark(seed=int(config["benchmark_seed"])),
        data_dir / "borgesbench_v1.json",
        root_seed=int(config["benchmark_seed"]),
    )
    validation_path = data_dir / "benchmark_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )

    algorithms = list(config["algorithms"])
    seeds = list(config["seeds"])
    budget = int(config["evaluation_budget"])
    population_size = int(config["population_size"])
    beam_width = int(config.get("beam_width", population_size))
    if quick:
        seeds = seeds[:2]
        budget = min(120, budget)
        population_size = min(20, population_size)
        beam_width = min(20, beam_width)
    primary_tasks = [
        (
            instance,
            algorithm,
            int(seed),
            budget,
            beam_width if algorithm == "beam" else population_size,
            algorithm not in BASELINE_ALGORITHMS,
        )
        for instance in instances
        for algorithm in algorithms
        for seed in seeds
    ]
    selected_workers = workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    primary_runs = _execute_tasks(primary_tasks, workers=selected_workers)
    primary_fronts = _reference_fronts(primary_runs)
    primary_rows = _run_rows(primary_runs, primary_fronts)
    raw_results_path = results_dir / "raw_results.csv"
    pd.DataFrame(primary_rows).sort_values(
        ["instance_id", "algorithm", "seed"]
    ).to_csv(raw_results_path, index=False)
    archives_path = results_dir / "archives.jsonl"
    _write_archives(archives_path, primary_runs)
    fronts_path = results_dir / "reference_fronts.json"
    _write_reference_fronts(fronts_path, primary_fronts)

    ablation_replicates = int(config["ablation_instances_per_cell"])
    ablation_seeds = list(config["ablation_seeds"])
    if quick:
        ablation_replicates = 1
        ablation_seeds = ablation_seeds[:2]
    ablation_instances = [
        instance for instance in instances if instance.replicate < ablation_replicates
    ]
    ablation_tasks = [
        (instance, "nsga2", int(seed), budget, population_size, repair)
        for instance in ablation_instances
        for repair in (True, False)
        for seed in ablation_seeds
    ]
    ablation_runs = _execute_tasks(ablation_tasks, workers=selected_workers)
    combined_fronts = {
        instance_id: values.copy() for instance_id, values in primary_fronts.items()
    }
    for instance in ablation_instances:
        extra = [
            _objectives(record)
            for run in ablation_runs
            if run["instance_id"] == instance.instance_id
            for record in run["archive"]
        ]
        combined_fronts[instance.instance_id] = nondominated(
            np.vstack([combined_fronts[instance.instance_id], np.asarray(extra, dtype=float)])
            if extra
            else combined_fronts[instance.instance_id]
        )
    ablation_rows = _run_rows(ablation_runs, combined_fronts)
    for row in ablation_rows:
        row["algorithm_variant"] = "repair" if row["repair"] else "no_repair"
    ablation_path = results_dir / "ablation_results.csv"
    pd.DataFrame(ablation_rows).sort_values(
        ["instance_id", "algorithm_variant", "seed"]
    ).to_csv(ablation_path, index=False)
    ablation_archives_path = results_dir / "ablation_archives.jsonl"
    _write_archives(ablation_archives_path, ablation_runs)

    finished_at = datetime.now(timezone.utc)
    manifest = {
        "experiment": "BorgesBench primary comparison and repair ablation",
        "mode": "quick" if quick else "full",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "wall_time_sec": (finished_at - started_at).total_seconds(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "workers": selected_workers,
        "primary_run_count": len(primary_runs),
        "ablation_run_count": len(ablation_runs),
        "primary_instance_count": len(instances),
        "evaluation_budget": budget,
        "population_size": population_size,
        "beam_width": beam_width,
        "baseline_repair": False,
        "population_method_repair": True,
        "config_sha256": _sha256(config_path),
        "benchmark_sha256": _sha256(benchmark_path),
        "artifacts": {},
    }
    manifest_path = output_root / "reproducibility_manifest.json"
    artifacts = {
        "benchmark": benchmark_path,
        "validation": validation_path,
        "raw_results": raw_results_path,
        "archives": archives_path,
        "reference_fronts": fronts_path,
        "ablation_results": ablation_path,
        "ablation_archives": ablation_archives_path,
    }
    manifest["artifacts"] = {
        name: {"path": str(path.relative_to(output_root)), "sha256": _sha256(path)}
        for name, path in artifacts.items()
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    artifacts["manifest"] = manifest_path
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/experiment.json"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--quick", action="store_true", help="Run a small integration experiment")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = run_experiment(
        args.config,
        args.output_root,
        workers=args.workers,
        quick=args.quick,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
