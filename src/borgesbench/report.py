"""Build compact publication tables and an evidence-grounded results report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .benchmark import load_benchmark


def _fmt(value: float, digits: int = 4) -> str:
    if abs(value) < 0.0001 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _instance_characteristics(instance) -> dict[str, float | int | str]:
    outdegree = {node_id: 0 for node_id in instance.nodes}
    indegree = {node_id: 0 for node_id in instance.nodes}
    spans = []
    for edge in instance.edges:
        outdegree[edge["source"]] += 1
        indegree[edge["target"]] += 1
        spans.append(
            instance.nodes[edge["target"]]["order"]
            - instance.nodes[edge["source"]]["order"]
        )
    return {
        "family": instance.family,
        "size": instance.size,
        "difficulty": instance.difficulty,
        "edge_count": len(instance.edges),
        "mean_out_degree": len(instance.edges) / len(instance.nodes),
        "invalid_edge_ratio": sum(not edge["valid"] for edge in instance.edges)
        / len(instance.edges),
        "mean_edge_span": float(np.mean(spans)),
        "branch_node_ratio": sum(value > 1 for value in outdegree.values())
        / len(outdegree),
        "reconvergence_node_ratio": sum(value > 1 for value in indegree.values())
        / len(indegree),
        "mandatory_count": len(instance.mandatory_events),
        "long_range_count": len(instance.long_range_constraints),
        "incompatible_count": len(instance.incompatible_pairs),
        "witness_length": len(instance.witness_path),
    }


def make_benchmark_table(benchmark_path: Path) -> pd.DataFrame:
    records = [_instance_characteristics(item) for item in load_benchmark(benchmark_path)]
    raw = pd.DataFrame(records)
    group = ["family", "size", "difficulty"]
    metrics = [column for column in raw.columns if column not in group]
    table = raw.groupby(group, sort=True, observed=True)[metrics].mean().reset_index()
    table.insert(3, "n_instances", raw.groupby(group).size().to_numpy())
    return table


def make_key_comparison_table(statistics: pd.DataFrame, ablation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = statistics[
        statistics["metric"].isin(["hypervolume", "igd_plus"])
        & statistics["algorithm_b"].eq("random")
    ]
    for _, item in selected.iterrows():
        rows.append(
            {
                "analysis": "primary",
                "metric": item["metric"],
                "method_a": item["algorithm_a"],
                "method_b": item["algorithm_b"],
                "p_holm": item["p_holm"],
                "a12_a_better": item["a12_a_better"],
                "effect_magnitude": item["effect_magnitude"],
            }
        )
    for _, item in ablation[
        ablation["metric"].isin(
            ["hypervolume", "igd_plus", "feasible_rate", "best_coherence"]
        )
    ].iterrows():
        rows.append(
            {
                "analysis": "repair_ablation",
                "metric": item["metric"],
                "method_a": "repair",
                "method_b": "no_repair",
                "p_holm": item["p_holm"],
                "a12_a_better": 1.0 - item["a12_a_better"],
                "effect_magnitude": item["effect_magnitude"],
            }
        )
    return pd.DataFrame(rows)


def build_report(project_root: Path) -> dict[str, Path]:
    analysis_dir = project_root / "results" / "analysis"
    aggregates = pd.read_csv(analysis_dir / "aggregate_summary.csv")
    main = aggregates[
        aggregates["scope"].eq("overall")
    ].pivot(index="algorithm", columns="metric", values="median")
    main = main.rename(columns={column: f"{column}_median" for column in main.columns})
    stats = pd.read_csv(analysis_dir / "table_statistics.csv")
    ablation = pd.read_csv(analysis_dir / "table_ablation.csv").set_index("metric")
    benchmark_table = make_benchmark_table(project_root / "data" / "borgesbench_v1.json")
    benchmark_table_path = analysis_dir / "table_benchmark.csv"
    benchmark_table.to_csv(benchmark_table_path, index=False)
    key_table = make_key_comparison_table(stats, ablation.reset_index())
    key_table_path = analysis_dir / "table_key_comparisons.csv"
    key_table.to_csv(key_table_path, index=False)

    hv_friedman = stats.loc[stats["metric"].eq("hypervolume")].iloc[0]
    igd_friedman = stats.loc[stats["metric"].eq("igd_plus")].iloc[0]
    beam_dpso = stats[
        stats["metric"].eq("hypervolume")
        & stats["algorithm_a"].eq("beam")
        & stats["algorithm_b"].eq("dpso")
    ].iloc[0]
    dpso_random = stats[
        stats["metric"].eq("hypervolume")
        & stats["algorithm_a"].eq("dpso")
        & stats["algorithm_b"].eq("random")
    ].iloc[0]

    difficulty = aggregates[
        aggregates["scope"].eq("difficulty")
        & aggregates["metric"].eq("hypervolume")
    ].pivot(index="algorithm", columns="difficulty", values="median")
    family = aggregates[
        aggregates["scope"].eq("family") & aggregates["metric"].eq("hypervolume")
    ].pivot(index="algorithm", columns="family", values="median")
    hv_no = float(str(ablation.loc["hypervolume", "variant_a_median_iqr"]).split()[0])
    hv_yes = float(str(ablation.loc["hypervolume", "variant_b_median_iqr"]).split()[0])
    runtime_no = float(str(ablation.loc["runtime_sec", "variant_a_median_iqr"]).split()[0])
    runtime_yes = float(str(ablation.loc["runtime_sec", "variant_b_median_iqr"]).split()[0])

    report = f"""# BorgesBench: experimental results report

## Executive result

The experiment supports a qualified conclusion rather than a single-algorithm
winner. DPSO obtained the highest overall median hypervolume
({_fmt(main.loc['dpso', 'hypervolume_median'])}), but its difference from
multiobjective beam search ({_fmt(main.loc['beam', 'hypervolume_median'])}) was
not significant after Holm correction (p={_fmt(beam_dpso['p_holm'])},
A12={_fmt(1.0 - beam_dpso['a12_a_better'], 3)}, negligible). The robust result
is that all four informed search methods outperformed random search on front
quality, while the domain repair operator determined whether population-based
search produced valid narratives consistently.

## Experimental integrity

- Frozen benchmark: 90 instances (3 families x 2 sizes x 3 difficulties x 5
  replications), all passing schema, DAG, reachability, difficulty, and witness
  validation.
- Primary experiment: 4,500 runs (90 x 5 methods x 10 seeds), 800 complete-path
  evaluations per run.
- Repair ablation: 360 NSGA-II runs on 36 stratified instances, 5 seeds, with
  and without repair.
- Objectives: coherence, thematic coverage, and fixed structural novelty in
  [0,1]. The fixed novelty definition makes hypervolume and IGD+ comparable
  across algorithms.
- Inference: seed medians were collapsed within instance before Friedman and
  paired Wilcoxon-Holm tests, avoiding seed-level pseudoreplication.

## Benchmark behaviour

Difficulty is structurally monotone: mean out-degree increases from 2.0 to 2.8
and 3.6; invalid-edge ratio increases from 0.100 to 0.202 and 0.299; mandatory
and long-range constraints increase from one to three. The families also have
distinct topology. Double has short parallel corridors (mean edge span 2.14,
branch-node ratio 0.926), Labyrinth has local forks and reconvergences (span
4.17, branch ratio 0.879), and Circular Time has sparse long-range dependencies
(span 21.90, branch ratio 0.375).

## Primary comparison

The global method effect was significant for hypervolume
(Friedman chi-square={_fmt(hv_friedman['friedman_statistic'], 2)},
p={hv_friedman['friedman_p']:.2e}) and IGD+
(chi-square={_fmt(igd_friedman['friedman_statistic'], 2)},
p={igd_friedman['friedman_p']:.2e}). Overall medians were:

| Method | Hypervolume | IGD+ | Feasible rate | Best coherence | Runtime (s) |
|---|---:|---:|---:|---:|---:|
"""
    for algorithm in ["random", "beam", "nsga2", "moead", "dpso"]:
        row = main.loc[algorithm]
        report += (
            f"| {algorithm} | {_fmt(row['hypervolume_median'])} | "
            f"{_fmt(row['igd_plus_median'])} | {_fmt(row['feasible_rate_median'])} | "
            f"{_fmt(row['best_coherence_median'])} | {_fmt(row['runtime_sec_median'])} |\n"
        )
    report += f"""

DPSO improved hypervolume over random search with a small effect
(Holm p={dpso_random['p_holm']:.2e}, A12={_fmt(dpso_random['a12_a_better'], 3)}).
The other informed methods also significantly exceeded random search, but
effect sizes for hypervolume remained small. Differences among beam, NSGA-II,
MOEA/D, and DPSO were mostly negligible in magnitude even when rank tests were
significant. The evidence therefore does not justify claiming a decisive
evolutionary or swarm winner.

Beam search reveals a useful quality-yield distinction. It reached competitive
front quality with a median feasible rate of only
{_fmt(main.loc['beam', 'feasible_rate_median'])}; the repaired population
methods reached 1.0. Beam therefore finds a small number of strong paths, while
repair produces valid paths reliably across the full evaluation stream.

## Difficulty and family effects

Hypervolume fell with difficulty for every method. DPSO declined from
{_fmt(difficulty.loc['dpso', 'low'])} on low instances to
{_fmt(difficulty.loc['dpso', 'high'])} on high instances; random search fell
from {_fmt(difficulty.loc['random', 'low'])} to
{_fmt(difficulty.loc['random', 'high'])}. The separation is consequently driven
mainly by medium and high difficulty rather than easy graphs.

The Double family was the hardest in front quality (DPSO median
{_fmt(family.loc['dpso', 'double'])}), whereas Labyrinth and Circular Time were
higher ({_fmt(family.loc['dpso', 'labyrinth'])} and
{_fmt(family.loc['dpso', 'circular_time'])}). This is consistent with the dense
short paired corridors creating many locally plausible but globally constrained
alternatives.

## Repair ablation

Repair raised NSGA-II median hypervolume from {_fmt(hv_no)} to {_fmt(hv_yes)}
({100 * (hv_yes / hv_no - 1):.1f}% relative), reduced median IGD+ from
0.02273 to 0, and raised feasible rate from 0.2906 to 1.0. Effects were medium
for hypervolume and large for IGD+, feasibility, and best coherence; every
quality comparison remained significant after correction. The cost was runtime:
median time increased from {_fmt(runtime_no)} s to {_fmt(runtime_yes)} s
({100 * (runtime_yes / runtime_no - 1):.1f}%). The repair operator is therefore
the clearest technical contribution, but it trades computational cost and some
exploratory freedom for guaranteed feasibility.

## Interpretation and publication position

The strongest paper narrative is not “NSGA-II wins.” It is:

1. structured search consistently beats uninformed sampling as constraints
   become harder;
2. a simple beam method can match population methods in Pareto quality but not
   in feasible yield;
3. representation-aware repair changes the problem more than the choice among
   NSGA-II, MOEA/D, and DPSO;
4. family topology changes where each method is effective, so no optimizer is
   uniformly dominant.

This supports a benchmark/method paper with a negative-comparative result:
population-based optimization is useful, but its practical advantage comes
primarily from constraint handling rather than from a universally superior
metaheuristic.

## Limitations

- The benchmark is synthetic and structurally scored; it does not establish
  human-perceived literary quality or imitate Borges's prose.
- Reference fronts are empirical unions, not exact global Pareto fronts. IGD+
  reaches zero for several methods at the 800-evaluation budget, so hypervolume
  is the more discriminative final-budget metric.
- Repair can fall back to a certified witness path. Perfect feasible rates for
  repaired methods should be interpreted as a property of the representation
  and repair pipeline, not of the optimizer alone.
- Baselines do not use repair because the operator is defined for mutated and
  recombined path genotypes. The ablation isolates this asymmetry, but primary
  comparisons remain comparisons of complete search pipelines.
- Runtime reflects this Python implementation and should be treated as a
  secondary measure.

## Reproducible artifacts

The package contains the frozen JSON benchmark, all raw run-level CSV/JSONL
outputs, empirical fronts, compact and detailed statistical tables, the single
publication figure in PNG/PDF, configuration, source code, tests, and a
hash-based reproducibility manifest.
"""
    report_path = project_root / "publication_results_report.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "report": report_path,
        "benchmark_table": benchmark_table_path,
        "key_comparisons": key_table_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    for name, path in build_report(args.project_root).items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
