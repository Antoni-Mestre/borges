# BorgesBench statistical protocol

## Analysis unit and estimands

The benchmark instance is the experimental unit. The ten stochastic runs of an
algorithm on an instance are repeated measurements, not independent observations.
For every metric, the pipeline therefore takes the median over seeds before any
cross-algorithm comparison. This yields one value per `instance_id` and algorithm.

The five preregistered outcomes and their favourable directions are:

| Metric | Direction | Role |
|---|---:|---|
| `hypervolume` | higher | Primary Pareto-front quality outcome |
| `igd_plus` | lower | Pareto-front distance/coverage |
| `feasible_rate` | higher | Constraint satisfaction |
| `best_coherence` | higher | Best narrative coherence reached |
| `runtime_sec` | lower | Secondary computational-cost outcome |

No metric may be selected or removed after examining significance. Hypervolume is
the primary endpoint; all other results are supporting evidence.

## Descriptive summaries and uncertainty

Results are reported as median and interquartile range (Q1–Q3), with the mean and
sample standard deviation retained in the machine-readable aggregate file. A 95%
percentile bootstrap confidence interval is calculated for the median by resampling
benchmark instances with replacement. Bootstrap resampling happens after seed
collapse and therefore respects the instance-level unit of analysis. The default is
10,000 resamples and random seed 2026. Cells containing a single instance receive a
degenerate interval and should not be interpreted inferentially.

`aggregate_summary.csv` contains five scopes: overall algorithm, algorithm × family,
algorithm × graph size, algorithm × difficulty, and the full algorithm × family ×
size × difficulty stratum. The compact `table_main.csv` reports overall values.
`table_statistics.csv` contains overall tests suitable for the paper, while
`statistical_tests_detailed.csv` additionally retains the difficulty-specific tests.

## Omnibus and post-hoc comparisons

For each metric, the global null hypothesis that all algorithms perform equally is
tested with the Friedman test over complete instance blocks. An instance is included
only if all compared algorithms have a score for that metric. The same test is also
reported descriptively within each difficulty level; the overall comparison is the
confirmatory analysis.

After a significant omnibus test, every algorithm pair is compared with a two-sided
paired Wilcoxon signed-rank test on the same complete instance blocks.
Exact/asymptotic handling is delegated to SciPy's `method="auto"`; an all-zero paired
difference is assigned p = 1. Pairwise p-values are corrected by Holm's step-down
procedure separately within every metric and analysis scope. The pairwise values are
retained for transparency even when the omnibus test is not significant, but the
`reject_holm` flag is gated by the omnibus result. A direct two-condition ablation
does not need this gate. The family-wise significance threshold is 0.05. Statistical
claims require both the adjusted p-value and the effect size; p-values alone are not
treated as evidence of practical importance.

## Effect size

Vargha–Delaney A12 is reported as the probability that a randomly selected
instance-level score from algorithm A is better than one from algorithm B, with ties
counting one half. Lower-is-better metrics are sign-reversed before calculation, so
values above 0.5 always favour A. Magnitudes use the conventional absolute distance
from 0.5: negligible (< 0.06), small (< 0.14), medium (< 0.21), and large (>= 0.21).
Because A12 is an all-pairs probability rather than a paired effect, the table also
reports the median paired advantage of A in the favourable direction.

## Repair ablation

The preregistered ablation compares NSGA-II with and without path repair using the
same instances, budgets, and seeds. The file may identify conditions through
`algorithm_variant`; if that column is absent, `algorithm` is used. When several base
algorithms reuse the same variant labels, the pipeline constructs unambiguous
`algorithm::variant` conditions. Seed medians, paired Wilcoxon tests, Holm correction,
and A12 are applied exactly as in the main comparison. `table_ablation.csv` includes
all five outcomes, while feasibility and hypervolume are the principal ablation
readouts.

## Figure

The only inferential figure is `figure_hypervolume_by_difficulty`: for each algorithm
and difficulty level it shows median instance-level hypervolume and its 95% bootstrap
interval. Its PDF and PNG files are alternative renderings of the same figure, not
separate figures. Difficulty is ordered low, medium, high when those labels are used.

## Input contract and execution

Main results require one row per run and these columns:

```text
instance_id,family,size,difficulty,algorithm,seed,hypervolume,igd_plus,
feasible_rate,best_coherence,runtime_sec
```

The ablation file has the same design and metric columns and may replace the compared
label with `algorithm_variant`. Duplicate `(instance_id, method, seed)` keys,
non-finite numeric values, inconsistent instance metadata, non-positive graph sizes,
and negative runtimes are rejected rather than silently aggregated.

Run from an installed project (or with `PYTHONPATH=src`):

```bash
python -m borgesbench.analysis \
  --results results/raw_results.csv \
  --ablation results/ablation_results.csv \
  --output-dir results/analysis \
  --bootstrap-resamples 10000 \
  --random-seed 2026
```

Programmatic entry points are `load_results`, `collapse_seeds`, `aggregate_results`,
`statistical_tests`, `make_main_table`, `make_ablation_table`,
`plot_hypervolume_by_difficulty`, and the end-to-end `analyze_results` function.
