# BorgesBench

Reproducible experimental package for **BorgesBench: Evolutionary Search in
Controlled Branching Narrative Spaces**.

The package contains a frozen benchmark of 90 controlled narrative graphs,
five search methods, a repair ablation, all 4,860 executed runs, statistical
analysis, and publication-ready tables and one figure.

## Main result

DPSO achieved the highest median hypervolume (0.3677), but it was not
significantly better than multiobjective beam search after Holm correction.
All informed methods outperformed random search. The largest practical effect
came from path repair: it raised NSGA-II feasibility from 0.2906 to 1.0 and
hypervolume from 0.2974 to 0.3589, at a 77% runtime cost.

See `publication_results_report.md` for the complete interpretation.

## Reproduce

From this directory, using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
python -m borgesbench.experiment --config config/experiment.json --output-root . --workers 8
python -m borgesbench.analysis --results results/raw_results.csv --ablation results/ablation_results.csv --output-dir results/analysis --bootstrap-resamples 2000
python -m borgesbench.report --project-root .
```

Use `--quick` with the experiment command for a small integration run.

## Key artifacts

- `data/borgesbench_v1.json`: frozen 90-instance benchmark.
- `data/benchmark_validation.json`: exhaustive validation report.
- `results/raw_results.csv`: 4,500 primary run-level results.
- `results/ablation_results.csv`: 360 repair-ablation results.
- `results/archives.jsonl`: returned feasible Pareto archives.
- `results/reference_fronts.json`: empirical reference fronts.
- `results/analysis/table_main.csv`: compact primary results table.
- `results/analysis/table_benchmark.csv`: benchmark characteristics.
- `results/analysis/table_key_comparisons.csv`: compact inferential results.
- `results/analysis/table_ablation.csv`: repair-ablation table.
- `results/analysis/figure_hypervolume_by_difficulty.png`: single figure.
- `reproducibility_manifest.json`: environment, hashes, and run counts.

## Design note

Structural novelty is fixed at the instance level instead of being recomputed
from each algorithm's population. This makes the same path receive the same
objective vector everywhere and permits a shared empirical reference front.
Pairwise path-set diversity is retained as a descriptive diagnostic.

The project does not imitate Borges's prose. The names *Labyrinth*, *Double*,
and *Circular Time* identify controlled graph structures only.

