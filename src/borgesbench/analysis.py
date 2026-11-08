"""Statistical analysis and publication artifacts for BorgesBench.

The module treats the benchmark instance, rather than an individual random seed,
as the experimental unit.  Repeated seeds are first collapsed with their median.
This avoids pseudo-replication while retaining the paired benchmark design.

Example
-------
python -m borgesbench.analysis \
    --results results/raw_results.csv \
    --ablation results/ablation_results.csv \
    --output-dir results/analysis
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)


DESIGN_COLUMNS = ("instance_id", "family", "size", "difficulty")
RUN_COLUMNS = (*DESIGN_COLUMNS, "algorithm", "seed")
METRIC_DIRECTIONS: Mapping[str, str] = {
    "hypervolume": "higher",
    "igd_plus": "lower",
    "feasible_rate": "higher",
    "best_coherence": "higher",
    "runtime_sec": "lower",
}
METRICS = tuple(METRIC_DIRECTIONS)


def _stable_seed(root_seed: int, *parts: object) -> int:
    """Return a deterministic uint32 seed independent of Python hash randomisation."""

    payload = "|".join([str(root_seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def load_results(
    path: str | Path,
    *,
    label_column: str = "algorithm",
    run_key_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load and validate a raw run-level results CSV.

    Parameters
    ----------
    path:
        CSV with one row per instance, algorithm (or variant), and seed.
    label_column:
        Column identifying the compared method.  Use ``algorithm_variant`` for
        an ablation file.
    run_key_columns:
        Method columns that, together with instance and seed, uniquely identify
        a run. Defaults to the label column. Ablations shared by several base
        algorithms can use ``("algorithm", "algorithm_variant")``.

    Returns
    -------
    pandas.DataFrame
        Validated data with numeric metric, size, and seed columns.

    Raises
    ------
    ValueError
        If required columns are missing, numeric values are invalid, or run keys
        are duplicated.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Results file does not exist: {path}")
    frame = pd.read_csv(path)
    method_keys = list(run_key_columns or (label_column,))
    required = set(DESIGN_COLUMNS) | {label_column, "seed", *method_keys} | set(METRICS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no result rows")

    frame = frame.copy()
    string_columns = list(
        dict.fromkeys(("instance_id", "family", "difficulty", label_column, *method_keys))
    )
    for column in string_columns:
        if frame[column].isna().any():
            raise ValueError(f"Column {column!r} contains missing values in {path}")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"Column {column!r} contains empty labels in {path}")

    for column in ("size", "seed", *METRICS):
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all():
            bad_rows = frame.index[converted.isna() | ~np.isfinite(converted)].tolist()[:5]
            raise ValueError(
                f"Column {column!r} contains non-finite/non-numeric values in rows {bad_rows}"
            )
        frame[column] = converted

    if (frame["runtime_sec"] < 0).any():
        raise ValueError("runtime_sec must be non-negative")
    if (frame["size"] <= 0).any():
        raise ValueError("size must be positive")

    duplicate_key = ["instance_id", *method_keys, "seed"]
    duplicates = frame.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        example = frame.loc[duplicates, duplicate_key].head(5).to_dict("records")
        raise ValueError(f"Duplicate run keys {duplicate_key}; examples: {example}")
    return frame


def collapse_seeds(
    frame: pd.DataFrame, *, label_column: str = "algorithm"
) -> pd.DataFrame:
    """Collapse repeated seeds to one median score per instance and method."""

    metadata_counts = frame.groupby("instance_id", observed=True)[
        ["family", "size", "difficulty"]
    ].nunique(dropna=False)
    if (metadata_counts > 1).any(axis=None):
        offenders = metadata_counts.index[(metadata_counts > 1).any(axis=1)].tolist()[:5]
        raise ValueError(f"Instance metadata changes across runs: {offenders}")

    keys = [*DESIGN_COLUMNS, label_column]
    collapsed = (
        frame.groupby(keys, observed=True, sort=True)[list(METRICS)]
        .median()
        .reset_index()
    )
    seed_counts = (
        frame.groupby(keys, observed=True, sort=True)["seed"]
        .nunique()
        .rename("n_seeds")
        .reset_index()
    )
    return collapsed.merge(seed_counts, on=keys, how="left", validate="one_to_one")


def bootstrap_ci(
    values: Iterable[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.median,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    random_seed: int = 2026,
) -> tuple[float, float]:
    """Percentile bootstrap interval over independent benchmark instances."""

    sample = np.asarray(list(values), dtype=float)
    sample = sample[np.isfinite(sample)]
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least one")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    if sample.size == 0:
        return (np.nan, np.nan)
    if sample.size == 1:
        value = float(statistic(sample))
        return (value, value)

    rng = np.random.default_rng(random_seed)
    boot = np.empty(n_resamples, dtype=float)
    # Chunking keeps memory bounded for large user-supplied result files.
    chunk_size = min(2_000, n_resamples)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, sample.size, size=(stop - start, sample.size))
        resampled = sample[indices]
        if statistic is np.median:
            boot[start:stop] = np.median(resampled, axis=1)
        elif statistic is np.mean:
            boot[start:stop] = np.mean(resampled, axis=1)
        else:
            boot[start:stop] = np.apply_along_axis(statistic, 1, resampled)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(boot, [tail, 1.0 - tail])
    return float(low), float(high)


def _summary_rows(
    instance_scores: pd.DataFrame,
    *,
    label_column: str,
    group_columns: Sequence[str],
    scope: str,
    confidence: float,
    n_resamples: int,
    random_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped = instance_scores.groupby(list(group_columns), observed=True, sort=True)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identifiers = dict(zip(group_columns, group_key))
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            ci_low, ci_high = bootstrap_ci(
                values,
                confidence=confidence,
                n_resamples=n_resamples,
                random_seed=_stable_seed(random_seed, scope, *group_key, metric),
            )
            row: dict[str, object] = {
                "scope": scope,
                label_column: identifiers[label_column],
                "family": identifiers.get("family", "all"),
                "size": identifiers.get("size", "all"),
                "difficulty": identifiers.get("difficulty", "all"),
                "metric": metric,
                "direction": METRIC_DIRECTIONS[metric],
                "n_instances": int(values.size),
                "median": float(median),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
            rows.append(row)
    return rows


def aggregate_results(
    instance_scores: pd.DataFrame,
    *,
    label_column: str = "algorithm",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Summarise each method overall and by family, size, difficulty, and stratum."""

    scopes = {
        "overall": [label_column],
        "family": [label_column, "family"],
        "size": [label_column, "size"],
        "difficulty": [label_column, "difficulty"],
        "full_stratum": [label_column, "family", "size", "difficulty"],
    }
    rows: list[dict[str, object]] = []
    for scope, columns in scopes.items():
        rows.extend(
            _summary_rows(
                instance_scores,
                label_column=label_column,
                group_columns=columns,
                scope=scope,
                confidence=confidence,
                n_resamples=n_resamples,
                random_seed=random_seed,
            )
        )
    return pd.DataFrame(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down family-wise-error correction."""

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(p.shape, np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(p))
    if finite_positions.size == 0:
        return adjusted
    order = finite_positions[np.argsort(p[finite_positions], kind="mergesort")]
    running_max = 0.0
    m = order.size
    for rank, position in enumerate(order):
        candidate = min(1.0, (m - rank) * p[position])
        running_max = max(running_max, candidate)
        adjusted[position] = running_max
    return adjusted


def vargha_delaney_a12(
    first: Iterable[float], second: Iterable[float], *, higher_is_better: bool = True
) -> float:
    """Probability that a random score from ``first`` outperforms ``second``.

    The conventional A12 definition compares all cross-sample pairs.  For
    lower-is-better metrics the scores are sign-reversed before comparison.
    """

    x = np.asarray(list(first), dtype=float)
    y = np.asarray(list(second), dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    if not higher_is_better:
        x, y = -x, -y
    combined = np.concatenate((x, y))
    ranks = stats.rankdata(combined, method="average")
    rank_sum_x = float(ranks[: x.size].sum())
    u_x = rank_sum_x - x.size * (x.size + 1) / 2.0
    return float(u_x / (x.size * y.size))


def _effect_magnitude(a12: float) -> str:
    if not np.isfinite(a12):
        return "undefined"
    distance = abs(a12 - 0.5)
    if distance < 0.06:
        return "negligible"
    if distance < 0.14:
        return "small"
    if distance < 0.21:
        return "medium"
    return "large"


def _safe_wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size == 0 or y.size == 0:
        return np.nan, np.nan
    difference = x - y
    if np.allclose(difference, 0.0, rtol=0.0, atol=1e-15):
        return 0.0, 1.0
    try:
        result = stats.wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan


def statistical_tests(
    instance_scores: pd.DataFrame,
    *,
    label_column: str = "algorithm",
    alpha: float = 0.05,
    include_difficulty_scopes: bool = True,
) -> pd.DataFrame:
    """Run Friedman and paired Wilcoxon-Holm tests on instance-level medians.

    Each scope uses complete blocks: an instance enters a comparison only when
    every method has a score for the metric.  Post-hoc p-values are Holm-adjusted
    separately within each scope and metric.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", instance_scores)]
    if include_difficulty_scopes:
        scopes.extend(
            (f"difficulty={difficulty}", group)
            for difficulty, group in instance_scores.groupby(
                "difficulty", observed=True, sort=True
            )
        )

    output: list[dict[str, object]] = []
    for scope_name, scoped in scopes:
        for metric in METRICS:
            matrix = scoped.pivot_table(
                index="instance_id",
                columns=label_column,
                values=metric,
                aggfunc="median",
            ).sort_index(axis=1)
            complete = matrix.dropna(axis=0, how="any")
            algorithms = [str(value) for value in complete.columns]
            n_blocks, n_algorithms = complete.shape
            friedman_stat = np.nan
            friedman_p = np.nan
            if n_blocks >= 2 and n_algorithms >= 3:
                result = stats.friedmanchisquare(
                    *(complete[column].to_numpy(dtype=float) for column in complete.columns)
                )
                friedman_stat, friedman_p = float(result.statistic), float(result.pvalue)

            pair_rows: list[dict[str, object]] = []
            raw_p_values: list[float] = []
            for first, second in combinations(complete.columns, 2):
                x = complete[first].to_numpy(dtype=float)
                y = complete[second].to_numpy(dtype=float)
                wilcoxon_stat, raw_p = _safe_wilcoxon(x, y)
                a12 = vargha_delaney_a12(
                    x,
                    y,
                    higher_is_better=METRIC_DIRECTIONS[metric] == "higher",
                )
                oriented_difference = x - y
                if METRIC_DIRECTIONS[metric] == "lower":
                    oriented_difference = -oriented_difference
                pair_rows.append(
                    {
                        "scope": scope_name,
                        "metric": metric,
                        "direction": METRIC_DIRECTIONS[metric],
                        "n_complete_instances": n_blocks,
                        "n_algorithms": n_algorithms,
                        "friedman_statistic": friedman_stat,
                        "friedman_p": friedman_p,
                        "algorithm_a": str(first),
                        "algorithm_b": str(second),
                        "wilcoxon_statistic": wilcoxon_stat,
                        "p_raw": raw_p,
                        "a12_a_better": a12,
                        "effect_magnitude": _effect_magnitude(a12),
                        "median_paired_advantage_a": float(np.median(oriented_difference)),
                    }
                )
                raw_p_values.append(raw_p)

            if not pair_rows:
                pair_rows.append(
                    {
                        "scope": scope_name,
                        "metric": metric,
                        "direction": METRIC_DIRECTIONS[metric],
                        "n_complete_instances": n_blocks,
                        "n_algorithms": n_algorithms,
                        "friedman_statistic": friedman_stat,
                        "friedman_p": friedman_p,
                        "algorithm_a": "",
                        "algorithm_b": "",
                        "wilcoxon_statistic": np.nan,
                        "p_raw": np.nan,
                        "a12_a_better": np.nan,
                        "effect_magnitude": "",
                        "median_paired_advantage_a": np.nan,
                    }
                )
                adjusted = np.array([np.nan])
            else:
                adjusted = holm_adjust(raw_p_values)
            for row, p_holm in zip(pair_rows, adjusted):
                omnibus_gate = bool(
                    n_algorithms < 3
                    or (np.isfinite(friedman_p) and friedman_p < alpha)
                )
                row["p_holm"] = p_holm
                row["omnibus_gate_passed"] = omnibus_gate
                row["reject_holm"] = bool(
                    omnibus_gate and np.isfinite(p_holm) and p_holm < alpha
                )
                output.append(row)
    return pd.DataFrame(output)


def make_main_table(aggregates: pd.DataFrame, *, label_column: str = "algorithm") -> pd.DataFrame:
    """Create the compact publication table of overall algorithm performance."""

    overall = aggregates.loc[aggregates["scope"].eq("overall")].copy()
    records: list[dict[str, object]] = []
    for label, group in overall.groupby(label_column, observed=True, sort=True):
        record: dict[str, object] = {label_column: label}
        for metric in METRICS:
            row = group.loc[group["metric"].eq(metric)]
            if row.empty:
                continue
            item = row.iloc[0]
            record["n_instances"] = int(item["n_instances"])
            record[f"{metric}_median_iqr"] = (
                f"{item['median']:.4g} [{item['q1']:.4g}, {item['q3']:.4g}]"
            )
        records.append(record)
    return pd.DataFrame(records)


def make_ablation_table(
    instance_scores: pd.DataFrame,
    *,
    label_column: str = "algorithm_variant",
    alpha: float = 0.05,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    random_seed: int = 2026,
) -> pd.DataFrame:
    """Create descriptive and paired inferential results for repair ablations."""

    aggregates = aggregate_results(
        instance_scores,
        label_column=label_column,
        confidence=confidence,
        n_resamples=n_resamples,
        random_seed=random_seed,
    )
    tests = statistical_tests(
        instance_scores,
        label_column=label_column,
        alpha=alpha,
        include_difficulty_scopes=False,
    )
    overall = aggregates.loc[aggregates["scope"].eq("overall")].copy()
    rows: list[dict[str, object]] = []
    for _, test in tests.iterrows():
        first, second, metric = test["algorithm_a"], test["algorithm_b"], test["metric"]
        if not first or not second:
            continue
        summaries = overall.loc[overall["metric"].eq(metric)].set_index(label_column)
        first_row, second_row = summaries.loc[first], summaries.loc[second]
        rows.append(
            {
                "metric": metric,
                "direction": METRIC_DIRECTIONS[metric],
                "variant_a": first,
                "variant_a_median_iqr": (
                    f"{first_row['median']:.4g} [{first_row['q1']:.4g}, {first_row['q3']:.4g}]"
                ),
                "variant_b": second,
                "variant_b_median_iqr": (
                    f"{second_row['median']:.4g} [{second_row['q1']:.4g}, {second_row['q3']:.4g}]"
                ),
                "n_complete_instances": test["n_complete_instances"],
                "wilcoxon_statistic": test["wilcoxon_statistic"],
                "p_holm": test["p_holm"],
                "reject_holm": test["reject_holm"],
                "a12_a_better": test["a12_a_better"],
                "effect_magnitude": test["effect_magnitude"],
            }
        )
    return pd.DataFrame(rows)


def _difficulty_order(values: Sequence[object]) -> list[str]:
    labels = [str(value) for value in pd.unique(pd.Series(values))]
    canonical = ["low", "medium", "high"]
    lowered = {label.lower(): label for label in labels}
    ordered = [lowered[name] for name in canonical if name in lowered]
    return ordered + sorted(label for label in labels if label not in ordered)


def plot_hypervolume_by_difficulty(
    instance_scores: pd.DataFrame,
    output_path: str | Path,
    *,
    label_column: str = "algorithm",
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    random_seed: int = 2026,
) -> Path:
    """Plot median instance hypervolume with percentile-bootstrap intervals."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    algorithms = sorted(instance_scores[label_column].astype(str).unique())
    difficulties = _difficulty_order(instance_scores["difficulty"].tolist())
    if not algorithms or not difficulties:
        raise ValueError("Cannot plot an empty analysis data set")

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    x = np.arange(len(difficulties), dtype=float)
    spread = 0.72
    offset_step = spread / max(1, len(algorithms))
    offsets = (np.arange(len(algorithms)) - (len(algorithms) - 1) / 2.0) * offset_step
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(3, len(algorithms))))

    for algorithm_index, algorithm in enumerate(algorithms):
        medians, lower_errors, upper_errors = [], [], []
        for difficulty in difficulties:
            values = instance_scores.loc[
                instance_scores[label_column].astype(str).eq(algorithm)
                & instance_scores["difficulty"].astype(str).eq(difficulty),
                "hypervolume",
            ].to_numpy(dtype=float)
            if values.size == 0:
                medians.append(np.nan)
                lower_errors.append(np.nan)
                upper_errors.append(np.nan)
                continue
            median = float(np.median(values))
            low, high = bootstrap_ci(
                values,
                confidence=confidence,
                n_resamples=n_resamples,
                random_seed=_stable_seed(random_seed, "figure", algorithm, difficulty),
            )
            medians.append(median)
            lower_errors.append(max(0.0, median - low))
            upper_errors.append(max(0.0, high - median))
        ax.errorbar(
            x + offsets[algorithm_index],
            medians,
            yerr=np.asarray([lower_errors, upper_errors]),
            label=algorithm,
            color=colors[algorithm_index],
            marker="o",
            markersize=5,
            linewidth=1.5,
            capsize=3,
        )

    ax.set_xticks(x, difficulties)
    ax.set_xlabel("Instance difficulty")
    ax.set_ylabel("Hypervolume")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="Algorithm", frameon=False, ncols=min(3, len(algorithms)))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _prepare_ablation(path: str | Path) -> tuple[pd.DataFrame, str]:
    raw_columns = pd.read_csv(path, nrows=0).columns
    label = "algorithm_variant" if "algorithm_variant" in raw_columns else "algorithm"
    key_columns: Sequence[str] | None = None
    if label == "algorithm_variant" and "algorithm" in raw_columns:
        key_columns = ("algorithm", "algorithm_variant")
    frame = load_results(path, label_column=label, run_key_columns=key_columns)
    # If variants are reused under several base algorithms, make conditions unique.
    if label == "algorithm_variant" and "algorithm" in frame.columns:
        bases = frame["algorithm"].astype(str).nunique()
        if bases > 1:
            frame[label] = frame["algorithm"].astype(str) + "::" + frame[label].astype(str)
            duplicate_key = ["instance_id", label, "seed"]
            if frame.duplicated(duplicate_key).any():
                raise ValueError("Ablation conditions are not unique after combining algorithm and variant")
    return frame, label


def analyze_results(
    results_csv: str | Path,
    output_dir: str | Path,
    *,
    ablation_csv: str | Path | None = None,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    random_seed: int = 2026,
) -> dict[str, Path]:
    """Run the complete BorgesBench analysis and write reproducible artifacts.

    Returns a mapping from artifact names to paths.  Files produced are
    ``instance_level_scores.csv``, ``aggregate_summary.csv``,
    ``table_main.csv``, ``table_statistics.csv``, detailed statistical results,
    the single hypervolume figure (PNG and vector PDF), and, when supplied,
    ``table_ablation.csv``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_results(results_csv)
    instance_scores = collapse_seeds(raw)
    aggregates = aggregate_results(
        instance_scores,
        confidence=confidence,
        n_resamples=n_resamples,
        random_seed=random_seed,
    )
    tests = statistical_tests(instance_scores, alpha=alpha)
    statistics_table = tests.loc[tests["scope"].eq("overall")].copy()
    main_table = make_main_table(aggregates)

    artifacts = {
        "instance_scores": output_dir / "instance_level_scores.csv",
        "aggregates": output_dir / "aggregate_summary.csv",
        "main_table": output_dir / "table_main.csv",
        "statistics_table": output_dir / "table_statistics.csv",
        "statistical_tests_detailed": output_dir / "statistical_tests_detailed.csv",
        "hypervolume_figure_png": output_dir / "figure_hypervolume_by_difficulty.png",
        "hypervolume_figure_pdf": output_dir / "figure_hypervolume_by_difficulty.pdf",
    }
    instance_scores.to_csv(artifacts["instance_scores"], index=False)
    aggregates.to_csv(artifacts["aggregates"], index=False)
    main_table.to_csv(artifacts["main_table"], index=False)
    statistics_table.to_csv(artifacts["statistics_table"], index=False)
    tests.to_csv(artifacts["statistical_tests_detailed"], index=False)
    plot_hypervolume_by_difficulty(
        instance_scores,
        artifacts["hypervolume_figure_png"],
        confidence=confidence,
        n_resamples=n_resamples,
        random_seed=random_seed,
    )
    plot_hypervolume_by_difficulty(
        instance_scores,
        artifacts["hypervolume_figure_pdf"],
        confidence=confidence,
        n_resamples=n_resamples,
        random_seed=random_seed,
    )

    if ablation_csv is not None:
        ablation_raw, label_column = _prepare_ablation(ablation_csv)
        ablation_scores = collapse_seeds(ablation_raw, label_column=label_column)
        ablation_table = make_ablation_table(
            ablation_scores,
            label_column=label_column,
            alpha=alpha,
            confidence=confidence,
            n_resamples=n_resamples,
            random_seed=random_seed,
        )
        artifacts["ablation_table"] = output_dir / "table_ablation.csv"
        ablation_table.to_csv(artifacts["ablation_table"], index=False)
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="Raw main-results CSV")
    parser.add_argument("--ablation", type=Path, help="Optional repair-ablation CSV")
    parser.add_argument(
        "--output-dir",
        "--outdir",
        dest="output_dir",
        type=Path,
        default=Path("results/analysis"),
        help="Artifact directory",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=2026)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = analyze_results(
        args.results,
        args.output_dir,
        ablation_csv=args.ablation,
        confidence=args.confidence,
        n_resamples=args.bootstrap_resamples,
        alpha=args.alpha,
        random_seed=args.random_seed,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
