import json

import pytest

from borgesbench.benchmark import (
    DIFFICULTIES,
    FAMILIES,
    NarrativeInstance,
    difficulty_summary,
    evaluate_path,
    generate_benchmark,
    generate_instance,
    load_benchmark,
    repair_path,
    save_benchmark,
    set_diversity,
    validate_benchmark,
    validate_instance,
)


@pytest.fixture(scope="module")
def benchmark_instances():
    return generate_benchmark(seed=417)


def test_factorial_design_has_exactly_90_unique_instances(benchmark_instances):
    assert len(benchmark_instances) == 90
    assert len({instance.instance_id for instance in benchmark_instances}) == 90
    cells = {
        (instance.family, instance.size, instance.difficulty, instance.replicate)
        for instance in benchmark_instances
    }
    assert len(cells) == 3 * 2 * 3 * 5
    assert {instance.family for instance in benchmark_instances} == set(FAMILIES)


def test_generation_is_deterministic_and_cell_independent():
    first = generate_instance("The Labyrinth", 30, "medium", 2, seed=99)
    second = generate_instance("labyrinth", 30, "medium", 2, seed=99)
    assert first.to_dict() == second.to_dict()
    assert first.to_dict() != generate_instance("labyrinth", 30, "medium", 3, seed=99).to_dict()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_each_family_and_difficulty_is_a_valid_dag_with_feasible_witness(family, difficulty):
    instance = generate_instance(family, 30, difficulty, seed=5)
    report = validate_instance(instance)
    assert report.valid, report.errors
    assert 6 <= len(instance.witness_path) <= 10
    assert evaluate_path(instance, instance.witness_path)["feasible"]
    assert all(
        instance.nodes[edge["source"]]["order"] < instance.nodes[edge["target"]]["order"]
        for edge in instance.edges
    )
    assert any(edge["valid"] for edge in instance.edges)
    assert any(not edge["valid"] for edge in instance.edges)


def test_evaluation_reports_controlled_hard_violations_and_bounded_objectives():
    instance = generate_instance("double", 30, "high", seed=23)
    good = evaluate_path(instance, instance.witness_path)
    assert good["feasible"]
    assert good["violation_count"] == 0
    assert set(good["objectives"]) == {
        "coherence",
        "thematic_coverage",
        "structural_novelty",
    }
    assert all(0.0 <= value <= 1.0 for value in good["objectives"].values())

    bad = evaluate_path(instance, list(reversed(instance.witness_path)))
    assert not bad["feasible"]
    assert bad["violation_count"] > 0
    assert {item["type"] for item in bad["violations"]} >= {
        "invalid_start",
        "invalid_end",
        "missing_transition",
    }


def test_repair_returns_a_feasible_bounded_path():
    instance = generate_instance("circular_time", 60, "high", seed=8)
    malformed = [instance.end_node, "does_not_exist", instance.start_node, instance.start_node]
    repaired = repair_path(instance, malformed)
    result = evaluate_path(instance, repaired)
    assert result["feasible"], result["violations"]
    assert instance.min_path_length <= len(repaired) <= instance.max_path_length


def test_static_novelty_and_descriptive_set_diversity_are_separate():
    instance = generate_instance("labyrinth", 30, "low", seed=31)
    first = evaluate_path(instance, instance.witness_path)["structural_novelty"]
    # Evaluating unrelated paths cannot alter the static novelty score.
    evaluate_path(instance, [instance.start_node, instance.end_node])
    second = evaluate_path(instance, instance.witness_path)["structural_novelty"]
    assert first == second
    assert set_diversity([instance.witness_path]) == 0.0
    assert 0.0 <= set_diversity([instance.witness_path, instance.witness_path[:-1]]) <= 1.0


def test_difficulty_controls_are_monotone(benchmark_instances):
    summary = difficulty_summary(benchmark_instances)
    ratios = [summary[level]["mean_invalid_edge_ratio"] for level in DIFFICULTIES]
    branching = [summary[level]["mean_out_degree"] for level in DIFFICULTIES]
    mandatory = [summary[level]["mean_mandatory_count"] for level in DIFFICULTIES]
    assert ratios[0] < ratios[1] < ratios[2]
    assert branching[0] < branching[1] < branching[2]
    assert mandatory[0] < mandatory[1] < mandatory[2]


def test_json_round_trip_preserves_every_instance(tmp_path, benchmark_instances):
    destination = save_benchmark(benchmark_instances, tmp_path / "borgesbench.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["instance_count"] == 90
    restored = load_benchmark(destination)
    assert [item.to_dict() for item in restored] == [item.to_dict() for item in benchmark_instances]
    assert NarrativeInstance.from_dict(restored[0].to_dict()).to_dict() == restored[0].to_dict()


def test_validation_detects_tampering(benchmark_instances):
    tampered = NarrativeInstance.from_dict(benchmark_instances[0].to_dict())
    tampered.edges[0]["target"] = tampered.edges[0]["source"]
    report = validate_instance(tampered)
    assert not report.valid
    assert any("DAG order" in error for error in report.errors)


def test_full_benchmark_validation(benchmark_instances):
    report = validate_benchmark(benchmark_instances)
    assert report["valid"], report["errors"]
    assert report["instance_count"] == 90
