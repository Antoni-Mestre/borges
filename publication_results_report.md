# BorgesBench: experimental results report

## Executive result

The experiment supports a qualified conclusion rather than a single-algorithm
winner. DPSO obtained the highest overall median hypervolume
(0.3677), but its difference from
multiobjective beam search (0.3656) was
not significant after Holm correction (p=0.0524,
A12=0.518, negligible). The robust result
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
(Friedman chi-square=82.37,
p=5.47e-17) and IGD+
(chi-square=85.12,
p=1.43e-17). Overall medians were:

| Method | Hypervolume | IGD+ | Feasible rate | Best coherence | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| random | 0.3372 | 0.0139 | 0.0037 | 0.894 | 0.5427 |
| beam | 0.3656 | 0 | 0.0094 | 0.9072 | 0.1745 |
| nsga2 | 0.3657 | 0 | 1 | 0.9098 | 0.5286 |
| moead | 0.3637 | 0.0008 | 1 | 0.9074 | 0.3908 |
| dpso | 0.3677 | 0 | 1 | 0.911 | 0.3185 |


DPSO improved hypervolume over random search with a small effect
(Holm p=2.33e-10, A12=0.635).
The other informed methods also significantly exceeded random search, but
effect sizes for hypervolume remained small. Differences among beam, NSGA-II,
MOEA/D, and DPSO were mostly negligible in magnitude even when rank tests were
significant. The evidence therefore does not justify claiming a decisive
evolutionary or swarm winner.

Beam search reveals a useful quality-yield distinction. It reached competitive
front quality with a median feasible rate of only
0.0094; the repaired population
methods reached 1.0. Beam therefore finds a small number of strong paths, while
repair produces valid paths reliably across the full evaluation stream.

## Difficulty and family effects

Hypervolume fell with difficulty for every method. DPSO declined from
0.4178 on low instances to
0.335 on high instances; random search fell
from 0.4206 to
0.2565. The separation is consequently driven
mainly by medium and high difficulty rather than easy graphs.

The Double family was the hardest in front quality (DPSO median
0.3175), whereas Labyrinth and Circular Time were
higher (0.3929 and
0.383). This is consistent with the dense
short paired corridors creating many locally plausible but globally constrained
alternatives.

## Repair ablation

Repair raised NSGA-II median hypervolume from 0.2974 to 0.3589
(20.7% relative), reduced median IGD+ from
0.02273 to 0, and raised feasible rate from 0.2906 to 1.0. Effects were medium
for hypervolume and large for IGD+, feasibility, and best coherence; every
quality comparison remained significant after correction. The cost was runtime:
median time increased from 0.2954 s to 0.5233 s
(77.1%). The repair operator is therefore
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
