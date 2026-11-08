# BorgesBench design decisions

## Scope

The frozen benchmark contains 90 instances: three narrative families, two graph
sizes, three difficulty levels, and five deterministic replications per cell.
Every instance is validated before the optimization phase and contains at least
one feasible start-to-end path of length 6--10.

## Objective semantics

All three objectives are maximized and normalized to `[0, 1]`:

1. narrative coherence: soft causal, temporal, character-continuity, and
   transition-quality constraints satisfied by a path;
2. thematic coverage: required motifs covered by the path;
3. structural novelty: the mean rarity of the selected branch transitions,
   using instance-level edge rarity fixed before optimization.

The proposal initially described diversity relative to the current population.
That would make the same path receive different objective values under different
algorithms and would invalidate a shared reference front for hypervolume and
IGD+. BorgesBench therefore uses fixed structural novelty as the third search
objective and reports mean pairwise Jaccard distance of returned paths as an
additional descriptive diversity diagnostic.

## Hard constraints

A feasible path must start and end correctly, respect the length interval, use
existing directed edges, introduce a character before dependent events, include
all mandatory events, and avoid incompatible event pairs. Invalid transitions
may exist in the graph but cannot occur in a feasible solution.

## Fairness

Each method receives 800 complete-path evaluations per instance and seed. The
population methods use 40 individuals. The five primary methods are random
search, multiobjective beam search, NSGA-II, MOEA/D, and discrete PSO. The
primary comparison uses repair for all stochastic optimizers. A targeted
NSGA-II ablation compares repair against penalty-only handling on a stratified
subset.

## Reference front

For each instance, the empirical reference set is the non-dominated union of
all objective vectors produced by all algorithms and primary seeds. Hypervolume
uses the origin as the reference point. IGD+ is computed against the empirical
reference set. Both metrics are therefore instance-specific and are compared
between algorithms only after blocking on instance and seed.

## Statistical unit

The run is the raw observation, but inferential comparisons are blocked by
instance and seed. Global comparisons use Friedman tests. Pairwise tests use
two-sided Wilcoxon signed-rank tests with Holm correction and Vargha--Delaney
`A12` effect sizes. Medians, IQRs, and instance-clustered bootstrap confidence
intervals accompany the tests.
