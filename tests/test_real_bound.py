"""Real-scorer checks for the branch-and-bound bound and solver.

``test_bnb.py`` verifies the bound and the search against exhaustive
enumeration on small graphs scored with ``fake_dag_builder``. This file
repeats the checks that need the course's real latency curves: leaf
exactness on the official MLP and attention placements, admissibility
against explicit completions of partial states on the real MLP, and
BnB versus exhaustive on small graphs scored by ``dag_builder``.
It skips itself when the course file is missing.
"""

import random
import sys
import unittest
from pathlib import Path

# Support test discovery from both the repository root and the tests directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import dag_builder as course
except ModuleNotFoundError:  # The course file comes from Canvas; skip without it.
    course = None

from branch_and_bound import lower_bound, solve_bnb
from exhaustive import best_completion
from placement import EMPTY, Chain, PlacementModel, candidate_chains, decision_edges

EPS = 1e-6


def full_state(model, placement):
    """A complete chain assignment: every edge decided, EMPTY where absent."""
    return {edge: placement.get(edge, EMPTY) for edge in model.edge_counts}


@unittest.skipIf(course is None, "course dag_builder.py is not present")
class LeafExactnessTests(unittest.TestCase):
    """The bound equals the scorer's latency at complete placements."""

    def test_leaf_bound_matches_scorer_on_official_placements(self):
        """At the official baselines the bound equals evaluate_dag exactly."""
        cases = (
            ("MLP", course.get_mlp_empty, course.get_mlp_baseline_inserted, (0, 6, 14)),
            ("Attention", course.get_attn_empty, course.get_attn_baseline_inserted, (0, 9, 14)),
        )
        for name, get_empty, get_inserted, levels in cases:
            model = PlacementModel(*get_empty())
            inserted_nodes, inserted_edges = get_inserted()
            for level in levels:
                with self.subTest(block=name, level=level):
                    placement = model.baseline_placement(inserted_nodes, inserted_edges, level)
                    score = model.evaluate(placement, level)
                    self.assertIsNotNone(score)
                    bound = lower_bound(model, level, full_state(model, placement))
                    self.assertIsNotNone(bound)
                    self.assertAlmostEqual(bound, score.latency, places=6)

    def test_leaf_bound_matches_scorer_on_random_placements(self):
        """Leaf exactness holds for arbitrary feasible placements, not just baselines.

        Uniformly random chain assignments are almost never feasible on a
        graph this size, so half of the samples mutate one to three edges of
        the official placement instead.
        """
        model = PlacementModel(*course.get_mlp_empty())
        inserted_nodes, inserted_edges = course.get_mlp_baseline_inserted()
        edges = decision_edges(model)
        rng = random.Random(20261117)
        for level in (0, 6, 14):
            baseline = model.baseline_placement(inserted_nodes, inserted_edges, level)
            checked = 0
            for attempt in range(4000):
                if attempt % 2 == 0:
                    state = {}
                    for edge in edges:
                        chains = candidate_chains(model.edge_counts[edge], model.l_max)
                        weights = [6 if c == EMPTY else 1 for c in chains]
                        state[edge] = rng.choices(chains, weights=weights)[0]
                else:
                    state = {e: baseline.get(e, EMPTY) for e in edges}
                    for edge in rng.sample(edges, rng.randint(1, 3)):
                        state[edge] = rng.choice(
                            candidate_chains(model.edge_counts[edge], model.l_max))
                placement = {e: c for e, c in state.items() if c != EMPTY}
                score = model.evaluate(placement, level)
                if score is None:
                    continue
                checked += 1
                bound = lower_bound(model, level, state)
                self.assertIsNotNone(bound)
                self.assertAlmostEqual(bound, score.latency, places=6)
                if checked >= 25:
                    break
            self.assertGreater(checked, 0, f"no feasible random placement at level {level}")


@unittest.skipIf(course is None, "course dag_builder.py is not present")
class AdmissibilityTests(unittest.TestCase):
    """The bound never exceeds a real completion's latency on the full MLP."""

    def test_bound_never_exceeds_a_real_completion(self):
        """On the real MLP, bound(partial) <= latency of any feasible completion."""
        model = PlacementModel(*course.get_mlp_empty())
        inserted_nodes, inserted_edges = course.get_mlp_baseline_inserted()
        edges = decision_edges(model)
        rng = random.Random(20261117)
        for level in (0, 6, 14):
            baseline = model.baseline_placement(inserted_nodes, inserted_edges, level)
            self.assertIsNotNone(model.evaluate(baseline, level))
            cases = []
            # Prefixes of the baseline assignment, and random partial states.
            for depth in range(1, min(8, len(edges)) + 1):
                cases.append({e: baseline.get(e, EMPTY) for e in edges[:depth]})
            for _ in range(20):
                depth = rng.randint(1, min(8, len(edges)))
                cases.append({
                    e: rng.choice(candidate_chains(model.edge_counts[e], model.l_max))
                    for e in edges[:depth]
                })
            for partial in cases:
                bound = lower_bound(model, level, partial)
                completions = []
                # Two explicit completions: baseline chains, then all EMPTY.
                for filler in (baseline.get, lambda e: EMPTY):
                    completion = dict(partial)
                    completion.update(
                        (e, filler(e)) for e in edges if e not in partial
                    )
                    placement = {e: c for e, c in completion.items() if c != EMPTY}
                    score = model.evaluate(placement, level)
                    if score is not None:
                        completions.append(score.latency)
                if not completions:
                    continue  # No witness completion to compare against.
                self.assertIsNotNone(bound, f"bound is None with feasible completions: {partial}")
                for latency in completions:
                    self.assertLessEqual(bound, latency + EPS)


@unittest.skipIf(course is None, "course dag_builder.py is not present")
class RealCurveSearchTests(unittest.TestCase):
    """BnB matches exhaustive enumeration when the real curves do the scoring."""

    @staticmethod
    def fork_model(l_max):
        """Source feeding two branches that rejoin (real cost curves)."""
        nodes = [
            ("src", "Identity", 0, "src"),
            ("a", "Identity", 0, "a"),
            ("b", "Identity", 0, "b"),
            ("join", "Mul", 1, "join"),
        ]
        edges = [("src", "a", 8), ("src", "b", 8), ("a", "join", 8), ("b", "join", 8)]
        return PlacementModel(nodes, edges, l_max)

    @staticmethod
    def chain_model(l_max):
        """A straight chain whose sink needs five input levels (real cost curves)."""
        nodes = [
            ("src", "Identity", 0, "src"),
            ("mid", "Mul", 1, "mid"),
            ("dst", "PolyEval1", 3, "dst"),
        ]
        edges = [("src", "mid", 8), ("mid", "dst", 8)]
        return PlacementModel(nodes, edges, l_max)

    @staticmethod
    def join_model(l_max):
        """Two unequal-cost paths merging at a node with its own need (real curves)."""
        nodes = [
            ("src", "Identity", 0, "src"),
            ("p", "Identity", 0, "p"),
            ("q", "Mul", 1, "q"),
            ("join", "CCMM2", 2, "join"),
        ]
        edges = [("src", "p", 8), ("src", "q", 8), ("p", "join", 8), ("q", "join", 8)]
        return PlacementModel(nodes, edges, l_max)

    def test_bnb_matches_exhaustive_with_real_curves(self):
        """BnB equals exhaustive optima when the real curves score the leaves."""
        cases = ((self.chain_model, 5, 0), (self.join_model, 3, 1), (self.fork_model, 2, 2))
        for build, l_max, l_init in cases:
            with self.subTest(graph=build.__name__, l_max=l_max, l_init=l_init):
                model = build(l_max)
                best = best_completion(model, l_init)
                self.assertIsNotNone(best, "the instance must admit a feasible placement")
                result = solve_bnb(model, l_init)
                self.assertTrue(result.proven_optimal)
                self.assertAlmostEqual(result.score.latency, best[1].latency, places=6)

    def test_bnb_seed_is_never_worsened_with_real_curves(self):
        """Seeding with the optimum cannot make the result worse."""
        model = self.join_model(3)
        best = best_completion(model, 1)
        result = solve_bnb(model, 1, seed_placement=best[0])
        self.assertLessEqual(result.score.latency, best[1].latency + EPS)


if __name__ == "__main__":
    unittest.main()
