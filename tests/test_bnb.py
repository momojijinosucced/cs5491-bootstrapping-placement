"""Exhaustive cross-checks for the branch-and-bound solver and its lower bound.

The TAs asked for small instances where enumeration can verify the CSP /
branch-and-bound implementation, the shared-prefix accounting, and that the
pruning bound never exceeds a known completion cost. These tests enumerate
every completion of every partial state on small graphs and compare against
``branch_and_bound``. They run against ``fake_dag_builder`` (synthetic cost
curves) so they execute without the course file; re-run with the real
``dag_builder.py`` present before reporting any course numbers.
"""

import sys
import unittest
from itertools import product
from pathlib import Path

# Support test discovery from both the repository root and the tests directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fake_dag_builder as fake
from src.branch_and_bound import lower_bound, solve_bnb
from src.exhaustive import best_completion
from src.placement import EMPTY, Chain, PlacementModel, candidate_chains, decision_edges

EPS = 1e-6


def fork_model(l_max):
    """Source feeding two branches that rejoin: the shared-prefix example."""
    nodes = [
        ("src", "Identity", 0, "src"),
        ("a", "Identity", 0, "a"),
        ("b", "Identity", 0, "b"),
        ("join", "Mul", 1, "join"),
    ]
    edges = [("src", "a", 8), ("src", "b", 8), ("a", "join", 8), ("b", "join", 8)]
    return PlacementModel(nodes, edges, l_max, scorer=fake)


def chain_model(l_max):
    """A straight chain whose sink needs five input levels."""
    nodes = [
        ("src", "Identity", 0, "src"),
        ("mid", "Mul", 1, "mid"),
        ("dst", "PolyEval1", 3, "dst"),
    ]
    edges = [("src", "mid", 8), ("mid", "dst", 8)]
    return PlacementModel(nodes, edges, l_max, scorer=fake)


def join_model(l_max):
    """Two paths of unequal cost merging at a node with its own level need."""
    nodes = [
        ("src", "Identity", 0, "src"),
        ("p", "Identity", 0, "p"),
        ("q", "Mul", 1, "q"),
        ("join", "CCMM2", 2, "join"),
    ]
    edges = [("src", "p", 8), ("src", "q", 8), ("p", "join", 8), ("q", "join", 8)]
    return PlacementModel(nodes, edges, l_max, scorer=fake)


def demand_model(l_max):
    """Two heavy branches from one source: only shared refreshes are cheap."""
    nodes = [
        ("src", "Identity", 0, "src"),
        ("a", "PolyEval1", 3, "a"),
        ("b", "PolyEval1", 3, "b"),
    ]
    edges = [("src", "a", 8), ("src", "b", 8)]
    return PlacementModel(nodes, edges, l_max, scorer=fake)


class SharedPrefixTests(unittest.TestCase):
    """Shared same-source prefixes are materialized and charged once."""

    def test_one_shared_bootstrap_and_per_branch_switch(self):
        """The worked example: one shared B2, then a switch on one branch only."""
        model = fork_model(14)
        placement = {("src", "a"): Chain(boot=2), ("src", "b"): Chain(boot=2, target=5)}
        result = model.evaluate(placement, l_init=2)

        self.assertIsNotNone(result)
        ops = [op for _, op, *_ in result.nodes]
        self.assertEqual(ops.count("Bootstrap2"), 1)  # Shared: built and charged once.
        self.assertEqual(ops.count("Modswitch"), 1)
        self.assertEqual((result.n_b1, result.n_b2), (0, 4))  # B2 counts pairs.
        self.assertEqual(result.budget, 2 + 4 * 13)
        self.assertEqual(result.depth, 1)
        self.assertEqual(result.l_in["a"], 13)
        self.assertEqual(result.l_in["b"], 5)
        self.assertAlmostEqual(
            result.latency, 4 * 1293.13 + fake.LATENCY["Mul"](5), places=6
        )

    def test_bound_charges_a_shared_bootstrap_once(self):
        """The bound must not double-charge a refresh shared by two edges."""
        model = fork_model(14)
        partial = {("src", "a"): Chain(boot=2), ("src", "b"): Chain(boot=2, target=5)}
        bound = lower_bound(model, 2, partial)
        self.assertIsNotNone(bound)
        # One B2 node is 4 pairs; a double charge would add 4 more pairs.
        once = 4 * 1293.13 + min(fake.LATENCY["Mul"](l) for l in range(1, 15))
        self.assertAlmostEqual(bound, once, places=6)
        self.assertLess(bound, 8 * 1293.13)


class BoundTests(unittest.TestCase):
    """Lower-bound invariants cross-checked against exhaustive enumeration."""

    def test_leaf_bound_equals_reference_score(self):
        """At complete states the bound equals the scorer's latency exactly."""
        for build, l_max, l_init in ((chain_model, 5, 0), (join_model, 3, 1)):
            with self.subTest(graph=build.__name__, l_max=l_max, l_init=l_init):
                model = build(l_max)
                edges = decision_edges(model)
                checked = 0
                for chosen in product(*(candidate_chains(model.edge_counts[e], l_max)
                                        for e in edges)):
                    state = dict(zip(edges, chosen))
                    placement = {e: c for e, c in state.items() if c != EMPTY}
                    score = model.evaluate(placement, l_init)
                    if score is None:
                        continue
                    checked += 1
                    bound = lower_bound(model, l_init, state)
                    self.assertIsNotNone(bound)
                    self.assertAlmostEqual(bound, score.latency, places=6)
                self.assertGreater(checked, 0)

    def test_bound_never_exceeds_best_completion(self):
        """For every partial state with a feasible completion, bound <= optimum."""
        cases = ((chain_model, 5, 0, 1), (fork_model, 2, 2, 2), (join_model, 2, 2, 2))
        for build, l_max, l_init, max_depth in cases:
            model = build(l_max)
            edges = decision_edges(model)
            for depth in range(1, max_depth + 1):
                for chosen in product(*(candidate_chains(model.edge_counts[e], l_max)
                                        for e in edges[:depth])):
                    state = dict(zip(edges[:depth], chosen))
                    bound = lower_bound(model, l_init, state)
                    best = best_completion(model, l_init, state)
                    if best is None:
                        continue  # States without completions are unconstrained.
                    self.assertIsNotNone(bound)
                    self.assertLessEqual(bound, best[1].latency + EPS)

    def test_illegal_partial_states_have_no_bound(self):
        """Partial states that break the chain rules must have no bound."""
        model = fork_model(14)
        # Bootstrap2 needs 8 or 16 ciphertexts; give one edge a wrong count.
        nodes = [("src", "Identity", 0, "src"), ("dst", "Identity", 0, "dst")]
        tiny = PlacementModel(nodes, [("src", "dst", 4)], 14, scorer=fake)
        self.assertIsNone(lower_bound(tiny, 2, {("src", "dst"): Chain(boot=2)}))
        # A switch that cannot lower the post-bootstrap level is infeasible.
        self.assertIsNone(lower_bound(model, 2, {("src", "a"): Chain(boot=1, target=14)}))
        # A bare switch above the source's output level lowers nothing.
        self.assertIsNone(lower_bound(model, 2, {("src", "a"): Chain(target=2)}))


class BranchAndBoundTests(unittest.TestCase):
    """The solver equals exhaustive search on small graphs and stays honest under budgets."""

    def test_matches_exhaustive_on_small_graphs(self):
        """BnB reproduces the exhaustive optimum and proves it on small graphs."""
        cases = ((chain_model, 5, 0), (fork_model, 3, 2), (join_model, 2, 2))
        for build, l_max, l_init in cases:
            with self.subTest(graph=build.__name__, l_max=l_max, l_init=l_init):
                model = build(l_max)
                result = solve_bnb(model, l_init)
                best = best_completion(model, l_init)
                self.assertTrue(result.proven_optimal)
                self.assertEqual(result.stop_reason, "optimal")
                self.assertIsNotNone(best)
                self.assertAlmostEqual(result.score.latency, best[1].latency, places=6)
                self.assertAlmostEqual(result.lower_bound, best[1].latency, places=6)

    def test_budget_stops_search_but_keeps_the_certificate_honest(self):
        """A budgeted run must not claim optimality, and its certificate must hold."""
        # Here every completion refreshes mid-chain, but no single edge is
        # forced at the root (mid's output can reach the sink's need only
        # after an upstream refresh), so the root bound stays strictly below
        # the optimum and a zero-node search cannot claim optimality.
        model = chain_model(6)
        optimum = best_completion(model, 0)
        self.assertIsNotNone(optimum)
        self.assertLess(lower_bound(model, 0, {}), optimum[1].latency - EPS)
        result = solve_bnb(model, 0, node_budget=0)
        self.assertFalse(result.proven_optimal)
        self.assertEqual(result.stop_reason, "node_budget")
        self.assertIsNone(result.score)  # No leaf was evaluated.
        self.assertLessEqual(result.lower_bound, optimum[1].latency + EPS)

        seeded = solve_bnb(model, 0, node_budget=0, seed_placement=optimum[0])
        self.assertFalse(seeded.proven_optimal)
        self.assertEqual(seeded.stop_reason, "node_budget")
        self.assertAlmostEqual(seeded.score.latency, optimum[1].latency, places=6)
        self.assertLessEqual(seeded.lower_bound, optimum[1].latency + EPS)
        self.assertLessEqual(seeded.lower_bound, seeded.score.latency + EPS)
        self.assertAlmostEqual(
            seeded.gap_percent,
            100.0 * (seeded.score.latency - seeded.lower_bound) / seeded.score.latency,
            places=6,
        )

    def test_seed_is_never_worsened(self):
        """The search never returns a placement worse than its seed."""
        model = fork_model(3)
        best = best_completion(model, 2)
        self.assertIsNotNone(best)
        result = solve_bnb(model, 2, seed_placement=best[0])
        self.assertTrue(result.proven_optimal)
        # Tie-optimal placements may differ; the cost must not get worse.
        self.assertAlmostEqual(result.score.latency, best[1].latency, places=6)

    def test_search_finds_the_shared_refresh(self):
        """Only a shared B2 keeps both heavy branches affordable."""
        model = demand_model(14)
        result = solve_bnb(model, 2)
        best = best_completion(model, 2)
        self.assertTrue(result.proven_optimal)
        self.assertIsNotNone(best)
        self.assertAlmostEqual(result.score.latency, best[1].latency, places=6)
        ops = [op for _, op, *_ in result.score.nodes]
        self.assertEqual(ops.count("Bootstrap2"), 1)  # Both branches share it.
        self.assertEqual(ops.count("Bootstrap1"), 0)
        # Identical switch targets share one Modswitch node as well.
        self.assertEqual(ops.count("Modswitch"), 1)


if __name__ == "__main__":
    unittest.main()
