"""Regression checks for MLP search and the course DAG placement rules."""

import sys
import unittest
from pathlib import Path

# Support test discovery from both the repository root and the milestone directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import dag_builder as course
except ModuleNotFoundError:  # The course file comes from Canvas; skip without it.
    course = None

from src.placement import Chain, InvalidPlacement, PlacementModel

try:
    from src.solver import solve_mlp
except ModuleNotFoundError:
    solve_mlp = None


@unittest.skipIf(course is None, "course dag_builder.py is not present")
class PlacementTests(unittest.TestCase):
    """Placement semantics checked against the course scorer's own baselines."""

    def test_matches_provided_baselines(self):
        """Rebuilding the official placements reproduces evaluate_dag exactly."""
        cases = (
            ("MLP", course.get_mlp_empty, course.get_mlp_baseline_inserted, 6),
            ("Attention", course.get_attn_empty, course.get_attn_baseline_inserted, 9),
        )
        for name, get_empty, get_baseline, level in cases:
            with self.subTest(block=name, level=level):
                nodes, edges = get_empty()
                baseline_nodes, baseline_edges = get_baseline()
                model = PlacementModel(nodes, edges)
                placement = model.baseline_placement(baseline_nodes, baseline_edges, level)
                result = model.evaluate(placement, level)
                reference = course.evaluate_dag(baseline_nodes, baseline_edges, level, 14)
                n_b1, n_b2, budget = course.budget_of(baseline_nodes, baseline_edges, level, 14)
                depth = course.depth_of(baseline_nodes, baseline_edges)

                self.assertIsNotNone(result)
                self.assertAlmostEqual(result.latency, reference["latency"], places=7)
                self.assertEqual((result.n_b1, result.n_b2, result.budget), (n_b1, n_b2, budget))
                self.assertEqual(result.depth, depth)
                self.assertAlmostEqual(result.utilization, depth / budget)
                for node, *_ in nodes:
                    self.assertEqual(result.l_in[node], reference["l_in"][node], node)
                    self.assertEqual(result.l_out[node], reference["l_out"][node], node)

    def test_removing_initial_mlp_bootstrap_depends_on_input_level(self):
        """The first MLP bootstrap is redundant only at a full-height input."""
        model = PlacementModel(*course.get_mlp_empty())
        inserted_nodes, inserted_edges = course.get_mlp_baseline_inserted()
        first_bootstrap = ("Conj_X_mlp", "Mul_mask_mlp")
        for level in (6, 14):
            with self.subTest(level=level):
                placement = model.baseline_placement(inserted_nodes, inserted_edges, level)
                baseline = model.evaluate(placement, level)
                self.assertIsNotNone(baseline)
                self.assertEqual(placement.pop(first_bootstrap), Chain(boot=1))
                modified = model.evaluate(placement, level)
                if level == 6:
                    self.assertIsNone(modified)
                else:
                    self.assertIsNotNone(modified)
                    self.assertEqual(modified.n_b1, baseline.n_b1 - 4)
                    self.assertEqual(modified.n_b2, baseline.n_b2)
                    self.assertEqual(modified.l_out["X_out_mlp"], baseline.l_out["X_out_mlp"])
                    self.assertAlmostEqual(baseline.latency - modified.latency, 4 * 1040.15)

    def test_shared_bootstrap_and_modswitch_prefixes_count_once(self):
        """Edges sharing a bootstrap-and-switch prefix pay for one node of each."""
        nodes = [(name, "Identity", 0, name) for name in ("src", "a", "b", "c")]
        nodes.append(("join", "Mul", 1, "join"))
        edges = [("src", branch, 8) for branch in ("a", "b", "c")]
        edges += [(branch, "join", 8) for branch in ("a", "b", "c")]
        model = PlacementModel(nodes, edges)
        result = model.evaluate(
            {
                ("src", "a"): Chain(boot=2, target=3),
                ("src", "b"): Chain(boot=2, target=3),
                ("src", "c"): Chain(boot=2),
            },
            l_init=2,
        )

        self.assertIsNotNone(result)
        ops = [op for _, op, *_ in result.nodes]
        self.assertEqual(ops.count("Bootstrap2"), 1)
        self.assertEqual(ops.count("Modswitch"), 1)
        self.assertEqual((result.n_b1, result.n_b2), (0, 4))
        self.assertEqual(result.budget, 2 + 4 * 13)
        self.assertEqual(result.depth, 1)
        self.assertEqual([result.l_in[name] for name in ("a", "b", "c", "join")], [3, 3, 13, 3])
        self.assertEqual(result.l_out["join"], 2)
        self.assertAlmostEqual(result.latency, 4 * 1293.13 + course.LATENCY["Mul"](3))

    def test_illegal_bootstrap_and_modswitch_are_rejected(self):
        """B2 on wrong counts and upward switches are rejected as infeasible."""
        nodes = [(name, "Identity", 0, name) for name in ("src", "dst")]
        cases = (
            ("odd ciphertext count", 3, Chain(boot=2)),
            ("unsupported even ciphertext count", 4, Chain(boot=2)),
            ("upward Modswitch", 8, Chain(target=4)),
            ("upward Modswitch after B2", 8, Chain(boot=2, target=14)),
        )
        for label, count, chain in cases:
            with self.subTest(case=label):
                model = PlacementModel(nodes, [("src", "dst", count)])
                placement = {("src", "dst"): chain}
                with self.assertRaises(InvalidPlacement):
                    model.compile(placement, l_init=2)
                self.assertIsNone(model.evaluate(placement, l_init=2))


@unittest.skipIf(course is None, "course dag_builder.py is not present")
class SearchTests(unittest.TestCase):
    """The milestone local search improves the baseline and is deterministic."""

    def test_bounded_search_improves_baseline_and_repeats_its_result(self):
        """One search step strictly improves the baseline and repeats identically."""
        baseline = solve_mlp(14, max_steps=0)
        self.assertEqual((baseline.evaluated, baseline.steps), (0, 0))
        first = solve_mlp(14, max_steps=1)
        second = solve_mlp(14, max_steps=1)

        self.assertEqual(first.steps, 1)
        self.assertGreater(first.evaluated, 0)
        self.assertLess(first.score.latency, baseline.score.latency)
        self.assertEqual(first.placement, second.placement)
        self.assertEqual(first.moves, second.moves)
        self.assertEqual(first.evaluated, second.evaluated)

        checked = course.evaluate_dag(first.score.nodes, first.score.edges, 14, 14)
        self.assertIsNotNone(checked)
        self.assertAlmostEqual(first.score.latency, checked["latency"], places=7)


if __name__ == "__main__":
    unittest.main()
