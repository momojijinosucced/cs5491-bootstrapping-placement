"""Reproduce the reference harness, MLP search metrics, and milestone plots."""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import dag_builder as db
from matplotlib import pyplot as plt
from placement import PlacementModel
from solver import solve_mlp


L_MAX = 14
BLOCKS = {
    "MLP": (db.get_mlp_empty, db.get_mlp_baseline_inserted),
    "Attention": (db.get_attn_empty, db.get_attn_baseline_inserted),
}


def baseline_row(block, level, model, inserted):
    """Check rebuilt baseline levels and score against the given inserted DAG."""
    given_nodes, given_edges = inserted
    given = db.evaluate_dag(given_nodes, given_edges, level, L_MAX)
    if given is None:
        raise RuntimeError(f"Provided {block} baseline infeasible at {level}")
    normalized = model.baseline_placement(given_nodes, given_edges, level)
    verified = model.evaluate(normalized, level)
    given_counts = db.budget_of(given_nodes, given_edges, level, L_MAX)
    if (verified is None or abs(verified.latency - given["latency"]) > 1e-7
            or (verified.n_b1, verified.n_b2, verified.budget) != given_counts
            or verified.depth != db.depth_of(given_nodes, given_edges)
            or any((verified.l_in[node], verified.l_out[node]) !=
                   (given["l_in"][node], given["l_out"][node]) for node, *_ in model.nodes)):
        raise RuntimeError(f"{block} scorer or rebuild mismatch at {level}")
    rounded_expected = round(28129.6897 + 0.16 * level, 2) if block == "MLP" else 53558.17
    if abs(round(verified.latency, 2) - rounded_expected) > 1e-7:
        raise RuntimeError(f"{block} disagrees with reference values at {level}")
    return {
        "block": block, "l_init": level, "nodes": len(model.nodes), "edges": len(model.edges),
        "feasible": True, "latency_ms": round(verified.latency, 6),
        "b1": verified.n_b1, "b2": verified.n_b2, "depth": verified.depth,
        "budget": verified.budget, "utilization": round(verified.utilization, 8),
    }


def write_rows(path, rows):
    """Write a stable-header CSV for a nonempty list of result records."""
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_plot(path, rows):
    """Plot the estimated latency saved by the search at each input level."""
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 12}):
        fig, ax = plt.subplots(figsize=(5.9, 2.85), layout="constrained")
        x = [r["l_init"] for r in rows]
        y = [r["saved_percent"] for r in rows]
        ax.bar(x, y, color="#245781", width=0.7)
        ax.axhline(0, color="#596675", lw=0.7)
        ax.set_xticks(x)
        ax.set_xlabel("Input level")
        ax.set_ylabel("Estimated latency saved (%)")
        ax.set_ylim(0, max(1, max(y) * 1.16))
        ax.grid(axis="y", alpha=0.24, lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(path, dpi=240, facecolor="white")
        plt.close(fig)


def run(levels, max_steps, output, make_plots=True):
    """Evaluate both given baselines and the original MLP local-search prototype."""
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    baselines = []
    for block, (get_empty, get_baseline) in BLOCKS.items():
        model, inserted = PlacementModel(*get_empty()), get_baseline()
        baselines.extend(baseline_row(block, level, model, inserted) for level in levels)
    write_rows(output / "baselines.csv", baselines)
    baseline_for = {r["l_init"]: r for r in baselines if r["block"] == "MLP"}

    rows, details = [], []
    examples = {}
    for level in levels:
        result = solve_mlp(level, max_steps=max_steps)
        score, base = result.score, baseline_for[level]
        if abs(result.initial_latency_ms - base["latency_ms"]) > 1e-6:
            raise RuntimeError(f"MLP seed differs from baseline at {level}")
        saved = result.initial_latency_ms - score.latency
        if saved < -1e-8:
            raise RuntimeError(f"MLP result worse than baseline at {level}")
        rows.append({
            "l_init": level, "feasible": True, "baseline_ms": base["latency_ms"],
            "candidate_ms": round(score.latency, 6), "saved_ms": round(saved, 6),
            "saved_percent": round(100 * saved / result.initial_latency_ms, 8),
            "baseline_bootstraps": base["b1"] + base["b2"],
            "candidate_bootstraps": score.n_b1 + score.n_b2,
            "candidate_b1": score.n_b1, "candidate_b2": score.n_b2,
            "depth": score.depth, "budget": score.budget,
            "utilization": round(score.utilization, 8),
            "evaluated_candidates": result.evaluated,
            "feasible_candidates": result.feasible_candidates,
            "accepted_moves": result.steps, "stop_reason": result.stop_reason,
            "solver_seconds": round(result.solver_seconds, 6),
        })
        details.append({
            "l_init": level,
            "placement": [{"edge": edge, **asdict(chain)} for edge, chain in sorted(result.placement.items())],
            "moves": [asdict(move) for move in result.moves],
            "edge_levels": [{"edge": edge, "level": lev} for edge, lev in sorted(score.levels.items())],
        })
        if level in (0, 14):
            examples[level] = score

    write_rows(output / "mlp_search.csv", rows)
    (output / "mlp_search_details.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    if make_plots:
        comparison_plot(output / "mlp_comparison.png", rows)
        for level, score in examples.items():
            db.visualize_empty_dag(
                score.nodes, score.edges,
                title=(f"MLP placement at L_init={level}: estimated {score.latency:,.2f} ms, "
                       f"B1={score.n_b1}, B2={score.n_b2}, depth={score.depth}, "
                       f"budget={score.budget}, utilisation={score.utilization:.3f}"),
                filename=str(output / f"mlp_l{level}.png"), x_spacing=9.0, y_spacing=2.4,
                levels=score.edge_lvl,
            )

    summary = {
        "input_levels": levels, "step_cap": max_steps,
        "baseline_cases": len(baselines), "mlp_cases": len(rows),
        "mlp_improved_cases": sum(r["saved_ms"] > 0 for r in rows),
        "best_case": max(rows, key=lambda r: r["saved_percent"]),
        "mlp_search_seconds": round(sum(r["solver_seconds"] for r in rows), 6),
        "wall_seconds": round(perf_counter() - started, 6),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    """Parse command-line options and run the milestone experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", choices=range(L_MAX + 1), default=list(range(L_MAX + 1)))
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum accepted MLP search moves per instance")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV and JSON results")
    args = parser.parse_args()
    if args.max_steps < 0:
        parser.error("--max-steps must be nonnegative")
    summary = run(sorted(set(args.levels)), args.max_steps, args.output_dir, not args.no_plots)
    print(f"Verified {summary['baseline_cases']} baseline cases; MLP improved in "
          f"{summary['mlp_improved_cases']}/{summary['mlp_cases']} cases.")
    print(f"Output: {args.output_dir.resolve()} ({summary['wall_seconds']:.2f} s)")


if __name__ == "__main__":
    main()
