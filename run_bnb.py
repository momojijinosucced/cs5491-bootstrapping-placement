"""Exact branch-and-bound runs on the MLP and attention blocks.

Extends the milestone's local-search experiments with the certified
branch-and-bound solver. Each case is seeded with a feasible placement (the
milestone local search for MLP, the official placement for attention) and
searched under a published wall-clock budget. Every row reports the best
placement found together with the certified global lower bound and the
optimality gap whenever the budget stops the search before optimality is
proven.

Usage:
    python run_bnb.py --block both --levels 0-14 --time-budget 60
    python run_bnb.py --seed-pool results/bnb_search.json --output results/bnb_search.csv
"""

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import dag_builder as course
from branch_and_bound import solve_bnb
from placement import EMPTY, Chain, PlacementModel
from solver import solve_mlp

BLOCKS = {
    "MLP": (course.get_mlp_empty, course.get_mlp_baseline_inserted),
    "Attention": (course.get_attn_empty, course.get_attn_baseline_inserted),
}

FIELDS = (
    "block", "l_init", "baseline_ms", "seed_ms", "seed_source", "bnb_ms", "saved_vs_seed_ms",
    "b1", "b2", "depth", "budget", "utilization",
    "proven_optimal", "lower_bound_ms", "gap_percent",
    "expansions", "leaves", "solver_seconds", "stop_reason",
    "time_budget_s", "node_budget",
)


def load_placement(item):
    """Rebuild a Chain placement from a ``run_bnb.py`` details entry."""
    return {tuple(entry["edge"]): Chain(boot=entry["boot"], target=entry["target"])
            for entry in item["placement"]}


def load_seed_pool(paths):
    """Placements from earlier runs, tried as seeds at every level."""
    pool = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        for item in json.loads(path.read_text(encoding="utf-8")):
            pool.append((item["block"], load_placement(item)))
        details = path.with_name("mlp_search_details.json")
        if details.exists():
            for item in json.loads(details.read_text(encoding="utf-8")):
                pool.append(("MLP", {
                    tuple(entry["edge"]): Chain(boot=entry["boot"], target=entry["target"])
                    for entry in item["placement"]
                }))
    return pool


def seed_placement(model, block, level, pool):
    """Best feasible seed: local search (MLP) / official placement, plus the pool.

    A placement found at one input level often stays feasible at another, so
    every collected placement is tried and the cheapest feasible one seeds
    the search.
    """
    candidates = []
    inserted_nodes, inserted_edges = BLOCKS[block][1]()
    official = model.baseline_placement(inserted_nodes, inserted_edges, level)
    candidates.append(("official", official))
    if block == "MLP":
        candidates.append(("local-search", solve_mlp(level).placement))
    candidates.extend((f"pool:{i}", placement)
                      for i, (pool_block, placement) in enumerate(pool)
                      if pool_block == block)
    best = None
    for source, placement in candidates:
        score = model.evaluate(placement, level)
        if score is not None and (best is None or score.latency < best[2].latency):
            best = (source, placement, score)
    return best[0], best[1], best[2].latency


def run_case(block, level, time_budget, node_budget, pool):
    """Solve one (block, level) case under budget; return its CSV row and placement detail."""
    model = PlacementModel(*BLOCKS[block][0]())
    seed_source, placement, seed_ms = seed_placement(model, block, level, pool)
    baseline = model.evaluate(
        model.baseline_placement(*BLOCKS[block][1](), level), level)
    result = solve_bnb(model, level, seed_placement=placement,
                       time_budget=time_budget, node_budget=node_budget)
    assert result.score is not None, "the seeded incumbent must survive the search"
    row = {
        "block": block, "l_init": level,
        "baseline_ms": round(baseline.latency, 6), "seed_ms": round(seed_ms, 6),
        "seed_source": seed_source,
        "bnb_ms": round(result.score.latency, 6),
        "saved_vs_seed_ms": round(seed_ms - result.score.latency, 6),
        "b1": result.score.n_b1, "b2": result.score.n_b2,
        "depth": result.score.depth, "budget": result.score.budget,
        "utilization": round(result.score.utilization, 8),
        "proven_optimal": result.proven_optimal,
        "lower_bound_ms": round(result.lower_bound, 6) if result.lower_bound is not None else "",
        "gap_percent": round(result.gap_percent, 8) if result.gap_percent is not None else "",
        "expansions": result.expansions, "leaves": result.leaves,
        "solver_seconds": round(result.solver_seconds, 6),
        "stop_reason": result.stop_reason,
        "time_budget_s": time_budget, "node_budget": node_budget,
    }
    detail = {
        "block": block, "l_init": level,
        "placement": [{"edge": edge, **asdict(chain)}
                      for edge, chain in sorted(result.placement.items())],
    }
    return row, detail


def parse_levels(items):
    """Accept plain level numbers and inclusive ranges like ``0-14``."""
    levels = []
    for item in items:
        if "-" in item:
            lo, hi = item.split("-", 1)
            levels.extend(range(int(lo), int(hi) + 1))
        else:
            levels.append(int(item))
    return levels


def main():
    """Run the 2x15 sweep and write the results CSV plus JSON placement details."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=["MLP", "Attention", "both"], default="both")
    parser.add_argument("--levels", nargs="+", default=[str(i) for i in range(15)])
    parser.add_argument("--time-budget", type=float, default=60.0,
                        help="wall-clock seconds per case (published with the results)")
    parser.add_argument("--node-budget", type=int, default=200_000,
                        help="search-node cap per case; also bounds memory use")
    parser.add_argument("--seed-pool", nargs="+", default=[],
                        help="bnb_search.json files from earlier runs; their placements "
                             "are tried as seeds at every level")
    parser.add_argument("--output", default="results/bnb_search.csv")
    args = parser.parse_args()

    levels = parse_levels(args.levels)
    blocks = ["MLP", "Attention"] if args.block == "both" else [args.block]
    pool = load_seed_pool(args.seed_pool)
    rows, details = [], []
    for block in blocks:
        for level in levels:
            row, detail = run_case(block, level, args.time_budget, args.node_budget, pool)
            rows.append(row)
            details.append(detail)
            status = ("optimal" if row["proven_optimal"]
                      else "gap {:.2f}%".format(row["gap_percent"]))
            print(f"{block:9s} L={level:2d}  {row['bnb_ms']:12.4f} ms  "
                  f"{status}  "
                  f"({row['expansions']} nodes, {row['solver_seconds']:.1f} s, "
                  f"{row['stop_reason']})")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
