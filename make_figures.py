"""Render placement figures for chosen BnB result rows.

Reads the ``run_bnb.py`` outputs (``bnb_search.csv`` + ``bnb_search.json``),
re-evaluates each saved placement with the course scorer, and draws the DAG
with levels on the edges via ``dag_builder.visualize_empty_dag``.

Usage:
    python make_figures.py --results results --out figures \
        --cases MLP:0 MLP:14 Attention:0 Attention:14
"""

import argparse
import csv
import json
from pathlib import Path

import dag_builder as course
from placement import Chain, PlacementModel

BLOCKS = {
    "MLP": course.get_mlp_empty,
    "Attention": course.get_attn_empty,
}


def main():
    """Draw annotated placement diagrams for the requested result rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results",
                        help="directory holding bnb_search.csv and bnb_search.json")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--cases", nargs="+", default=["MLP:0", "MLP:14"],
                        help="BLOCK:LEVEL entries to draw")
    args = parser.parse_args()

    results = Path(args.results)
    rows = {(r["block"], int(r["l_init"])): r
            for r in csv.DictReader((results / "bnb_search.csv").open(encoding="utf-8"))}
    details = {(d["block"], d["l_init"]): d
               for d in json.loads((results / "bnb_search.json").read_text(encoding="utf-8"))}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        block, level = case.split(":")
        key = (block, int(level))
        row, detail = rows[key], details[key]
        model = PlacementModel(*BLOCKS[block]())
        placement = {
            tuple(item["edge"]): Chain(boot=item["boot"], target=item["target"])
            for item in detail["placement"]
        }
        score = model.evaluate(placement, int(level))
        assert score is not None, f"saved placement is no longer feasible: {case}"
        filename = out / f"bnb_{block.lower()}_l{level}.png"
        course.visualize_empty_dag(
            score.nodes, score.edges,
            title=(f"{block} placement at L_init={level}: estimated "
                   f"{score.latency:,.2f} ms, B1={score.n_b1}, B2={score.n_b2}, "
                   f"depth={score.depth}, budget={score.budget}, "
                   f"utilisation={score.utilization:.3f}"),
            filename=str(filename), x_spacing=9.0, y_spacing=2.4,
            levels=score.edge_lvl,
        )
        print(f"{case}: {filename}  ({row['bnb_ms']} ms)")


if __name__ == "__main__":
    main()
