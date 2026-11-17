# Automated Bootstrapping Placement for Encrypted Inference

CS5491 Artificial Intelligence, Group G003 — Xinghang ZENG, Chengyu BIE, and Weijia ZHANG.

In CKKS-style encrypted inference every multiplication consumes a ciphertext level, and an
expensive Bootstrap must refresh the ciphertext once the levels run out. This repository
solves the course's placement task: for the supplied MLP and Attention blocks and every input
level 0–14, insert Bootstrap1, Bootstrap2 and Modswitch operations on the DAG edges so that
every operator has enough levels while the estimated inference latency is as low as possible.

The work has two phases, both fully reproducible here:

- **Milestone** — reproduce the course baselines and improve the official MLP placement with
  a single-edge local search (all 15 input levels, 3.6–18.4% saved).
- **Final** — a branch-and-bound solver with a certified, admissible lower bound. It beats the
  milestone local search at all 15 MLP levels (best 21,911.66 ms at level 14) and improves
  three Attention levels (best 51,642.57 ms at level 6). Whenever a time or node budget stops
  the search, the result carries a certified global lower bound and an honest optimality gap;
  a placement is called *proven optimal* only when the search actually exhausts the space.

All numbers in the final report are produced by the commands below on a laptop CPU
(Ryzen 7 8845H, Python 3.11.3). No LLM and no GPU are involved.

## Setup

Run these commands from the project directory. Python 3.11 or newer (the final experiments
used 3.11.3, the milestone 3.12.7) and Matplotlib 3.9.2:

```sh
python -m venv .venv
```

Activate the environment: run `./.venv/Scripts/Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` on macOS/Linux. Then install the dependency:

```sh
python -m pip install -r requirements.txt
```

Everything runs on a CPU and uses the course's `dag_builder.py` for the graphs, the official
placements and the latency model. That file is course material and is therefore **not**
committed to this repository: after cloning, download `dag_builder.py` from Canvas under
Files → Project → Automated Bootstrapping Placement and put it next to `benchmark.py`. The
test suite passes without it (the branch-and-bound tests score with `tests/fake_dag_builder.py`,
a synthetic double) except `tests/test_milestone.py`, which skips itself.

## Reproduce the final results (report Tables 1–3 and Figure 1)

```sh
# 1. Regression tests: 19 tests, about 2 s.
python -m unittest discover -s tests -v

# 2. Milestone benchmark (seeds and baselines; also writes results/mlp_search.csv):
#    about 6 s including plots.
python benchmark.py --output-dir results

# 3. Branch-and-bound, pass 1 (official + local-search seeds): 30 cases x 60 s,
#    about 30 minutes. Writes results/bnb_pass1.csv and .json.
python run_bnb.py --block both --levels 0-14 --time-budget 60 --node-budget 200000 \
    --output results/bnb_pass1.csv

# 4. Pass 2 (adds every pass-1 placement as a cross-level seed): same budget,
#    about 30 minutes. Writes results/bnb_search.csv and .json — the report's numbers.
python run_bnb.py --block both --levels 0-14 --time-budget 60 --node-budget 200000 \
    --seed-pool results/bnb_pass1.json --output results/bnb_search.csv

# 5. Budget sensitivity (report Table 3): two cases at 10 s and 300 s, about 12 minutes.
#    Each run re-seeds from the pass-2 placements. Writes results/budget_*.csv.
python run_bnb.py --block MLP --levels 6 --time-budget 10 \
    --seed-pool results/bnb_search.json --output results/budget_mlp6_t10.csv
python run_bnb.py --block MLP --levels 6 --time-budget 300 --node-budget 200000 \
    --seed-pool results/bnb_search.json --output results/budget_mlp6_t300.csv
python run_bnb.py --block Attention --levels 9 --time-budget 10 \
    --seed-pool results/bnb_search.json --output results/budget_attn9_t10.csv
python run_bnb.py --block Attention --levels 9 --time-budget 300 --node-budget 200000 \
    --seed-pool results/bnb_search.json --output results/budget_attn9_t300.csv

# 6. Placement diagrams (report uses the overview figure; these are the annotated DAGs):
#    about 5 s. Writes results/figures/*.png.
python make_figures.py --results results --out results/figures \
    --cases MLP:6 MLP:14 Attention:6 Attention:7

# 7. The four-page final report PDF (also renders results/figures/report_overview.png):
#    about 5 s. Requires reportlab (below).
python build_final_report.py --out CS5491_Project_Final.pdf
```

Install the optional PDF dependency first if needed:

```sh
python -m pip install reportlab==4.4.9
```

Expected outputs of steps 3–5 (estimated inference latency, course model; solver time is
reported separately in the CSVs):

| Case | Official (ms) | Branch-and-bound (ms) | B1/B2 | Certified LB (ms) | Gap |
| --- | ---: | ---: | :---: | ---: | ---: |
| MLP L=0 | 28,129.69 | 26,070.02 | 4/4 | 15,818.66 | 39.3% |
| MLP L=6 | 28,130.65 | 26,070.98 | 4/4 | 10,646.14 | 59.2% |
| MLP L=14 | 28,131.93 | 21,911.66 | 0/4 | 10,646.14 | 51.4% |
| Attention L=6 | 53,558.17 | 51,642.57 | 4/12 | 20,357.33 | 60.6% |
| Attention L=9 | 53,558.17 | 53,558.17 (tie) | 9/8 | 20,357.33 | 62.0% |

The complete 30-row table is `results/bnb_search.csv`; the full tables, findings, and the
budget-sensitivity analysis are in the final report. Runs are deterministic except for
wall-clock cutoffs, which can shift a row by a fraction of a millisecond across machines; the
tie-or-improve relationship against the baselines is stable.

## Reproduce the milestone

```sh
python -m unittest discover -s tests -v
python benchmark.py --output-dir results
```

The search accepts up to five moves per input level. When two outgoing edges share the same
inserted operations, the evaluator charges for the common prefix once.

| Output | Contents |
| --- | --- |
| `results/baselines.csv` | 30 rows: both official placements at each input level. |
| `results/mlp_search.csv` | 15 rows: MLP local-search results by input level. |
| `results/mlp_search_details.json` | MLP placements and accepted moves for every input level. |
| `results/mlp_comparison.png` | Baseline versus search for all MLP input levels. |
| `results/mlp_l0.png`, `results/mlp_l14.png` | Resulting placements at the two representative input levels. |
| `results/summary.json` | Run summary. |

For a shorter run over three representative levels:

```sh
python benchmark.py --levels 0 6 14 --output-dir results-smoke --no-plots
```

Milestone check numbers: local search beats the official placement at all 15 MLP levels
(3.63–3.64% for levels 0–13, 18.42% at level 14 where the first bootstrap can be removed);
the Attention baseline is 53,558.17 ms at every level.

## Rebuild the report PDFs

```sh
python build_report.py --results results --out CS5491_Project_Milestone.pdf       # milestone, 2 pages
python build_final_report.py --out CS5491_Project_Final.pdf                       # final, 4 pages + refs
```

Use `--authors "Name 1, Name 2, and Name 3"` on `build_report.py` to include the group author
line (the final builder defaults to the group's names). On Windows both builders use Times New
Roman from the system font directory; on another platform, supply `--fonts` with the directory
containing the Times New Roman files (`times.ttf`, `timesbd.ttf`, `timesi.ttf`, `timesbi.ttf`).
The final builder reads its table values from `results/` (`bnb_search.csv`, `mlp_search.csv`,
`budget_*.csv`), so the PDF always matches the artifacts.

## Code map

- `placement.py`: Represents and validates edge insertion chains and compiles them into a
  graph for evaluation. `candidate_chains` and `decision_edges` define the search domain and
  branching order shared by every solver.
- `solver.py`: Milestone local search — cost-reducing, feasible single-edge changes to the MLP
  placement.
- `branch_and_bound.py`: Exact branch-and-bound search over edge chains. The pruning lower
  bound is admissible (interval propagation + cheapest achievable operator costs + shared
  bootstrap accounting + forced-future-bootstrap cover); an optional node/time budget stops
  the search, which then reports the certified global lower bound and the optimality gap.
- `exhaustive.py`: Enumerates every feasible completion of a partial state on small graphs;
  the oracle used to verify the branch-and-bound solver and its bound.
- `benchmark.py`: Milestone benchmark — tables, figures, and placements.
- `run_bnb.py`: Branch-and-bound runs on both blocks under published time and node budgets.
  Seeds: official placement, milestone local search (MLP), and any earlier run's placements
  via `--seed-pool` (cross-level multi-start). Writes one CSV row per case plus JSON details.
- `make_figures.py`: Renders annotated placement diagrams for selected result rows.
- `build_report.py`: Builds the two-page milestone report PDF from the benchmark outputs.
- `build_final_report.py`: Builds the four-page final report PDF from the branch-and-bound
  outputs, including the overview figure.
- `tests/`: 19 regression tests. `tests/test_bnb.py` cross-checks the solver and its bound
  against exhaustive enumeration on small graphs (runs without the course file, scored by
  `tests/fake_dag_builder.py`); `tests/test_real_bound.py` repeats the leaf-exactness and
  admissibility checks against the real latency curves; `tests/test_milestone.py` checks
  placement semantics against the course scorer and skips itself when `dag_builder.py` is
  missing.
