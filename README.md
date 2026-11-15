# Automated Bootstrapping Placement for Encrypted Inference

CS5491 Artificial Intelligence, Project Milestone

Bootstrapping restores ciphertext levels during encrypted inference, but it is expensive. We start with the course's MLP placement and try changes to one edge at a time. Each round keeps the feasible change with the lowest estimated latency. We also reproduce the course baselines for both the MLP and attention blocks at input levels 0 through 14.

## Setup

Run these commands from the project directory. We used Python 3.12.7 and Matplotlib 3.9.2.

```sh
python -m venv .venv
```

Activate the environment: run `./.venv/Scripts/Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux. Then install the dependency:

```sh
python -m pip install -r requirements.txt
```

The benchmark runs on a CPU. It uses the course's `dag_builder.py` for the graphs, baselines and latency model. After cloning the public repository, download `dag_builder.py` from Canvas under Files → Project → Automated Bootstrapping Placement and put it next to `benchmark.py`.

## Reproduce the milestone

```sh
python -m unittest discover -s tests -v
python benchmark.py --output-dir results
```

The search accepts up to five moves per input level. When two outgoing edges share the same inserted operations, the evaluator charges for the common prefix once.

| Output | Contents |
| --- | --- |
| `results/baselines.csv` | 30 rows: both official placements at each input level. |
| `results/mlp_search.csv` | 15 rows: MLP search results by input level. |
| `results/mlp_search_details.json` | MLP placements and accepted moves for every input level. |
| `results/mlp_comparison.png` | Baseline versus search for all MLP input levels. |
| `results/mlp_l0.png`, `results/mlp_l14.png` | Resulting placements at the two representative input levels. |
| `results/summary.json` | Run summary. |

For a shorter run over three representative levels:

```sh
python benchmark.py --levels 0 6 14 --output-dir results-smoke --no-plots
```

## Check the results

| MLP input level | Baseline (ms) | Local search (ms) | Reduction |
| --- | ---: | ---: | ---: |
| 0 | 28,129.69 | 27,107.76 | 3.63% |
| 6 | 28,130.65 | 27,107.76 | 3.64% |
| 14 | 28,131.93 | 22,949.40 | 18.42% |

All 15 MLP cases improved. At level 14, the search could remove the first bootstrap, bringing the count from nine to five. Every run stopped after two or three moves at a local optimum for single-edge changes. The attention baseline was 53,558.17 ms at every level.

The table shows estimated inference latency from the course model. Our benchmark took about six seconds including plots on a Ryzen 7 8845H. The MLP search accounted for about four seconds.

## Rebuild the draft PDF

After running the full benchmark, install the optional PDF dependency and run:

```sh
python -m pip install reportlab==4.4.9
python build_report.py --results results --out CS5491_Project_Milestone.pdf
```

Use `--authors "Name 1, Name 2, and Name 3"` to include the group author line. On Windows the builder uses Times New Roman from the system font directory. On another platform, supply `--fonts` with the directory containing the Times New Roman font files (`times.ttf`, `timesbd.ttf`, `timesi.ttf`, and `timesbi.ttf`).

## Code map

- `placement.py`: Represents and validates edge insertion chains and compiles them into a graph for evaluation.
- `solver.py`: Searches for cost-reducing, feasible changes to the MLP placement.
- `benchmark.py`: Runs the benchmark and writes tables, figures, and placements.
- `build_report.py`: Builds the report PDF from the default benchmark outputs.
- `tests/`: Regression tests for placement handling and search behavior.
