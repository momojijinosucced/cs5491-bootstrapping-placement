"""Single entry point for every runnable task in this repository.

Usage:
    python main.py test
    python main.py benchmark --output-dir results
    python main.py bnb --block both --levels 0-14 --time-budget 60
    python main.py figures --cases MLP:6 MLP:14 Attention:6 Attention:7
    python main.py report-milestone --out CS5491_Project_Milestone.pdf
    python main.py report-final --out CS5491_Project_Final.pdf

Run commands from the repository root; README.md has the full reproduction
recipe with expected outputs. Every subcommand forwards its arguments to the
matching module in ``src/``; ``test`` runs the regression tests in ``tests/``.
"""

import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "test": "run the 19 regression tests under tests/",
    "benchmark": "milestone benchmark: tables, figures, and placements",
    "bnb": "branch-and-bound runs under published time/node budgets",
    "figures": "annotated placement diagrams for selected result rows",
    "report-milestone": "rebuild the two-page milestone report PDF",
    "report-final": "rebuild the four-page final report PDF",
}

MODULES = {
    "benchmark": "src.benchmark",
    "bnb": "src.run_bnb",
    "figures": "src.make_figures",
    "report-milestone": "src.build_report",
    "report-final": "src.build_final_report",
}


def run_tests():
    """Discover and run every test under tests/; return the process exit code."""
    suite = unittest.TestLoader().discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    """Dispatch the first argument to a command and forward the rest to it."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__.strip())
        print("\ncommands:")
        for name, help_text in COMMANDS.items():
            print(f"  {name:16s} {help_text}")
        return 0
    command, rest = args[0], args[1:]
    if command == "test":
        return run_tests()
    if command not in MODULES:
        print(f"unknown command {command!r}; choose from: {', '.join(COMMANDS)}",
              file=sys.stderr)
        return 2
    sys.argv = [f"main.py {command}"] + rest
    importlib.import_module(MODULES[command]).main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
