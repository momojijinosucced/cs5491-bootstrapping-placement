"""Deterministic, baseline-seeded local search for the MLP block."""

from dataclasses import dataclass
from time import perf_counter

import dag_builder as db
from src.placement import EMPTY, Chain, Evaluation, PlacementModel, candidate_chains


Edge = tuple[str, str]
EPSILON = 1e-8  # Ignore changes due solely to the summation order.


@dataclass(frozen=True)
class Move:
    """One accepted replacement of the insertion chain on an original edge."""

    edge: Edge
    before: Chain
    after: Chain
    latency_ms: float


@dataclass(frozen=True)
class SearchResult:
    """Best feasible placement found and the measured search effort."""

    placement: dict[Edge, Chain]
    score: Evaluation
    initial_latency_ms: float
    evaluated: int
    feasible_candidates: int
    steps: int
    moves: tuple[Move, ...]
    solver_seconds: float
    stop_reason: str


def solve_mlp(level: int, max_steps: int = 5) -> SearchResult:
    """Keep the best improving legal single-edge move per sweep, up to a cap."""
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")

    start = perf_counter()
    model = PlacementModel(*db.get_mlp_empty())
    placement = model.baseline_placement(*db.get_mlp_baseline_inserted(), level)
    score = model.evaluate(placement, level)
    if score is None:
        raise RuntimeError(f"Provided baseline is not feasible at level {level}")
    initial_latency = score.latency
    moves: list[Move] = []
    evaluated = feasible = 0
    stop = "step_cap"
    neighborhoods = [((u, v), candidate_chains(ct, model.l_max)) for u, v, ct in model.edges]

    for _ in range(max_steps):
        best_placement, best_score, best_move = placement, score, None
        for edge, options in neighborhoods:
            before = placement.get(edge, EMPTY)
            for after in options:
                if after == before:
                    continue
                candidate = placement.copy()
                if after == EMPTY:
                    candidate.pop(edge, None)
                else:
                    candidate[edge] = after
                evaluated += 1
                candidate_score = model.evaluate(candidate, level)
                if candidate_score is None:
                    continue
                feasible += 1
                if candidate_score.latency < best_score.latency - EPSILON:
                    best_placement, best_score = candidate, candidate_score
                    best_move = Move(edge, before, after, best_score.latency)
        if best_move is None:
            stop = "no_improving_single_edge_move"
            break
        moves.append(best_move)
        placement, score = best_placement, best_score

    return SearchResult(placement, score, initial_latency, evaluated, feasible,
                        len(moves), tuple(moves), perf_counter() - start, stop)
