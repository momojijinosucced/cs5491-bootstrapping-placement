"""Exhaustive enumeration of edge chains for small graphs.

This is the validation oracle requested by the TAs: for small instances it
enumerates every legal completion of a partial state, so the branch-and-bound
solver and its pruning lower bound can be checked against ground truth. Keep
instances tiny; the combination count grows quickly with the edge count.
"""

from collections.abc import Iterator, Mapping
from itertools import product

from placement import Chain, Evaluation, Placement, PlacementModel, candidate_chains, decision_edges


class TooManyCombinations(ValueError):
    """The requested enumeration exceeds the accepted combination count."""


def combinations_of(model: PlacementModel) -> int:
    """Count the complete chain assignments without materializing them."""
    total = 1
    for u, v in model.edge_counts:
        total *= len(candidate_chains(model.edge_counts[u, v], model.l_max))
    return total


def completions(
    model: PlacementModel,
    l_init: int,
    partial: Mapping[tuple[str, str], Chain] | None = None,
    *,
    max_combinations: int = 500_000,
) -> Iterator[tuple[dict[tuple[str, str], Chain], Evaluation]]:
    """Yield ``(placement, evaluation)`` for every feasible completion of ``partial``.

    ``partial`` fixes chains on some edges (``Chain()`` means a decided empty
    chain); undecided edges range over ``candidate_chains``. Infeasible
    completions are skipped. Raises ``TooManyCombinations`` when the remaining
    product exceeds ``max_combinations``.
    """
    partial = dict(partial or {})
    for edge in partial:
        if edge not in model.edge_counts:
            raise KeyError(f"Unknown placement edge {edge!r}")
    remaining = [edge for edge in decision_edges(model) if edge not in partial]
    product_size = 1
    for edge in remaining:
        product_size *= len(candidate_chains(model.edge_counts[edge], model.l_max))
    if product_size > max_combinations:
        raise TooManyCombinations(f"{product_size} combinations exceed the cap")

    fixed = {edge: chain for edge, chain in partial.items() if chain != Chain()}
    for chosen in product(*(candidate_chains(model.edge_counts[e], model.l_max) for e in remaining)):
        placement = dict(fixed)
        placement.update((e, c) for e, c in zip(remaining, chosen) if c != Chain())
        score = model.evaluate(placement, l_init)
        if score is not None:
            yield placement, score


def best_completion(
    model: PlacementModel,
    l_init: int,
    partial: Mapping[tuple[str, str], Chain] | None = None,
    *,
    max_combinations: int = 500_000,
) -> tuple[dict[tuple[str, str], Chain], Evaluation] | None:
    """Return the cheapest feasible completion of ``partial``, or None."""
    best = None
    for placement, score in completions(model, l_init, partial, max_combinations=max_combinations):
        if best is None or score.latency < best[1].latency:
            best = (placement, score)
    return best
