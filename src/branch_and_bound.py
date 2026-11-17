"""Branch-and-bound placement search with a certified global lower bound.

The search branches on the insertion chain of one original edge at a time in
the order of ``placement.decision_edges``. A partial state is pruned when its
admissible lower bound cannot beat the incumbent, which is seeded from a
feasible placement (for example the milestone local search). When a node or
time budget stops the search, the result reports a certified global lower
bound: the minimum bound over the unexplored frontier, so the optimality gap
is honest even without a complete search.
"""

from dataclasses import dataclass
from time import perf_counter

from src.placement import (
    EMPTY,
    _MIN_LEVEL,
    Chain,
    Evaluation,
    Placement,
    PlacementModel,
    candidate_chains,
    decision_edges,
)

EPSILON = 1e-8  # Ignore changes due solely to the summation order.


@dataclass(frozen=True)
class BnBResult:
    """Best feasible placement found, the certificate, and the search effort."""

    placement: dict[tuple[str, str], Chain] | None
    score: Evaluation | None
    proven_optimal: bool
    lower_bound: float | None
    gap_percent: float | None
    expansions: int
    leaves: int
    solver_seconds: float
    stop_reason: str


def ordered_chains(model: PlacementModel, edge: tuple[str, str]) -> tuple[Chain, ...]:
    """Order an edge's chains by estimated insertion cost, cheapest first.

    Empty chains come before free switches (higher targets first, since they
    constrain later nodes least), then Bootstrap2 and Bootstrap1 variants.
    """
    count = model.edge_counts[edge]

    def key(chain: Chain) -> tuple[float, bool, int]:
        """Sort by estimated insertion cost, then free switches, high targets first."""
        if chain.boot == 1:
            estimate = count * model.latency["Bootstrap1"](0)
        elif chain.boot == 2:
            estimate = (count // 2) * model.latency["Bootstrap2"](0)
        else:
            estimate = 0.0
        has_target = chain.target is not None
        return (estimate, has_target, -(chain.target if has_target else 0))

    return tuple(sorted(candidate_chains(count, model.l_max), key=key))


def _post_interval(model, u, chain, lo_in, hi_in):
    """Level interval arriving at the child of edge ``(u, v)``, or None if impossible."""
    cost_u = model.node_by_name[u][2]
    l_max = model.l_max
    if chain is None:  # Undecided edge: any legal level is still possible.
        return (0, l_max)
    boot, target = chain.boot, chain.target
    level = l_max - (boot - 1) if boot else None
    if target is not None:
        if level is not None:
            return (target, target) if 0 <= target < level else None
        # A bare Modswitch must strictly lower the source's output level.
        lo_out, hi_out = lo_in[u] - cost_u, hi_in[u] - cost_u
        return None if target >= hi_out else (target, target)
    if level is not None:
        return (level, level)
    return (lo_in[u] - cost_u, hi_in[u] - cost_u)


def lower_bound(
    model: PlacementModel, l_init: int, partial: Placement
) -> float | None:
    """An admissible cost bound for every feasible completion of ``partial``.

    Returns None when the partial state provably admits no feasible
    completion. The bound prices every original node at the cheapest level
    its input can still reach, charges each forced shared bootstrap once, and
    charges the cheapest refresh that undecided edges must still insert when
    no refresh-free chain can deliver what their child needs.  It never
    exceeds the cost of any completion.
    """
    for edge, chain in partial.items():
        if edge not in model.edge_counts:
            raise KeyError(f"Unknown placement edge {edge!r}")
        count = model.edge_counts[edge]
        if type(chain.boot) is not int or not 0 <= chain.boot <= 2:
            return None
        if chain.boot == 2 and count not in (8, 16):
            return None
        if chain.target is not None and (type(chain.target) is not int
                                         or not 0 <= chain.target <= model.l_max):
            return None
    lo_in: dict[str, int] = {}
    hi_in: dict[str, int] = {}
    for name in model.topological_order:
        _, op, cost, _ = model.node_by_name[name]
        need = max(cost, _MIN_LEVEL.get(op, 0))
        incoming = model.incoming[name]
        if not incoming:
            lo = hi = l_init
        else:
            lows, highs = [], []
            for u, v, _ in incoming:
                interval = _post_interval(model, u, partial.get((u, v), None), lo_in, hi_in)
                if interval is None:
                    return None
                lows.append(interval[0])
                highs.append(interval[1])
            lo = max(need, min(lows))
            hi = min(highs)
        if lo > hi:
            return None
        lo_in[name], hi_in[name] = lo, hi

    total = 0.0
    for name in model.topological_order:
        curve = model.latency[model.node_by_name[name][1]]
        total += min(curve(level) for level in range(lo_in[name], hi_in[name] + 1))

    forced: dict[tuple[str, int, int], float] = {}
    for (u, v), chain in partial.items():
        if chain.boot:
            count = model.edge_counts[u, v]
            key = (u, count, chain.boot)
            if key not in forced:
                forced[key] = _bootstrap_cost(model, u, count, chain.boot, lo_in, hi_in)
    total += sum(forced.values())

    # Bootstrap cost that undecided edges still have to pay.  A node needs at
    # least its own ``need`` on every incoming edge (its input is the minimum
    # of them), so an undecided edge whose source cannot deliver that much
    # without a refresh is forced to take one.  One shared node per source and
    # ciphertext count can cover every forced edge of the group, so only the
    # cheapest cover is charged -- anything more would not be admissible.
    needs = {name: max(model.node_by_name[name][2], _MIN_LEVEL.get(model.node_by_name[name][1], 0))
             for name in model.topological_order}
    groups: dict[tuple[str, int], dict] = {}
    for v in model.topological_order:
        for u, _, count in model.incoming[v]:
            if (u, v) in partial:
                continue  # Decided; charged above if it boots.
            if hi_in[u] - model.node_by_name[u][2] >= needs[v]:
                continue  # A refresh-free chain can still deliver the need.
            group = groups.setdefault((u, count), {"needs": set(), "boots": set()})
            group["needs"].add(needs[v])
    for (u, count), group in groups.items():
        for (x, y), chain in partial.items():
            if x == u and chain.boot and model.edge_counts[x, y] == count:
                group["boots"].add(chain.boot)
        options = []
        for choose in (frozenset({1}), frozenset({2}), frozenset({1, 2})):
            used = group["boots"] | set(choose)
            if all(any(model.l_max - (boot - 1) >= need for boot in used)
                   for need in group["needs"]):
                options.append(sum(
                    _bootstrap_cost(model, u, count, boot, lo_in, hi_in)
                    for boot in choose if boot not in group["boots"]
                ))
        if options:
            total += min(options)
    return total


def _bootstrap_cost(model, u, count, boot, lo_in, hi_in):
    """The cheapest a refresh on ``(u, *)`` with ``count`` ciphertexts can cost."""
    curve = model.latency["Bootstrap1" if boot == 1 else "Bootstrap2"]
    cost_u = model.node_by_name[u][2]
    unit = min(curve(level) for level in range(lo_in[u] - cost_u, hi_in[u] - cost_u + 1))
    return unit * (count if boot == 1 else count // 2)


def solve_bnb(
    model: PlacementModel,
    l_init: int,
    *,
    seed_placement: Placement | None = None,
    node_budget: int | None = None,
    time_budget: float | None = None,
) -> BnBResult:
    """Search for a minimum-latency placement, or the best one within budget.

    ``node_budget`` caps the number of expanded search nodes and
    ``time_budget`` caps the wall-clock seconds; whichever runs out first
    stops the search. The result then reports ``lower_bound`` as the certified
    global bound and ``gap_percent`` as ``100 * (best - bound) / best``.
    """
    if node_budget is not None and (type(node_budget) is not int or node_budget < 0):
        raise ValueError("node_budget must be a nonnegative integer")
    if time_budget is not None and time_budget < 0:
        raise ValueError("time_budget must be nonnegative")

    started = perf_counter()
    edges = decision_edges(model)
    options = [ordered_chains(model, edge) for edge in edges]

    best_placement: dict | None = None
    best_score: Evaluation | None = None
    if seed_placement is not None:
        seed_score = model.evaluate(dict(seed_placement), l_init)
        if seed_score is None:
            raise ValueError("The seed placement is not feasible at this input level")
        best_placement = {e: c for e, c in seed_placement.items() if c != EMPTY}
        best_score = seed_score

    root_bound = lower_bound(model, l_init, {})
    if root_bound is None:
        raise RuntimeError("No feasible placement exists at this input level")

    stack: list[tuple[dict, int, float]] = [({}, 0, root_bound)]
    expansions = leaves = 0
    stop_reason = "search_exhausted"

    while stack:
        if node_budget is not None and expansions >= node_budget:
            stop_reason = "node_budget"
            break
        if time_budget is not None and perf_counter() - started >= time_budget:
            stop_reason = "time_budget"
            break
        partial, depth, bound = stack.pop()
        if best_score is not None and bound >= best_score.latency - EPSILON:
            continue
        if depth == len(edges):
            expansions += 1
            leaves += 1
            placement = {e: c for e, c in partial.items() if c != EMPTY}
            score = model.evaluate(placement, l_init)
            if score is not None and (best_score is None or
                                      score.latency < best_score.latency - EPSILON):
                best_placement, best_score = placement, score
            continue
        expansions += 1
        edge = edges[depth]
        children = []
        for chain in options[depth]:
            child = dict(partial)
            child[edge] = chain
            child_bound = lower_bound(model, l_init, child)
            if child_bound is None:
                continue
            if best_score is not None and child_bound >= best_score.latency - EPSILON:
                continue
            children.append((child, depth + 1, child_bound))
        stack.extend(reversed(children))  # Pop the cheapest chain first.

    open_bound = min((bound for _, _, bound in stack), default=float("inf"))
    if best_score is not None and open_bound >= best_score.latency - EPSILON:
        proven_optimal, stop_reason = True, "optimal"
        certificate = best_score.latency
    elif stop_reason == "search_exhausted":
        if best_score is None:
            proven_optimal, certificate = True, None  # Proven infeasible.
            stop_reason = "no_feasible_placement"
        else:
            proven_optimal, certificate = True, best_score.latency
    else:
        proven_optimal = False
        certificate = min(best_score.latency, open_bound) if best_score is not None else open_bound
        if certificate == float("inf"):
            certificate = None

    gap = None
    if best_score is not None and certificate is not None:
        gap = 100.0 * (best_score.latency - certificate) / best_score.latency
    return BnBResult(
        best_placement, best_score, proven_optimal, certificate, gap,
        expansions, leaves, perf_counter() - started, stop_reason,
    )
