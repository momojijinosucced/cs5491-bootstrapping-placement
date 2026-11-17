"""A synthetic stand-in for the course's dag_builder.py, for logic tests only.

The course file is not committed (it comes from Canvas), so the search and
bound logic is verified against this double, which implements the same model
semantics documented in the milestone records:

- source nodes start at ``l_init``; a node with several parents receives the
  minimum incoming level; an operator needs ``max(cost, min_level)`` levels;
- estimated latency sums per-node costs: Bootstrap1 charges per ciphertext,
  Bootstrap2 per pair, Modswitch is free, original operators charge
  ``LATENCY[op](input level)``;
- ``depth_of`` is the longest path of original level costs (insertions add 0);
- ``budget_of`` returns per-ciphertext B1, per-pair B2, and
  ``l_init + l_max * n_b1 + (l_max - 1) * n_b2``.

Operator cost curves below are synthetic; results obtained with this double
prove search and bound properties, not course latencies. Re-run the tests with
the real ``dag_builder.py`` present before reporting numbers.
"""

from collections import deque

# Mirrors the placement rules in placement.py (copied so the double stands alone).
MIN_LEVEL = {
    "PCMM1": 2, "PCMM2": 2, "PCMM3": 2, "PCMM4": 2,
    "CCMM1": 4, "CCMM2": 2,
    "PolyEval1": 5, "PolyEval2": 5, "PolyEval3": 8, "PolyEval4": 8,
    "Mul": 1,
}
INSERTED = ("Bootstrap1", "Bootstrap2", "Modswitch")

LATENCY = {
    # Insertion operators: unit costs match the course model constants.
    "Bootstrap1": lambda level: 1040.15,
    "Bootstrap2": lambda level: 1293.13,
    "Modswitch": lambda level: 0.0,
    # Original operators: synthetic nondecreasing curves over the input level.
    "Identity": lambda level: 0.0,
    "Mul": lambda level: 100.0 + 5.0 * level,
    "PCMM1": lambda level: 200.0 + 10.0 * level,
    "PCMM2": lambda level: 210.0 + 10.0 * level,
    "PCMM3": lambda level: 220.0 + 10.0 * level,
    "PCMM4": lambda level: 230.0 + 10.0 * level,
    "CCMM1": lambda level: 300.0 + 15.0 * level,
    "CCMM2": lambda level: 310.0 + 15.0 * level,
    "PolyEval1": lambda level: 400.0 + 25.0 * level,
    "PolyEval2": lambda level: 420.0 + 25.0 * level,
    "PolyEval3": lambda level: 500.0 + 40.0 * level,
    "PolyEval4": lambda level: 520.0 + 40.0 * level,
}


def _topological_order(nodes, edges):
    """Return node names in topological order, rejecting cyclic graphs."""
    by_name = {node[0]: node for node in nodes}
    outgoing = {name: [] for name in by_name}
    degree = {name: 0 for name in by_name}
    for u, v, _ in edges:
        if u not in by_name or v not in by_name:
            raise ValueError(f"Unknown endpoint in edge {(u, v)!r}")
        outgoing[u].append(v)
        degree[v] += 1
    ready = deque(name for name, count in degree.items() if count == 0)
    order = []
    while ready:
        name = ready.popleft()
        order.append(name)
        for child in outgoing[name]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    if len(order) != len(nodes):
        raise ValueError("The graph contains a cycle")
    return order, by_name


def _incoming(edges):
    """Map each child to its (parent, ciphertext count) list."""
    parents = {}
    for u, v, count in edges:
        parents.setdefault(v, []).append((u, count))
    return parents


def evaluate_dag(nodes, edges, l_init, l_max, latency=None):
    """Propagate levels and score the graph, or return None if infeasible."""
    table = dict(LATENCY if latency is None else latency)
    order, by_name = _topological_order(nodes, edges)
    parents = _incoming(edges)
    l_in, l_out, edge_lvl = {}, {}, {}
    for name in order:
        _, op, cost, _ = by_name[name]
        if parents.get(name):
            lin = min(l_out[u] for u, _ in parents[name])
        else:
            lin = l_init
        if not max(cost, MIN_LEVEL.get(op, 0)) <= lin <= l_max:
            return None
        l_in[name] = lin
        if op == "Bootstrap1":
            l_out[name] = l_max  # Absolute restoration, like the course scorer.
        elif op == "Bootstrap2":
            l_out[name] = l_max - 1
        else:
            l_out[name] = lin - cost
    total = 0.0
    for u, v, count in edges:
        edge_lvl[u, v] = l_out[u]
    for name in order:
        _, op, _, _ = by_name[name]
        if op == "Bootstrap1":
            total += table[op](l_in[name]) * count_of(edges, name)
        elif op == "Bootstrap2":
            total += table[op](l_in[name]) * (count_of(edges, name) // 2)
        elif op == "Modswitch":
            total += 0.0
        else:
            total += table[op](l_in[name])
    return {"latency": total, "l_in": l_in, "l_out": l_out, "edge_lvl": edge_lvl}


def count_of(edges, node):
    """Ciphertext count on the (single) edge entering an inserted node."""
    counts = [count for u, v, count in edges if v == node]
    return counts[0]


def depth_of(nodes, edges):
    """Longest path of original level costs; insertion operators add zero."""
    order, by_name = _topological_order(nodes, edges)
    parents = _incoming(edges)
    depth = {}
    total = 0
    for name in order:
        _, op, cost, _ = by_name[name]
        weight = 0 if op in INSERTED else cost
        depth[name] = weight + max((depth[u] for u, _ in parents.get(name, [])), default=0)
        total = max(total, depth[name])
    return total


def budget_of(nodes, edges, l_init, l_max):
    """Per-ciphertext B1, per-pair B2, and the course budget formula."""
    order, by_name = _topological_order(nodes, edges)
    n_b1 = n_b2 = 0
    for name in order:
        op = by_name[name][1]
        if op == "Bootstrap1":
            n_b1 += count_of(edges, name)
        elif op == "Bootstrap2":
            n_b2 += count_of(edges, name) // 2
    return n_b1, n_b2, l_init + l_max * n_b1 + (l_max - 1) * n_b2
