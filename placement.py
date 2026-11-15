"""Placement legality, shared-prefix insertion, and reference scoring."""

from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import isclose

import dag_builder


Edge = tuple[str, str]
TensorEdge = tuple[str, str, int]
Node = tuple[str, str, int, str]
LatencyTable = Mapping[str, Callable[[int], float]]

_BOOT = {1: "Bootstrap1", 2: "Bootstrap2"}
_BOOT_NUMBER = {op: number for number, op in _BOOT.items()}
_INSERTED = {*_BOOT_NUMBER, "Modswitch"}
_MIN_LEVEL = {
    "PCMM1": 2, "PCMM2": 2, "PCMM3": 2, "PCMM4": 2,
    "CCMM1": 4, "CCMM2": 2,
    "PolyEval1": 5, "PolyEval2": 5, "PolyEval3": 8, "PolyEval4": 8,
    "Mul": 1,
}


class InvalidPlacement(ValueError):
    """A chain, graph, or induced level violates the placement rules."""


@dataclass(frozen=True)
class Chain:
    """A bootstrap (0 = none), optionally followed by a switch to `target`."""

    boot: int = 0
    target: int | None = None


EMPTY = Chain()
Placement = Mapping[Edge, Chain]


@dataclass(frozen=True)
class CompiledPlacement:
    """The concrete graph and the post-chain levels on its original edges."""

    nodes: tuple[Node, ...]
    edges: tuple[TensorEdge, ...]
    levels: dict[Edge, int]


@dataclass(frozen=True)
class Evaluation(CompiledPlacement):
    """Latency and statistics from the supplied scorer."""

    latency: float
    n_b1: int
    n_b2: int
    depth: int
    budget: int
    utilization: float
    l_in: dict[str, int]
    l_out: dict[str, int]
    edge_lvl: dict[Edge, int]


class PlacementModel:
    """An unmodified block DAG and its legal insertion and scoring rules."""

    def __init__(
        self,
        nodes: Iterable[Node],
        edges: Iterable[TensorEdge],
        l_max: int = 14,
        *,
        latency: LatencyTable | None = None,
    ) -> None:
        if type(l_max) is not int or not 1 <= l_max <= 14:
            raise InvalidPlacement("l_max must be an integer in [1, 14]")
        self.l_max = l_max
        self.latency = dict(dag_builder.LATENCY if latency is None else latency)
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)
        self.node_by_name = {node[0]: node for node in self.nodes}
        if not self.nodes or len(self.node_by_name) != len(self.nodes):
            raise InvalidPlacement("The empty DAG needs nonempty, distinct node names")
        for name, op, cost, _ in self.nodes:
            if not isinstance(name, str) or not name:
                raise InvalidPlacement("Node names must be nonempty strings")
            if op in _INSERTED or op not in self.latency:
                raise InvalidPlacement(f"Unsupported original operator {op!r} at {name!r}")
            if type(cost) is not int or cost < 0:
                raise InvalidPlacement(f"Invalid level cost at {name!r}: {cost!r}")
        if not _INSERTED.issubset(self.latency):
            raise InvalidPlacement("Missing latency for one or more insertion operators")

        self.edge_counts: dict[Edge, int] = {}
        incoming: dict[str, list[TensorEdge]] = {name: [] for name in self.node_by_name}
        outgoing: dict[str, list[TensorEdge]] = {name: [] for name in self.node_by_name}
        for u, v, count in self.edges:
            if u not in outgoing or v not in incoming or (u, v) in self.edge_counts:
                raise InvalidPlacement(f"Unknown endpoint or duplicate edge {(u, v)!r}")
            if type(count) is not int or count < 1:
                raise InvalidPlacement(f"Invalid ciphertext count on {(u, v)!r}: {count!r}")
            self.edge_counts[u, v] = count
            incoming[v].append((u, v, count))
            outgoing[u].append((u, v, count))
        self.incoming = {name: tuple(edges) for name, edges in incoming.items()}
        self.outgoing = {name: tuple(edges) for name, edges in outgoing.items()}

        degree = {name: len(edges) for name, edges in incoming.items()}
        ready = deque(name for name, count in degree.items() if count == 0)
        ordered = []
        while ready:
            name = ready.popleft()
            ordered.append(name)
            for _, child, _ in outgoing[name]:
                degree[child] -= 1
                if degree[child] == 0:
                    ready.append(child)
        if len(ordered) != len(self.nodes):
            raise InvalidPlacement("The original graph contains a cycle")
        self.topological_order = tuple(ordered)

    def _check_l_init(self, l_init: int) -> None:
        if type(l_init) is not int or not 0 <= l_init <= self.l_max:
            raise InvalidPlacement(f"l_init must be an integer in [0, {self.l_max}]")

    def compile(self, placement: Placement, l_init: int) -> CompiledPlacement:
        """Propagate levels and materialize each distinct same-source prefix once."""
        self._check_l_init(l_init)
        for edge in placement:
            if edge not in self.edge_counts:
                raise InvalidPlacement(f"Unknown placement edge {edge!r}")

        nodes = list(self.nodes)
        edges: list[TensorEdge] = []
        levels: dict[Edge, int] = {}
        used_names = set(self.node_by_name)

        def insert(parent: str, count: int, op: str, cost: int, label: str) -> str:
            index = len(nodes) - len(self.nodes) + 1
            name = f"__placement_{index}"
            while name in used_names:
                index += 1
                name = f"__placement_{index}"
            used_names.add(name)
            nodes.append((name, op, cost, label))
            edges.append((parent, name, count))
            return name

        for u in self.topological_order:
            _, op, cost, _ = self.node_by_name[u]
            lin = min((levels[a, b] for a, b, _ in self.incoming[u]), default=l_init)
            if not max(cost, _MIN_LEVEL.get(op, 0)) <= lin <= self.l_max:
                raise InvalidPlacement(f"{u} ({op}) cannot run at input level {lin}")
            lout = lin - cost
            boot_nodes: dict[tuple[int, int], str] = {}
            switch_nodes: dict[tuple[int, int, int], str] = {}
            for _, v, count in self.outgoing[u]:
                edge = u, v
                chain = placement.get(edge, EMPTY)
                if not isinstance(chain, Chain) or type(chain.boot) is not int or not 0 <= chain.boot <= 2:
                    raise InvalidPlacement(f"Invalid chain on {edge!r}: {chain!r}")
                boot, target = chain.boot, chain.target
                if boot == 2 and count not in (8, 16):
                    raise InvalidPlacement(f"Bootstrap2 needs 8 or 16 ciphertexts on {edge!r}")
                last, level = u, lout
                if boot:
                    level = self.l_max - (boot - 1)
                    boot_key = count, boot
                    if boot_key not in boot_nodes:
                        boot_nodes[boot_key] = insert(u, count, _BOOT[boot], 0, f"B{boot}")
                    last = boot_nodes[boot_key]
                if target is not None:
                    if type(target) is not int or not 0 <= target < level:
                        raise InvalidPlacement(f"Modswitch on {edge!r} must lower {level}: {target!r}")
                    switch_key = count, boot, target
                    if switch_key not in switch_nodes:
                        switch_nodes[switch_key] = insert(
                            last, count, "Modswitch", level - target, f"Switch to {target}"
                        )
                    last, level = switch_nodes[switch_key], target
                levels[edge] = level
                edges.append((last, v, count))
        return CompiledPlacement(tuple(nodes), tuple(edges), levels)

    def evaluate(self, placement: Placement, l_init: int) -> Evaluation | None:
        """Score a feasible placement with the course code, or return None."""
        try:
            compiled = self.compile(placement, l_init)
        except InvalidPlacement:
            return None
        nodes, edges = compiled.nodes, compiled.edges
        scored = dag_builder.evaluate_dag(nodes, edges, l_init, self.l_max, self.latency)
        if scored is None:
            raise RuntimeError("The legality pass and reference scorer disagree")
        depth = dag_builder.depth_of(nodes, edges)
        n_b1, n_b2, budget = dag_builder.budget_of(nodes, edges, l_init, self.l_max)
        return Evaluation(
            nodes, edges, compiled.levels, scored["latency"], n_b1, n_b2, depth, budget,
            depth / budget if budget else 0.0, scored["l_in"], scored["l_out"], scored["edge_lvl"],
        )

    def baseline_placement(
        self,
        inserted_nodes: Iterable[Node],
        inserted_edges: Iterable[TensorEdge],
        l_init: int,
    ) -> dict[Edge, Chain]:
        """Extract absolute-level chains from a supplied concrete placement."""
        self._check_l_init(l_init)
        inserted_nodes, inserted_edges = tuple(inserted_nodes), tuple(inserted_edges)
        by_name = {node[0]: node for node in inserted_nodes}
        if len(by_name) != len(inserted_nodes):
            raise InvalidPlacement("The inserted DAG contains duplicate names")
        for name, original in self.node_by_name.items():
            if name not in by_name or by_name[name][1:3] != original[1:3]:
                raise InvalidPlacement(f"The inserted DAG changes or omits original node {name!r}")
        insertions = by_name.keys() - self.node_by_name.keys()
        for name in insertions:
            _, op, cost, _ = by_name[name]
            if op not in _INSERTED or type(cost) is not int or (cost <= 0 if op == 'Modswitch' else cost != 0):
                raise InvalidPlacement(f"Invalid inserted operator at {name!r}")
        incoming: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for u, v, count in inserted_edges:
            if u not in by_name or v not in by_name:
                raise InvalidPlacement(f"Unknown endpoint in inserted edge {(u, v)!r}")
            incoming[v].append((u, count))
        if any(len(incoming[name]) != 1 for name in insertions):
            raise InvalidPlacement("Inserted operators must have one parent")

        paths: dict[Edge, tuple[str, ...]] = {}
        used_insertions = set()
        for v in self.node_by_name:
            for parent, count in incoming[v]:
                u = parent
                chain = []
                while u in insertions:
                    if u in chain:
                        raise InvalidPlacement("Cycle in an insertion chain")
                    chain.append(u)
                    u, upstream_count = incoming[u][0]
                    if upstream_count != count:
                        raise InvalidPlacement("Insertion changes the ciphertext count")
                edge = u, v
                if edge in paths or self.edge_counts.get(edge) != count:
                    raise InvalidPlacement(f"Inserted DAG changes or duplicates edge {edge!r}")
                paths[edge] = tuple(reversed(chain))
                used_insertions.update(chain)
        if paths.keys() != self.edge_counts.keys() or used_insertions != insertions:
            raise InvalidPlacement("The inserted DAG omits original edges or has unused insertions")

        scored = dag_builder.evaluate_dag(
            inserted_nodes, inserted_edges, l_init, self.l_max, self.latency
        )
        if scored is None:
            raise InvalidPlacement("The inserted DAG is infeasible at this input level")
        placement = {}
        for edge, path in paths.items():
            boot, target = 0, None
            for name in path:
                op = by_name[name][1]
                if op == "Modswitch" and target is None:
                    target = scored["l_out"][name]
                elif op in _BOOT_NUMBER and not boot and target is None:
                    boot = _BOOT_NUMBER[op]
                else:
                    raise InvalidPlacement(f"Unsupported placement chain on {edge!r}")
            if boot or target is not None:
                placement[edge] = Chain(boot, target)

        normalized = self.evaluate(placement, l_init)
        if normalized is None or not isclose(normalized.latency, scored["latency"], rel_tol=1e-12):
            raise InvalidPlacement("The extracted placement does not reproduce the inserted DAG score")
        return placement
