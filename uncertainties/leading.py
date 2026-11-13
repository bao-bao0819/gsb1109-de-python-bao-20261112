"""
Robust leading evidence between measurements that share error terms.

A measurement batch is said to *robustly lead* another batch when the
nominal difference between the two remains positive after subtracting
``k`` times the standard deviation of that difference:

    margin(a, b) = (nominal(a) - nominal(b)) - k * std_dev(a - b) > 0

The factor ``k`` is specified by the report.  Because the measurements
may share error terms (common bias variables, calibration offsets, ...),
the standard deviation of the *difference* must be computed on the
difference itself, through the shared linear error model.  Judging the
same pair from the two marginal error bars would ignore the
cancellation of shared terms and can give a different (conservative and
incorrect for this criterion) answer.

The module turns a collection of measurements into queryable evidence:

* :func:`robust_leading_pairs` returns every batch pair that directly
  satisfies the criterion, together with its margin.
* :class:`LeadRegistry` stores the minimum set of direct evidence from
  which every leading relation can be recovered (the transitive
  reduction of the leading relation).  Relations whose direct evidence
  is not stored are answered through the shortest proving chain, while
  their true endpoint margin is always recomputed separately from the
  two endpoint measurements.

Tied batches (zero or non-positive margin) never produce a leading edge.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

from .core import AffineScalarFunc, nominal_value

__all__ = [
    "LeadEvidence",
    "LeadChain",
    "LeadRegistry",
    "lead_margin",
    "pair_uncertainty",
    "robust_leading_pairs",
    "minimal_leading_evidence",
]


def _derivatives(measurement) -> dict:
    """
    Return the expanded ``{Variable: derivative}`` mapping of a
    measurement, without triggering the deprecation warnings attached
    to the public ``derivatives`` property.

    Plain numbers have no error variables and yield an empty mapping.
    """
    if not isinstance(measurement, AffineScalarFunc):
        return {}

    linear_part = measurement._linear_part
    if not linear_part.expanded():
        linear_part.expand()
        # A missing variable then raises a KeyError instead of
        # silently creating a zero entry:
        linear_part.linear_combo.default_factory = None
    return linear_part.linear_combo


def _difference_variance(leader, laggard) -> float:
    """
    Variance of ``leader - laggard`` computed from the full linear
    error model, so that error terms shared by the two measurements
    cancel through the difference of their coefficients.
    """
    derivatives_leader = _derivatives(leader)
    derivatives_laggard = _derivatives(laggard)

    variance = 0.0
    for variable in derivatives_leader.keys() | derivatives_laggard.keys():
        coefficient = derivatives_leader.get(variable, 0.0) - derivatives_laggard.get(
            variable, 0.0
        )
        variance += (coefficient * variable.std_dev) ** 2
    return variance


def pair_uncertainty(leader, laggard) -> float:
    """
    Standard deviation of the difference ``leader - laggard``.

    Error variables appearing in both measurements contribute with the
    difference of their coefficients (shared bias therefore cancels,
    partially or fully); variables appearing in only one measurement
    contribute with their full coefficient.  This is identical to
    ``(leader - laggard).std_dev`` but does not build the intermediate
    affine expression.
    """
    return math.sqrt(_difference_variance(leader, laggard))


def lead_margin(leader, laggard, k: float) -> float:
    """
    Robust leading margin of ``leader`` over ``laggard``.

    The margin is the nominal difference minus ``k`` times the
    standard deviation of that difference, with shared error terms
    fully accounted for.  A strictly positive margin means that
    ``leader`` robustly leads ``laggard``.
    """
    nominal_difference = nominal_value(leader) - nominal_value(laggard)
    return nominal_difference - k * pair_uncertainty(leader, laggard)


@dataclass(frozen=True)
class LeadEvidence:
    """A single batch pair that directly satisfies the lead criterion."""

    leader: str
    laggard: str
    margin: float


@dataclass(frozen=True)
class LeadChain:
    """
    Answer to a leading query.

    ``chain`` gives the shortest sequence of batches that proves the
    relation through stored direct evidence; ``hop_margins`` lists the
    direct margin of every edge along it.  ``endpoint_margin`` is the
    true margin of the two queried endpoints, recomputed separately
    from their shared error model rather than inferred from the hops.
    ``direct`` indicates whether the pair itself satisfies the
    criterion (whether or not its direct evidence is stored).
    """

    leader: str
    laggard: str
    chain: tuple
    hop_margins: tuple
    endpoint_margin: float
    direct: bool

    @property
    def hops(self) -> Sequence[tuple]:
        return tuple(zip(self.chain, self.chain[1:]))


class LeadRegistry:
    """
    Queryable repository of robust leading evidence for a set of
    measurements evaluated with a report-specified factor ``k``.

    Parameters
    ----------
    measurements:
        Mapping from batch name to a measurement with uncertainties.
        Measurements built from the same ``ufloat`` error variables
        share those error terms.  Plain numbers are accepted and
        treated as exact measurements.
    k:
        Non-negative robustness factor specified by the report.
    """

    def __init__(self, measurements: Mapping, k: float):
        if k < 0:
            raise ValueError("k must be non-negative")
        self._measurements = dict(measurements)
        self._k = float(k)

        # Margin of every ordered pair that directly satisfies the
        # criterion.  A positive margin requires a strictly larger
        # nominal value, so this relation is acyclic.
        self._margins: dict[tuple[str, str], float] = {}
        names = list(self._measurements)
        for leader in names:
            for laggard in names:
                if leader == laggard:
                    continue
                margin = lead_margin(
                    self._measurements[leader],
                    self._measurements[laggard],
                    self._k,
                )
                if margin > 0.0:
                    self._margins[(leader, laggard)] = margin

        self._minimal_edges = self._transitive_reduction(frozenset(self._margins))
        self._adjacency = self._build_adjacency(self._minimal_edges)

    @property
    def k(self) -> float:
        return self._k

    @staticmethod
    def _build_adjacency(edges) -> dict[str, frozenset]:
        adjacency: dict[str, set] = {}
        for leader, laggard in edges:
            adjacency.setdefault(leader, set()).add(laggard)
        return {leader: frozenset(laggards) for leader, laggards in adjacency.items()}

    @staticmethod
    def _has_alternative_path(adjacency, source, target) -> bool:
        """
        Tell whether ``target`` is reachable from ``source`` without
        using the direct ``source -> target`` edge.
        """
        pending = [node for node in adjacency.get(source, ()) if node != target]
        seen = {source}
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node in seen:
                continue
            seen.add(node)
            pending.extend(adjacency.get(node, ()))
        return False

    def _transitive_reduction(self, edges) -> frozenset:
        adjacency = self._build_adjacency(edges)
        return frozenset(
            edge for edge in edges if not self._has_alternative_path(adjacency, *edge)
        )

    def _require_known(self, name):
        if name not in self._measurements:
            raise KeyError(f"Unknown batch: {name!r}")

    def _shortest_chain(self, leader, laggard):
        """
        Shortest path over the stored minimal evidence, breaking ties
        lexicographically so that queries are deterministic.
        """
        predecessors = {leader: None}
        pending = deque([leader])
        while pending:
            node = pending.popleft()
            for following in sorted(self._adjacency.get(node, ())):
                if following in predecessors:
                    continue
                predecessors[following] = node
                if following == laggard:
                    chain = []
                    current = laggard
                    while current is not None:
                        chain.append(current)
                        current = predecessors[current]
                    return tuple(reversed(chain))
                pending.append(following)
        return None

    def pairs(self) -> list[LeadEvidence]:
        """All batch pairs directly satisfying the criterion."""
        return [
            LeadEvidence(leader, laggard, self._margins[(leader, laggard)])
            for leader, laggard in sorted(self._margins)
        ]

    def evidence(self) -> list[LeadEvidence]:
        """
        Minimum set of direct evidence from which every leading
        relation remains recoverable (the transitive reduction).
        """
        return [
            LeadEvidence(leader, laggard, self._margins[(leader, laggard)])
            for leader, laggard in sorted(self._minimal_edges)
        ]

    def is_leading(self, leader: str, laggard: str) -> bool:
        """Whether ``leader`` robustly leads ``laggard``."""
        self._require_known(leader)
        self._require_known(laggard)
        return (leader, laggard) in self._margins

    def margin(self, leader: str, laggard: str) -> float:
        """True margin of a pair, recomputed from its endpoints."""
        self._require_known(leader)
        self._require_known(laggard)
        return lead_margin(
            self._measurements[leader],
            self._measurements[laggard],
            self._k,
        )

    def prove(self, leader: str, laggard: str) -> LeadChain:
        """
        Return the shortest stored-evidence chain proving that
        ``leader`` leads ``laggard``.

        The endpoint margin is always recomputed separately from the
        two endpoint measurements and their shared error terms.

        Raises
        ------
        KeyError
            If a batch name is unknown or no leading relation between
            the two batches can be established.
        """
        self._require_known(leader)
        self._require_known(laggard)
        if leader == laggard:
            raise KeyError(f"A batch cannot lead itself: {leader!r}")

        chain = self._shortest_chain(leader, laggard)
        if chain is None:
            raise KeyError(f"No leading relation from {leader!r} to {laggard!r}")

        hop_margins = tuple(
            self._margins[(start, end)] for start, end in zip(chain, chain[1:])
        )
        return LeadChain(
            leader=leader,
            laggard=laggard,
            chain=chain,
            hop_margins=hop_margins,
            endpoint_margin=self.margin(leader, laggard),
            direct=(leader, laggard) in self._margins,
        )


def robust_leading_pairs(measurements: Mapping, k: float) -> list[LeadEvidence]:
    """
    Return every ordered batch pair that directly satisfies the robust
    leading criterion, each with its margin.
    """
    return LeadRegistry(measurements, k).pairs()


def minimal_leading_evidence(measurements: Mapping, k: float) -> list[LeadEvidence]:
    """
    Return the minimum set of direct leading evidence (transitive
    reduction) from which every leading relation is recoverable.
    """
    return LeadRegistry(measurements, k).evidence()
