"""Finite-domain constraint solver for integer pixel-shift adjustments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from affine import Affine

from .model import Sheet, Tie

_COST_EPS = 1e-12


def _apply_linear(
    linear: tuple[tuple[float, float], tuple[float, float]],
    shift: tuple[int, int],
) -> tuple[float, float]:
    """World displacement of pixel ``(dx, dy)`` under the linear part."""
    (a, b), (d, e) = linear
    dx, dy = shift
    return a * dx + b * dy, d * dx + e * dy


class InfeasibleAdjustmentError(Exception):
    """No shift combination satisfies every seam within the given ranges."""


@dataclass(frozen=True)
class _TieGeometry:
    tie: Tie
    base: tuple[float, float]
    shift_a: tuple[tuple[float, float], tuple[float, float]]
    shift_b: tuple[tuple[float, float], tuple[float, float]]
    direction: tuple[float, float] | None

    def residual(self, va: tuple[int, int], vb: tuple[int, int]) -> tuple[float, float]:
        dax, day = _apply_linear(self.shift_a, va)
        dbx, dby = _apply_linear(self.shift_b, vb)
        return (
            self.base[0] + dax - dbx,
            self.base[1] + day - dby,
        )

    def satisfied(self, va: tuple[int, int], vb: tuple[int, int]) -> bool:
        rx, ry = self.residual(va, vb)
        tol = self.tie.tolerance
        if self.direction is not None:
            scalar = rx * self.direction[0] + ry * self.direction[1]
            return abs(scalar) <= tol + _residual_eps(tol, scalar)
        if self.tie.radial:
            magnitude = math.hypot(rx, ry)
            return magnitude <= tol + _residual_eps(tol, magnitude)
        eps = _residual_eps(tol, max(abs(rx), abs(ry)))
        return abs(rx) <= tol + eps and abs(ry) <= tol + eps


def _residual_eps(tolerance: float, value: float) -> float:
    return 1e-10 * max(1.0, abs(tolerance), abs(value))


class ShiftSolver:
    """Solve the seam adjustment as a lexicographic finite-domain problem.

    The objective hierarchy is:

    1. minimum number of sheets moved away from the zero shift,
    2. minimum total world displacement ``sum(norm(A_i @ d_i))``.
    """

    def __init__(self, sheets: list[Sheet], ties: list[Tie]) -> None:
        if not sheets:
            raise ValueError("at least one sheet is required")
        names = [sheet.name for sheet in sheets]
        if len(set(names)) != len(names):
            raise ValueError("sheet names must be unique")
        self.sheets = sheets
        self.ties = ties
        self.index = {name: i for i, name in enumerate(names)}
        for tie in ties:
            if tie.sheet_a not in self.index or tie.sheet_b not in self.index:
                raise ValueError(f"tie {tie.name!r} references an unknown sheet")
        self.zero = (0, 0)
        self.domains: list[list[tuple[int, int]]] = [sheet.domain() for sheet in sheets]
        self.domain_sets: list[set[tuple[int, int]]] = [set(d) for d in self.domains]
        self.linear = [self._linear_part(sheet.transform) for sheet in sheets]
        self.cost_table = [
            {value: self._shift_norm(i, value) for value in self.domains[i]}
            for i in range(len(sheets))
        ]
        self.geometries = [self._build_geometry(tie) for tie in ties]
        self.neighbors: list[set[int]] = [set() for _ in sheets]
        self.supports: dict[tuple[int, int], dict] = {}
        self._build_supports()
        root_domains = [set(d) for d in self.domain_sets]
        if not self._ac3(root_domains, list(self.supports.keys())):
            raise InfeasibleAdjustmentError(
                "arc consistency failed: the seam constraints have no solution"
            )
        self.root_domains = root_domains

    @staticmethod
    def _linear_part(
        transform: Affine,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return (transform.a, transform.b), (transform.d, transform.e)

    def _shift_norm(self, sheet_idx: int, value: tuple[int, int]) -> float:
        return math.hypot(*_apply_linear(self.linear[sheet_idx], value))

    def _build_geometry(self, tie: Tie) -> _TieGeometry:
        ia = self.index[tie.sheet_a]
        ib = self.index[tie.sheet_b]
        world_a = self.sheets[ia].transform @ tie.pixel_a
        world_b = self.sheets[ib].transform @ tie.pixel_b
        return _TieGeometry(
            tie=tie,
            base=(world_a[0] - world_b[0], world_a[1] - world_b[1]),
            shift_a=self.linear[ia],
            shift_b=self.linear[ib],
            direction=tie.along_track,
        )

    def _build_supports(self) -> None:
        pair_ties: dict[tuple[int, int], list[_TieGeometry]] = {}
        for geometry in self.geometries:
            ia = self.index[geometry.tie.sheet_a]
            ib = self.index[geometry.tie.sheet_b]
            pair_ties.setdefault((ia, ib), []).append(geometry)
            self.neighbors[ia].add(ib)
            self.neighbors[ib].add(ia)
        for (ia, ib), geoms in pair_ties.items():
            support_ab: dict[tuple[int, int], set[tuple[int, int]]] = {}
            support_ba: dict[tuple[int, int], set[tuple[int, int]]] = {}
            for va in self.domain_sets[ia]:
                support_ab[va] = {
                    vb
                    for vb in self.domain_sets[ib]
                    if all(g.satisfied(va, vb) for g in geoms)
                }
            for vb in self.domain_sets[ib]:
                support_ba[vb] = {
                    va
                    for va in self.domain_sets[ia]
                    if all(g.satisfied(va, vb) for g in geoms)
                }
            self.supports[(ia, ib)] = support_ab
            self.supports[(ib, ia)] = support_ba

    def _ac3(self, domains: list[set], seed_arcs: list[tuple[int, int]]) -> bool:
        queue = deque(seed_arcs)
        while queue:
            i, j = queue.popleft()
            support = self.supports[(i, j)]
            kept = {vi for vi in domains[i] if support[vi] & domains[j]}
            if not kept:
                return False
            if len(kept) != len(domains[i]):
                domains[i] = kept
                for k in self.neighbors[i]:
                    if k != j:
                        queue.append((k, i))
        return True

    def _ordered_values(self, sheet_idx: int, values: set) -> list[tuple[int, int]]:
        return [v for v in self.domains[sheet_idx] if v in values]

    def _lower_bound(self, domains: list[set], assigned: list) -> float:
        total = 0.0
        for i, value in enumerate(assigned):
            if value is None:
                total += min(self.cost_table[i][v] for v in domains[i])
        return total

    def _search(
        self,
        budget: int,
        target_cost: float | None = None,
        fixed: dict[int, tuple[int, int]] | None = None,
    ) -> dict[int, tuple[int, int]] | None:
        domains = [set(d) for d in self.root_domains]
        assigned: list[tuple[int, int] | None] = [None] * len(self.sheets)
        fixed = fixed or {}
        initial_changed = 0
        initial_cost = 0.0
        for i, value in fixed.items():
            if value not in domains[i]:
                return None
            domains[i] = {value}
            assigned[i] = value
            initial_changed += value != self.zero
            initial_cost += self.cost_table[i][value]
        if initial_changed > budget:
            return None
        if fixed and not self._ac3(
            domains,
            [(j, i) for i in fixed for j in self.neighbors[i]],
        ):
            return None
        best: dict[int, tuple[int, int]] | None = None
        best_cost = math.inf

        def recurse(
            domains: list[set],
            assigned: list,
            changed: int,
            cost: float,
        ) -> None:
            nonlocal best, best_cost
            if changed > budget:
                return
            unassigned = [i for i, value in enumerate(assigned) if value is None]
            if not unassigned:
                if target_cost is None:
                    if cost < best_cost - _COST_EPS:
                        best_cost = cost
                        best = dict(enumerate(assigned))
                else:
                    tolerance = max(_COST_EPS, 1e-10 * abs(target_cost))
                    if abs(cost - target_cost) <= tolerance:
                        best = dict(enumerate(assigned))
                return
            lower_bound = cost + self._lower_bound(domains, assigned)
            if target_cost is None:
                if best is not None and lower_bound >= best_cost - _COST_EPS:
                    return
            else:
                tolerance = max(_COST_EPS, 1e-10 * abs(target_cost))
                if lower_bound > target_cost + tolerance:
                    return
            variable = min(
                unassigned,
                key=lambda i: (
                    len(domains[i]),
                    -len(self.neighbors[i]),
                    i,
                ),
            )
            force_zero = changed == budget
            for value in self._ordered_values(variable, domains[variable]):
                if force_zero and value != self.zero:
                    continue
                child_domains = [set(d) for d in domains]
                child_domains[variable] = {value}
                if not self._ac3(
                    child_domains,
                    [(j, variable) for j in self.neighbors[variable]],
                ):
                    continue
                child_assigned = list(assigned)
                child_assigned[variable] = value
                recurse(
                    child_domains,
                    child_assigned,
                    changed + (value != self.zero),
                    cost + self.cost_table[variable][value],
                )
                if target_cost is not None and best is not None:
                    return

        recurse(domains, assigned, initial_changed, initial_cost)
        return best

    def solve_optimal(self) -> tuple[dict[int, tuple[int, int]], int, float]:
        """Return the lexicographically optimal assignment."""
        for budget in range(len(self.sheets) + 1):
            assignment = self._search(budget)
            if assignment is not None:
                changed = sum(v != self.zero for v in assignment.values())
                cost = sum(self.cost_table[i][v] for i, v in assignment.items())
                return assignment, changed, cost
        raise InfeasibleAdjustmentError("no shift combination satisfies all seams")

    def possible_values(
        self,
        sheet_idx: int,
        budget: int,
        target_cost: float,
    ) -> set[tuple[int, int]]:
        """All shifts of one sheet that occur in globally optimal solutions."""
        candidates: set[tuple[int, int]] = set()
        for value in self._ordered_values(sheet_idx, self.root_domains[sheet_idx]):
            assignment = self._search(
                budget,
                target_cost=target_cost,
                fixed={sheet_idx: value},
            )
            if assignment is not None:
                candidates.add(value)
        return candidates
