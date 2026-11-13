"""Exact LRU cache warmup (prefetch) optimizer for small jobs.

Before an early-morning backfill only some pages can be pre-read.  Given
candidate pages with a read cost and a size, the real access sequence and
an initially empty LRU cache, this module finds the warmup sequence that
minimizes the total read cost of the whole round within a byte budget.

Charging model:

- Prefetching a page is charged at ``night_discount * cost`` (the night
  discount applies only to the warmup phase) and loads the page into the
  LRU cache, evicting least recently used pages as usual.
- Each later access is free on a cache hit and charged the full normal
  ``cost`` on a miss; a miss loads the page into the cache, again
  evicting LRU pages as needed.
- The sum of the sizes of the prefetched pages must not exceed
  ``prefetch_budget`` bytes.

Ties between equally cheap optimal plans are broken in favour of fewer
prefetched bytes, then lexicographically by warmup order to keep the
result deterministic.

The optimizer enumerates every ordered warmup sequence exactly, which is
intended for small jobs (at most ``DEFAULT_MAX_CANDIDATES`` cacheable,
actually accessed candidate pages).

JSON input format (see ``main()``)::

    {
      "cache_capacity": 3,
      "prefetch_budget": 3,
      "night_discount": 0.5,
      "pages": [
        {"name": "A", "size": 1, "cost": 6},
        {"name": "B", "size": 2, "cost": 10}
      ],
      "access_sequence": ["B", "A", "B", "A", "B"]
    }

Costs and the discount are parsed through ``fractions.Fraction``, so
values like ``0.5``, ``"0.1"`` or ``"1/3"`` are handled exactly.
"""

import argparse
import json
import sys
from collections import OrderedDict, namedtuple
from fractions import Fraction
from itertools import combinations, permutations

DEFAULT_MAX_CANDIDATES = 9

Page = namedtuple("Page", ["name", "size", "cost"])
BillEvent = namedtuple("BillEvent", ["kind", "page", "charge", "evicted"])
Plan = namedtuple("Plan", ["prefetch_order", "prefetch_bytes", "bill"])


class Problem:
    def __init__(
        self,
        pages,
        access_sequence,
        cache_capacity,
        prefetch_budget,
        night_discount,
    ):
        self.pages = dict(pages)
        self.access_sequence = tuple(access_sequence)
        self.cache_capacity = int(cache_capacity)
        self.prefetch_budget = int(prefetch_budget)
        self.night_discount = night_discount


class Bill:
    def __init__(self, events, prefetch_cost, access_cost):
        self.events = list(events)
        self.prefetch_cost = prefetch_cost
        self.access_cost = access_cost

    @property
    def total_cost(self):
        return self.prefetch_cost + self.access_cost


class LruCache:
    """Minimal LRU cache with byte capacity, initially empty."""

    def __init__(self, capacity):
        self.capacity = capacity
        self._entries = OrderedDict()
        self.size = 0

    def access(self, name):
        """Mark `name` as used; return True on a hit."""
        if name in self._entries:
            self._entries.move_to_end(name)
            return True
        return False

    def load(self, page):
        """Insert `page`, evicting LRU pages; return evicted page names."""
        if page.name in self._entries:
            self._entries.move_to_end(page.name)
            return ()
        evicted = []
        while self._entries and self.size + page.size > self.capacity:
            name, size = self._entries.popitem(last=False)
            self.size -= size
            evicted.append(name)
        if page.size <= self.capacity:
            self._entries[page.name] = page.size
            self.size += page.size
        return tuple(evicted)


def simulate(problem, prefetch_order):
    """Run one round: warmup at the night rate, then charged accesses."""
    cache = LruCache(problem.cache_capacity)
    events = []
    prefetch_cost = Fraction(0)
    for name in prefetch_order:
        page = problem.pages[name]
        charge = page.cost * problem.night_discount
        evicted = cache.load(page)
        prefetch_cost += charge
        events.append(BillEvent("prefetch", name, charge, evicted))
    access_cost = Fraction(0)
    for name in problem.access_sequence:
        page = problem.pages[name]
        if cache.access(name):
            events.append(BillEvent("hit", name, Fraction(0), ()))
        else:
            evicted = cache.load(page)
            access_cost += page.cost
            events.append(BillEvent("miss", name, page.cost, evicted))
    return Bill(events, prefetch_cost, access_cost)


def candidate_pages(problem):
    """Pages worth prefetching: accessed later and cacheable at all.

    Prefetching anything else can never lower the total cost and only
    spends budget bytes, so it is dominated under the tie-break rule.
    """
    accessed = set(problem.access_sequence)
    return sorted(
        name
        for name, page in problem.pages.items()
        if name in accessed and page.size <= problem.cache_capacity
    )


def optimize(problem, max_candidates=DEFAULT_MAX_CANDIDATES):
    """Find the cheapest warmup sequence within the byte budget.

    Returns a ``Plan``.  Ties on total cost are broken by fewer
    prefetched bytes, then by lexicographic warmup order.
    """
    candidates = candidate_pages(problem)
    if len(candidates) > max_candidates:
        raise ValueError(
            f"{len(candidates)} candidate pages exceed the exact-search "
            f"limit of {max_candidates}; pass a higher max_candidates "
            "to override"
        )
    sizes = {name: problem.pages[name].size for name in candidates}
    costs = {name: problem.pages[name].cost for name in candidates}
    best_key = None
    best_plan = None
    for count in range(len(candidates) + 1):
        for subset in combinations(candidates, count):
            prefetch_bytes = sum(sizes[name] for name in subset)
            if prefetch_bytes > problem.prefetch_budget:
                continue
            prefetch_cost = problem.night_discount * sum(
                costs[name] for name in subset
            )
            if best_key is not None and prefetch_cost > best_key[0]:
                continue
            for order in permutations(subset):
                bill = simulate(problem, order)
                key = (bill.total_cost, prefetch_bytes, order)
                if best_key is None or key < best_key:
                    best_key = key
                    best_plan = Plan(order, prefetch_bytes, bill)
    return best_plan


def format_cost(value):
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_report(problem, plan):
    """Human-readable warmup order followed by the billing breakdown."""
    lines = []
    discount = problem.night_discount
    if plan.prefetch_order:
        order = ", ".join(plan.prefetch_order)
    else:
        order = "(none)"
    lines.append(
        f"Warmup order: {order} "
        f"(prefetch bytes {plan.prefetch_bytes}/{problem.prefetch_budget}, "
        f"night discount {format_cost(discount)})"
    )
    for event in plan.bill.events:
        if event.kind != "prefetch":
            continue
        page = problem.pages[event.page]
        line = (
            f"  prefetch {event.page}: charge {format_cost(event.charge)}"
            f" (normal {format_cost(page.cost)} x {format_cost(discount)})"
        )
        if event.evicted:
            line += f"; evicts {', '.join(event.evicted)}"
        lines.append(line)
    lines.append(f"Warmup subtotal: {format_cost(plan.bill.prefetch_cost)}")
    lines.append("Accesses (normal rate):")
    step = 0
    for event in plan.bill.events:
        if event.kind == "prefetch":
            continue
        step += 1
        line = f"  {step}. {event.page}: {event.kind}, charge "
        line += format_cost(event.charge)
        if event.evicted:
            line += f"; evicts {', '.join(event.evicted)}"
        lines.append(line)
    lines.append(f"Access subtotal: {format_cost(plan.bill.access_cost)}")
    lines.append(f"Grand total: {format_cost(plan.bill.total_cost)}")
    return "\n".join(lines)


def plan_to_json(problem, plan):
    return {
        "prefetch_order": list(plan.prefetch_order),
        "prefetch_bytes": plan.prefetch_bytes,
        "prefetch_cost": format_cost(plan.bill.prefetch_cost),
        "access_cost": format_cost(plan.bill.access_cost),
        "total_cost": format_cost(plan.bill.total_cost),
        "events": [
            {
                "kind": event.kind,
                "page": event.page,
                "charge": format_cost(event.charge),
                "evicted": list(event.evicted),
            }
            for event in plan.bill.events
        ],
    }


def _fraction(value):
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def load_problem(data):
    pages = {}
    for entry in data["pages"]:
        pages[entry["name"]] = Page(
            entry["name"], int(entry["size"]), _fraction(entry["cost"])
        )
    sequence = tuple(data["access_sequence"])
    unknown = [name for name in sequence if name not in pages]
    if unknown:
        raise ValueError(f"access sequence mentions unknown pages: {unknown}")
    return Problem(
        pages=pages,
        access_sequence=sequence,
        cache_capacity=data["cache_capacity"],
        prefetch_budget=data["prefetch_budget"],
        night_discount=_fraction(data.get("night_discount", 1)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Exact LRU warmup optimizer for small jobs."
    )
    parser.add_argument(
        "input",
        help="JSON workload file, or '-' to read from standard input",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the plan as JSON instead of a text report",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="exact-search limit on cacheable, accessed candidate pages",
    )
    args = parser.parse_args(argv)
    if args.input == "-":
        data = json.load(sys.stdin)
    else:
        with open(args.input, encoding="utf-8") as stream:
            data = json.load(stream)
    problem = load_problem(data)
    plan = optimize(problem, max_candidates=args.max_candidates)
    if args.json:
        json.dump(plan_to_json(problem, plan), sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(format_report(problem, plan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
