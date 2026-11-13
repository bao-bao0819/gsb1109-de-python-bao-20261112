import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from fractions import Fraction

import warmup_optimizer as wo


def make_problem(pages, sequence, capacity, budget, discount=1):
    return wo.Problem(
        pages={name: wo.Page(name, size, Fraction(cost)) for name, size, cost in pages},
        access_sequence=tuple(sequence),
        cache_capacity=capacity,
        prefetch_budget=budget,
        night_discount=Fraction(discount),
    )


class OptimizeTest(unittest.TestCase):
    def test_prefetch_both_pages_all_hits(self):
        # capacity 3, budget 3, night discount 1/2
        # no prefetch: 16, prefetch A: 13, prefetch B: 11, both: 8
        problem = make_problem(
            [("A", 1, 6), ("B", 2, 10)],
            "BABAB",
            capacity=3,
            budget=3,
            discount=Fraction(1, 2),
        )
        plan = wo.optimize(problem)
        self.assertEqual(plan.prefetch_order, ("A", "B"))
        self.assertEqual(plan.prefetch_bytes, 3)
        self.assertEqual(plan.bill.prefetch_cost, 8)
        self.assertEqual(plan.bill.access_cost, 0)
        self.assertEqual(plan.bill.total_cost, 8)

    def test_tie_break_prefers_fewer_prefetch_bytes(self):
        # {A, B} (2 bytes) and {A, B, C} (3 bytes) both reach total 0;
        # the cheaper-bytes plan wins, lexicographic order breaks the rest.
        problem = make_problem(
            [("A", 1, 6), ("B", 1, 10), ("C", 1, 0)],
            "ABABC",
            capacity=3,
            budget=3,
            discount=0,
        )
        plan = wo.optimize(problem)
        self.assertEqual(plan.prefetch_order, ("A", "B"))
        self.assertEqual(plan.prefetch_bytes, 2)
        self.assertEqual(plan.bill.total_cost, 0)

    def test_prefetch_not_worth_it_picks_empty_plan(self):
        # prefetching A costs exactly what it saves, so the empty plan
        # wins the byte tie-break.
        problem = make_problem(
            [("A", 3, 10)], "A", capacity=4, budget=6, discount=1
        )
        plan = wo.optimize(problem)
        self.assertEqual(plan.prefetch_order, ())
        self.assertEqual(plan.prefetch_bytes, 0)
        self.assertEqual(plan.bill.total_cost, 10)

    def test_uncacheable_page_is_never_prefetched(self):
        problem = make_problem(
            [("A", 5, 7)], "AA", capacity=3, budget=10, discount=0
        )
        plan = wo.optimize(problem)
        self.assertEqual(plan.prefetch_order, ())
        self.assertEqual(plan.bill.total_cost, 14)

    def test_never_accessed_page_is_never_prefetched(self):
        problem = make_problem(
            [("A", 1, 6), ("Z", 1, 100)],
            "AA",
            capacity=2,
            budget=2,
            discount=0,
        )
        plan = wo.optimize(problem)
        self.assertEqual(plan.prefetch_order, ("A",))
        self.assertEqual(plan.bill.total_cost, 0)

    def test_fractional_night_discount_is_exact(self):
        problem = make_problem(
            [("A", 1, 10)], "A", capacity=1, budget=1, discount=Fraction(1, 3)
        )
        plan = wo.optimize(problem)
        self.assertEqual(plan.prefetch_order, ("A",))
        self.assertEqual(plan.bill.prefetch_cost, Fraction(10, 3))
        self.assertEqual(plan.bill.total_cost, Fraction(10, 3))

    def test_miss_then_hit_at_normal_rate(self):
        problem = make_problem(
            [("A", 1, 4)], "AA", capacity=1, budget=0, discount=0
        )
        plan = wo.optimize(problem)
        self.assertEqual(plan.prefetch_order, ())
        self.assertEqual(plan.bill.access_cost, 4)
        self.assertEqual(plan.bill.total_cost, 4)

    def test_too_many_candidates_raises(self):
        pages = [(f"P{i}", 1, 1) for i in range(10)]
        problem = make_problem(
            pages, [f"P{i}" for i in range(10)], capacity=10, budget=10
        )
        with self.assertRaises(ValueError):
            wo.optimize(problem)


class SimulateTest(unittest.TestCase):
    def test_prefetch_order_controls_eviction(self):
        # capacity 4 cannot hold both 3-byte pages; the page prefetched
        # last stays cached.
        problem = make_problem(
            [("A", 3, 10), ("B", 3, 20)],
            "A",
            capacity=4,
            budget=6,
            discount=1,
        )
        bill_ab = wo.simulate(problem, ("A", "B"))
        self.assertEqual(bill_ab.total_cost, 40)
        self.assertEqual(bill_ab.events[-1].kind, "miss")
        bill_ba = wo.simulate(problem, ("B", "A"))
        self.assertEqual(bill_ba.total_cost, 30)
        self.assertEqual(bill_ba.events[-1].kind, "hit")

    def test_miss_records_evictions(self):
        problem = make_problem(
            [("A", 1, 6), ("B", 1, 10)],
            "AB",
            capacity=1,
            budget=0,
            discount=1,
        )
        bill = wo.simulate(problem, ())
        self.assertEqual(bill.events[1].evicted, ("A",))


class ReportTest(unittest.TestCase):
    def test_format_report_breakdown(self):
        problem = make_problem(
            [("A", 1, 6), ("B", 2, 10)],
            "BABAB",
            capacity=3,
            budget=3,
            discount=Fraction(1, 2),
        )
        plan = wo.optimize(problem)
        report = wo.format_report(problem, plan)
        self.assertIn("Warmup order: A, B", report)
        self.assertIn("prefetch A: charge 3 (normal 6 x 1/2)", report)
        self.assertIn("prefetch B: charge 5 (normal 10 x 1/2)", report)
        self.assertIn("Warmup subtotal: 8", report)
        self.assertIn("1. B: hit, charge 0", report)
        self.assertIn("Access subtotal: 0", report)
        self.assertIn("Grand total: 8", report)


class CliTest(unittest.TestCase):
    def workload(self):
        return {
            "cache_capacity": 3,
            "prefetch_budget": 3,
            "night_discount": 0.5,
            "pages": [
                {"name": "A", "size": 1, "cost": 6},
                {"name": "B", "size": 2, "cost": 10},
            ],
            "access_sequence": ["B", "A", "B", "A", "B"],
        }

    def test_main_text_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "workload.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(self.workload(), stream)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(wo.main([path]), 0)
        self.assertIn("Grand total: 8", buffer.getvalue())

    def test_main_json_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "workload.json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(self.workload(), stream)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                self.assertEqual(wo.main([path, "--json"]), 0)
        output = json.loads(buffer.getvalue())
        self.assertEqual(output["prefetch_order"], ["A", "B"])
        self.assertEqual(output["prefetch_bytes"], 3)
        self.assertEqual(output["total_cost"], "8")

    def test_load_problem_rejects_unknown_page(self):
        data = self.workload()
        data["access_sequence"] = ["A", "X"]
        with self.assertRaises(ValueError):
            wo.load_problem(data)


if __name__ == "__main__":
    unittest.main()
