import random

import pytest

import portion as P


class IntInterval(P.AbstractDiscreteInterval):
    _step = 1


D = P.create_api(IntInterval)


def rule_key(rule):
    return str(rule.source), repr(rule.interval), str(rule.grade)


def signature(grading):
    return [
        (
            repr(segment.interval),
            segment.status,
            repr(segment.grade),
            tuple(sorted(rule_key(rule) for rule in segment.basis)),
            tuple(sorted(rule_key(rule) for rule in segment.reasons)),
        )
        for segment in grading.segments()
    ]


class TestGradeRule:
    def test_rejects_empty_interval(self):
        with pytest.raises(ValueError):
            P.GradeRule(P.open(1, 1), "grade")

    def test_rejects_non_interval(self):
        with pytest.raises(TypeError):
            P.GradeRule((0, 1), "grade")

    def test_accepts_tuple_forms(self):
        from_tuple = P.compile_grading([(P.closed(0, 1), "a")])
        from_triple = P.compile_grading([(P.closed(0, 1), "a", "lab")])
        from_rule = P.compile_grading([P.GradeRule(P.closed(0, 1), "a", "lab")])
        assert from_tuple.grade_for(0) == "a"
        assert from_triple.rules[0].source == "lab"
        assert from_rule.rules[0].source == "lab"

    def test_rejects_unsupported_item(self):
        with pytest.raises(TypeError):
            P.compile_grading([object()])


class TestSegmentation:
    def test_disjoint_rules_with_gaps(self):
        grading = P.compile_grading(
            [
                (P.closedopen(0, 10), "low"),
                (P.closedopen(10, 20), "mid"),
            ]
        )
        statuses = [(segment.interval, segment.status) for segment in grading]
        assert statuses == [
            (P.open(-P.inf, 0), P.GAP),
            (P.closedopen(0, 10), P.UNIQUE),
            (P.closedopen(10, 20), P.UNIQUE),
            (P.closedopen(20, P.inf), P.GAP),
        ]
        assert grading.grade_for(5) == "low"
        assert grading.grade_for(10) == "mid"

    def test_open_and_closed_endpoints(self):
        grading = P.compile_grading([(P.open(0, 1), "x")])
        unique = [s for s in grading if s.is_unique]
        assert len(unique) == 1
        assert unique[0].interval == P.open(0, 1)
        assert 0 not in grading
        assert 1 not in grading
        assert 0.5 in grading

    def test_singleton_overrides_broader_rule_on_one_point(self):
        grading = P.compile_grading(
            [
                (P.closedopen(0, 10), "base"),
                (P.singleton(5), "edge"),
            ]
        )
        assert grading.grade_for(4) == "base"
        assert grading.grade_for(5) == "edge"
        assert grading.grade_for(6) == "base"
        point = grading.classify(5)
        assert point.interval == P.singleton(5)
        assert point.basis[0].grade == "edge"

    def test_singleton_on_boundary_point(self):
        grading = P.compile_grading(
            [
                (P.closedopen(0, 10), "base"),
                (P.singleton(0), "edge"),
            ]
        )
        assert grading.grade_for(0) == "edge"
        assert grading.grade_for(1) == "base"

    def test_nested_rule_overrides_by_strict_containment(self):
        grading = P.compile_grading(
            [
                (P.closedopen(0, 100), "normal"),
                (P.openclosed(50, 60), "alert"),
            ]
        )
        assert grading.grade_for(50) == "normal"
        assert grading.grade_for(55) == "alert"
        assert grading.grade_for(60) == "alert"
        assert grading.grade_for(61) == "normal"

    def test_crossing_same_grade_is_unique_with_shared_basis(self):
        grading = P.compile_grading(
            [
                (P.closedopen(0, 5), "L"),
                (P.closedopen(3, 8), "L"),
            ]
        )
        overlap = grading.classify(4)
        assert overlap.is_unique
        assert overlap.grade == "L"
        assert len(overlap.basis) == 2

    def test_crossing_different_grades_conflict_only_on_overlap(self):
        grading = P.compile_grading(
            [
                (P.closedopen(0, 5), "L"),
                (P.closedopen(3, 8), "H"),
            ]
        )
        assert grading.grade_for(2) == "L"
        assert grading.classify(4).is_conflict
        assert grading.grade_for(6) == "H"
        assert grading.conflicts() == P.closedopen(3, 5)

    def test_same_region_same_grade_keeps_common_basis(self):
        grading = P.compile_grading(
            [
                (P.closed(0, 10), "A", "lab1"),
                (P.closed(0, 10), "A", "lab2"),
            ]
        )
        segment = grading.classify(5)
        assert segment.is_unique
        assert segment.grade == "A"
        assert {rule.source for rule in segment.basis} == {"lab1", "lab2"}

    def test_same_region_different_grades_conflict(self):
        grading = P.compile_grading(
            [
                (P.closed(0, 10), "A", "lab1"),
                (P.closed(0, 10), "B", "lab2"),
            ]
        )
        segment = grading.classify(5)
        assert segment.is_conflict
        assert {rule.grade for rule in segment.reasons} == {"A", "B"}
        with pytest.raises(P.GradingConflict):
            grading.grade_for(5)

    def test_closed_touching_point_conflicts(self):
        grading = P.compile_grading(
            [
                (P.closed(0, 5), "a"),
                (P.closed(5, 10), "b"),
            ]
        )
        point = grading.classify(5)
        assert point.is_conflict
        assert point.interval == P.singleton(5)

    def test_open_touching_point_is_a_gap(self):
        grading = P.compile_grading(
            [
                (P.closedopen(0, 5), "a"),
                (P.openclosed(5, 10), "b"),
            ]
        )
        point = grading.classify(5)
        assert point.is_gap
        assert 5 not in grading

    def test_infinite_bounds(self):
        grading = P.compile_grading(
            [
                (P.open(-P.inf, 0), "neg"),
                (P.closedopen(0, P.inf), "nonneg"),
            ]
        )
        assert grading.grade_for(-1) == "neg"
        assert grading.grade_for(0) == "nonneg"
        assert grading.grade_for(1) == "nonneg"

    def test_multi_atomic_rule(self):
        grading = P.compile_grading([(P.closed(0, 1) | P.closed(3, 4), "z")])
        assert grading.grade_for(0) == "z"
        assert grading.grade_for(4) == "z"
        assert grading.classify(2).is_gap

    def test_empty_input_is_one_gap(self):
        grading = P.compile_grading([])
        assert len(grading) == 1
        assert grading.segments()[0].is_gap
        assert 0 not in grading

    def test_grades_need_not_be_hashable(self):
        grading = P.compile_grading(
            [
                (P.closed(0, 5), ["low"]),
                (P.singleton(3), ["high"]),
            ]
        )
        assert grading.grade_for(0) == ["low"]
        assert grading.grade_for(3) == ["high"]
        overlap = P.compile_grading(
            [
                (P.closedopen(0, 5), ["x"]),
                (P.closedopen(3, 8), ["x"]),
            ]
        ).classify(4)
        assert overlap.is_unique
        assert len(overlap.basis) == 2


class TestQueries:
    def setup_method(self):
        self.grading = P.compile_grading(
            [
                (P.closedopen(0, 4), "L", "s1"),
                (P.closedopen(3, 7), "H", "s2"),
            ]
        )

    def test_classify_and_status_helpers(self):
        assert self.grading.classify(1).is_unique
        assert self.grading.classify(3).is_conflict
        assert self.grading.classify(10).is_gap

    def test_grade_for_gap_raises_keyerror(self):
        with pytest.raises(KeyError):
            self.grading.grade_for(10)

    def test_contains_only_covered_values(self):
        assert 2 in self.grading
        assert 10 not in self.grading

    def test_covered_gaps_and_conflicts(self):
        assert self.grading.gaps() == (P.open(-P.inf, 0) | P.closedopen(7, P.inf))
        assert self.grading.conflicts() == P.closedopen(3, 4)
        assert self.grading.covered() == P.closedopen(0, 7)

    def test_unique_intervals_is_interval_dict(self):
        mapping = self.grading.unique_intervals()
        assert isinstance(mapping, P.IntervalDict)
        assert mapping[1] == "L"
        assert mapping[5] == "H"

    def test_domain_clipping(self):
        grading = P.compile_grading([(P.closed(0, 10), "a")], domain=P.closedopen(2, 8))
        assert [segment.interval for segment in grading] == [P.closedopen(2, 8)]
        assert all(segment.is_unique for segment in grading)

    def test_duplicate_rules_are_deduplicated(self):
        grading = P.compile_grading(
            [
                (P.closed(0, 1), "a", "s"),
                (P.closed(0, 1), "a", "s"),
            ]
        )
        assert len(grading.rules) == 1


class TestOrderIndependence:
    def _random_rules(self, rng):
        rules = []
        for _ in range(12):
            low = rng.randint(0, 6)
            high = rng.randint(low, 6)
            interval = P.Interval.from_atomic(
                rng.choice([P.CLOSED, P.OPEN]),
                low,
                high,
                rng.choice([P.CLOSED, P.OPEN]),
            )
            if interval.empty:
                continue
            rules.append(
                (
                    interval,
                    rng.choice(["L", "M", "H"]),
                    rng.choice(["s1", "s2", None]),
                )
            )
        return rules

    def test_shuffling_keeps_map_identical(self):
        rng = random.Random(2026)
        for _ in range(100):
            rules = self._random_rules(rng)
            reference = signature(P.compile_grading(rules))
            for _ in range(5):
                shuffled = rules[:]
                rng.shuffle(shuffled)
                assert signature(P.compile_grading(shuffled)) == reference

    def test_reversed_input_keeps_map_identical(self):
        rules = [
            (P.closedopen(0, 5), "L"),
            (P.singleton(3), "M"),
            (P.closedopen(4, 9), "H"),
        ]
        assert P.compile_grading(rules) == P.compile_grading(list(reversed(rules)))

    def test_boundary_samples_keep_their_grade(self):
        rng = random.Random(99)
        for _ in range(100):
            rules = self._random_rules(rng)
            reference = P.compile_grading(rules)
            shuffled = rules[:]
            rng.shuffle(shuffled)
            reordered = P.compile_grading(shuffled)
            for value in [index / 2 for index in range(-2, 16)]:
                expected = reference.classify(value)
                actual = reordered.classify(value)
                assert (actual.status, actual.grade) == (
                    expected.status,
                    expected.grade,
                )


class TestDiscreteInterval:
    def test_compilation_with_discrete_domain(self):
        grading = P.compile_grading(
            [
                (D.closedopen(0, 5), "L"),
                (D.singleton(2), "H"),
            ],
            klass=IntInterval,
        )
        assert grading.grade_for(0) == "L"
        assert grading.grade_for(2) == "H"
        assert grading.grade_for(4) == "L"
        assert grading.classify(5).is_gap
