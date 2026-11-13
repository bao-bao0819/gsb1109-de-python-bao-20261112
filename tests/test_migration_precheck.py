"""Tests for the strptime-to-Arrow migration pre-check.

These tests pin every registered difference class to observed behavior on all
three comparison layers (acceptance domain, converted instant, business day)
and certify the year-less batches by sweeping explicit years rather than
assuming a default.  A field that both parsers accept is never accepted as
compatibility evidence by itself.
"""

from datetime import datetime, timezone

import pytest

from precheck import migration_precheck as mp
from precheck.migration_precheck import (
    ALL_YEARS,
    BATCHES,
    CASES,
    MAX_YEAR,
    MIN_YEAR,
    Rejected,
    arrow_parse,
    certify_batch,
    collect_verdicts,
    legacy_parse,
    render_report,
    run_case,
)

VERDICTS = {case.case_id: run_case(case) for case in CASES}
DOMAIN_CASES = [case for case in CASES if case.layer == "domain"]
INSTANT_CASES = [case for case in CASES if case.layer == "instant"]


@pytest.fixture(scope="module")
def certifications():
    return {batch.batch_id: certify_batch(batch) for batch in BATCHES}


class TestRegistryShape:
    def test_case_ids_unique_and_fields_non_empty(self):
        ids = [case.case_id for case in CASES]
        assert len(ids) == len(set(ids))
        for case in CASES:
            assert case.field
            assert case.legacy_fmt.startswith("%")
            assert case.arrow_fmt

    def test_one_minimal_field_per_difference_class(self):
        # Each registered class reproduces through a single field string; no
        # class is allowed to lean on a second field to expose the difference.
        for case in CASES:
            verdict = run_case(case)
            if case.layer == "domain":
                assert verdict.relation == case.claim
            else:
                assert case.claim == "both_accept_diff"
                assert verdict.relation == "both_accept"

    def test_claims_are_known_relations(self):
        for case in CASES:
            assert case.claim in {
                "arrow_only",
                "legacy_only",
                "both_accept_diff",
                "both_reject",
            }

    @pytest.mark.parametrize("case", DOMAIN_CASES, ids=lambda c: c.case_id)
    def test_domain_claim_matches_observed_relation(self, case):
        verdict = VERDICTS[case.case_id]
        assert verdict.relation == case.claim


class TestAcceptanceDomain:
    @pytest.mark.parametrize(
        "case_id",
        ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D10",
         "D11", "D12"],
    )
    def test_arrow_accepts_what_strptime_rejects(self, case_id):
        verdict = VERDICTS[case_id]
        assert verdict.legacy_status == "REJ"
        assert verdict.arrow_status == "OK"
        assert isinstance(verdict.arrow_dt, datetime)

    @pytest.mark.parametrize("case_id", ["D13", "D14", "D15", "D16", "D17"])
    def test_strptime_accepts_what_arrow_rejects(self, case_id):
        verdict = VERDICTS[case_id]
        assert verdict.legacy_status == "OK"
        assert verdict.arrow_status == "REJ"
        assert isinstance(verdict.legacy_dt, datetime)

    def test_day_of_year_floor_is_year_1000_in_arrow(self):
        verdict = VERDICTS["D17"]
        assert verdict.legacy_dt == datetime(1, 3, 1)
        assert verdict.arrow_status == "REJ"
        # A four-digit year is the Arrow DDD/DDDD boundary.
        assert arrow_parse("1000-060", "YYYY-DDDD") == datetime(1000, 3, 1)

    def test_end_of_day_rolls_forward_only_in_arrow(self):
        verdict = VERDICTS["D01"]
        assert verdict.arrow_dt == datetime(2026, 1, 2, 0, 0)

    def test_seven_digit_fraction_carries_business_day(self):
        verdict = VERDICTS["D03"]
        day = (verdict.arrow_dt.year, verdict.arrow_dt.month, verdict.arrow_dt.day)
        assert day == (2026, 1, 2)
        assert verdict.arrow_dt.microsecond == 0

    def test_arrow_token_shape_split_for_offsets(self):
        # One strptime %z covers both shapes; Arrow needs Z vs ZZ explicitly.
        colon = run_case(
            mp.Case("t", "domain", "2026-01-01T00:00:00+02:00",
                    "%Y-%m-%dT%H:%M:%S%z", "YYYY-MM-DDTHH:mm:ssZZ",
                    "both_accept_diff", "")
        )
        no_colon = VERDICTS["D16"]
        assert colon.arrow_status == "OK"
        assert no_colon.arrow_status == "REJ"


class TestInstantAndBusinessDay:
    @pytest.mark.parametrize("case", INSTANT_CASES, ids=lambda c: c.case_id)
    def test_both_accept_but_layers_differ(self, case):
        verdict = VERDICTS[case.case_id]
        assert verdict.relation == "both_accept"
        assert (
            verdict.instant_equal is False
            or verdict.business_day_equal is False
            or verdict.awareness_equal is False
        )

    def test_missing_year_defaults_1900_vs_1(self):
        verdict = VERDICTS["I01"]
        assert verdict.legacy_dt.year == 1900
        assert verdict.arrow_dt.year == 1
        assert verdict.business_day_equal is False
        assert verdict.instant_equal is False

    def test_weekday_name_anchor_epochs_differ(self):
        verdict = VERDICTS["I03"]
        assert verdict.legacy_dt == datetime(1900, 1, 1)
        assert verdict.arrow_dt == datetime(1970, 1, 5)

    def test_numeric_weekday_digit_is_consumed_but_ignored_by_arrow(self):
        verdict = VERDICTS["I04"]
        assert verdict.arrow_dt == datetime(1, 1, 1)
        assert verdict.legacy_dt == datetime(1900, 1, 1)

    def test_gmt_naive_vs_aware_blocks_automatic_migration(self):
        verdict = VERDICTS["I05"]
        assert verdict.legacy_dt.tzinfo is None
        assert verdict.arrow_dt.tzinfo is not None
        assert verdict.awareness_equal is False
        # UTC instant only agrees after the importer attaches its reference zone
        assert verdict.legacy_dt.replace(tzinfo=timezone.utc).timestamp() == (
            verdict.arrow_dt.timestamp()
        )


class TestCompatibilityEvidencePolicy:
    def test_no_plain_both_success_counts_as_evidence(self):
        for verdict in collect_verdicts():
            if verdict.relation == "both_accept":
                fully_equal = (
                    verdict.instant_equal
                    and verdict.business_day_equal
                    and verdict.awareness_equal
                )
                if not fully_equal:
                    # registered on purpose as a divergence, never as evidence
                    assert verdict.case.claim == "both_accept_diff"

    def test_fully_equal_control_case_would_pass_all_three_layers(self):
        # Sanity control: identical, aware results are the only thing that may
        # serve as positive evidence.  Built outside the divergence registry.
        control = mp.Case(
            "CTRL", "control", "2024-02-29T00:00:00+0000",
            "%Y-%m-%dT%H:%M:%S%z", "YYYY-MM-DDTHH:mm:ssZ",
            "both_reject", "",
        )
        verdict = run_case(control)
        assert verdict.relation == "both_accept"
        assert verdict.instant_equal is True
        assert verdict.business_day_equal is True
        assert verdict.awareness_equal is True

    def test_ledger_excludes_both_success_from_agreement(self):
        report = render_report()
        assert "both-success rows are NOT counted as compatibility evidence" in report


class TestMissingYearCertification:
    def test_sweep_covers_full_datetime_year_range(self):
        assert list(ALL_YEARS)[0] == MIN_YEAR == 1
        assert list(ALL_YEARS)[-1] == MAX_YEAR == 9999
        assert len(list(ALL_YEARS)) == 9999

    def test_month_day_batch_certifiable_on_every_year(self, certifications):
        cert = certifications["B-NOYEAR-MD"]
        assert cert["certifiable_count"] == 9999
        assert cert["certifiable_min"] == 1
        assert cert["certifiable_max"] == 9999
        assert cert["sets_agree"] is True

    def test_leap_day_batch_certifiable_only_on_leap_years(self, certifications):
        cert = certifications["B-NOYEAR-LEAP"]
        leap_years = [
            y for y in ALL_YEARS
            if y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
        ]
        assert cert["arrow_accepted_years"] == leap_years
        assert cert["legacy_accepted_years"] == leap_years
        assert cert["certifiable_count"] == 2424
        assert cert["certifiable_min"] == 4
        assert cert["certifiable_max"] == 9996
        assert cert["sets_agree"] is True

    def test_bare_day_of_year_is_rejected_by_arrow(self, certifications):
        cert = certifications["B-NOYEAR-DOY"]
        assert cert["bare_legacy"].startswith("OK")
        assert cert["bare_arrow"].startswith("REJ")

    def test_day_of_year_certifiable_intersection_starts_at_1000(self, certifications):
        cert = certifications["B-NOYEAR-DOY"]
        # Arrow's unpadded rebuild rejects years 1..999 even though stdlib takes them.
        assert cert["certifiable_years"][0] == 1000
        assert cert["certifiable_years"][-1] == 9999
        assert cert["certifiable_count"] == 9000
        assert cert["sets_agree"] is False
        assert cert["legacy_only_years"] == list(range(1, 1000))
        assert cert["arrow_only_years"] == []

    def test_day_366_rollover_agrees_inside_certified_band(self, certifications):
        cert = certifications["B-NOYEAR-DOY366"]
        assert cert["certifiable_years"][0] == 1000
        assert cert["certifiable_years"][-1] == 9998
        assert cert["certifiable_count"] == 8999
        assert cert["sets_agree"] is False
        assert cert["legacy_only_years"] == list(range(1, 1000))
        # On a non-leap year both parsers roll day 366 into the next year.
        completed = "2023-366"
        assert legacy_parse(completed, "%Y-%j") == datetime(2024, 1, 1)
        assert arrow_parse(completed, "YYYY-DDDD") == datetime(2024, 1, 1)

    def test_recorded_default_batch_is_flagged_for_explicit_year(self, certifications):
        cert = certifications["B-DEFAULT-1900"]
        assert cert["default_year"] == 1900
        assert "complete_field" not in cert or "certifiable_min" not in cert
        assert cert["bare_legacy"].startswith("OK")
        assert cert["bare_arrow"].startswith("OK")

    def test_parser_failures_are_normalized_to_rejected(self):
        with pytest.raises(mp.Rejected):
            legacy_parse("not-a-date", "%Y-%m-%d")
        with pytest.raises(Rejected):
            arrow_parse("not-a-date", "YYYY-MM-DD")


class TestReport:
    def test_report_has_three_layers_and_certification_sections(self):
        report = render_report()
        for heading in (
            "[1] ACCEPTANCE-DOMAIN DIVERGENCES",
            "[2] INSTANT / [3] BUSINESS-DAY DIVERGENCES",
            "[4] MISSING-YEAR BATCH CERTIFICATION",
            "[5] COMPATIBILITY-EVIDENCE LEDGER",
        ):
            assert heading in report

    def test_report_is_deterministic(self):
        assert render_report() == render_report()

    def test_report_lists_every_registered_case(self):
        report = render_report()
        for case in CASES:
            assert case.case_id in report
            assert case.field in report
