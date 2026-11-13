from datetime import date

import pytest

from arrow.infer import Anchor, ColumnRelation, FormatInferer

FORMATS = ["MM/DD/YY", "DD/MM/YY", "YY/MM/DD"]


class TestColumnViability:
    def test_unparseable_values_eliminate_formats(self):
        inferer = FormatInferer(
            columns={"placed": ["13/04/05", "02/03/04"]},
            formats=FORMATS,
        )
        result = inferer.infer()
        # 13 cannot be a month, so MM/DD/YY is out for this column.
        assert result.formats["placed"] == ("DD/MM/YY", "YY/MM/DD")

    def test_all_formats_survive_fully_ambiguous_column(self):
        inferer = FormatInferer(
            columns={"placed": ["03/04/05"]},
            formats=FORMATS,
        )
        result = inferer.infer()
        assert result.formats["placed"] == tuple(sorted(FORMATS))
        assert not result.determined


class TestAnchors:
    def test_anchor_narrows_candidates(self):
        inferer = FormatInferer(
            columns={"placed": ["03/04/05"]},
            formats=FORMATS,
            anchors=[Anchor("placed", 0, date(2003, 4, 5))],
        )
        result = inferer.infer()
        assert result.formats["placed"] == ("YY/MM/DD",)
        assert result.determined

    def test_calibration_record_shrinks_candidates(self):
        inferer = FormatInferer(
            columns={"placed": ["03/04/05", "02/01/06"]},
            formats=FORMATS,
        )
        before = inferer.infer()
        after = inferer.calibrate("placed", 0, date(2005, 3, 4))
        assert set(after.formats["placed"]) < set(before.formats["placed"])
        assert after.formats["placed"] == ("MM/DD/YY",)

    def test_anchor_row_out_of_range(self):
        inferer = FormatInferer(
            columns={"placed": ["03/04/05"]},
            formats=FORMATS,
            anchors=[Anchor("placed", 0, date(2003, 4, 5))],
        )
        with pytest.raises(ValueError):
            inferer.add_anchor("placed", 7, date(2003, 4, 5))


class TestCrossColumnRelations:
    def test_shipment_cannot_precede_order(self):
        columns = {
            "ordered": ["05/06/07", "01/02/03"],
            "shipped": ["06/05/07", "02/01/03"],
        }
        relation = ColumnRelation(earlier="ordered", later="shipped")
        inferer = FormatInferer(
            columns=columns, formats=FORMATS, relations=[relation]
        )
        result = inferer.infer()
        for assignment in result.assignments:
            ordered = inferer._parsed_column("ordered", assignment["ordered"])
            shipped = inferer._parsed_column("shipped", assignment["shipped"])
            for first, second in zip(ordered, shipped):
                assert first <= second
        # DD/MM/YY for both columns puts shipment (2007-05-06) before the
        # order (2007-06-05) in row 0, so that combination is excluded.
        assert all(
            not (
                a["ordered"] == "DD/MM/YY" and a["shipped"] == "DD/MM/YY"
            )
            for a in result.assignments
        )

    def test_relation_plus_anchor_pins_both_columns(self):
        columns = {"ordered": ["05/06/07"], "shipped": ["04/05/06"]}
        inferer = FormatInferer(
            columns=columns,
            formats=FORMATS,
            anchors=[Anchor("ordered", 0, date(2005, 6, 7))],
            relations=[ColumnRelation(earlier="ordered", later="shipped")],
        )
        result = inferer.infer()
        assert result.formats["ordered"] == ("YY/MM/DD",)
        # Shipment must be on/after 2005-06-07; YY/MM/DD (2004-05-06) is
        # too early, the other two readings (2006) survive.
        assert result.formats["shipped"] == ("DD/MM/YY", "MM/DD/YY")


class TestRowDecisions:
    def test_same_partition_rows_released_others_get_two_interpretations(self):
        inferer = FormatInferer(
            columns={"placed": ["04/04/05", "03/07/05"]},
            formats=["MM/DD/YY", "DD/MM/YY"],
        )
        result = inferer.infer()
        released, ambiguous = result.decisions["placed"]

        assert released.resolved
        assert released.partition == (2005, 4)

        assert not ambiguous.resolved
        assert ambiguous.partition is None
        assert len(ambiguous.interpretations) == 2
        assert {i.value for i in ambiguous.interpretations} == {
            date(2005, 3, 7),
            date(2005, 7, 3),
        }

    def test_interpretations_are_actually_feasible(self):
        inferer = FormatInferer(
            columns={"placed": ["03/07/05"]},
            formats=FORMATS,
        )
        result = inferer.infer()
        (decision,) = result.decisions["placed"]
        assert not decision.resolved
        for interpretation in decision.interpretations:
            assignment = dict(interpretation.formats)
            assert assignment in result.assignments


class TestOrderIndependence:
    def test_format_list_order_does_not_change_result(self):
        columns = {
            "ordered": ["05/06/07", "01/02/03"],
            "shipped": ["06/05/07", "02/01/03"],
        }
        anchors = [Anchor("ordered", 1, date(2003, 1, 2))]
        relations = [ColumnRelation(earlier="ordered", later="shipped")]
        results = []
        orderings = (
            FORMATS,
            list(reversed(FORMATS)),
            ["DD/MM/YY", "YY/MM/DD", "MM/DD/YY"],
        )
        for formats in orderings:
            inferer = FormatInferer(
                columns=columns,
                formats=formats,
                anchors=anchors,
                relations=relations,
            )
            results.append(inferer.infer())
        assert results[0] == results[1] == results[2]

    def test_duplicate_formats_are_ignored(self):
        inferer = FormatInferer(
            columns={"placed": ["03/04/05"]},
            formats=["MM/DD/YY", "MM/DD/YY", "DD/MM/YY"],
        )
        assert inferer.formats == ("DD/MM/YY", "MM/DD/YY")
