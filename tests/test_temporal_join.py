import unittest
from datetime import date

from cachetools import CacheInfo

from temporal_join import DimRecord, TemporalDimTable, TemporalJoinCache

OPEN_ENDED = date.max


def province_record(store, region, valid_from, revision, valid_to=OPEN_ENDED):
    return DimRecord(store, valid_from, valid_to, revision, {"region": region})


def tax_record(region, rate, valid_from, revision, valid_to=OPEN_ENDED):
    return DimRecord(region, valid_from, valid_to, revision, {"rate": rate})


class TemporalDimTableTest(unittest.TestCase):
    def test_resolve_picks_record_effective_at_order_time(self):
        table = TemporalDimTable("tax")
        table.add(tax_record("East", 0.13, date(2026, 1, 1), "T1"))
        self.assertEqual(table.resolve("East", date(2026, 3, 15)).revision, "T1")
        self.assertIsNone(table.resolve("West", date(2026, 3, 15)))
        self.assertIsNone(table.resolve("East", date(2025, 12, 31)))

    def test_retroactive_correction_takes_precedence_inside_interval(self):
        table = TemporalDimTable("tax")
        table.add(tax_record("East", 0.13, date(2026, 1, 1), "T1"))
        table.add(
            tax_record("East", 0.09, date(2026, 3, 1), "T2", valid_to=date(2026, 5, 1))
        )
        self.assertEqual(table.resolve("East", date(2026, 2, 15)).revision, "T1")
        self.assertEqual(table.resolve("East", date(2026, 3, 15)).revision, "T2")
        self.assertEqual(table.resolve("East", date(2026, 5, 1)).revision, "T1")


class TemporalJoinCacheTest(unittest.TestCase):
    def setUp(self):
        self.joins = TemporalJoinCache()
        self.joins.province.add(
            province_record("S1", "East", date(2026, 1, 1), "P-S1-v1")
        )
        self.joins.province.add(
            province_record("S2", "East", date(2026, 1, 1), "P-S2-v1")
        )
        self.joins.tax.add(tax_record("East", 0.13, date(2026, 1, 1), "T-East-v1"))

    def enrich(self, store, month, day):
        return self.joins.enrich(store, date(2026, month, day))

    def test_backfilled_march_order_reuses_march_interpretation(self):
        first = self.enrich("S1", 3, 15)
        self.assertEqual(
            first,
            {
                "region": "East",
                "tax_rate": 0.13,
                "province_revision": "P-S1-v1",
                "tax_revision": "T-East-v1",
            },
        )
        # the same March order, only keyed in May, hits the cache
        self.assertEqual(self.enrich("S1", 3, 15), first)
        self.assertEqual(
            self.joins.cache_info(),
            CacheInfo(hits=1, misses=1, maxsize=1024, currsize=1),
        )

    def test_retroactive_tax_correction(self):
        self.assertEqual(self.enrich("S1", 2, 15)["tax_rate"], 0.13)
        self.assertEqual(self.enrich("S1", 3, 15)["tax_rate"], 0.13)

        # retroactive correction: East tax was 9% in March and April
        self.joins.tax.add(
            tax_record(
                "East", 0.09, date(2026, 3, 1), "T-East-v2", valid_to=date(2026, 5, 1)
            )
        )

        # before the corrected interval the old entry is still hit
        self.assertEqual(self.enrich("S1", 2, 15)["tax_rate"], 0.13)
        # inside the interval the interpretation changed: miss, recompute
        corrected = self.enrich("S1", 3, 15)
        self.assertEqual(corrected["tax_rate"], 0.09)
        self.assertEqual(corrected["tax_revision"], "T-East-v2")
        self.assertEqual(self.enrich("S1", 4, 15), corrected)
        # after the interval the old entry is hit again
        self.assertEqual(self.enrich("S1", 5, 15)["tax_rate"], 0.13)
        # another store depending on the corrected tax record gets the
        # new result as well
        other = self.enrich("S2", 3, 20)
        self.assertEqual(other["tax_rate"], 0.09)
        self.assertEqual(other["tax_revision"], "T-East-v2")
        self.assertEqual(self.enrich("S2", 2, 20)["tax_rate"], 0.13)
        self.assertEqual(self.enrich("S2", 3, 20), other)

        self.assertEqual(
            self.joins.cache_info(),
            CacheInfo(hits=5, misses=4, maxsize=1024, currsize=4),
        )

    def test_retroactive_province_correction(self):
        self.joins.tax.add(tax_record("West", 0.10, date(2026, 1, 1), "T-West-v1"))
        # S1 belonged to region West during April and May
        self.joins.province.add(
            province_record(
                "S1", "West", date(2026, 4, 1), "P-S1-v2", valid_to=date(2026, 6, 1)
            )
        )

        moved = self.enrich("S1", 4, 15)
        self.assertEqual(moved["region"], "West")
        self.assertEqual(moved["tax_rate"], 0.10)
        self.assertEqual(self.enrich("S1", 4, 15), moved)
        # outside the corrected interval S1 still resolves to East
        self.assertEqual(self.enrich("S1", 3, 15)["region"], "East")
        self.assertEqual(self.enrich("S1", 7, 15)["tax_rate"], 0.13)

        self.assertEqual(
            self.joins.cache_info(),
            CacheInfo(hits=2, misses=2, maxsize=1024, currsize=2),
        )

    def test_missing_dimension_record_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.enrich("unknown-store", 3, 15)
        self.joins.province.add(
            province_record("S3", "North", date(2026, 1, 1), "P-S3-v1")
        )
        with self.assertRaises(KeyError):
            self.enrich("S3", 3, 15)
        # failed lookups must not populate the cache
        self.assertEqual(
            self.joins.cache_info(),
            CacheInfo(hits=0, misses=0, maxsize=1024, currsize=0),
        )


if __name__ == "__main__":
    unittest.main()
