# Arrow migration pre-check

Pre-flight comparison for moving the registered date importer from the
standard-library `datetime.strptime` parser to Arrow.  The check is fully
data-driven and reproducible; it does not modify the `arrow` package.

## What it compares

Every registered field syntax is exercised on three layers:

1. **Acceptance domain** — which field strings each parser accepts or rejects.
2. **Converted instant** — POSIX microseconds after a reference UTC zone is
   attached to any naive result.
3. **Business day** — the civil `(year, month, day)`, including rollovers such
   as `24:00` end-of-day and seven-digit fractional-second carry.

A naive-vs-aware result is tracked separately as an *awareness* mismatch even
when the UTC instant happens to coincide.

### Compatibility-evidence rule

A field on which both parsers merely succeed is **not** compatibility evidence.
A syntax counts as certified only when both parsers reject together, or both
accept with an identical instant, business day, and awareness.  See the ledger
in section `[5]` of `report.txt`.

## Missing-year batches

`strptime` fills an omitted year with its 1900 default; Arrow falls back to
year 1, and some tokens (bare `DDD`/`DDDD`) require an explicit year.  Batches
that recorded no default year are therefore certified by sweeping **every
explicit** datetime-valid year (1..9999) through both parsers.  The reported
certifiable range is the intersection of the two accepted-year sets;
parser-only bands are reported separately rather than merged away.

## Running

```sh
# regenerate precheck/report.txt
.venv/bin/python precheck/migration_precheck.py

# run the executable oracles (64 tests)
.venv/bin/python -m pytest tests/test_migration_precheck.py -o addopts=""
```

The repository-wide `tox.ini` adds coverage flags requiring `pytest-cov`; the
`-o addopts=""` override above runs without that optional plugin.

## Layout

- `migration_precheck.py` — registry of minimal reproducing fields, three-layer
  comparison engine, year-sweep certification, and report renderer.
- `report.txt` — generated deterministic report.
- `../tests/test_migration_precheck.py` — locks every registered claim as an
  executable oracle.

## Registered difference classes

- `D01`–`D12`: Arrow accepts a field that `strptime` rejects.
- `D13`–`D17`: `strptime` accepts a field that Arrow rejects (including the
  year-1000 floor of Arrow's unpadded day-of-year rebuild).
- `I01`–`I05`: both accept, but the instant, business day, or awareness differs.
