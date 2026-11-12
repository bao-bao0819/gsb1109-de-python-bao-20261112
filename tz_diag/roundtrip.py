"""Round-trip proof: polluted in-memory datetime vs. the document's own VTIMEZONE."""
import datetime
import subprocess
import sys

import vobject
from vobject.icalendar import getTzid, utc

WALL = datetime.datetime(2026, 1, 15, 2, 0, 0)

# Phase 1: in THIS process, import A first (registers Local=+10), then B.
vobject.readOne(open("a.ics").read())
cal_b = vobject.readOne(open("b.ics").read())
ev = cal_b.vevent_list[0]
dt = ev.dtstart.value
polluted_key = dt.astimezone(utc)
blob = cal_b.serialize()
open("b_reserialized.ics", "w").write(blob)
print("Polluted B in-memory tzinfo offset:", dt.tzinfo.utcoffset(WALL),
      " tzname:", dt.tzinfo.tzname(WALL))
print("B UTC storage key after A,B order :", polluted_key.isoformat())
print("Correct B key if its own -05 used : 2026-01-15T07:00:00+00:00")
print("Drift vs correct                  :",
      (polluted_key.replace(tzinfo=None) - datetime.datetime(2026, 1, 15, 7, 0)).total_seconds() / 3600, "hours")
print("Serialized doc keeps its own VTIMEZONE TZOFFSETTO:",
      [ln for ln in blob.splitlines() if ln.startswith("TZOFFSETTO")])
print("Serialized VEVENT DTSTART line    :",
      [ln for ln in blob.splitlines() if ln.startswith("DTSTART")])

# Phase 2: fresh process re-reads that same serialized file (empty registry).
code = (
    "import vobject, datetime\n"
    "from vobject.icalendar import utc\n"
    "cal=vobject.readOne(open('b_reserialized.ics').read())\n"
    "dt=cal.vevent_list[0].dtstart.value\n"
    "print('FRESH reparse offset:', dt.tzinfo.utcoffset(datetime.datetime(2026,1,15,2)),\n"
    "      'tzname:', dt.tzinfo.tzname(datetime.datetime(2026,1,15,2)),\n"
    "      'UTC key:', dt.astimezone(utc).isoformat())\n"
)
out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
print("\nAfter re-parse in a brand-new process (global map reset):")
print(out.stdout.strip())
