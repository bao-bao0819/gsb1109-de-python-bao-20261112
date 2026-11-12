"""Diagnose cross-file VTIMEZONE/TZID pollution in vobject.

Each process is fresh, so the module-global __tzidMap starts empty (only UTC).
An appointment's tzinfo is classified by its effective rule signature
(offset + tzname at the wall time) because TimezoneComponent.gettzinfo
rebuilds a fresh dateutil object on every attribute access.

Categories:
  own      : global registry rule == this document's own VTIMEZONE rule
  external : registry rule differs from (or document lacks) an own VTIMEZONE
  none     : no usable definition -> tzinfo is None (floating)
"""
import datetime
import sys

import vobject
from vobject.icalendar import getTzid, registerTzid, utc

FILES = {
    "A": "a.ics",  # VTIMEZONE TZID=Local, fixed +10:00
    "B": "b.ics",  # VTIMEZONE TZID=Local, fixed -05:00
    "C": "c.ics",  # TZID=Local referenced, NO VTIMEZONE shipped
}
WALL = datetime.datetime(2026, 1, 15, 2, 0, 0)

reg_log = []
_orig_register = registerTzid


def spy_register(tzid, tzinfo):
    name = tzinfo.tzname(None) if hasattr(tzinfo, "tzname") else "?"
    reg_log.append((tzid, name))
    return _orig_register(tzid, tzinfo)


import vobject.icalendar as ical  # noqa: E402

ical.registerTzid = spy_register


def sig(tzi, dt=WALL):
    if tzi is None:
        return None
    off = tzi.utcoffset(dt)
    total = int(off.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{total % 3600 // 60:02d} / {tzi.tzname(dt)}"


def own_tzinfo(cal):
    vtzs = getattr(cal, "vtimezone_list", [])
    return vtzs[0].tzinfo if vtzs else None


def vevent_dtstart_line(block):
    inside = False
    for ln in block.splitlines():
        if ln == "BEGIN:VEVENT":
            inside = True
        elif ln == "END:VEVENT":
            inside = False
        elif inside and ln.startswith("DTSTART"):
            return ln
    return "(none)"


def run(order):
    print("=" * 80)
    print(f"ORDER {order}   (shared process, global vobject __tzidMap)")
    print("=" * 80)
    own_sig_by_file = {}
    for label in order:
        with open(FILES[label]) as fh:
            cal = vobject.readOne(fh.read())
        own_tzi = own_tzinfo(cal)
        own_sig = sig(own_tzi) if own_tzi is not None else None
        if own_sig:
            own_sig_by_file[label] = own_sig
        glob_tzi = getTzid("Local", smart=False)
        glob_sig = sig(glob_tzi) if glob_tzi is not None else None

        print(f"\n[{label}] {FILES[label]}  own VTIMEZONE: "
              f"{own_sig if own_sig else 'NONE IN FILE'}")
        print(f"      registry 'Local' before/at parse: {glob_sig}")
        for ev in cal.vevent_list:
            dt = ev.dtstart.value
            ev_sig = sig(dt.tzinfo, dt.replace(tzinfo=None))
            if dt.tzinfo is None:
                category = "NONE (no usable definition -> floating time)"
            elif own_sig is not None and ev_sig == own_sig and glob_sig == own_sig:
                category = "OWN file definition (first registrant wins)"
            elif own_sig is not None and ev_sig != own_sig:
                category = "EXTERNAL registered definition (own VTIMEZONE ignored)"
            elif own_sig is None:
                category = "EXTERNAL registered definition (file has none)"
            else:
                category = "OWN file definition"
            utc_key = dt.astimezone(utc).isoformat() if dt.tzinfo else "N/A (floating)"
            print(f"      {ev.uid.value}")
            print(f"         local wall time : {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"         effective rule  : {ev_sig}")
            print(f"         source          : {category}")
            print(f"         UTC storage key : {utc_key}")

        out = cal.serialize()
        print("      re-serialized VEVENT DTSTART :", vevent_dtstart_line(out))
        vtz = [ln for ln in out.splitlines() if ln.startswith(("TZID", "TZOFFSETTO", "TZNAME"))]
        print("      re-serialized VTIMEZONE kept:", vtz)

    print("\n  registerTzid() calls this process (Local only registered on first win):")
    for tzid, name in reg_log:
        print(f"     - {tzid!r} tzname={name}")
    print()


if __name__ == "__main__":
    run(sys.argv[1].split(","))
