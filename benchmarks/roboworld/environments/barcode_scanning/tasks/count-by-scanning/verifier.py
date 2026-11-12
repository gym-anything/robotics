"""Outcome of count-by-scanning: the slot's count booked under its location from each unit's own barcode, every unit
read once and correctly, the shelf left faced as found and the scanner back in its cradle.

verify(e, q, rec, reads, checks, metrics) judges an episode: e is the task.Evaluator, q the bodies' poses (n x 7:
position, quaternion xyzw), rec the host app's record, reads the firmware's reads, and checks and metrics what every
task checks and measures (Evaluator.evaluate). It adds the task's own checks and returns the evaluation: success, score,
progress, contract, checks and metrics. The rules (bookings, stock, score) are in task.py's module docstring; the stock
rules are Evaluator._shelf, which count-key-quantity shares."""
from barcode_scanning.catalog import SKUS
from barcode_scanning.scoring import DISTURBED_COST, NOT_RETURNED, FAILED     # count-by-scanning's score constants

CONTRACT = "count-by-scanning-v1"    # the scoring contract its evaluation reports (evaluation["contract"])


def verify(e, q, rec, reads, checks, metrics):
    """count-by-scanning (module docstring): the count booked under the slot's location from each unit's own
    symbol, the shelf left as found, the scanner back; score = 1 on success, else FAILED x progress."""
    from barcode_scanning.catalog import sku_gtin14
    sc = e.world.scene
    target = e.meta.get("target_slot")
    want = {sku_gtin14(SKUS[k]): n for k, n in e.meta.get("truth", {}).items()}
    got = dict((rec.get("location_counts") or {}).get(target, {}))
    metrics["record"], metrics["truth"] = got, want
    checks["record_matches"] = got == want
    units = [n for n, it in sc.items.items() if it.get("slot") == target and it.get("sku")]
    # bookings only: a read the host refused and a location label book nothing (device.HostApp.on_scan)
    per_unit, foreign, hits, errors = {}, [], set(), []
    for r in reads:
        loc = r.get("booked_location")
        if not loc:
            continue
        a = r.get("attribution") or {}
        mine = a.get("kind") == "product" and a.get("item") in units
        if mine:
            per_unit[a["item"]] = per_unit.get(a["item"], 0) + 1
        else:
            foreign.append(a.get("label"))
        if mine and loc == target and not a.get("misread") and a["item"] not in hits:
            hits.add(a["item"])
        else:
            errors.append(dict(label=a.get("label"), location=loc, misread=bool(a.get("misread"))))
    metrics["unit_reads"], metrics["foreign_reads"] = per_unit, foreign
    checks["each_unit_once"] = all(per_unit.get(u, 0) == 1 for u in units) and not foreign
    # misreads the record holds data from: a booked misread, or a misread location label whose code a booking was
    # made under before any correct read of that code in the same stretch (consecutive location reads setting one
    # code). A refused read, a stray misread of the code already correctly active, or a misread label re-scanned
    # before anything was booked leaves nothing in the record. An unbacked stretch of the counted slot's code costs
    # one error (its bookings would otherwise be hits); bookings under another code are errors already.
    in_record, stretch = {i for i, r in enumerate(reads) if r.get("booked_location")
                          and (r.get("attribution") or {}).get("misread")}, None
    for i, r in enumerate(reads):
        code = r.get("sets_location")
        if code:
            if stretch is None or stretch["code"] != code:
                stretch = dict(code=code, backed=False, misread=[], charged=False)
            if (r.get("attribution") or {}).get("misread"):
                stretch["misread"].append(i)
            else:
                stretch["backed"] = True
        elif r.get("booked_location") and stretch is not None and not stretch["backed"]:
            in_record.update(stretch["misread"])                 # unbacked: every read in the stretch misread
            if stretch["code"] == target and not stretch["charged"]:
                stretch["charged"] = True
                errors.append(dict(label=(reads[stretch["misread"][0]].get("attribution") or {}).get("label"),
                                   location=target, misread=True))
    metrics["misreads_in_record"] = len(in_record)
    checks["no_misreads"] = not in_record
    disturbed = e._shelf(q, target, units, checks, metrics)
    # progress: the count booked, less wrong bookings and what is left disturbed, by any route
    returned = checks["scanner_seated"] and checks["no_key_pressed"]
    progress = max(0., len(hits) - len(errors) - DISTURBED_COST*len(disturbed))/max(1, len(units)) \
        * (1. if returned else NOT_RETURNED)
    metrics.update(units=len(units), hits=sorted(hits), errors=errors, disturbed=disturbed, scanner_returned=bool(returned))
    ok = all(checks.values())
    return dict(success=bool(ok), score=1. if ok else FAILED*progress, progress=float(progress),
                contract=CONTRACT, checks={k: bool(v) for k, v in checks.items()},
                metrics=metrics)
