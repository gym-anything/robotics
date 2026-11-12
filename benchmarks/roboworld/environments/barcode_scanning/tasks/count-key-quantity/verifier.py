"""Outcome of count-key-quantity: the quantity keyed for each product under the counted slot's location equals the
units in the slot, nothing keyed under another location, the shelf left faced as found (count-by-scanning's stock
rules, Evaluator._shelf) and the scanner back in its cradle.

verify(e, q, rec, reads, checks, metrics) judges an episode: e is the task.Evaluator, q the bodies' poses (n x 7:
position, quaternion xyzw), rec the host app's record, reads the firmware's reads, and checks and metrics what every
task checks and measures (Evaluator.evaluate). It adds the task's own checks and returns the evaluation: success, score,
progress, contract, checks and metrics."""
from barcode_scanning import scoring as S
from barcode_scanning.catalog import SKUS

CONTRACT = "count-key-quantity-v1"    # the scoring contract its evaluation reports (evaluation["contract"])
SCAN_SHARE = .3         # count-key-quantity: share of a product's credit for a correct read of it under the slot


def quantity_credit(scanned, keyed, truth):
    """count-key-quantity, one product with `truth` units in the slot: SCAN_SHARE for a correct read of it under the
    counted location, and the rest by how close the quantity the host holds for it under that location is to the
    truth (nothing at an error of the whole truth or more). ENTER overwrites, so a corrected quantity earns it all."""
    return SCAN_SHARE*bool(scanned) + (1 - SCAN_SHARE)*max(0., 1 - abs(keyed - truth)/truth)


def key_bookings(entries, reads, locations):
    """count-key-quantity: replay the host app's entries (HostApp.on_scan and HostApp.on_key, in the order the host
    handled them) beside each scan's attribution (`reads`: the firmware's reads, in the same order as the host's scan
    entries). Returns (scans, record): every item scan the host accepted as (active location, code, read), and the
    quantities it saved, {location: {code: (quantity, read of the scan it was keyed for)}}, a later ENTER for the same
    item under the same location replacing the earlier one as in the host."""
    reads = list(reads)
    n_scans = sum(1 for e in entries if e["kind"] == "scan")
    if n_scans != len(reads):
        raise ValueError(f"{n_scans} scan entries in the host record against {len(reads)} firmware reads")
    locations, reads = set(locations), iter(reads)
    location = pending = None
    buf, scans, record = "", [], {}
    for e in entries:
        if e["kind"] == "scan":
            r = next(reads)
            if e["format"] == "Code128" and e["data"] in locations:
                location = e["data"]                    # a location label: the pending item, if any, stays pending
            elif location is not None:                  # before any location the host refuses the scan
                pending, buf = (location, e["data"], r), ""
                scans.append(pending)
        elif e["kind"] == "key" and pending is not None:
            key = e["key"]
            if key.isdigit():
                buf = (buf + key)[:4]
            elif key in (".", "-"):
                buf = buf[:-1]
            elif key == "ENTER" and buf:
                record.setdefault(pending[0], {})[pending[1]] = (int(buf), pending[2])
                pending, buf = None, ""
    return scans, record


def verify(e, q, rec, reads, checks, metrics):
    """count-key-quantity: the quantities the host holds under the slot's location equal the truth,
    nothing keyed under any other location, the shelf by count-by-scanning's stock rules, the scanner back."""
    returned = e._returned(checks, metrics)
    from barcode_scanning.catalog import sku_gtin14
    sc = e.world.scene
    target = e.meta.get("target_slot")
    want = {sku_gtin14(SKUS[k]): n for k, n in e.meta.get("truth", {}).items()}
    counts = rec.get("location_counts") or {}
    got = dict(counts.get(target, {}))
    metrics["record"], metrics["truth"] = got, want
    checks["record_matches"] = got == want
    # quantities keyed under another location are wrong data in the record (the counted slot's own quantities are
    # judged, not which location label was read last, as in count-by-scanning)
    elsewhere = {loc: dict(c) for loc, c in counts.items() if loc != target and c}
    metrics["counts_under_other_locations"] = elsewhere
    checks["no_counts_under_other_locations"] = not elsewhere
    units = [n for n, it in sc.items.items() if it.get("slot") == target and it.get("sku")]
    disturbed = e._shelf(q, target, units, checks, metrics)
    # which scan each saved quantity was keyed for: the host's entries replayed beside the reads' attribution
    scans, booked = key_bookings(rec["entries"], [r.get("attribution") or {} for r in reads], sc.slots.keys())
    if {loc: {c: v[0] for c, v in d.items()} for loc, d in booked.items()} != counts:
        raise RuntimeError(f"count-key-quantity: the replayed entries {booked} disagree with the host record "
                           f"{counts}")
    scanned = {code for loc, code, a in scans if loc == target and not a.get("misread")}
    errors = [dict(location=loc, code=code, misread=bool(a.get("misread"))) for loc, d in booked.items()
              for code, (_, a) in d.items() if loc != target or code not in want or a.get("misread")]
    credits = [quantity_credit(g in scanned, got.get(g, 0), n) for g, n in want.items()]
    metrics.update(products=len(want), scanned=sorted(scanned), errors=errors, units=len(units),
                   disturbed=disturbed)
    return e._done(checks, metrics, S.item_progress(credits, len(errors), len(disturbed), returned))
