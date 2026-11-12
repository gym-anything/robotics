"""Outcome of library-checkin: every book checked in from its library label (never its ISBN), each read once and
correctly, every book in the returns bin and the bin upright on the desk.

verify(e, q, rec, reads, checks, metrics) judges an episode: e is the task.Evaluator, q the bodies' poses (n x 7:
position, quaternion xyzw), rec the host app's record, reads the firmware's reads, and checks and metrics what every
task checks and measures (Evaluator.evaluate). It adds the task's own checks and returns the evaluation: success, score,
progress, contract, checks and metrics."""
from barcode_scanning import scoring as S

CONTRACT = "library-checkin-v1"    # the scoring contract its evaluation reports (evaluation["contract"])
ISBN_FORMATS = ("EAN13", "UPCA")    # library-checkin: a book's ISBN symbologies (its library label is Codabar)


def verify(e, q, rec, reads, checks, metrics):
    returned = e._returned(checks, metrics)
    att = e._attributions(reads)
    w, on_floor = e.world, metrics["on_floor"]
    books = {n: it for n, it in w.scene.items.items() if it.get("library_id")}
    want = sorted(it["library_id"] for it in books.values())
    metrics["checked_in"], metrics["truth"] = sorted(rec["checked_in"]), want
    checks["checkin_matches"] = sorted(rec["checked_in"]) == want
    checks["no_isbn_reads"] = not any(r["format"] in ISBN_FORMATS for r in reads)
    in_bin = set()
    if w.scene.items.get("returns_bin"):
        inside = {n for n, it in books.items() if e._in_container(q, "returns_bin", it["body"])}
        bin_ok = e._container_on_counter(q, "returns_bin")
        metrics["not_returned"] = [n for n in books if n not in inside]
        checks["books_in_bin"] = not metrics["not_returned"] and bin_ok
        in_bin = inside if bin_ok else set()
    # progress: each book checked in (its library label read correctly) and binned; one error per other read
    # (an ISBN, a misread, a repeat check-in, a label that is no book's library label)
    checked, errors = S.tally(att, books, "library", lambda a: a.get("format") in ISBN_FORMATS)
    metrics.update(checked=sorted(checked), read_errors=errors)
    credits = [S.put_away_credit(b in checked, b in in_bin) for b in books]
    return e._done(checks, metrics, S.item_progress(credits, len(errors), len(on_floor), returned))
