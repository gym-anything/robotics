"""Score arithmetic of the eight tasks.

task.Evaluator measures the state, the host app's record and the read log; the functions here turn those measurements
into progress. They are pure (no physics, no third-party imports), so the rules are tested without a simulator.

Every task: score = 1 if every check passes, else FAILED x progress. Progress lies in [0, 1], is 0 at an untouched
start and depends only on the end state, the host record, what was read and the episode history (the scanner's
steady hold, the water case's highest rise), never on the order of the work, the hand, a trigger pull or a hands-free
presentation. count-by-scanning has its own progress (task.Evaluator._count_by_scanning); the other seven use its
constants: DISTURBED_COST per disturbed unit (every free body on the floor, the scanner included; in count-key-quantity
also every unit, case or facing lane its stock rules find off its place) and x NOT_RETURNED while the scanner is not
seated and released or a key is held down.
"""

DISTURBED_COST = .5     # a unit left disturbed costs half a counted unit
NOT_RETURNED = .9       # progress factor while the scanner is not seated and released, or a key is held down
FAILED = .9             # an unsuccessful episode scores FAILED x progress: every failure below every success

HOLD_S = 5.             # hold-scanner: the steady hold the instruction asks for
RETURN_SHARE = .3       # hold-scanner, scan-location: share of the progress for the scanner put back afterwards
READ_FACTOR = .5        # hold-scanner: progress factor once anything was read (the task forbids reads)
WRONG_LOCATION_COST = .25   # scan-location: each read of another slot's location label
PUT_AWAY_SHARE = .4     # checkout, library-checkin: share of an item's credit for putting it in the tote / bin
UNPAID_COST = .5        # checkout: an item bagged without being rung up (an unpaid item leaving the counter)
LIFT_FREE_M = .010      # checkout-bulky: the case's highest rise with full case credit (the case_not_lifted limit) ...
LIFT_ZERO_M = .050      # ... falling linearly to no case credit at this rise
SCAN_SHARE = .3         # count-key-quantity: share of a product's credit for a correct read of it under the slot


def clip01(x):
    return min(1., max(0., float(x)))


def score(success, progress):
    """1 on success, else FAILED x progress."""
    return 1. if success else FAILED*float(progress)


def returned_factor(returned):
    return 1. if returned else NOT_RETURNED


def hold_progress(steady_s, returned, reads, on_floor):
    """hold-scanner: min(1, longest steady hold / HOLD_S) x (0.7 + 0.3 x returned) x (READ_FACTOR once anything was
    read), less DISTURBED_COST per free body on the floor. Putting the scanner back earns nothing without a hold."""
    hold = min(1., max(0., float(steady_s))/HOLD_S)
    p = hold*(1 - RETURN_SHARE + RETURN_SHARE*bool(returned))*(READ_FACTOR if reads else 1.)
    return clip01(p - DISTURBED_COST*on_floor)


def location_progress(hit, returned, wrong_reads, on_floor):
    """scan-location: hit x (0.7 + 0.3 x returned), less WRONG_LOCATION_COST per read of another slot's location label
    and DISTURBED_COST per free body on the floor; hit = the target's label read correctly and its code still the
    recorded location. Aiming earns nothing; a second read of the right label costs nothing."""
    p = bool(hit)*(1 - RETURN_SHARE + RETURN_SHARE*bool(returned))
    return clip01(p - WRONG_LOCATION_COST*wrong_reads - DISTURBED_COST*on_floor)


def tally(reads, items, kind, wrong):
    """Sort the reads into the items whose own `kind` label was read correctly at least once, and one error per other
    read: a `wrong` read (an inner unit's symbol through a pack's film, a book's ISBN), a misread, a repeat of an item
    already read, or a label that belongs to none of `items`. Each read is its attribution with its format (keys
    format, label, kind, item, misread). The order of the reads changes neither result's size."""
    done, errors = set(), []
    for r in reads:
        own = r.get("kind") == kind and r.get("item") in items
        why = ("wrong" if wrong(r) else "misread" if r.get("misread") else
               ("repeat" if r["item"] in done else None) if own else "not an item")
        if why is None:
            done.add(r["item"])
        else:
            errors.append(dict(label=r.get("label"), error=why))
    return done, errors


def put_away_credit(read, put_away):
    """checkout, library-checkin: an item rung up / checked in earns 1 - PUT_AWAY_SHARE, and the rest once it is in
    the tote / bin; an item put away without being read earns nothing (checkout charges it as unpaid)."""
    return float(bool(read))*(1 - PUT_AWAY_SHARE + PUT_AWAY_SHARE*bool(put_away))


def lift_credit(rise_m):
    """checkout-bulky: 1 up to LIFT_FREE_M of the case's highest rise in the episode, falling linearly to 0 at
    LIFT_ZERO_M (read from the history, so a lift cannot be undone by setting the case back)."""
    return clip01((LIFT_ZERO_M - float(rise_m))/(LIFT_ZERO_M - LIFT_FREE_M))


def quantity_credit(scanned, keyed, truth):
    """count-key-quantity, one product with `truth` units in the slot: SCAN_SHARE for a correct read of it under the
    counted location, and the rest by how close the quantity the host holds for it under that location is to the
    truth (nothing at an error of the whole truth or more). ENTER overwrites, so a corrected quantity earns it all."""
    return SCAN_SHARE*bool(scanned) + (1 - SCAN_SHARE)*max(0., 1 - abs(keyed - truth)/truth)


def item_progress(credits, errors, disturbed, returned):
    """(sum of the per-item credits - errors - DISTURBED_COST x disturbed) / number of items, clipped to [0, 1], x
    NOT_RETURNED unless the scanner is back and no key is down."""
    return clip01((sum(credits) - errors - DISTURBED_COST*disturbed)/max(1, len(credits)))*returned_factor(returned)


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
