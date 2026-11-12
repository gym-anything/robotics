"""Score arithmetic the tasks share.

task.Evaluator measures the state, the host app's record and the read log; the functions here turn those measurements
into progress. They are pure (no physics, no third-party imports), so the rules are tested without a simulator. The
arithmetic only one task uses is in that task's verifier.py (tasks/<task>/), as pure as this.

Every task: score = 1 if every check passes, else FAILED x progress. Progress lies in [0, 1], is 0 at an untouched
start and depends only on the end state, the host record, what was read and the episode history (the scanner's
steady hold, the water case's highest rise), never on the order of the work, the hand, a trigger pull or a hands-free
presentation. count-by-scanning has its own progress (tasks/count-by-scanning/verifier.py); the other seven use its
constants: DISTURBED_COST per disturbed unit (every free body on the floor, the scanner included; in count-key-quantity
also every unit, case or facing lane its stock rules find off its place) and x NOT_RETURNED while the scanner is not
seated and released or a key is held down.
"""

DISTURBED_COST = .5     # a unit left disturbed costs half a counted unit
NOT_RETURNED = .9       # progress factor while the scanner is not seated and released, or a key is held down
FAILED = .9             # an unsuccessful episode scores FAILED x progress: every failure below every success

RETURN_SHARE = .3       # hold-scanner, scan-location: share of the progress for the scanner put back afterwards
PUT_AWAY_SHARE = .4     # checkout, library-checkin: share of an item's credit for putting it in the tote / bin
UNPAID_COST = .5        # checkout: an item bagged without being rung up (an unpaid item leaving the counter)


def clip01(x):
    return min(1., max(0., float(x)))


def score(success, progress):
    """1 on success, else FAILED x progress."""
    return 1. if success else FAILED*float(progress)


def returned_factor(returned):
    return 1. if returned else NOT_RETURNED


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


def item_progress(credits, errors, disturbed, returned):
    """(sum of the per-item credits - errors - DISTURBED_COST x disturbed) / number of items, clipped to [0, 1], x
    NOT_RETURNED unless the scanner is back and no key is down."""
    return clip01((sum(credits) - errors - DISTURBED_COST*disturbed)/max(1, len(credits)))*returned_factor(returned)
