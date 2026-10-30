"""Score arithmetic of the eight barcode-scanning tasks (barcode_scanning/scoring.py).

No physics, no renderer, no scipy: the progress rules as pure functions of what the evaluator measures. Every task's
progress is 0 at an untouched start, 1 at the measurements of a success, graded in between, and independent of the order
of the reads. The count-key-quantity replay is checked against the host app itself (device.HostApp) on random sequences
of scans and keys. The simulator-backed minimal pairs are in test_barcode_scanning_evaluator.py.

  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_barcode_scanning_scoring.py
"""

import ast
import itertools
import json
import random
import unittest

from barcode_scanning import scoring as S
from tests.packages import PACKAGES

ROOT = PACKAGES


def read(label, kind="product", item=None, misread=False, fmt="UPCA"):
    return dict(label=label, kind=kind, item=item, misread=misread, format=fmt)


class SharedConstantsTest(unittest.TestCase):
    def test_constants_are_count_by_scanning_s(self):
        """The seven tasks share count-by-scanning's constants."""
        self.assertEqual((S.FAILED, S.NOT_RETURNED, S.DISTURBED_COST), (.9, .9, .5))

    def test_every_failure_scores_below_every_success(self):
        self.assertEqual(S.score(True, .2), 1.)
        self.assertAlmostEqual(S.score(False, 1.), .9)
        self.assertEqual(S.score(False, 0.), 0.)

    def test_every_task_reports_a_contract_and_the_catalogue_lists_the_eight(self):
        from barcode_scanning.config import TASKS
        tree = ast.parse((ROOT/"barcode_scanning"/"task.py").read_text())
        contracts = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                         and any(getattr(t, "id", None) == "CONTRACTS" for t in n.targets))
        tasks = {t for ts in TASKS.values() for t in ts}
        self.assertEqual(set(contracts), tasks)
        self.assertEqual(contracts["count-by-scanning"], "count-by-scanning-v1")
        self.assertEqual(len(set(contracts.values())), 8)
        bench = json.loads((ROOT/"barcode_scanning"/"benchmark.json").read_text())
        entry = bench["task_set"]
        self.assertEqual(set(entry["tasks"]), tasks)
        self.assertEqual(entry["contracts"], contracts)


class HoldScannerProgressTest(unittest.TestCase):
    def test_untouched_start_is_zero(self):
        """The scanner never left its cradle: seated, nothing read, nothing on the floor (6 of 7 checks pass) -> 0."""
        self.assertEqual(S.hold_progress(0., True, 0, 0), 0.)

    def test_hold_returned_is_one(self):
        self.assertEqual(S.hold_progress(5.3, True, 0, 0), 1.)

    def test_partial_credit(self):
        self.assertAlmostEqual(S.hold_progress(5., False, 0, 0), .7)              # held 5 s, left in the hand
        self.assertAlmostEqual(S.score(False, S.hold_progress(5., False, 0, 0)), .63)
        self.assertAlmostEqual(S.hold_progress(2.5, True, 0, 0), .5)              # half the hold, put back
        self.assertAlmostEqual(S.hold_progress(5.3, True, 1, 0), .5)              # a read during the hold
        self.assertAlmostEqual(S.hold_progress(5.3, True, 0, 1), .5)              # a unit knocked onto the floor
        self.assertEqual(S.hold_progress(0., True, 0, 1), 0.)                     # clipped at 0

    def test_return_without_hold_earns_nothing(self):
        self.assertEqual(S.hold_progress(0., True, 0, 0), S.hold_progress(0., False, 0, 0))


class ScanLocationProgressTest(unittest.TestCase):
    def test_untouched_start_is_zero(self):
        self.assertEqual(S.location_progress(False, True, 0, 0), 0.)

    def test_success_is_one(self):
        self.assertEqual(S.location_progress(True, True, 0, 0), 1.)

    def test_partial_credit(self):
        self.assertAlmostEqual(S.location_progress(True, False, 0, 0), .7)        # read, scanner not back
        self.assertAlmostEqual(S.location_progress(True, True, 1, 0), .75)        # another slot's label read first
        self.assertEqual(S.location_progress(False, True, 1, 0), 0.)              # only the wrong label
        self.assertAlmostEqual(S.location_progress(True, True, 0, 1), .5)         # something knocked onto the floor


class TallyTest(unittest.TestCase):
    ITEMS = {"a", "b", "c"}

    def reads(self):
        return [read("a_code", item="a"), read("b_code", item="b"), read("a_code", item="a"),
                read("pack_inner_0_0", kind="inner", item="b"), read("c_code", item="c", misread=True),
                read("loc_A03-3-B", kind="location", fmt="Code128")]

    def test_first_reads_and_errors(self):
        done, errors = S.tally(self.reads(), self.ITEMS, "product", lambda r: r.get("kind") == "inner")
        self.assertEqual(done, {"a", "b"})
        self.assertEqual(sorted(e["error"] for e in errors), ["misread", "not an item", "repeat", "wrong"])

    def test_order_of_reads_changes_nothing(self):
        results = set()
        for perm in itertools.permutations(self.reads()):
            done, errors = S.tally(list(perm), self.ITEMS, "product", lambda r: r.get("kind") == "inner")
            results.add((frozenset(done), len(errors)))
        self.assertEqual(len(results), 1)

    def test_isbn_is_an_error_for_the_library(self):
        reads = [read("book_0_library", kind="library", item="book_0", fmt="Codabar"),
                 read("book_1_code", item="book_1", fmt="EAN13")]
        done, errors = S.tally(reads, {"book_0", "book_1"}, "library", lambda r: r["format"] in ("EAN13", "UPCA"))
        self.assertEqual((done, [e["error"] for e in errors]), ({"book_0"}, ["wrong"]))


class CheckoutProgressTest(unittest.TestCase):
    N = 5

    def progress(self, rung, bagged, errors=0, on_floor=0, returned=True, case=None):
        items = [f"i{k}" for k in range(self.N)]
        credits = [S.put_away_credit(i in rung, i in bagged) for i in items]
        if case is not None:
            credits.append(case)
        unpaid = len(set(bagged) - set(rung))
        return S.item_progress(credits, errors + S.UNPAID_COST*unpaid, on_floor, returned)

    def test_untouched_start_is_zero(self):
        self.assertEqual(self.progress(set(), set()), 0.)
        self.assertEqual(self.progress(set(), set(), case=0.), 0.)

    def test_success_is_one(self):
        every = {f"i{k}" for k in range(self.N)}
        self.assertEqual(self.progress(every, every), 1.)
        self.assertEqual(self.progress(every, every, case=S.lift_credit(0.)), 1.)

    def test_partial_credit(self):
        every = {f"i{k}" for k in range(self.N)}
        self.assertAlmostEqual(self.progress(every, set()), .6)                   # rung up, left in the basket
        self.assertAlmostEqual(self.progress({"i0", "i1"}, {"i0"}), (1. + .6)/self.N)
        self.assertAlmostEqual(self.progress(every, every, errors=1), (self.N - 1)/self.N)     # one item rung twice
        self.assertAlmostEqual(self.progress(every - {"i0"}, every), (self.N - 1 - .5)/self.N)  # bagged unpaid
        self.assertAlmostEqual(self.progress(every, every, on_floor=1), (self.N - .5)/self.N)
        self.assertAlmostEqual(self.progress(every, every, returned=False), .9)

    def test_case_credit_falls_with_the_case_s_rise(self):
        self.assertEqual(S.lift_credit(0.), 1.)
        self.assertEqual(S.lift_credit(.010), 1.)
        self.assertAlmostEqual(S.lift_credit(.030), .5)
        self.assertEqual(S.lift_credit(.050), 0.)
        self.assertEqual(S.lift_credit(.200), 0.)
        every = {f"i{k}" for k in range(self.N)}
        self.assertAlmostEqual(self.progress(every, every, case=0.), self.N/(self.N + 1))      # case never scanned


class LibraryProgressTest(unittest.TestCase):
    def test_untouched_checked_in_binned(self):
        books = ["b0", "b1", "b2"]
        p = lambda checked, binned, errors=0: S.item_progress(
            [S.put_away_credit(b in checked, b in binned) for b in books], errors, 0, True)
        self.assertEqual(p(set(), set()), 0.)
        self.assertEqual(p(set(books), set(books)), 1.)
        self.assertAlmostEqual(p(set(books), set()), .6)                          # checked in, left on the desk
        self.assertAlmostEqual(p({"b0"}, {"b0"}), 1/3)
        self.assertAlmostEqual(p(set(books), set(books), errors=1), 2/3)           # an ISBN read as well
        self.assertEqual(p(set(), set(books)), 0.)                                 # binned without check-in


class CountKeyQuantityProgressTest(unittest.TestCase):
    TRUTH = {"g1": 6, "g2": 4}

    def progress(self, scanned, keyed, errors=0, disturbed=0, returned=True):
        credits = [S.quantity_credit(g in scanned, keyed.get(g, 0), n) for g, n in self.TRUTH.items()]
        return S.item_progress(credits, errors, disturbed, returned)

    def test_untouched_start_is_zero(self):
        self.assertEqual(self.progress(set(), {}), 0.)

    def test_success_is_one(self):
        self.assertEqual(self.progress({"g1", "g2"}, dict(self.TRUTH)), 1.)

    def test_partial_credit(self):
        self.assertAlmostEqual(self.progress({"g1"}, {}), .3/2)                    # scanned, nothing keyed
        self.assertAlmostEqual(self.progress({"g1"}, {"g1": 6}), 1/2)              # one product complete
        self.assertAlmostEqual(self.progress({"g1", "g2"}, {"g1": 5, "g2": 4}), (.3 + .7*5/6 + 1)/2)  # off by one
        self.assertAlmostEqual(self.progress({"g1", "g2"}, {"g1": 12, "g2": 4}), (.3 + 1)/2)  # off by the whole count
        self.assertAlmostEqual(self.progress({"g1", "g2"}, dict(self.TRUTH), errors=1), 1/2)   # keyed elsewhere too
        self.assertAlmostEqual(self.progress({"g1", "g2"}, dict(self.TRUTH), disturbed=1), .75)
        self.assertAlmostEqual(self.progress({"g1", "g2"}, dict(self.TRUTH), returned=False), .9)


class KeyBookingsTest(unittest.TestCase):
    """The replay of the host's entries must reproduce the host's own record (device.HostApp)."""
    LOCATIONS = ("A03-3-A", "A03-3-B", "A03-3-C")
    PRODUCTS = ("078412003512", "085314002263", "072941001584")
    KEYS = ("1", "2", "3", "4", "5", "6", "9", "0", ".", "-", "ENTER")

    def run_host(self, events):
        from barcode_scanning.device import HostApp
        host = HostApp("count-key-quantity", {}, self.LOCATIONS)
        reads = []
        for t, (kind, value) in enumerate(events):
            if kind == "scan":
                fmt = "Code128" if value in self.LOCATIONS else "UPCA"
                host.on_scan(dict(format=fmt, text=value, t_report=float(t)))
                reads.append(dict(label=value, misread=t % 7 == 3))
            else:
                host.on_key(value, float(t))
        return host, reads

    def test_replay_matches_the_host_on_random_sequences(self):
        rng = random.Random(0)
        for trial in range(400):
            events = [("scan", rng.choice(self.LOCATIONS + self.PRODUCTS)) if rng.random() < .35 else
                      ("key", rng.choice(self.KEYS)) for _ in range(rng.randint(0, 40))]
            host, reads = self.run_host(events)
            scans, record = S.key_bookings(host.record["entries"], reads, self.LOCATIONS)
            with self.subTest(trial=trial):
                self.assertEqual({loc: {c: v[0] for c, v in d.items()} for loc, d in record.items()},
                                 host.record["location_counts"])
                for loc, d in record.items():                   # each quantity carries the read it was keyed for
                    for code, (_, r) in d.items():
                        self.assertEqual(r["label"].zfill(14), code)
                self.assertTrue(all(loc in self.LOCATIONS and r["label"] in self.PRODUCTS for loc, _, r in scans))

    def test_quantity_belongs_to_the_scan_it_was_keyed_for(self):
        events = [("scan", "078412003512"), ("key", "5"), ("key", "ENTER"),           # refused: no location yet
                  ("scan", "A03-3-B"), ("scan", "078412003512"), ("key", "6"), ("key", "ENTER"),
                  ("scan", "085314002263"), ("key", "4"), ("scan", "A03-3-A"), ("key", "ENTER"),
                  ("scan", "078412003512"), ("key", "7"), ("key", "-"), ("key", "6"), ("key", "ENTER")]
        host, reads = self.run_host(events)
        scans, record = S.key_bookings(host.record["entries"], reads, self.LOCATIONS)
        g = lambda code: code.zfill(14)
        self.assertEqual([(loc, code) for loc, code, _ in scans], [("A03-3-B", g("078412003512")),
                                                                    ("A03-3-B", g("085314002263")),
                                                                    ("A03-3-A", g("078412003512"))])
        # the 4 keyed after the location changed still belongs to the item scanned under A03-3-B
        self.assertEqual({loc: {c: v[0] for c, v in d.items()} for loc, d in record.items()},
                         {"A03-3-B": {g("078412003512"): 6, g("085314002263"): 4}, "A03-3-A": {g("078412003512"): 6}})
        self.assertEqual(record["A03-3-B"][g("085314002263")][1], reads[3])

    def test_entries_and_reads_must_agree(self):
        host, reads = self.run_host([("scan", "A03-3-B"), ("scan", "078412003512")])
        with self.assertRaises(ValueError):
            S.key_bookings(host.record["entries"], reads[:1], self.LOCATIONS)


if __name__ == "__main__":
    unittest.main()
