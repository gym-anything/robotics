"""The scanner's firmware and the host application it reports to.

Firmware (DS2208 Product Reference Guide), evaluated on the physics timeline:
  DOCKED        seat sensor true: hands-free (presentation) mode, frames while the presentation zone changes,
                same symbol refused until out of view 0.5 s, different symbol after 0.1 s, session 0.25-2.5 s
  LIFTED_IDLE   out of the cradle, not moving for 2 s: aim off
  AIM           moved ("Auto Aim" default): aim line on
  SESSION       trigger switch closed: illumination on, frames at 30 Hz, aim off during each exposure,
                decode every frame; a read, a trigger release or the 9.9 s timeout ends the session
  WAIT_RELEASE  after a read / timeout: illumination off until the trigger opens (one read per pull)
The frames themselves are rendered and decoded elsewhere (env.py -> render.py -> optics.py -> decoder.py);
this module only decides WHEN frames are taken and what a decode does. It never decodes from geometry.

Host app: the inventory / POS / library application the scanner feeds (a keyboard-wedge host). Its record is
what the task evaluator judges; its screen is rendered on the cart's display for the policy cameras.
"""

import math
from pathlib import Path
import numpy as np
from . import params as P


class Firmware:
    def __init__(self, seated=False):
        # power-on reads the stand's seat sensor (a scanner that starts in its cradle is DOCKED from the first tick;
        # it reported LIFTED_IDLE for one tick before)
        self.state = "DOCKED" if seated else "LIFTED_IDLE"
        self.session_start = None
        self.last_motion = -1e9
        self.last_frame_t = -1e9
        self.next_frame_t = None
        self.good_read_until = -1e9
        self.reads = []                 # decoded reports (dicts)
        self.frame_log = []             # every acquired frame (metadata)
        self.seen = {}                  # hands-free: symbol -> last time seen in a frame
        self.last_found = set()         # hands-free: symbols decodable in the latest frame
        self.last_report = (None, -1e9)
        self.handsfree_active_until = -1e9

    # ---------------------------------------------------------------- per physics tick
    def tick(self, t, trigger_closed, seated, moving, zone_change):
        """Advance the state machine one physics tick. Returns True when a frame starts at this tick."""
        if moving:
            self.last_motion = t
        if seated:
            if self.state != "DOCKED":
                self.state = "DOCKED"
            if zone_change:
                self.handsfree_active_until = max(self.handsfree_active_until, t + P.HANDSFREE_SESSION_MIN_S)
            if t <= self.handsfree_active_until:
                return self._frame_due(t)
            # between sessions nothing in the zone has changed, so what the last frame saw is still in view: a symbol
            # left stationary has not been "out of the field of view for the timeout period" and is not
            # re-read when it next moves (DS2208 PRG p. 5-22; a can held still for 3 s was otherwise re-read as soon as
            # it moved)
            for key in self.last_found:
                if key in self.seen:
                    self.seen[key] = t
            return False
        if self.state == "DOCKED":
            self.state = "AIM"
        if self.state == "WAIT_RELEASE":
            if not trigger_closed:
                self.state = "AIM"
            return False
        if trigger_closed:
            if self.state != "SESSION":
                self.state, self.session_start = "SESSION", t
                self.next_frame_t = None
            if t - self.session_start > P.DECODE_SESSION_TIMEOUT_S:
                self.state = "WAIT_RELEASE"
                return False
            return self._frame_due(t)
        if self.state == "SESSION":
            self.state = "AIM"
        self.state = "AIM" if t - self.last_motion < P.AIM_IDLE_OFF_S else "LIFTED_IDLE"
        return False

    def _frame_due(self, t):
        """Frames on a fixed 1/30 s grid from the start of acquisition (restarting the interval at the next whole
        physics tick would run the imager at 29.4 Hz)."""
        period = 1/P.FRAME_RATE_HZ
        if self.next_frame_t is None or t - self.next_frame_t > period:
            self.next_frame_t = t
        if t >= self.next_frame_t - 1e-9:
            self.next_frame_t += period
            self.last_frame_t = t
            return True
        return False

    # ---------------------------------------------------------------- light state
    def illumination_on(self, t):
        """Illumination LEDs lit: a trigger session, or a hands-free session in the cradle (motion in the zone)."""
        return self.state == "SESSION" or (self.state == "DOCKED" and t <= self.handsfree_active_until)

    def led_intensity(self, t):
        """Renderer intensity of each illumination LED. Hands-free exposures are capped 5.8x shorter (0.22 vs
        1.27 ms, params.py), so the presentation drive is 5.8x brighter for the same signal."""
        if not self.illumination_on(t):
            return 0.
        return P.LED_INTENSITY*(P.T_MAX_HANDHELD_S/P.T_MAX_HANDSFREE_S if self.state == "DOCKED" else 1.)

    def aim_on(self, t):
        return self.state in ("AIM", "SESSION")

    def good_read_led(self, t):
        return t < self.good_read_until

    # ---------------------------------------------------------------- decode outcome of one frame
    def on_frame_result(self, t_frame, result, symbols_in_view):
        """Apply a frame's decode (the chosen symbol or None). Returns the report dict or None."""
        docked = self.state == "DOCKED"
        if result is None:
            return None
        key = (result["format"], result["text"])
        if docked:
            # `seen` holds REPORTED symbols only, refreshed while they stay in view: a reported symbol is not re-reported
            # until it has been out of view for the same-symbol timeout; a different symbol waits only for the
            # different-symbol timeout after the last report (a refused symbol is not marked seen: otherwise a
            # second item held in view right after a read could never be read)
            last = self.seen.get(key)
            if last is not None and t_frame - last < P.SAME_SYMBOL_TIMEOUT_S:
                self.seen[key] = t_frame
                return None
            prev_key, prev_t = self.last_report
            if prev_key is not None and prev_key != key and t_frame - prev_t < P.DIFFERENT_SYMBOL_TIMEOUT_S:
                return None
            self.seen[key] = t_frame
            self.handsfree_active_until = t_frame + P.HANDSFREE_SESSION_MIN_S
        else:
            if self.state != "SESSION":
                return None
            self.state = "WAIT_RELEASE"
        rep = dict(result, t_frame=t_frame, t_report=t_frame + P.DECODE_LATENCY_S, mode="handsfree" if docked else "handheld")
        self.reads.append(rep)
        self.last_report = (key, t_frame)
        self.good_read_until = t_frame + P.GOOD_READ_LED_S
        return rep

    def note_seen(self, t_frame, found):
        """Hands-free bookkeeping: symbols decodable in this frame (even if not reported) refresh their timers."""
        self.last_found = {(f["format"], f["text"]) for f in found}
        for key in self.last_found:
            if key in self.seen:
                self.seen[key] = t_frame


class HostApp:
    """Keyboard-wedge host: collects scans and keypad entries into the task record and draws its screen."""

    def __init__(self, task, catalog_names, location_codes=()):
        self.task = task
        self.names = catalog_names
        self.locations = set(location_codes)
        self.record = {"location": None, "counts": {}, "location_counts": {}, "entries": [], "errors": [], "checked_in": [],
                       "sale": []}
        self.pending_location = None
        self.pending_item = None
        self.keybuf = ""
        self.last = None
        first = {"checkout-present": "Present items", "checkout-bulky": "Present items", "checkout-counter": "Present items",
                 "library-checkin": "Scan item barcode",
                 "hold-scanner": "Scanner ready"}.get(task, "Scan location label")
        self.screen = {"title": "READY", "lines": [first]}
        self._render_seq = 0

    def on_scan(self, rep):
        from .decoder import normalise
        code = normalise(rep["format"], rep["text"])
        self.last = code
        r = self.record
        entry = dict(t=rep["t_report"], kind="scan", format=rep["format"], data=code, raw=rep["text"])
        r["entries"].append(entry)
        if self.task in ("count-by-scanning", "count-key-quantity", "scan-location", "hold-scanner"):
            if rep["format"] == "Code128" and code in self.locations:
                r["location"] = code
                rep["sets_location"] = code                   # the read made this the active location
                self.screen = {"title": f"LOCATION {code}", "lines": ["Scan item" if self.task != "scan-location" else "Location recorded"]}
                return
            if r["location"] is None:
                r["errors"].append(dict(entry, error="scan location first"))
                self.screen = {"title": "ERROR", "lines": ["Scan location first", code]}
                return
            if self.task == "count-key-quantity":
                self.pending_item, self.pending_location = code, r["location"]
                self.keybuf = ""
                self.screen = {"title": f"LOC {r['location']}", "lines": [self.names.get(code, code), "QTY: _", "key count, ENTER"]}
                return
            r["counts"][code] = r["counts"].get(code, 0) + 1
            loc = r["location_counts"].setdefault(r["location"], {})
            loc[code] = loc.get(code, 0) + 1
            rep["booked_location"] = r["location"]        # the read's booking, judged per read by count-by-scanning
            self.screen = {"title": f"LOC {r['location']}", "lines": [self.names.get(code, code), f"COUNT {r['counts'][code]}",
                                                                       f"UNITS {sum(r['counts'].values())}"]}
        elif self.task in ("checkout-present", "checkout-bulky", "checkout-counter"):
            r["sale"].append(code)
            self.screen = {"title": f"ITEMS {len(r['sale'])}", "lines": [self.names.get(code, code)]}
        elif self.task == "library-checkin":
            r["checked_in"].append(code)
            known = code in self.names
            self.screen = {"title": "CHECK-IN" if known else "ITEM NOT FOUND", "lines": [self.names.get(code, code)]}
            if not known:
                r["errors"].append(dict(entry, error="not a library item"))

    def on_key(self, key, t):
        r = self.record
        r["entries"].append(dict(t=t, kind="key", key=key))
        if self.task != "count-key-quantity" or self.pending_item is None:
            return
        if key.isdigit():
            self.keybuf = (self.keybuf + key)[:4]
        elif key in (".", "-"):
            self.keybuf = self.keybuf[:-1]
        elif key == "ENTER" and self.keybuf:
            qty = int(self.keybuf)
            r["counts"][self.pending_item] = qty
            r["location_counts"].setdefault(self.pending_location, {})[self.pending_item] = qty
            self.screen = {"title": f"LOC {r['location']}", "lines": [self.names.get(self.pending_item, self.pending_item),
                                                                       f"QTY {qty} SAVED", "Scan next item"]}
            self.pending_item, self.keybuf = None, ""
            return
        if self.pending_item is not None:
            self.screen["lines"] = [self.names.get(self.pending_item, self.pending_item), f"QTY: {self.keybuf or '_'}",
                                    "key count, ENTER"]

    @staticmethod
    def draw_screen(screen, path):
        """Render the app screen (200 x 130 mm at 4 px/mm) to a PNG: large text for the head camera."""
        from PIL import Image, ImageDraw
        from .artwork import _font, FONT_BOLD, FONT_SANS
        im = Image.new("RGB", (800, 520), (8, 22, 40))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 800, 96], fill=(20, 90, 160))
        d.text((24, 48), str(screen.get("title", ""))[:22], font=_font(FONT_BOLD, 60), fill=(255, 255, 255), anchor="lm")
        for k, line in enumerate(screen.get("lines", [])[:4]):
            d.text((24, 140 + k*92), str(line)[:26], font=_font(FONT_BOLD if k == 0 else FONT_SANS, 56 if k else 50),
                   fill=(235, 240, 245), anchor="lm")
        im.save(path)
        return path
