"""Task instructions and the outcome evaluator.

The evaluator reads only the simulated world: where every loose piece is and what it weighs, the poses of tools,
containers and furniture, and the one-second rest window. It never reads actions, robot commands or task flags, and a
piece counts for a place only by being there: in a container means inside its walls below its rim; in the taped
square means resting on the floor inside the tape. Thresholds are authored and listed in THRESHOLDS.

closing-sweep has criteria of its own: what was swept up counts only in the barrel while it stands upright (the pencil
is not counted in the mass and is put away in the barrel or lying on a table, desk or counter top), every crumpled paper
and half of the paper scraps go in the barrel, a table's identical chairs each stand at one of its places (where one of
them started, pushed in at the table, or the way between), the barrel may stand anywhere, the tools are put away where
they belong and not held in a robot hand, and fine dust left in a dustpan or kept in a broom is not yet put away.
Its score is a reward contract (REWARD_CONTRACTS): 1 on success, else 0.9 x progress, where progress (Evaluator.progress)
measures the final state's work product whatever route made it; what lies in a dustpan counts at a reduced rate. The
other tasks score the share of their criteria.
"""

import itertools
import math
import numpy as np
from scipy.spatial.transform import Rotation
from classroom_arrangement import params as CP
from . import params as P
from . import layout as L
from . import toolkit as T

THRESHOLDS = dict(
    gathered={"gather-open": 0.90, "corner-and-edge": 0.90, "under-table": 0.85},
    scraps=0.80, scattered=0.02, edges=0.03, edge_band=0.10,
    contained={"pan-and-barrel": 0.95, "heavy-spill": 0.97, "closing-sweep": 0.90, "broken-mug": 0.98, "table-scraps": 0.90},
    beads=57, floor_left_g={"pan-and-barrel": 1.5, "broken-mug": 1.5, "heavy-spill": 4.0}, spilled=0.01, spill_radius=0.5,
    chair=(0.10, 15.0), table=0.03, supply=0.10, container=0.5, hang=0.03, pan_home=0.30,
    rest_speed=0.005, rest_motion=0.002, on_floor_z=0.050,
    dust_left=0.5,             # a broom is not asked for a dust-free floor: half of the film may stay
    # closing-sweep: tables and the desk nudged while swept under; containers' tilt; a hand (fingers, hand, link 7)
    # pressing on a tool with more than touch_n holds it; a piece "on a top" lies within top_band of a table's, the
    # desk's or the counter's top; a counter tool's body origin within counter_band of the counter top, over it; the
    # share of the paper scraps, by count, in the barrel; at most lodged_g of what bristles hold counts as put away; a
    # chair's place reaches push_in past its tucked spot toward the table, where its back meets the top; the
    # lobby dustpan stands anywhere along the floor under the rail, within pan_home of the line at its laid distance
    # from the wall
    closing=dict(table=0.10, upright_deg=10.0, touch_n=1.0, top_band=(-0.010, 0.012), counter_band=(-0.020, 0.200), scraps=0.5,
                 lodged_g=0.05, push_in=0.32))
CONTAINER_OF = {"pan-and-barrel": "barrel", "heavy-spill": "barrel", "closing-sweep": "barrel", "broken-mug": "pail",
                "table-scraps": "wastebasket", "beads": "pail"}
# closing-sweep's score: 1 on success, else FAIL_SCORE_SCALE x progress, so every failure ranks below every success.
# progress = clip(debris D + litter L + dust U - penalty, 0, 1) of the final state, weights in PROGRESS. Swept up is
# what lies in the barrel while its mouth faces up (tilt under open_deg); what lies in a dustpan counts at the rate
# in_pan, as swept up and not yet emptied. D the share of the counted mass swept up, U the share of the start's fine
# dust swept up (bristles' dust up to the verifier's lodged_g), each over its success threshold and capped at 1; L the
# mean of (the crumpled papers swept up, the scraps swept up over their success share, the pencil put away). The penalty:
# per_item for each tool not put away and for each piece of furniture or container outside what is accepted (graded over
# graded_m or graded_deg beyond its tolerance), capped at cap, and per_item more for each one tipped over, not capped.
# An untouched room scores 0.
REWARD_CONTRACTS = {"closing-sweep": "progress-v1"}
FAIL_SCORE_SCALE = 0.9
PROGRESS = dict(debris=0.6, litter=0.1, dust=0.3, in_pan=0.5, open_deg=60.0, per_item=0.03, graded_m=0.40, graded_deg=45.0, cap=0.20)
PANS = {"lobby_pan": (P.LOBBY_PAN["pan"], 0.025), "hand_pan": (P.HAND_PAN["pan"], 0.022)}     # (width, depth, height), lip ramp (toolkit)
PAN_LINE = ((L.X1 - 0.30, P.RAIL["y"][0]), (L.X1 - 0.30, P.RAIL["y"][1]))   # closing-sweep: under the rail, at the pan's laid distance


def seg_dist(p, a, b):
    """Distance from point p to the segment a-b (2-D)."""
    p, a, b = (np.asarray(v, float)[:2] for v in (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab)/max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t*ab)))


def instruction(config, spec):
    """The task's text as a session serves it. A variant that takes a tool away gets a text without that tool."""
    task, v = spec["task"], spec.get("variant", "default")
    text = L.TASKS[task]["instruction"]
    if v == "push-broom-only":
        text += " Only the push broom is on the rail."
    elif v == "hand-pan":
        text += " The lobby dustpan is away: use the hand dustpan from the back counter instead."
        if task == "pan-and-barrel":
            text = ("A pile of rice, grit and cereal has been swept into the taped square. The lobby dustpan is away: pick the pile up "
                    "with the hand dustpan from the back counter, empty it into the grey barrel, and hang the brooms back on the rail. "
                    "The hand dustpan may be left anywhere.")
    elif v == "dusty":
        text += " Fine dust lies over the same floor: at least half of it has to leave the floor too."
    elif v == "one-arm":
        text = ("The craft table is covered in paper scraps, eraser crumbs and beads. There is no dustpan: with the counter brush, brush "
                "them over the table's end straight into the wastebasket that stands against it. Leave the pencil cup, the paper, the "
                "scissors and the glue where they are.")
    if task == "pan-and-barrel":
        text += " Fine dust lies around the pile: at least half of it has to leave the floor too."
    return text


class Evaluator:
    def __init__(self, env):
        self.env = env
        self.spec = env.spec
        self.task = self.spec["task"]
        w = env.world
        self.start = {n: w.pose2d(n) for n in w.scene.bodies}
        self.start_debris = env.start_debris
        d = self.start_debris
        self.mass = d["mass"]
        self.total = float(self.mass.sum())
        self.cls = np.array([w.debris_classes[k].name for k in d["cls"]])
        self.tag = np.array(d["tag"])
        self.containers = {n: t["spec"]["inner"] for n, t in w.scene.tools.items() if t["spec"].get("inner")}
        self.hang = {n: np.asarray(L.hang_pose(n)[0], float) for n, t in self.spec["tools"].items() if t.get("hung")}
        # closing-sweep: each student chair's place pushed in at its own table (as the room's other chairs stand) with
        # that table's laid-out pose; the robot's bodies and each tool's bodies (a pan's handle and wheels too)
        lay = L.furniture()
        self.tucked = {n: (it["table"], lay[n]["pose"], lay[it["table"]]["pose"]) for n, it in self.spec["items"].items() if it.get("table")}
        self.robot = set(w.scene.robot_bodies)
        self.hands = {i for n, i in w.scene.robot_body.items() if n.endswith(("_fr3_link7", "_fr3_hand", "_fr3_leftfinger", "_fr3_rightfinger"))}
        self.tool_bodies = {n: {t["body"], t.get("handle_body"), *t.get("wheel_bodies", ())} - {None} for n, t in w.scene.tools.items()}

    # ------------------------------------------------------------------ where pieces are
    def inside(self, name, pos):
        """Mask of pieces inside container `name`: within its inner walls and between its bottom and its rim, in the
        container's own frame (so a tipped container holds only what is still in it)."""
        w = self.env.world
        p, R = w.body_pose(self.env.world.scene.tools[name]["body"])
        loc = (pos - p) @ R
        inner = self.containers[name]
        if inner["shape"] == "round":
            ok = np.hypot(loc[:, 0], loc[:, 1]) < inner.get("r_corner", inner["r"])
        else:
            ok = (np.abs(loc[:, 0]) < inner["hx"]) & (np.abs(loc[:, 1]) < inner["hy"])
        return ok & (loc[:, 2] > 0.0) & (loc[:, 2] < inner["h"])

    def in_square(self, pos):
        cx, cy = self.spec["square"]["centre"]
        h = self.spec["square"]["side"]/2
        return (np.abs(pos[:, 0] - cx) <= h) & (np.abs(pos[:, 1] - cy) <= h) & (pos[:, 2] < THRESHOLDS["on_floor_z"] + 0.05)

    def on_floor(self, pos, contained):
        return (pos[:, 2] < THRESHOLDS["on_floor_z"]) & ~contained

    def under(self, name, pos):
        """Mask of pieces on the floor under a table's or desk's top."""
        (x, y, yaw), _ = self.env.world.pose2d(name)
        kind = self.spec["items"][name]["kind"]
        hx, hy = (CP.TABLE_TOP[0]/2, CP.TABLE_TOP[1]/2) if kind == "table" else (CP.TEACHER_TOP[0]/2, CP.TEACHER_TOP[1]/2)
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = pos[:, 0] - x, pos[:, 1] - y
        return (np.abs(c*dx + s*dy) < hx) & (np.abs(-s*dx + c*dy) < hy) & (pos[:, 2] < 0.2)

    def on_counter(self, pos, band):
        """Mask of points over the back counter's top and within `band` (below, above) of its height."""
        x0, x1 = P.CASEWORK["x"]
        top = P.CASEWORK["top"]
        return ((pos[:, 0] > x0 - 0.0125) & (pos[:, 0] < x1 + 0.0125) & (pos[:, 1] > L.Y1 - P.CASEWORK["depth"] - 0.025)
                & (pos[:, 1] < L.Y1) & (pos[:, 2] > top + band[0]) & (pos[:, 2] < top + band[1]))

    def on_top(self, pos):
        """Mask of pieces lying on a table's or the teacher's desk's top, or on the back counter."""
        w, band = self.env.world, THRESHOLDS["closing"]["top_band"]
        ok = self.on_counter(pos, band)
        for n, it in self.spec["items"].items():
            if it["kind"] not in ("table", "teacher_desk"):
                continue
            (x, y, yaw), _ = w.pose2d(n)
            if it["kind"] == "table":
                hx, hy, top = CP.TABLE_TOP[0]/2, CP.TABLE_TOP[1]/2, L.TABLE_H
            else:
                hx, hy, top = CP.TEACHER_TOP[0]/2, CP.TEACHER_TOP[1]/2, CP.TEACHER_HEIGHT
            c, s = math.cos(yaw), math.sin(yaw)
            dx, dy = pos[:, 0] - x, pos[:, 1] - y
            ok |= ((np.abs(c*dx + s*dy) < hx) & (np.abs(-s*dx + c*dy) < hy) & (pos[:, 2] > top + band[0])
                   & (pos[:, 2] < top + band[1]))
        return ok

    def chair_off(self, name, xy, yaw):
        """How far a chair stands from where it counts as put back: the line from where it started to its place pushed
        in at its own table (at that table's present pose) and on, push_in further toward the table, where its back
        meets the top, and the nearer of the start's and the table's yaws. A chair
        without a table (the teacher's) has only its start."""
        (sx, sy, syaw), _ = self.start[name]
        pts = [np.array((sx, sy))]
        yaws = [syaw]
        if name in self.tucked:
            table, (cx, cy, cyaw), (lx, ly, lyaw) = self.tucked[name]
            (tx, ty, tyaw), _ = self.env.world.pose2d(table)
            d = tyaw - lyaw
            c, s = math.cos(d), math.sin(d)
            rot = lambda u, v: np.array((c*u - s*v, s*u + c*v))
            tucked = np.array((tx, ty)) + rot(cx - lx, cy - ly)
            inward = rot(0.0, -math.copysign(1.0, cy - ly))
            pts += [tucked, tucked + THRESHOLDS["closing"]["push_in"]*inward]
            yaws.append(cyaw + d)
        return min(seg_dist(xy, a, b) for a, b in zip(pts, pts[1:] or pts)), \
            min(abs((yaw - y + math.pi) % (2*math.pi) - math.pi) for y in yaws)

    def in_pans(self, pos):
        """Mask of pieces resting in a dustpan whose mouth faces up: on its floor slab behind the lip's ramp, inside its
        walls, in the pan's own frame."""
        w = self.env.world
        ok = np.zeros(len(pos), bool)
        for n, ((wdt, depth, h), ramp) in PANS.items():
            if n not in w.scene.tools:
                continue
            p, R = w.tool_pose(n)
            if R[2, 2] < math.cos(math.radians(PROGRESS["open_deg"])):
                continue
            loc = (pos - p) @ R
            ok |= ((np.abs(loc[:, 0]) < wdt/2) & (loc[:, 1] > -depth/2 - 0.002) & (loc[:, 1] < depth/2 - ramp) & (loc[:, 2] > 0.001)
                   & (loc[:, 2] < h + 0.03))
        return ok

    def held(self):
        """Names of the tools a robot hand presses on (fingers, hand or link 7; normal force above touch_n, the rigid
        world's contact table): a tool still in the hand is not put away. A tool at its place that the base or an arm
        merely rests against is not held."""
        w, lim = self.env.world, THRESHOLDS["closing"]["touch_n"]
        touched = {b if a in self.hands else a for a, b, f, _, _ in w.contact_table()
                   if f > lim and ((a in self.hands) != (b in self.hands)) and not (a in self.robot and b in self.robot)}
        return sorted(n for n, bodies in self.tool_bodies.items() if bodies & touched)

    def progress(self, parts, out_of_place, tipped):
        """closing-sweep's progress (REWARD_CONTRACTS, PROGRESS): a function of the final state only. `parts` holds what
        is swept up (in the barrel while its mouth faces up) and what lies in a dustpan."""
        th, W, h = THRESHOLDS, PROGRESS, PROGRESS["in_pan"]
        q = parts
        # each term is (1 - h) T(swept up) + h T(swept up + in dustpans): a dustpan's contents are worth half of what
        # emptying them adds, so emptying is always worth something
        mix = lambda T, binned, panned: (1.0 - h)*T(binned) + h*T(binned + panned)
        debris = mix(lambda x: min(1.0, x/th["contained"]["closing-sweep"]), q["mass_binned"], q["mass_in_pans"])
        wads = mix(lambda k: k/max(q["wads"], 1), q["wads_binned"], q["wads_in_pans"])
        scraps = mix(lambda k: min(1.0, k/max(th["closing"]["scraps"]*q["scraps"], 1e-9)), q["scraps_binned"], q["scraps_in_pans"]) if q["scraps"] else 1.0
        pencil = {"put away": 1.0, "in a dustpan": h}.get(q["pencil"], 0.0)
        litter = float(np.mean([wads, scraps, pencil]))
        dust = mix(lambda g: min(1.0, g/((1.0 - th["dust_left"])*max(q["dust_start"], 1e-12))), q["dust_binned"] + q["dust_lodged_credit"],
                   q["dust_in_pans"])
        graded = sum(v for k, v in out_of_place.items() if k not in tipped)
        penalty = min(W["cap"], W["per_item"]*float(graded)) + W["per_item"]*len(tipped)
        p = float(np.clip(W["debris"]*debris + W["litter"]*litter + W["dust"]*dust - penalty, 0.0, 1.0))
        return dict(progress=round(p, 4), debris=round(debris, 4), litter=round(litter, 4), dust=round(dust, 4), penalty=round(penalty, 4),
                    out_of_place={k: round(v, 3) for k, v in out_of_place.items() if v > 0}, tipped=sorted(tipped),
                    swept_up={(k.replace("dust_", "dust_g_") if k.startswith("dust_") else k):
                              (round(1e3*v, 4) if k.startswith("dust_") else round(v, 4) if isinstance(v, float) else v) for k, v in q.items()})

    # ------------------------------------------------------------------ evaluation
    def evaluate(self):
        env, w, th, task = self.env, self.env.world, THRESHOLDS, self.task
        cs = task == "closing-sweep"
        d = w.debris()
        pos, m = d["pos"], self.mass
        total = max(self.total, 1e-12)
        motion, window = env.window_motion()
        crit = {}
        cont = {n: self.inside(n, pos) for n in self.containers}
        out_of_place, tipped = {}, set()      # closing-sweep: what the progress penalty reads (0 accepted .. 1), what fell over
        if cs:
            # a barrel that does not stand holds nothing: upside down over the pile, every piece lay between its bottom
            # and its rim in its own frame. Progress reads what is still in it while its
            # mouth faces up (the dust model's own rule, dust.py), so a barrel leaning on something is not a cliff
            tilt_b = w.pose2d("barrel")[1]
            upright, opened = bool(tilt_b <= th["closing"]["upright_deg"]), bool(tilt_b < PROGRESS["open_deg"])
            binned = cont["barrel"] & opened
            cont["barrel"] = cont["barrel"] & upright
        any_cont = np.zeros(len(m), bool)
        for v in cont.values():
            any_cont |= v
        sq = self.in_square(pos) & ~any_cont
        floor = self.on_floor(pos, any_cont)
        frac = lambda mask: float(m[mask].sum()/total)
        zone = self.spec["zone"]
        in_zone = (pos[:, 0] > zone[0] - 0.3) & (pos[:, 0] < zone[2] + 0.3) & (pos[:, 1] > zone[1] - 0.3) & (pos[:, 1] < zone[3] + 0.3)
        target = CONTAINER_OF.get(task)
        if task in th["gathered"]:
            got = sq | cont.get("barrel", np.zeros(len(m), bool))
            crit["gathered"] = dict(check="mass in the taped square or the barrel", ok=frac(got) >= th["gathered"][task],
                                    fraction=round(frac(got), 4), need=th["gathered"][task])
            if task in ("gather-open", "under-table"):
                wad, scr = self.cls == "wad", self.cls == "scrap"
                okw = bool(got[wad].all()) if wad.any() else True
                fs = float(got[scr].mean()) if scr.any() else 1.0
                crit["wads_and_scraps"] = dict(check="every wad and 80 % of the scraps in the square or the barrel",
                                               ok=okw and fs >= th["scraps"], wads_in=int(got[wad].sum()), wads=int(wad.sum()),
                                               scraps_fraction=round(fs, 3))
            out = ~(in_zone | got)
            crit["not_scattered"] = dict(check="mass outside the task's floor and its target", ok=frac(out) <= th["scattered"],
                                         fraction=round(frac(out), 4), allowed=th["scattered"])
        if task == "corner-and-edge":
            x0, y0, x1, y1 = zone
            band = th["edge_band"]
            near = floor & (pos[:, 0] > x0) & (pos[:, 1] > y0) & ((pos[:, 0] > x1 - band) | (pos[:, 1] > y1 - band) | (
                (pos[:, 0] < x0 + band) & (pos[:, 1] > L.Y1 - P.CASEWORK["depth"])))
            crit["edges_clear"] = dict(check="mass left within 0.10 m of the alcove's walls", ok=frac(near) <= th["edges"],
                                       fraction=round(frac(near), 4), allowed=th["edges"])
        if task == "under-table":
            u = floor & self.under("table_B1", pos)
            crit["edges_clear"] = dict(check="mass left under the table", ok=frac(u) <= th["edges"], fraction=round(frac(u), 4),
                                       allowed=th["edges"])
        if task in th["contained"]:
            which = np.ones(len(m), bool)
            if task in ("table-scraps", "closing-sweep"):
                which = self.cls != "pencil"
            tot = max(float(m[which].sum()), 1e-12)
            f = float(m[which & cont[target]].sum()/tot)
            c = dict(check=f"mass in the {target}" + (" while it stands upright (the pencil not counted)" if cs else ""),
                     ok=f >= th["contained"][task], fraction=round(f, 4), need=th["contained"][task])
            if task == "broken-mug":
                big = np.isin(self.tag, ("shard", "handle"))
                c["fragments_in"] = int(cont[target][big].sum()); c["fragments"] = int(big.sum())
                c["ok"] = bool(c["ok"] and cont[target][big].all())
            if cs:
                c["barrel_upright"] = upright
            crit["in_container"] = c
        if cs:
            # the litter a swept classroom does not have on its floor: crumpled paper, a pencil (a school supply: in the
            # barrel, or kept on a table, the desk or the counter), and its paper scraps, of which half by count is
            # asked: assisted benches put 7 and 8 of 11 in the barrel, the rest stopping at the dustpan's lip
            wad, scr, pen = self.cls == "wad", self.cls == "scrap", self.cls == "pencil"
            inb, kept = cont["barrel"], self.on_top(pos) & ~any_cont
            panned = self.in_pans(pos) & ~(any_cont | binned)
            fs = float(inb[scr].mean()) if scr.any() else 1.0
            where = ("in the barrel" if inb[pen].all() else "kept on a top" if (inb | kept)[pen].all() else
                     "in a dustpan" if (inb | kept | panned)[pen].all() else "left out")
            crit["litter_picked_up"] = dict(check="every crumpled paper and half of the paper scraps in the barrel; the pencil in it or on a "
                                                  "table, desk or counter top",
                                            ok=bool(inb[wad].all()) and fs >= th["closing"]["scraps"] and where in ("in the barrel", "kept on a top"),
                                            wads_in=int(inb[wad].sum()), wads=int(wad.sum()), scraps_in=int(inb[scr].sum()), scraps=int(scr.sum()),
                                            scraps_fraction=round(fs, 3), need_scraps=th["closing"]["scraps"], pencil=where)
            counted = self.cls != "pencil"
            mc = max(float(m[counted].sum()), 1e-12)
            parts = dict(mass_binned=float(m[counted & binned].sum())/mc, mass_in_pans=float(m[counted & panned].sum())/mc,
                         wads=int(wad.sum()), wads_binned=int((binned & wad).sum()), wads_in_pans=int((panned & wad).sum()),
                         scraps=int(scr.sum()), scraps_binned=int((binned & scr).sum()), scraps_in_pans=int((panned & scr).sum()),
                         pencil="put away" if (binned | kept)[pen].all() else "in a dustpan" if (binned | kept | panned)[pen].all() else "left out")
        if task == "beads":
            n = int(cont["pail"].sum())
            crit["in_container"] = dict(check="beads in the pail", ok=n >= th["beads"], beads_in=n, need=th["beads"], beads=int(len(m)))
        if task in th["floor_left_g"]:
            g = 1e3*float(m[floor].sum())
            crit["floor_left_clean"] = dict(check="loose debris left on the floor", ok=g <= th["floor_left_g"][task], grams=round(g, 3),
                                            allowed_g=th["floor_left_g"][task])
        if task in ("pan-and-barrel", "broken-mug", "heavy-spill", "beads"):
            tp, _ = w.body_pose(w.scene.tools[target]["body"])
            zc = ((zone[0] + zone[2])/2, (zone[1] + zone[3])/2)
            zr = 0.5*math.hypot(zone[2] - zone[0], zone[3] - zone[1])
            far = floor & (np.hypot(pos[:, 0] - zc[0], pos[:, 1] - zc[1]) > zr + th["spill_radius"]) & (
                np.hypot(pos[:, 0] - tp[0], pos[:, 1] - tp[1]) > th["spill_radius"] + 0.3)
            crit["nothing_spilled_on_the_way"] = dict(check="mass on the floor away from the pick-up and the container",
                                                      ok=frac(far) <= th["spilled"], fraction=round(frac(far), 4), allowed=th["spilled"])
        # ---- furniture, supplies, containers, tools
        if task in ("under-table", "broken-mug", "beads", "closing-sweep"):
            bad, swapped = [], {}
            excess = lambda d_, y_, t_, lim_: 1.0 if t_ > 5.0 else min(1.0, max(0.0, (d_ - lim_[0])/PROGRESS["graded_m"],
                                                                                (math.degrees(y_) - lim_[1])/PROGRESS["graded_deg"]))
            # closing-sweep: a chair is back where it was, or pushed in at its own table, or anywhere between; a
            # table's chairs are identical, so its chairs are matched one-to-one to its places and any matching that
            # puts every chair at a place will do
            groups = {}
            for n, it in self.spec["items"].items():
                key = ("chairs of", self.tucked[n][0]) if cs and it["kind"] == "chair" and n in self.tucked else n
                groups.setdefault(key, []).append(n)
            for names in groups.values():
                pose = {n: w.pose2d(n) for n in names}
                kind = self.spec["items"][names[0]]["kind"]
                lim = th["chair"] if kind == "chair" else (th["closing"]["table"] if cs else th["table"], 5.0)
                off = {}
                for n in names:
                    (x, y, yaw), tilt = pose[n]
                    for place in names:
                        if cs and kind == "chair":
                            off[n, place] = self.chair_off(place, (x, y), yaw) + (tilt,)
                        elif place == n:
                            (sx, sy, syaw), _ = self.start[n]
                            off[n, n] = (math.hypot(x - sx, y - sy), abs((yaw - syaw + math.pi) % (2*math.pi) - math.pi), tilt)
                cost = lambda perm: (sum(excess(*off[n, q], lim) for n, q in zip(names, perm)), sum(off[n, q][0] for n, q in zip(names, perm)))
                best = min(itertools.permutations(names), key=cost) if len(names) > 1 else tuple(names)
                for n, place in zip(names, best):
                    dist, dyaw, tilt = off[n, place]
                    if place != n:
                        swapped[n] = place
                    if dist > lim[0] or math.degrees(dyaw) > lim[1] or tilt > 5.0:
                        bad.append(f"{n}: {dist*1e3:.0f} mm, {math.degrees(dyaw):.0f} deg, tilt {tilt:.0f} deg")
                    if cs:
                        out_of_place[n] = excess(dist, dyaw, tilt, lim)
                        if tilt > 5.0:
                            tipped.add(n)
            check = ("chairs upright within 0.10 m and 15 deg of a place of their table (where one of its chairs started, pushed in at the "
                     "table, or the way between), one chair to a place; tables and desk within 0.10 m and 5 deg, upright") if cs else \
                "chairs within 0.10 m and 15 deg of their start, tables and desk within 0.03 m, upright"
            crit["furniture_back"] = dict(check=check, ok=not bad, moved=bad[:8])
            if cs:
                crit["furniture_back"]["at_another_chairs_place"] = swapped
        if task == "table-scraps":
            bad = []
            top = L.TABLE_H
            for n in self.spec["supplies"]:
                (x, y, yaw), tilt = w.pose2d(n)
                (sx, sy, _), _ = self.start[n]
                z = float(w.body_pose(n)[0][2])
                if math.hypot(x - sx, y - sy) > th["supply"] or tilt > 10.0 or abs(z - top) > 0.02:
                    bad.append(n)
            pen = self.cls == "pencil"
            pen_ok = bool(((pos[pen, 2] > top - 0.01) & ~any_cont[pen]).all()) if pen.any() else True
            crit["supplies_kept"] = dict(check="cup, paper, scissors and glue upright within 0.10 m on the table; pencil on the table",
                                         ok=not bad and pen_ok, moved=bad, pencil_on_table=pen_ok)
        bad = []
        for n in self.containers:
            (x, y, _), tilt = w.pose2d(n)
            (sx, sy, _), _ = self.start[n]
            far = math.hypot(x - sx, y - sy)
            anywhere = cs and n == "barrel"          # closing-sweep: the barrel may be brought to the work
            if tilt > 10.0 or (far > th["container"] and not anywhere):
                bad.append(f"{n}: tilt {tilt:.0f} deg, {far:.2f} m")
            if cs:
                out_of_place[n] = 1.0 if tilt > 10.0 else 0.0 if anywhere else min(1.0, max(0.0, (far - th["container"])/PROGRESS["graded_m"]))
                if tilt > 10.0:
                    tipped.add(n)
        crit["containers_standing"] = dict(check="the barrel upright; the pail and the wastebasket upright within 0.5 m of their start" if cs else
                                           "barrel, pail and wastebasket upright within 0.5 m of their start", ok=not bad, problems=bad)
        if cs:
            held = self.held()
            bad = []
            for n, home in self.hang.items():
                off = float(np.linalg.norm(w.tool_pose(n)[0] - home))
                if off > th["hang"] + 0.02 or n in held:
                    bad.append(f"{n}: {off*1e3:.0f} mm from its place on the rail" + (", held by the robot" if n in held else ""))
            if "lobby_pan" in self.start:
                # its place is the floor under the rail, not one spot by it
                (x, y, _), tilt = w.pose2d("lobby_pan")
                off = seg_dist((x, y), *map(np.array, PAN_LINE))
                if off > th["pan_home"] or tilt > 10.0 or "lobby_pan" in held:
                    bad.append(f"lobby_pan: {off:.2f} m from the floor under the rail, tilt {tilt:.0f} deg"
                               + (", held by the robot" if "lobby_pan" in held else ""))
            for n in w.scene.tools:
                # a tool stood inside a container is not put away (the loaded dustpan stood in the barrel brought to
                # the rail)
                if n not in self.containers and any(self.inside(c, w.tool_pose(n)[0][None])[0] for c in self.containers) \
                        and not any(b.startswith(n + ":") for b in bad):
                    bad.append(f"{n}: inside a container")
            for n in ("counter_brush", "hand_pan"):
                if n in w.scene.tools:
                    p = w.tool_pose(n)[0]
                    on = bool(self.on_counter(p[None], th["closing"]["counter_band"])[0])
                    if not on or n in held:
                        bad.append(f"{n}: " + ("held by the robot" if n in held else f"not on the back counter, at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"))
            for b in bad:
                out_of_place[b.split(":")[0]] = 1.0
            crit["tools_put_away"] = dict(check="each broom back on its own prongs, the lobby dustpan standing on the floor under the rail, the "
                                                "counter brush and the hand dustpan on the back counter; none of them held by a robot hand or "
                                                "inside a container", ok=not bad, problems=bad,
                                          held_by_the_robot=held)
        if task == "pan-and-barrel":
            bad = []
            for n, home in self.hang.items():
                p, R = w.tool_pose(n)
                if np.linalg.norm(p - home) > th["hang"] + 0.02:
                    bad.append(f"{n}: {np.linalg.norm(p - home)*1e3:.0f} mm from its place on the rail")
            if "lobby_pan" in self.start:
                (x, y, _), tilt = w.pose2d("lobby_pan")
                (sx, sy, _), _ = self.start["lobby_pan"]
                if math.hypot(x - sx, y - sy) > th["pan_home"] or tilt > 10.0:
                    bad.append(f"lobby_pan: {math.hypot(x - sx, y - sy):.2f} m from its place, tilt {tilt:.0f} deg")
            crit["tools_hung_up"] = dict(check="every broom back on its prongs, the dustpan standing under the rail", ok=not bad, problems=bad)
        # ---- fine dust (a film, sweeping/dust.py): how much of it is still on the floor. On the floor: the film, the
        # lines at pan lips, and the ridge that rides in front of bristle tips standing on the floor.
        if w.dust is not None and cs:
            # closing-sweep: what is swept up goes in the barrel, the dust too: dust left in a dustpan is not put away.
            # Put away: the barrel's tally while the barrel stands (a tally is never taken
            # back out, dust.py), and what a broom's bristles hold (a broom always keeps some) up to lodged_g, above every
            # amount on record (0.049 g); more would be dust kept in a broom
            dd = w.dust
            ridge, start = dd.ridge_on_floor(), dd.total0
            tally = float(dd.tally.get("barrel", 0.0))
            barrel = tally if upright else 0.0
            lodged = float(dd.lodged.sum())
            credit = min(lodged, 1e-3*th["closing"]["lodged_g"])
            in_pans = sum(float(p["field"].sum()) for p in dd.pans.values())
            left = max(0.0, dd.total() - barrel - credit)
            crit["dust_swept"] = dict(check="fine dust not put in the barrel (on the floor, in front of a broom's tips, in a dustpan or elsewhere; "
                                            "what bristles hold counts as put away up to 0.05 g), of what lay on the floor at the start",
                                      ok=left <= th["dust_left"]*start,
                                      fraction=round(left/max(start, 1e-12), 4), allowed=th["dust_left"], grams_left=round(1e3*left, 3),
                                      grams_on_the_floor=round(1e3*(dd.on_floor() + ridge), 3), grams_in_the_ridge=round(1e3*ridge, 3),
                                      grams_in_dustpans=round(1e3*in_pans, 3), grams_in_the_barrel=round(1e3*barrel, 3),
                                      grams_lodged_in_bristles=round(1e3*lodged, 3), grams_credited_from_bristles=round(1e3*credit, 3),
                                      grams_at_start=round(1e3*start, 3))
            parts.update(dust_start=start, dust_binned=tally if opened else 0.0, dust_lodged_credit=credit, dust_in_pans=in_pans)
        elif w.dust is not None:
            ridge = w.dust.ridge_on_floor()
            left, start = w.dust.on_floor() + ridge, w.dust.total0
            crit["dust_swept"] = dict(check="fine dust left on the floor (film, lines at pan lips, the ridge in front of a broom's tips), "
                                            "of what lay there at the start", ok=left <= th["dust_left"]*start,
                                      fraction=round(left/max(start, 1e-12), 4), allowed=th["dust_left"], grams_left=round(1e3*left, 3),
                                      grams_in_the_ridge=round(1e3*ridge, 3), grams_at_start=round(1e3*start, 3))
        # ---- rest: nothing still moving (debris asleep or slow, watched bodies still for the last second)
        speed = float(np.linalg.norm(d["vel"], axis=1).max(initial=0.0))
        moving = [n for n, v in motion.items() if v > th["rest_motion"]]
        crit["at_rest"] = dict(check="debris slower than 5 mm/s; tools, containers and furniture still for 1 s",
                               ok=bool(window and speed <= th["rest_speed"] and not moving), fastest_piece_m_s=round(speed, 4),
                               awake=int(w.awake), moving=moving[:6])
        oks = [c["ok"] for c in crit.values()]
        summary = dict(total_g=round(1e3*total, 3), in_square_g=round(1e3*float(m[sq].sum()), 3),
                       on_floor_g=round(1e3*float(m[floor].sum()), 3),
                       in_container_g={n: round(1e3*float(m[v].sum()), 3) for n, v in cont.items()})
        if w.dust is not None:
            summary["fine_dust_g"] = w.dust.summary()
        out = {"succeeded": bool(all(oks)), "score": round(sum(oks)/len(oks), 4), "criteria": crit, "debris": summary,
               "window_complete": bool(window),
               "note": "world state only (piece positions and masses, body poses, contacts with the robot, 1 s rest window)" if cs else
               "world state only (piece positions and masses, body poses, 1 s rest window)"}
        if cs:
            reward = self.progress(parts, out_of_place, tipped)
            reward.update(contract=REWARD_CONTRACTS[task], score_rule="1 if success else 0.9 x progress",
                          mean_of_checks=out["score"])      # the share of criteria that hold
            out["score"] = 1.0 if out["succeeded"] else round(FAIL_SCORE_SCALE*reward["progress"], 4)
            out["reward"] = reward
        return out
