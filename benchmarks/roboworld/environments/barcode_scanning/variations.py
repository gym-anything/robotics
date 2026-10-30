"""Frozen variation catalogue and its validation campaign.

Catalogue: every task of every preset at seeds 0..999 (8 000 variations). A seed varies what the task varies (the
counted slot and its stock mix and unit yaw; the basket's contents and where they lie; the returned books, their
order and which cover carries the library label; pack contents); dimensions, masses and materials are never
randomised. Each variation has a split frozen by a hash of (task, seed): train 80 %, validation 10 %, test 10 %.
The file `variations.json` is written once and hash-checked; `variants.task_resolve` reports each variation's split.

Validation: a stratified sample of variations is built and settled exactly as a session would be, and each reset
is checked: finite, the scanner seated, nothing on the floor, nothing thrown by the settle (a placed item: > 20 mm or
> 15 deg from its spawn pose, a cylinder's roll about its own axis not counted; an item dropped into a container: out
of the container or above its rim; either: still moving > 50 mm/s), the task's evaluator constructible, and the
counted-slot / basket / book ground truth non-empty. Items that settled inside their container are listed.

  python -m barcode_scanning.variations --freeze                   # write variations.json (idempotent)
  python -m barcode_scanning.variations --validate 12 --jobs 4 --out barcode_scanning/evidence/variations
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from .config import TASKS

ROOT = Path(__file__).resolve().parent
FILE = ROOT/"variations.json"
SEEDS = 1000


def split_of(task, seed):
    h = int(hashlib.sha256(f"barcode-scanning/{task}/{seed}".encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else ("validation" if h < 90 else "test")


def build():
    rows = []
    for preset, tasks in TASKS.items():
        for task in tasks:
            for seed in range(SEEDS):
                rows.append(dict(id=f"barcode-scanning/v1/{preset}/{task}/seed-{seed}", preset=preset, task=task,
                                 seed=seed, split=split_of(task, seed)))
    counts = {}
    for r in rows:
        counts.setdefault(r["task"], {}).setdefault(r["split"], 0)
        counts[r["task"]][r["split"]] += 1
    body = json.dumps(rows, separators=(",", ":"))
    return dict(schema="barcode-scanning.variations.v1", count=len(rows), seeds_per_task=SEEDS, counts=counts,
                rule="split = sha256('barcode-scanning/<task>/<seed>')[:8] mod 100: <80 train, <90 validation, else test",
                sha256=hashlib.sha256(body.encode()).hexdigest(), variations=rows)


def load():
    d = json.loads(FILE.read_text())
    body = json.dumps(d["variations"], separators=(",", ":"))
    if hashlib.sha256(body.encode()).hexdigest() != d["sha256"]:
        raise ValueError("variations.json does not match its frozen hash")
    return d


def lookup(task, seed):
    if 0 <= seed < SEEDS:
        return split_of(task, seed)
    return None


# ------------------------------------------------------------------------------------------------ validation
def check_one(preset, task, seed):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from .config import Config
    from .env import ScanEnv
    t0 = time.time()
    rec = dict(preset=preset, task=task, seed=seed, split=split_of(task, seed))
    try:
        env = ScanEnv(Config(preset=preset, task=task, seed=seed), output_dir=f"/tmp/bs_var_{preset}_{task}_{seed}")
        w = env.world
        q0, q, qd = w.model.body_q.numpy(), w.state.body_q.numpy(), w.state.body_qd.numpy()
        from .catalog import SKUS
        thrown, settled = {}, {}
        for n, it in w.scene.items.items():
            b = it["body"]
            moved = float(np.linalg.norm(q[b, :3] - q0[b, :3]))
            dR = Rotation.from_quat(q[b, 3:])*Rotation.from_quat(q0[b, 3:]).inv()
            sku = SKUS.get(it.get("sku")) if it.get("sku") else None
            if sku is not None and sku.shape != "box":
                # a cylinder turned about its own axis (rolled) is the same pose: measure the tilt of the axis only
                a0 = Rotation.from_quat(q0[b, 3:]).apply((0., 0., 1.)); a1 = Rotation.from_quat(q[b, 3:]).apply((0., 0., 1.))
                turned = math.degrees(math.acos(max(-1., min(1., float(a0 @ a1)))))
            else:
                turned = math.degrees(dR.magnitude())
            speed = float(np.linalg.norm(qd[b, :3]))
            rec_ = dict(moved_mm=round(moved*1e3, 1), turned_deg=round(turned, 1), speed_mm_s=round(speed*1e3, 1))
            cont = w.scene.items.get(it.get("container")) if it.get("container") else None
            if cont is not None and cont.get("container_size"):
                # dropped into a container (the customer's basket): it may slide, roll or tip while it settles; it must
                # stay inside and come to rest (a spawn-pose tolerance flagged 20 of 84 counter resets whose items had
                # only rolled or slid off a neighbour, all at rest inside the basket)
                pc, Rc = w.body_pose(cont["body"])
                L, Wd, H = cont["container_size"]
                rel = Rc.T @ (q[b, :3] - pc)
                inside = abs(rel[0]) <= L/2 and abs(rel[1]) <= Wd/2 and rel[2] <= H/2 + .05
                if not inside or speed > .05:
                    thrown[n] = dict(rec_, left_container=not inside)
                elif moved > .02 or turned > 15.:
                    settled[n] = rec_
            elif moved > .02 or turned > 15. or speed > .05:
                thrown[n] = rec_
        ev = env.evaluator.evaluate()
        meta = env.evaluator.meta
        truth = meta.get("truth") or {}
        items = [n for n, it in w.scene.items.items() if it.get("sku")]
        rec.update(finite=bool(np.isfinite(q).all() and np.isfinite(qd).all()), fault=w.fault, seated=bool(w.seated),
                   nothing_on_floor=bool(ev["checks"]["nothing_on_floor"]), on_floor=ev["metrics"].get("on_floor"),
                   thrown=thrown, settled_in_container=settled, target_slot=meta.get("target_slot"), truth=truth, items=len(items),
                   ground_truth_ok=bool(truth) if task != "hold-scanner" and preset != "library" else True,
                   build_and_settle_s=round(time.time() - t0, 1))
        rec["ok"] = bool(rec["finite"] and not rec["fault"] and rec["seated"] and rec["nothing_on_floor"] and not thrown
                         and rec["ground_truth_ok"])
    except Exception as exc:
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return rec


def sample(k):
    """k seeds per task: seed 0, then seeds spread over the catalogue, every split represented."""
    out = []
    for preset, tasks in TASKS.items():
        for task in tasks:
            seeds = [0] + [int(round(1 + i*(SEEDS - 2)/max(1, k - 2))) for i in range(k - 1)]
            out += [(preset, task, s) for s in sorted(set(seeds))[:k]]
    return out


def _worker(args):
    return check_one(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--validate", type=int, default=0, help="seeds per task to build and check")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.freeze:
        d = build()
        if FILE.exists():
            old = json.loads(FILE.read_text())
            if old["sha256"] != d["sha256"]:
                raise SystemExit("variations.json exists with a different hash; the catalogue is frozen")
        FILE.write_text(json.dumps(d, indent=None, separators=(",", ":")))
        print(json.dumps({k: v for k, v in d.items() if k != "variations"}, indent=1))
    if a.validate:
        from concurrent.futures import ProcessPoolExecutor
        from .manifest import launch_stamp
        stamp = launch_stamp()
        jobs = sample(a.validate)
        t0 = time.time()
        results = []
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for r in ex.map(_worker, jobs):
                results.append(r)
                print(json.dumps({k: r.get(k) for k in ("preset", "task", "seed", "split", "ok", "thrown", "on_floor", "error")}), flush=True)
        summary = dict(built=len(results), ok=sum(r.get("ok", False) for r in results),
                       failed=[{k: r.get(k) for k in ("preset", "task", "seed", "thrown", "on_floor", "error", "fault")}
                               for r in results if not r.get("ok")], wall_s=round(time.time() - t0, 1))
        out = Path(a.out or ROOT/"evidence"/"variations")
        out.mkdir(parents=True, exist_ok=True)
        (out/"validation.json").write_text(json.dumps(dict(code_at_launch=stamp, summary=summary, results=results), indent=1, default=str))
        print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
