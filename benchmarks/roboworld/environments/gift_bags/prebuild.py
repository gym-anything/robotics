"""Start states of the current code: list them, build the missing ones, ship them with the code.

    python -m gift_bags.prebuild               list every start state: where it is (shipped, cache, missing)
    python -m gift_bags.prebuild --build       build the missing ones (one after another; hours on a fresh checkout
                                               when the generators changed; measured on 4 threads:
                                               about 40 min for the tote with its handles, 10-17 min for each of
                                               the 18 wrapped presents, 23 s for the carton)
    python -m gift_bags.prebuild --build --only home [--seeds 0,1,5]    a part of them
    python -m gift_bags.prebuild --ship        copy the current code's files from the cache into gift_bags/start_states
                                               and remove shipped files no key of the current code names

`gar doctor gift-bags` and the variation contract (`/v1/variations`) show the same list (variants.availability)."""
import sys
import shutil
import time
from pathlib import Path

from . import startcache as SC


def wrapped_variants():
    """The 18 wrapped-present variants a home seed can draw: [dict(box, clearance, wrap_stock, key, path, where)]."""
    from . import wrapmap as WM, scene as S, giftbox as GB
    from .variants import HOME_AXES
    out = []
    for box in HOME_AXES["box"]:
        for cl in HOME_AXES["clearance_m"]:
            for stock in HOME_AXES["wrap_stock"]:
                key = WM._cache_key(S.TABLE_Z, S.BOX_XY, "girth", 0, .008, GB.box_spec(box, cl), stock)
                path = SC.locate(f"wrapped_gift_{key}.npz")
                out.append(dict(box=box, clearance_m=cl, wrap_stock=stock, key=key, path=str(path), where=SC.where(path)))
    return out


def carton():
    from . import carton as CT, scene as S
    key = CT._cache_key(S.RECEIVING_Z, S.CARTON_XY)
    path = SC.locate(f"sealed_carton_{key}.npz")
    return dict(key=key, path=str(path), where=SC.where(path))


def table():
    """Every start state of the current code: {preset: [dict(name, key, where, ...)]}. Builds nothing."""
    from . import scene as S
    c = S.counter_readiness()
    counter = [dict(name="tote shipping state", **c["shipping_state"])]
    counter.append(dict(name="tote with its handles", **(c["handles"] or dict(key=None, where="missing",
                                                                              reason="needs the shipping state first"))))
    for r, stem in zip(counter, ("flat_tote_shipping_state", "flat_tote_handles")):   # the file names of bagnet / handles
        if r.get("key") and not r.get("path"):
            r["path"] = str(SC.locate(f"{stem}_{r['key']}.npz"))
    home = [dict(name=f"wrapped present: {v['box']}, lid clearance {1e3*v['clearance_m']:g} mm, {v['wrap_stock']}", **v)
            for v in wrapped_variants()]
    return dict(counter=counter, home=home, receiving=[dict(name="sealed carton", **carton())])


def seeds_of(variant, n=200):
    """The first home seeds (0..n-1) that draw this wrapped variant."""
    from .variants import home_variant
    return [s for s in range(n) if all(home_variant(s)[k] == variant[k2] for k, k2 in
                                       (("box", "box"), ("clearance", "clearance_m"), ("wrap_stock", "wrap_stock")))]


def build(only=None, seeds=None, log=print):
    from .config import Config, TASKS
    from . import scene as S
    done = []
    for preset, rows in table().items():
        if only and preset != only:
            continue
        if preset == "home":
            from .variants import home_variant
            todo = {}
            for s in (seeds if seeds is not None else range(400)):
                v = home_variant(s)
                todo.setdefault((v["box"], v["clearance"], v["wrap_stock"]), s)
            for (box, cl, stock), s in sorted(todo.items(), key=lambda kv: kv[1]):
                row = next(r for r in rows if (r["box"], r["clearance_m"], r["wrap_stock"]) == (box, cl, stock))
                if row["where"] != "missing":
                    continue
                t0 = time.perf_counter()
                log(f"building {row['name']} (seed {s}) ...")
                S.home(Config(preset="home", task="unwrap-gift", seed=s))
                done.append((row["name"], time.perf_counter() - t0))
                log(f"  built in {done[-1][1]:.0f} s")
        elif any(r["where"] == "missing" for r in rows):
            t0 = time.perf_counter()
            log(f"building the {preset} preset's start state ...")
            S.BUILDERS[preset](Config(preset=preset, task=TASKS[preset][0]))
            done.append((preset, time.perf_counter() - t0))
            log(f"  built in {done[-1][1]:.0f} s")
    return done


def ship(log=print):
    """Copy the current code's start states from the cache into gift_bags/start_states; drop shipped files that no
    key of the current code names. Returns (copied, removed, missing)."""
    SC.SHIPPED.mkdir(exist_ok=True)
    want, missing, copied = set(), [], []
    for rows in table().values():
        for r in rows:
            if r["key"] is None or r["where"] == "missing":
                missing.append(r["name"])
                continue
            name = Path(r["path"]).name
            want.add(name)
            if r["where"] == "cache":
                shutil.copy2(r["path"], SC.SHIPPED/name)
                copied.append(name)
    removed = []
    for f in sorted(SC.SHIPPED.glob("*.npz")):
        if f.name not in want:
            f.unlink()
            removed.append(f.name)
    log(f"shipped {len(copied)} file(s), removed {len(removed)} stale, {len(missing)} missing: {missing}")
    return copied, removed, missing


def main(argv):
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    seeds = [int(s) for s in argv[argv.index("--seeds") + 1].split(",")] if "--seeds" in argv else None
    if "--build" in argv:
        build(only, seeds)
    if "--ship" in argv:
        ship()
    n = dict(shipped=0, cache=0, missing=0)
    for preset, rows in table().items():
        print(preset)
        for r in rows:
            n[r["where"]] += 1
            extra = f"  ({r['reason']})" if r.get("reason") else ""
            print(f"  {r['where']:8s} {r['key'] or '-':16s} {r['name']}{extra}")
    print(f"{n['shipped']} shipped, {n['cache']} in the local cache only, {n['missing']} missing (a launch builds a "
          "missing one first)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
