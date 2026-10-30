"""Robot setups the forklift environment can host (robots.catalog ENVIRONMENTS['forklift'] validator).

The seated operator (forklift/operator.py) is built from whichever humanoid rig is selected: everything robot-specific
comes from the catalogue model (its frames, groups, `forklift` controller profile and seated pose) and the model's
entry in operator.SEAT_FIT; the driving-robot kit (rig forklift-drive-kit) is built by operator.add_kit from its kit fit.
The robots offered are ALIASES (public name -> the catalogue rig the operator builds), read from the catalogue. A setup
passed as an object (a RobotSetup, its dict or a rig JSON path) or a catalogue rig id is accepted only when it is one of
those rigs exactly (instances and attachments; the setup's id and description aside): the operator builds that rig and
nothing else, so a setup that differs (another controller profile, mount, pose, initial pose, parameters or an
attachment) is refused rather than run as the rig (unified contract, invariant 4). Other robots are refused with their
measured reason (catalog ENVIRONMENTS['forklift']['unsupported']).
"""

def _aliases():
    """The robots offered: robots.catalog ENVIRONMENTS['forklift']['aliases'] (public name -> catalogue rig), one table for
    the CLI, the backend, the config and the operator (a further humanoid that fits the seat needs an alias there and an
    operator.SEAT_FIT entry)."""
    from robots.catalog import ENVIRONMENTS
    return dict(ENVIRONMENTS["forklift"]["aliases"])


ALIASES = _aliases()


def describe(selected):
    from robots.catalog import resolve
    if isinstance(selected, str):
        return selected
    try:
        setup = resolve(selected)
        return dict(setup_id=setup.id, setup_fingerprint=setup.fingerprint)
    except Exception:
        return repr(selected)[:200]


def _built(setup):
    """What the operator builds from a setup: its instances and attachments (not its id or description)."""
    d = setup.to_dict()
    return dict(instances=d["instances"], attachments=d.get("attachments", {}))


def _differences(setup, rig):
    a, b = _built(setup), _built(rig)
    out = []
    if set(a["instances"]) != set(b["instances"]):
        out.append(f"instances {sorted(a['instances'])} (the rig's: {sorted(b['instances'])})")
    for name in set(a["instances"]) & set(b["instances"]):
        x, y = a["instances"][name], b["instances"][name]
        for k in sorted(set(x) | set(y)):
            if x.get(k) != y.get(k):
                out.append(f"{name}.{k} {x.get(k)!r} (the rig's: {y.get(k)!r})")
    if a["attachments"] != b["attachments"]:
        out.append(f"attachments {sorted(a['attachments'])} (the rig's: {sorted(b['attachments'])})")
    return out


def validate_setup(selected):
    from robots.catalog import ENVIRONMENTS, resolve
    unsupported = ENVIRONMENTS["forklift"].get("unsupported", {})
    if isinstance(selected, str):
        if selected in ALIASES:
            return selected
        if selected in unsupported:
            raise ValueError(unsupported[selected])
    try:
        setup = resolve(selected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"forklift: not a robot setup ({exc}); choose one of {', '.join(ALIASES)}") from exc
    for alias, rig in ALIASES.items():
        if _built(setup) == _built(resolve(rig)):
            return alias
    models = {i.get("model", "?") for i in setup.to_dict().get("instances", {}).values()}
    for alias, rig in ALIASES.items():
        want = {i["model"] for i in resolve(rig).to_dict()["instances"].values()}
        if models == want:
            raise ValueError(f"forklift builds the {alias} exactly as catalogue rig {rig}; this setup differs in "
                             f"{'; '.join(_differences(setup, resolve(rig)))}, and is refused rather than run as {rig}")
    bare = {m.split("@")[0] for m in models}
    for name in sorted(bare):
        if name in unsupported and len(bare) == 1:
            raise ValueError(unsupported[name])
    offered = ", ".join(f"{a} (rig {r}, model {', '.join(sorted(i['model'] for i in resolve(r).to_dict()['instances'].values()))})"
                        for a, r in ALIASES.items())
    raise ValueError(f"forklift hosts {offered}. Other robots are refused with their measured reason "
                     "(robots/catalog.py ENVIRONMENTS['forklift']['unsupported'])")
