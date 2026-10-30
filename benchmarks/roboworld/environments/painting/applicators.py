"""Applicator registry: one paint model per tool body in the scene."""


def build_applicators(system):
    from .roller import Roller
    from .brush import Brush
    from .aerosol import Aerosol
    sc = system.w.scene
    for name, it in sc.items.items():
        kind = it["kind"]
        if kind == "roller":
            system.applicators[name] = Roller(system, name, it["body"], sc.bodies[f"{name}_cover"])
        elif kind in ("sash", "round8", "stencil_brush", "longreach"):
            system.applicators[name] = Brush(system, name, it["body"], {"stencil_brush": "stencil"}.get(kind, kind))
        elif kind == "aerosol":
            system.applicators[name] = Aerosol(system, name, it["body"], sc.bodies[f"{name}_button"],
                                               sc.joints_extra[f"{name}_button"])
    # optional start load (probes and seeded 'roller pre-loaded' starts): fraction of a full load, as the paint is supplied
    for name, frac in (system.spec.get("preload") or {}).items():
        a = system.applicators[name]
        pt = system.paints[0]
        if a.kind == "roller":
            from .roller import SAT
            V = SAT*a.C0*float(frac)
            a.s[0][:] = V*pt.phi0; a.v[:] = V*(1 - pt.phi0)
        elif a.kind == "brush":
            from .brush import HOLD
            a.cap = 2*a.C_half*0.33                  # dipped to a third of the bristle length
            V = HOLD*a.cap*float(frac)
            a.s[0][:] = V*pt.phi0; a.v[:] = V*(1 - pt.phi0)
