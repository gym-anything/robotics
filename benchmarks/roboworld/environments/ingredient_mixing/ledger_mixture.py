"""The mixture's part of the parameter ledger: every constant that governs the food (materials.py, task.py, heat.py,
continuum.py), read from the modules themselves so that the table cannot drift from the code, each with a label and
its basis. `python -m ingredient_mixing.params` prints it after the ledger of geometry, robot and appliances.

Labels are those of params.py: read (a source opened and the number seen), search (a search summary only), derived,
estimate, open (no basis found; chosen), authored (a design choice), literature (a classical value, from memory unless
a source is named), measured (on this machine), calibrated (set on an independent observable,
named), engine (a numerical setting, not a property of a real thing).
"""

from dataclasses import fields


def entries():
    from . import continuum as CT
    from . import heat as H
    from . import materials as M
    from . import task as T
    out = []

    def add(name, value, unit, label, basis):
        out.append(dict(name=name, value=value, unit=unit, label=label, basis=basis))

    S, R = T.FLOUR, M.Recipe()
    solid = dict(
        rho=("kg/m3", "search", "particle density of wheat flour; not confirmed on an opened page"),
        void=("-", "search", "void ratio poured: 510 kg/m3 bulk, a cup of spooned flour at 120 g (King Arthur's weight chart); "
                              "an engineering table gives 600 kg/m3 for flour in bulk (opened)"),
        bound=("-", "literature", "liquid a wetted solid holds, as a void ratio: a flour's water absorption in a dough, about "
                                  "60 % of its mass (farinograph figures for all-purpose flour; from memory), is 0.88 ml per ml"),
        grain=("m", "estimate", "flour particle size"),
        friction_deg=("deg", "estimate", "internal friction of loose flour"),
        cohesion=("Pa", "estimate", "cohesion of dry flour at the stresses of a heap in a bowl; decides with the speed "
                                    "whether a whisk throws powder or cuts through it (E15 dry)"),
        cap_cohesion=("Pa", "estimate", "peak capillary cohesion of damp flour"),
        law_a=("-", "derived", "log10 K = a x + b fitted through the pancake batter of the study read (K 0.754 Pa s^n at its "
                               "flour share) and the muffin batter figure (3.35 Pa s at 60 1/s, search)"),
        law_b=("-", "derived", "as law_a"),
        n_slope=("-", "derived", "flow index n = 1 - n_slope x through the same two batters"),
        n_min=("-", "estimate", "floor of the flow index"),
        yield_x0=("-", "estimate", "wetted-solid share from which a batter has a yield stress"),
        yield_x1=("-", "estimate", "share at which it is a dough"),
        yield_max=("Pa", "estimate", "yield stress of a stiff dough"),
        hyd_gain=("-", "estimate", "thickening of a batter as its flour hydrates"),
        hyd_tau=("s", "estimate", "time of that; a recipe rests a batter 15 minutes"),
        work_gain=("-", "estimate", "stiffening by mixing work (gluten)"),
        work_scale=("J/kg", "authored", "work at which a flour batter has stiffened by a tenth: 25 fold strokes of the "
                                        "method (task.REFERENCE_STROKE_J_KG, measured); the muffin method's 15 to 20 "
                                        "light strokes (Quaker Oats, opened) with a margin"),
        set_tau=("s", "estimate", "cold-set time constant of a pudding mix; 0 for flour"),
        set_k=("Pa s^n", "estimate", "consistency of a set pudding"), set_n=("-", "estimate", "its flow index"),
        set_yield=("Pa", "estimate", "its yield stress"), set_xref=("-", "derived", "the box's mix-to-milk ratio"),
        starch=("-", "literature", "starch share of wheat flour"),
        gel_a=("1/s", "derived", "Arrhenius rate of gelatinisation, set so that pasting begins near the 59.9 C of the pancake study (read)"),
        gel_ea=("J/mol", "literature", "activation energy of starch gelatinisation"),
        gel_b=("Pa s^n", "estimate", "consistency of a cooked starch paste per gelatinised share"),
        gel_m=("-", "estimate", "its exponent"), gel_n=("-", "estimate", "flow index of the cooked paste"))
    for f in fields(S):
        if f.name == "name":
            continue
        unit, label, basis = solid[f.name]
        add(f"FLOUR.{f.name}", getattr(S, f.name), unit, label, basis)
    for name, obj, what in (("PUDDING", T.PUDDING, "instant pudding mix"), ("CAKE", T.CAKE, "cake mix"), ("STARCH", T.STARCH, "corn starch"),
                            ("SPELT", T.SPELT, "whole spelt flour (the cited table: K 6.9 Pa s^n, n 0.33)")):
        diff = {f.name: getattr(obj, f.name) for f in fields(obj) if f.name != "name" and getattr(obj, f.name) != getattr(S, f.name)}
        label = "read" if name == "SPELT" else "estimate"
        basis = (f"{what}: differs from the flour in these constants. " + (
            "set_tau 100 s: 95 % set after 5 minutes, the box's 'soft-set in 5 minutes' (search)" if name == "PUDDING" else
            "no measurement of this mix was found" if name != "SPELT" else "the batter law's two constants from the cited table"))
        add(name, diff, "see FLOUR", label, basis)
    recipe = dict(
        rho_aq=("kg/m3", "literature", "milk"), k_aq=("Pa s", "literature", "milk at 25 C, 2 mPa s"),
        rho_fat=("kg/m3", "literature", "vegetable oil"), rho_sugar=("kg/m3", "literature", "sucrose crystal"),
        void_sugar=("-", "read", "granulated sugar poured, 850 kg/m3 (an engineering table gives 800, opened)"),
        rho_minor=("kg/m3", "literature", "baking powder and salt, grain density"),
        void_minor=("-", "search", "poured baking powder 500-600 kg/m3, salt about 1200: 12 g to 3 g gives about 650"),
        cp=("J/kg/K", "literature", "water phase, fat, sugar, salts, the solid's four states"),
        visc_ea=("J/mol", "literature", "temperature dependence of a watery phase's viscosity"),
        lump_ke=("1/strain", "open", "erosion of a lump per unit strain above its cohesive stress; no measurement found"),
        lump_tauc=("Pa", "open", "cohesive stress of a wet flour lump; none found"),
        wire_strain=("-", "estimate", "strain a wire's boundary layer puts on what it sweeps"),
        d_mol=("m2/s", "literature", "sucrose in water"),
        wet_d=("m2/s", "calibrated", "liquid into dry powder between points. Bounds: capillary rise into a bed of 10 um pores "
                                     "gives L^2/t of about 6e-5 m2/s (Washburn; an upper figure, before the wetted flour "
                                     "swells and closes its pores); no measured figure for flour was found. "
                                     "Set on an outcome: the muffin method's 15 to 20 light strokes until the dry "
                                     "ingredients are just moistened (Quaker Oats, opened). Assisted strokes leave 2 % of "
                                     "the flour dry after 31 strokes at 5e-7, 17.6 at 2e-6 in development "
                                     "runs and 19.4 on the physics as delivered (E15; the "
                                     "spread between runs is about a stroke). Weakly sensitive: the rate goes as about the fifth root of the "
                                     "constant, the strokes decide"),
        conductivity=("W/m/K", "literature", "water-rich food"),
        sound=("m/s", "engine", "numerical speed of sound"), g_paste=("Pa", "engine", "elastic modulus of the trial stress, pastes"),
        g_powder=("Pa", "engine", "the same for powder"),
        k_powder=("Pa", "search", "bulk stiffness of loose flour (510 to 880 kg/m3 under 4.5 kPa; unconfirmed)"),
        k_dry=("Pa s", "engine", "rate term of flowing powder"),
        tau_rest=("Pa", "engine", "rest stress of every wet state: without it a thin liquid never comes to rest (E4). A thin "
                                  "liquid therefore holds to a wall like a 1 Pa paste until it is sheared faster than about 10 1/s"))
    for f in fields(R):
        if f.name in ("solid", "pastes"):
            continue
        unit, label, basis = recipe[f.name]
        add(f"RECIPE.{f.name}", getattr(R, f.name), unit, label, basis)
    for p in (T.MAYO, T.KETCHUP, T.RELISH, T.HONEY, T.MUSTARD, T.PORRIDGE, T.PETFOOD):
        add(f"PASTE {p.name}", dict(rho=p.rho, yield_pa=p.yield_pa, k=p.k, n=p.n, cp=p.cp), "kg/m3, Pa, Pa s^n, -, J/kg/K",
            "search" if p.name in ("mayonnaise", "ketchup", "honey") else "estimate",
            "Herschel-Bulkley constants: published ranges for mayonnaise, ketchup and honey (search "
            "level); estimates for the others")
    add("REFERENCE_STROKE_J_KG", T.REFERENCE_STROKE_J_KG, "J/kg", "measured",
        "work one fold stroke puts into the wetted flour of a muffin batter while it is being combined, the mean over the "
        "method's first 16 strokes; a stroke through batter already mixed "
        "puts in 0.20 (E8)")
    for name, t in T.TASKS.items():
        add(f"CRITERIA {name}", t["criteria"], "see task.py", "authored", "limits; their bases are listed above task.TASKS")
    c = CT.Continuum.__init__.__defaults__
    add("WALL_C", 8.0, "-", "calibrated", "wall rule: tangential velocity taken at wall_c nu/dx^2; the cup's Couette torque at "
                                          "3 Pa s")
    add("HOLD_T", 0.25, "cells", "calibrated", "distance outside a surface within which the wall rule acts (set on the cup's torque)")
    add("KEEP_OUT", "once a tick", "-", "engine",
        "a point of the mixture found inside a solid is set back on its surface along the surface's normal, loses its "
        "speed into the solid, and the solid is given the opposite impulse (continuum.k_keep_out). No constant. The grid "
        "rule alone lets points drift a cell deep into walls")
    add("CORNER_RULE", (0.5, 0.7), "cells, cosine", "engine",
        "wall rule at an inside corner: a point sliding along one face is not let into the next (continuum.k_wall looks "
        "half a cell ahead along the slide; where the surface found there is turned more than 45 degrees, cosine below "
        "0.7, the speed into it is removed). Without the 45 degree bound a curved wall was treated as a corner and the "
        "cup's torque fell 35 %; both numbers are the solver's, not properties of anything")
    add("WIRE_SWEEP", "wire radius", "-", "engine",
        "dry powder (wetted share at most 0.5) inside a wire's own cylinder is set on its surface and takes the wire's "
        "speed along that normal if it is slower; the wire is given the opposite impulse (continuum.k_wire_sweep). No "
        "constant. The wire's drag alone, spread over three cells, moves the powder it meets 0.1 mm a pass (E8)")
    add("WIRE_CARRY", CT.WIRE_CARRY, "share, wire diameters", "measured",
        "dry powder next to a wire goes the wire's way at no less than share * exp(-(r - d/2) / (length * d)) of the "
        "wire's speed, r its distance from the wire's axis (continuum.k_wire_sweep; the grid's own share counts toward "
        "it). Read off the displacement profile behind a wire RESOLVED by the same solver in the same flour (a solid "
        "capsule on cells of 0.3 and 0.4 mm, upright through the surface, drawn sideways at 0.25 m/s: E20): "
        "the powder in the wire's own track is left "
        "3.1 to 3.5 mm farther along, half a diameter to the side 1.8 mm, one diameter 1.1 mm, two 0.46 mm; summed over "
        "the points, 10.4 and 10.5 mm2 carried per length of wire on 0.40 and 0.30 mm cells (6.7 to 6.8 times the "
        "wire's cross-section). The solver's own wire under the rule, in the same bench on the tasks' 2.5 mm cells and "
        "on 1.25 mm cells (E8): 8.4 and 8.1 mm2, four fifths of the resolved wire's and "
        "the same on both cells within 4 %; a rule that only moves powder out of the wire's way, read the same way "
        "in the same bench, gives 0.27 and 3.5 mm2 (and lying 10 mm deep 2.7 and 6.2, where the rule gives 8.4 and 10.3). The bench itself is held to an "
        "outside answer: the same resolved wire drawn slowly through a purely cohesive material is resisted by 8.5 to "
        "8.7 times strength times diameter, where the limit load is 9.14 (smooth) to 11.94 (rough) (Randolph and "
        "Houlsby 1984). Not in the rule: at 0.6 m/s the resolved wire carries 14.1 mm2, the solver's 8.4 on 2.5 mm cells "
        "and 12.8 on 1.25 mm cells; a wire lying 10 mm under the powder carries the powder above it farther, 22.6 to "
        "23.2 mm2 resolved (24.3 at 0.6 m/s) against the solver's 8.4 and 10.3. Two wires side by side: three diameters "
        "apart the resolved pair carries 21.1 mm2 a wire and the powder midway 7.1 mm; the solver 19.2 mm2 and 3.6 mm on "
        "2.5 mm cells (one cell spans the gap and the grid carries the powder between them) and 10.1 mm2 and 1.9 mm on "
        "1.25 mm cells; five apart 11.4 against 9.3 and 12.8; eight apart the resolved slab's side walls stand 4.4 mm "
        "from each wire and the case is not used (4.9 against 7.9 and 6.9). The wire's drag in the flour, upright at "
        "0.25 m/s: resolved 3.7 to 3.8 N/m, the solver's 1.8 on 2.5 mm cells and 2.9 on 1.25 mm cells")
    add("WIRE_CUT", CT.WIRE_CUT, "wire diameters", "measured",
        "open volume a wire leaves in the dry powder or the stiff paste it goes through, per unit of the area it "
        "sweeps (continuum.k_wire_sweep: `cut`, on the points within a diameter of the wire's axis). The strip the "
        "resolved wire went through, one diameter wide, holds 0.43 to 0.50 of the powder it held (upright, 0.25 m/s; "
        "0.55 to 0.64 lying 10 mm deep; 0.13 to 0.15 in the cohesive material)")
    add("CUT_REACH_D", CT.CUT_REACH_D, "wire diameters", "measured",
        "how far a liquid follows a wire's cut into dry powder from where the cut borders free liquid "
        "(continuum._slit). With WIRE_CUT it sets how much liquid a wire takes with it into a heap: 0.5 d times 6 d, "
        "3.0 wire diameters squared per length of wire and entry. Measured with the resolved wire drawn from a pool of "
        "thin liquid (0.01 Pa s) into a bed of flour standing beside it (E8): "
        "71.0 and 71.2 mm3 of liquid in the bed for 11.9 mm of wire on 0.40 and 0.30 mm cells, "
        "6.0 mm2 per length, 3.05 diameters squared, all of it within two diameters of the cut; 68.4 mm3 (2.94) for a "
        "cut twice as long, so it is a dose at the entry, not a share of the cut's length; 73.3 mm3 (3.15) at 0.60 m/s; "
        "a round wire 4 mm across (a beater strip's width; no flat strip was resolved) at 0.60 m/s 2.63 of its diameter "
        "squared. Where the liquid lies along the cut is NOT "
        "settled by the bench: within 6 mm of the bed's face on 0.40 mm cells, along all 15 mm on 0.30 mm cells. A heap "
        "free to give way takes more: 12 mm wide, liquid in 10 of the cut's 15 stretches, 0.80 diameters per cut area, "
        "and the one wire pushed 13 % of the powder out of the heap's footprint; 17 mm wide, 7 of 15 stretches, 0.43, "
        "2.7 %. The solver applies the bed's figure everywhere: it understates what a whisk does to a small or loose heap")
    add("CUT_PASTE_COST", CT.CUT_PASTE_COST, "cells of reach per cell", "measured",
        "a cut through a stiff wet paste lets the liquid less far than one through dry powder: the resolved wire drawn "
        "from the pool into a bed of the cohesive material (200 Pa) took 42.0 mm3 with it, 3.5 mm2 per length, 1.80 "
        "diameters squared; 6 / 1.7 = 3.5 diameters of reach times 0.5 gives 1.76")
    add("CUT_HOLD_PA", CT.CUT_HOLD_PA, "Pa", "estimate",
        "yield stress from which a wet mixture keeps a wire's cut at all (below it the mixture closes behind the "
        "wire and only the viscous law acts). About the weight of 5 mm of batter; no measurement")
    add("CUT_THIN_PA_S", CT.CUT_THIN_PA_S, "Pa s^n", "measured",
        "consistency up to which free liquid is taken into a cut. The resolved wire took 91.5 mm3 of a liquid of 1 Pa s "
        "with it, against 71.2 of one of 0.01 Pa s (3.93 and 3.06 diameters squared): the rule takes the same dose for "
        "both and understates the thicker by about a quarter; thicker was not tried")
    add("WIRE_BEARING", 10.5, "-", "literature", "plastic resistance of a cylinder pushed sideways through a material of shear "
                                                 "strength s: 9.14 s d (smooth) to 11.94 s d (rough) per length (Randolph and Houlsby 1984)")
    add("MIX_C", "(0.17 dx / spacing)^2 / 0.7", "-", "literature", "sub-grid scalar transport bounded by the strain rate: "
                                                                    "Smagorinsky constant 0.17 (Lilly), turbulent Schmidt number 0.7; not calibrated here")
    add("FREE_SURFACE", CT.FREE_SURFACE, "-", "measured", "free-surface correction of the gathered fraction")
    add("HOB FLOOR_PA_S", H.FLOOR_PA_S, "Pa s", "measured", "the solver's numerical viscosity (0.04-0.07 Pa s), below which the "
                                                           "convection closure acts")
    add("HOB RENEW_STRAIN", H.RENEW_STRAIN, "-", "estimate", "strain of the bottom layer that renews a patch's film")
    add("HOB CHAR_STRAIN", H.CHAR_STRAIN, "-", "estimate", "strain of the layer on a burnt patch that scrapes its flecks into the food")
    add("HOB CARRIER", H.CARRIER, "kg kJ/(s m2 kPa kg)", "literature",
        "Carrier's evaporation coefficient for a still water surface (ASHRAE; quoted from memory, and used above a "
        "pool's temperatures: estimate). It is used below 50 C only, scaled by 0.95 to meet "
        "the measured pot there (HOB EVAPORATION). Against a table "
        "(Engineering ToolBox, heat loss from open water tanks in still air, 170 to 200 F) the law gives 0.83 to 0.84 "
        "of the table's loss; the measured pot says the same and more (0.59 at 98 C)")
    add("HOB EVAPORATION", dict(zip(H.EVAPORATION_C, H.EVAPORATION_G_H)), "C: g/h from a pot 25 cm across", "read",
        "evaporation from the food's surface from 50 C up: what 5 litres of water in an open stainless pot on an electric "
        "range lost at each temperature it was held at (A. Berick, Heat Losses In A Cook Pot at Constant Temperature, "
        "Aprovecho Research Center, figure 5: the numbers are printed on the chart), per "
        "area of the pot. Carrier's law (HOB CARRIER) alone gives 1.05 of these at 50 C, 0.80 at "
        "80 C, 0.76 at 90 C and 0.59 at 98 C. Measured on water, used for a sauce and a porridge, whose surfaces skin "
        "over: transferred")
    add("HOB AMBIENT", (H.AMBIENT_C, H.AMBIENT_RH), "C, -", "authored", "the kitchen's air")
    add("HOB PATCHES", (H.N_RING, H.N_SECTOR), "-", "engine", "the pan bottom's patches, about 2 cm2 each")
    from . import render as RD
    add("DRAWING PASTE_SCATTER", dict(RD.PASTE_SCATTER), "-", "estimate",
        "how strongly ketchup, relish and honey scatter light against mayonnaise, for the DRAWN colour of blended "
        "pastes (two-constant Kubelka-Munk; no effect on the mixture or on the evaluator). Arithmetic from "
        "droplet and particle sizes gives 0.02 to 0.1 for ketchup; no measurement of either paste was found. Checked "
        "against one observation that was opened: two parts of mayonnaise with one of ketchup are 'pale coral "
        "rather than tomato-red' (Kelvin's Kitchen, Salsa Rosada) and 'an even pink' (The Latin Spoon). Drawn sRGB of "
        "that blend: (0.92, 0.53, 0.52); with 0.05, (0.94, 0.62, 0.60); with 0.2, (0.90, 0.44, 0.43); with every paste "
        "scattering alike, (0.82, 0.27, 0.27)")
    add("DRAWING STEAM", dict(shells=RD.SHELLS, opacity_a_shell=0.06, light_of_its_own=150.0), "-", "authored",
        "how a puff of vapour is drawn (render.puffs and the 'steam' material): set by looking at the simmering "
        "custard from the overhead and the close cameras. The NUMBER of "
        "puffs is the hob model's vapour leaving the pan, one for every 4 mg/s")
    return out


if __name__ == "__main__":
    for e in entries():
        print(f"{e['name']:28s} {str(e['value'])[:60]:62s} {e['unit']:22s} {e['label']:11s} {e['basis']}")
