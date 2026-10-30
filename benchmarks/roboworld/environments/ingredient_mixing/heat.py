"""Hob, pan and the film of food on the pan bottom.

The knob's detent sets the hob's power (params.HOB["levels_w"]). The power is deposited in the pan's base in a ring
(mean radius 45 mm), only while the pan stands on the zone. The base is a plate of patches (rings x sectors, about
2 cm2 each) with the tri-ply's heat capacity and in-plane conduction; the wall is one more node.

Between a patch and the food above it there is a film thinner than a grid cell, and it decides whether food catches:
  * a still layer takes heat by conduction as a semi-infinite slab does, q = (T_w - T_b) sqrt(k rho c / (pi t_r)),
    with T_b the temperature of the food in the cell above and t_r the time since the patch was renewed (penetration
    theory, the scraped-surface model). Shear in the food on the patch renews it: t_r decays with the strain rate there.
  * a wet patch hotter than boiling passes heat as a boiling surface does, q = B (T_w - 100)^3 (nucleate boiling,
    Rohsenow's correlation for water on polished stainless steel): the plate runs some kelvin above 100 C and the heat
    goes into the food above it. While that food is below boiling the bubbles made on the plate condense in it
    (subcooled boiling: it is why a kettle sings before it boils), so nothing is lost.
  * food brought to 100 C makes vapour with whatever more it is given. Vapour made in the layer on a patch is what
    dries that patch's film: when the film's water is gone the patch has burnt on; it is marked and it stays. Where
    the vapour goes: it condenses in the food that is still below boiling (and heats it) or leaves as steam, by how
    near the whole pan is to the boil (params.HOB simmer_c to 100 C).
  * a burnt-on patch has its dried film between the plate and the food: a layer in series, of the film's thickness
    (less what a tool has scraped off) and the conductivity of a fouling deposit (params.HOB burnt_k). Conduction
    into the food and boiling go on from the layer's top, so the plate under it runs hotter by the layer's
    temperature drop (some tens of kelvin at the ring's heat flux) and settles there. The layer does not grow
    thicker with time as a real one does, and nothing chars further: left out.
A renewed patch that is not burnt gets a fresh film.

The hob has the two protections of the portable induction hob it is modelled on: it heats only while the pan stands on
the zone, and it shuts down when its sensor, under the glass at the zone's centre, reads more than 238 C (the manual's
error E2). The sensor follows the pan's base at the centre with the glass's lag. A hob that has cut out stays off until
its knob has been turned to 0.

The food's surface loses heat by evaporation at the top layer's temperature: from 50 C up what an open pot of water on
a range was measured to lose per area (EVAPORATION_*), below that Carrier's equation for a still water surface. It is
what makes a hot pan steam long before it boils, and what bends its heating curve from about 85 C on. The vapour leaving,
from the surface and from food at the boil, is the pan's steam (state: steam_g_s, boiling_w), which the renderer draws
and a camera sees. The hob puts its level's watts times its efficiency into the pan (params.HOB efficiency). A burnt-on patch that is scraped (shear in the layer on it) gives flecks of its film
to the food there (materials.ST_CHAR; drawn brown, carried with the food).

Liquids thinner than the continuum's numerical viscosity cannot convect as they should; for those the conduction
between points is multiplied by the Nusselt number of a heated plate facing up, Nu = 0.15 Ra^(1/3) (McAdams), computed
with the liquid's true viscosity. It fades out as the mixture thickens. This is a closure for a known defect of the
solver, labelled as one; E12 shows the run with and without it.
"""

import math
import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from . import params as P
from . import materials as M

N_RING, N_SECTOR = 8, 12
BOIL_C = 100.0
FLOOR_PA_S = 0.05                # the continuum's numerical viscosity (E3: 0.04-0.07 Pa s with the viscous wall rule;
                                 # 0.5-0.6 with every wall node held): below it the convection closure acts
AMBIENT_C, AMBIENT_RH = 20.0, 0.5
CARRIER = 0.089                  # kg/s per m2 per kPa x kJ/kg in still air: w = A (p_w - p_a) 0.089 / Y (Carrier 1918, as
                                 # the ASHRAE handbook gives it for pool surfaces; used here far above a pool's
                                 # temperature: estimate, from memory of the handbook)
# Evaporation from an open pot of hot water, MEASURED: grams an hour from 5 litres in a stainless pot 25 cm across on
# an electric range, held at each temperature, no lid (A. Berick, Heat Losses In A Cook Pot at Constant Temperature,
# Aprovecho Research Center, figure 5; the numbers are printed on its chart). Carrier's law above gives 1.05 of these at 50 C, 0.80 at 80 C, 0.76 at 90 C and 0.59
# at 98 C: a pool's law in still air does not know the plume that rises off a hot pot. From 50 C up the pan's
# surface therefore loses what this pot lost per area at its temperature (straight lines between the points, and
# the last line carried on to the boil); below 50 C Carrier's law, scaled to meet the table there.
EVAPORATION_C = (50.0, 60.0, 70.0, 80.0, 90.0, 92.0, 94.0, 96.0, 98.0)
EVAPORATION_G_H = (74.0, 146.0, 240.0, 410.0, 648.0, 706.0, 824.0, 954.0, 1128.0)
EVAPORATION_POT_M2 = math.pi*0.125**2
CHAR_STRAIN = 3.0                # strain of the layer on a burnt patch that scrapes its film's flecks into the food


def evaporation_kg_s_m2(t_c):
    """Water leaving a still surface at t_c into the kitchen's air, kg/s per m2: the measured pot from 50 C up
    (EVAPORATION_*), Carrier's law below, the two meeting at 50 C."""
    carrier = lambda t: max(0.0, p_sat_kpa(min(t, BOIL_C)) - AMBIENT_RH*p_sat_kpa(AMBIENT_C))*CARRIER/(P.HOB["latent"]*1e-3)
    t0, g0 = EVAPORATION_C[0], EVAPORATION_G_H[0]/3.6e6/EVAPORATION_POT_M2
    if t_c <= t0:
        return carrier(t_c)*g0/carrier(t0)
    t = min(float(t_c), BOIL_C)
    tc, gh = EVAPORATION_C, EVAPORATION_G_H
    if t >= tc[-1]:
        g = gh[-1] + (gh[-1] - gh[-2])/(tc[-1] - tc[-2])*(t - tc[-1])
    else:
        g = float(np.interp(t, tc, gh))
    return g/3.6e6/EVAPORATION_POT_M2


def p_sat_kpa(t_c):
    """Saturation pressure of water vapour (Magnus form), kPa."""
    return 0.61094*math.exp(17.625*t_c/(t_c + 243.04))
RENEW_STRAIN = 1.0               # strain of the bottom layer that renews a patch's film


@wp.kernel
def k_patch_gather(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), temp: wp.array(dtype=float),
                   heatcap: wp.array(dtype=float), gdot: wp.array(dtype=float), cons: wp.array(dtype=float),
                   vpos: wp.vec3, vrot: wp.quat, radius: float, layer: float, n_ring: int, n_sector: int,
                   z_top: float, sums: wp.array2d(dtype=float), bulk: wp.array(dtype=float)):
    """Per patch, over the points in the layer on the pan bottom: sum of heat capacity, of capacity x temperature, of
    capacity x strain rate; and over all points: capacity, capacity x temperature, capacity x consistency."""
    p = wp.tid()
    if flags[p] != 0:
        return
    pl = wp.quat_rotate_inv(vrot, x[p] - vpos)
    c = heatcap[p]
    wp.atomic_add(bulk, 0, c)
    wp.atomic_add(bulk, 1, c*temp[p])
    wp.atomic_add(bulk, 2, c*cons[p])
    wp.atomic_add(bulk, 3, c*pl[2])
    wp.atomic_add(bulk, 4, c*pl[2]*temp[p])
    wp.atomic_add(bulk, 5, c*pl[2]*pl[2])
    if pl[2] > z_top:                    # the top layer: what the surface's evaporation cools
        wp.atomic_add(bulk, 6, c)
        wp.atomic_add(bulk, 7, c*temp[p])
    if temp[p] < 99.5:                   # food below boiling: where vapour made elsewhere condenses
        wp.atomic_add(bulk, 8, c)
    if pl[2] > layer:
        return
    r = wp.sqrt(pl[0]*pl[0] + pl[1]*pl[1])
    if r >= radius:
        return
    i = wp.min(int(r/radius*float(n_ring)), n_ring - 1)
    a = wp.atan2(pl[1], pl[0])
    if a < 0.0:
        a = a + 6.2831853
    j = wp.min(int(a/6.2831853*float(n_sector)), n_sector - 1)
    k = i*n_sector + j
    wp.atomic_add(sums, k, 0, c)
    wp.atomic_add(sums, k, 1, c*temp[p])
    wp.atomic_add(sums, k, 2, c*gdot[p])


@wp.kernel
def k_patch_heat(x: wp.array(dtype=wp.vec3), flags: wp.array(dtype=int), temp: wp.array(dtype=float),
                 vpos: wp.vec3, vrot: wp.quat, radius: float, layer: float, n_ring: int, n_sector: int,
                 dtemp: wp.array(dtype=float), cap_c: float, lost: wp.array(dtype=float), heatcap: wp.array(dtype=float),
                 z_top: float, top_dt: float, dchar: wp.array(dtype=float), state: wp.array2d(dtype=float), st_char: int,
                 cond_dt: float, lostp: wp.array(dtype=float), st_peak: int):
    """Give each point of the bottom layer its patch's temperature rise and, over a burnt patch being scraped, its
    share of the film's flecks; cool the top layer by the surface's evaporation; give the food below boiling the heat
    of the vapour that condenses in it; cap the mixture at boiling (the surplus is vapour: counted per patch for the
    points of the bottom layer, in one sum for the rest); keep the highest temperature each point has had."""
    p = wp.tid()
    if flags[p] != 0:
        return
    pl = wp.quat_rotate_inv(vrot, x[p] - vpos)
    t = temp[p]
    if t < 99.5:
        t = t + cond_dt
    if pl[2] > z_top:
        t = t - top_dt
    k = int(-1)
    if pl[2] <= layer:
        r = wp.sqrt(pl[0]*pl[0] + pl[1]*pl[1])
        if r < radius:
            i = wp.min(int(r/radius*float(n_ring)), n_ring - 1)
            a = wp.atan2(pl[1], pl[0])
            if a < 0.0:
                a = a + 6.2831853
            j = wp.min(int(a/6.2831853*float(n_sector)), n_sector - 1)
            k = i*n_sector + j
            t = t + dtemp[k]
            state[p, st_char] = wp.min(1.0, state[p, st_char] + dchar[k])
    if t > cap_c:
        if k >= 0:
            wp.atomic_add(lostp, k, heatcap[p]*(t - cap_c))
        else:
            wp.atomic_add(lost, 0, heatcap[p]*(t - cap_c))
        t = cap_c
    temp[p] = t
    state[p, st_peak] = wp.max(state[p, st_peak], t)


class Hob:
    def __init__(self, world):
        self.w = world
        sc = world.scene
        self.pan = sc.vessel
        self.knob = sc.tools["knob"]
        s = P.PAN
        self.R = s["radius"]
        n = N_RING*N_SECTOR
        edges = np.linspace(0, self.R, N_RING + 1)
        self.ring_r = 0.5*(edges[:-1] + edges[1:])
        self.area = np.repeat(math.pi*(edges[1:]**2 - edges[:-1]**2)/N_SECTOR, N_SECTOR)        # m2 per patch
        steel_t = s["wall"] - s["core"]
        self.cap_area = s["core"]*2700.*900. + steel_t*7900.*500.                                # J/m2/K
        self.kt = s["core"]*P.AL_K + steel_t*P.STEEL_K                                            # W/K (in-plane)
        w = np.exp(-0.5*((self.ring_r - P.HOB["ring_r"])/(0.5*P.HOB["ring_w"]))**2)*self.ring_r
        self.deposit = np.repeat(w/(w*1.0).sum()/N_SECTOR, N_SECTOR)                             # share of power per patch
        self.T = np.full(n, 20.0)                # plate temperature per patch
        self.T_wall = 20.0
        self.wall_cap = 2*math.pi*self.R*s["depth"]*self.cap_area
        self.wall_area = 2*math.pi*self.R*s["depth"]
        self.t_r = np.full(n, 1.0)               # s since renewed
        self.burnt = np.zeros(n, bool)
        self.layer = 2.0*world.cont.dx
        self.film_water0 = None                  # kg/m2 of water in a fresh film (set from the mixture at first step)
        self.film = None
        # the ledger, joules since the build: electric_j the level's watts; hob_j what reaches the pan (times the
        # efficiency); to_food_j from the plate into the food (wall_boiling_j the part of it passed by boiling on
        # the plate); evaporated_j from the surface; boiled_j and vapour_j the vapour that LEFT as steam, made in the
        # layer on the pan bottom and in the rest of the food; condensed_j the vapour that condensed in food below
        # boiling (heat moved inside the pan, no loss); loss_j from the pan's wall to the room
        self.energy = dict(electric_j=0., hob_j=0., to_food_j=0., wall_boiling_j=0., boiled_j=0., loss_j=0., vapour_j=0.,
                           condensed_j=0., evaporated_j=0.)
        self.cond_j = 0.0                        # vapour made in the last step that condenses in the food in the next
        self.char = np.zeros(n)                  # share of each burnt patch's film already scraped into the food
        self.z_top = 1.0                         # vessel-frame height above which points are the top layer (set each step)
        self.surface_c, self.steam_g_s, self.boiling_w = 20.0, 0.0, 0.0
        # the highest mean temperature the sauce in the pan has had, taken after every step (record_food_peak): what a
        # probe standing in the sauce would have shown at its highest. A record for the evaluator (task.evaluate,
        # `cooked`); it changes nothing physical
        self.food_peak_c = None
        # seconds the sauce in the pan has been at the boil (its mean at simmer_c or above) while the hob delivered
        # power, in total: a record for the evaluator (`not_boiled_on`); it changes nothing physical
        self.boil_s = 0.0
        self.conduction_gain = 1.0
        self.nusselt = True
        self.power_w = 0.0
        self.on_zone = True
        # the hob's overheat cut-out (its manual's error E2): a sensor under the glass at the zone's centre follows the
        # pan's base there with the glass's lag; above the limit the hob shuts down and stays off until its knob has
        # been turned to 0
        self.sensor_c = 20.0
        self.cut_out = False
        dev = world.cont.device
        self._sums = wp.zeros((n, 3), dtype=float, device=dev)
        self._bulk = wp.zeros(10, dtype=float, device=dev)
        self._lostp = wp.zeros(n, dtype=float, device=dev)
        self._dtemp = wp.zeros(n, dtype=float, device=dev)
        self._dchar = wp.zeros(n, dtype=float, device=dev)
        self._lost = wp.zeros(1, dtype=float, device=dev)
        self.history = []

    def burnt_area(self):
        return float(self.area[self.burnt].sum())

    def scrape(self):
        """Every tick: a tool's working edge moving over a burnt-on patch whose film has not been scraped off yet is
        resisted by the work of scraping it, params.HOB scrape_j_m2 per area swept: a force of that times the edge's
        width, against the motion (a fraction of a newton: felt in the joint torques, not a brake)."""
        self.scrape_n = 0.0
        if not self.burnt.any():
            return
        w = self.w
        q = w.bq[self.pan.body]
        Rv = Rotation.from_quat(q[3:])
        for t in w.scene.tools.values():
            width = getattr(t, "edge_w", None)
            if width is None:
                continue
            body, off = t.tip
            qb = w.bq[body]
            Rb = Rotation.from_quat(qb[3:])
            tip = qb[:3] + Rb.apply(off)
            pl = Rv.inv().apply(tip - q[:3])
            r = float(np.hypot(pl[0], pl[1]))
            if pl[2] > self.layer + 0.004 or r >= self.R:
                continue
            i = min(int(r/self.R*N_RING), N_RING - 1)
            j = min(int((math.atan2(pl[1], pl[0]) % (2*math.pi))/(2*math.pi)*N_SECTOR), N_SECTOR - 1)
            k = i*N_SECTOR + j
            left = 1.0 - float(self.char[k])
            if not self.burnt[k] or left < 0.02:
                continue
            com = qb[:3] + Rb.apply(w.body_com[body])
            v = w.bqd[body, :3] + np.cross(w.bqd[body, 3:], tip - com)
            axis = Rv.apply([0., 0., 1.])
            v = v - axis*(v @ axis)
            speed = float(np.linalg.norm(v))
            if speed < 1e-3:
                continue
            F = -P.HOB["scrape_j_m2"]*width*left*v/speed
            w.add_force(body, F, tip)
            self.scrape_n += float(np.linalg.norm(F))

    def step(self, dt):
        w, cont, pan = self.w, self.w.cont, self.pan
        q = w.bq[pan.body]
        up = Rotation.from_quat(q[3:]).apply([0., 0., 1.])[2]
        zone = np.array(P.HOB_ZONE)
        self.on_zone = bool(np.linalg.norm(q[:2] - zone) < 0.04 and up > 0.985 and abs(q[2] - w.scene.vessel_start[2]) < 0.005)
        self.sensor_c += (float(self.T[:N_SECTOR].mean()) - self.sensor_c)*(1.0 - math.exp(-dt/P.HOB["sensor_tau"]))
        if self.knob.s_level == 0:
            self.cut_out = False
        elif self.sensor_c > P.HOB["cutout_c"] and not self.cut_out:
            self.cut_out = True
            w.log_event(dict(type="hob_cut_out", tick=int(w.tick), sensor_c=round(self.sensor_c, 1),
                             plate_max_c=round(float(self.T.max()), 1)))
        self.power_w = float(P.HOB["levels_w"][self.knob.s_level]) if self.on_zone and not self.cut_out else 0.0
        self.pan_w = P.HOB["efficiency"]*self.power_w             # what reaches the pan's base
        n = N_RING*N_SECTOR
        vp, vr = wp.vec3(*q[:3]), wp.quat(*q[3:])
        self._sums.zero_(); self._bulk.zero_()
        wp.launch(k_patch_gather, dim=cont.capacity, inputs=[cont.x, cont.flags, cont.temp, cont.heatcap, cont.gdot,
                                                             cont.cons, vp, vr, self.R, self.layer, N_RING, N_SECTOR,
                                                             float(self.z_top), self._sums, self._bulk], device=cont.device)
        s = self._sums.numpy().astype(np.float64)
        b = self._bulk.numpy().astype(np.float64)
        cap = s[:, 0]
        has = cap > 1e-9
        T_b = np.where(has, s[:, 1]/np.maximum(cap, 1e-12), 20.0)
        gdot = np.where(has, s[:, 2]/np.maximum(cap, 1e-12), 0.0)
        r = w.recipe
        k_food, rho_c = r.conductivity, 1030.0*3900.0
        if self.film_water0 is None:
            self.film_water0 = P.HOB["film"]*1030.0*0.85
            self.film = np.full(n, self.film_water0)
        # renewal of the film by shear in the layer on the patch
        self.t_r = (self.t_r + dt)*np.exp(-gdot*dt/RENEW_STRAIN)
        renewed = (self.t_r < 0.5) & ~self.burnt
        self.film[renewed] = np.minimum(self.film_water0, self.film[renewed] + self.film_water0*gdot[renewed]*dt/RENEW_STRAIN)
        h = np.sqrt(k_food*rho_c/(math.pi*np.maximum(self.t_r, 0.05)))            # W/m2/K, penetration theory
        # a liquid thinner than the solver's floor convects off a hot plate on its own (it cannot in the continuum):
        # the plate's natural-convection coefficient, Nu = 0.15 Ra^(1/3), which does not depend on the length scale,
        # with the liquid's true viscosity; it fades out as the mixture thickens past the floor
        k_true = getattr(self, "mean_k", 1.0)
        if self.nusselt and k_true < FLOOR_PA_S:
            fade = min(1.0, (FLOOR_PA_S - k_true)/(0.5*FLOOR_PA_S))
            nu_true = max(k_true, 1.0e-3)/1030.0
            dT = np.maximum(self.T - T_b, 0.0)
            h_nc = 0.15*k_food*(P.GRAVITY*4.0e-4*dT/(nu_true*k_food/rho_c))**(1/3)
            h = np.maximum(h, fade*h_nc)
        self.h_mean = float(h[has].mean()) if has.any() else 0.0
        h = np.where(has, h, 0.0)
        # plate: power in, in-plane conduction (ring to ring, sector to sector), to the food, to the wall
        T = self.T.reshape(N_RING, N_SECTOR)
        q_in = self.pan_w*self.deposit                                             # W per patch
        dr = self.R/N_RING
        lap = np.zeros_like(T)
        for i in range(N_RING):
            if i > 0:
                lap[i] += self.kt*(2*math.pi*(i*dr)/N_SECTOR)/dr*(T[i - 1] - T[i])
            if i < N_RING - 1:
                lap[i] += self.kt*(2*math.pi*((i + 1)*dr)/N_SECTOR)/dr*(T[i + 1] - T[i])
            arc = 2*math.pi*self.ring_r[i]/N_SECTOR
            lap[i] += self.kt*dr/arc*(np.roll(T[i], 1) + np.roll(T[i], -1) - 2*T[i])
        lap = lap.ravel()
        to_wall = np.zeros(n)
        outer = slice((N_RING - 1)*N_SECTOR, n)
        to_wall[outer] = self.kt*(2*math.pi*self.R/N_SECTOR)/(0.5*dr)*(self.T[outer] - self.T_wall)
        # a patch with food on it passes heat by conduction into the food and, above boiling, by nucleate boiling,
        # B (T - 100)^3 per area. On a burnt-on patch both start at the top of the dried film, which lies between the
        # plate and the food as a layer of conductance burnt_k over what is left of its thickness (the share already
        # scraped off is gone): G (T_plate - T_top) = h (T_top - T_food) + B (T_top - 100)^3 gives the top's
        # temperature (Newton, the function is monotone). The step is taken with the exchange linearised about the
        # plate's temperature (it is stiff: at 7 K above boiling the boiling term alone changes by 20 kW/m2 per
        # kelvin), and the food is given exactly what the plate gives up
        B = P.HOB["boil_b"]
        burnt = self.burnt & has
        G = P.HOB["burnt_k"]/(P.HOB["film"]*np.maximum(1.0 - self.char, 1.0e-3))  # W/m2/K, the layer left
        T_top = self.T.copy()
        if burnt.any():
            Tp, Tb, hb, Gb = self.T[burnt], T_b[burnt], h[burnt], G[burnt]
            ts = np.clip((Gb*Tp + hb*Tb)/(Gb + hb), np.minimum(Tp, Tb), np.maximum(Tp, Tb))
            for _ in range(30):
                sb = np.maximum(ts - BOIL_C, 0.0)
                f = Gb*(Tp - ts) - hb*(ts - Tb) - B*sb**3
                step = f/(Gb + hb + 3.0*B*sb**2)
                ts = ts + step
                if float(np.abs(step).max()) < 1.0e-9:
                    break
            T_top[burnt] = ts
        self.T_top = T_top
        sup = np.where(has, np.maximum(T_top - BOIL_C, 0.0), 0.0)
        q_food = h*self.area*(T_top - T_b)                                         # W per patch (may be negative)
        q_boil = B*self.area*sup**3                                                # W per patch
        direct = h + 3.0*B*sup**2                                                  # d(q)/dT_top per area, W/m2/K
        stiff = self.area*np.where(burnt, G*direct/(G + direct), direct)           # d(q_food + q_boil)/dT_plate, W/K
        boil_share = np.where(direct > 0.0, 3.0*B*sup**2/np.maximum(direct, 1.0e-12), 0.0)
        C = self.cap_area*self.area
        dT_plate = dt*(q_in + lap - to_wall - q_food - q_boil)/(C + dt*stiff)
        q_out = q_food + q_boil + stiff*dT_plate                                   # W per patch into the food
        self.T = self.T + dT_plate
        loss = P.HOB["loss_w_m2k"]*self.wall_area*(self.T_wall - 20.0)
        self.T_wall += dt*(to_wall.sum() - loss)/self.wall_cap
        # heat into the food of each patch's layer
        dtemp = np.where(has, q_out*dt/np.maximum(cap, 1e-12), 0.0)
        self._dtemp.assign(dtemp.astype(np.float32))
        # a burnt patch whose layer is sheared (the stirrer's edge on it) gives up flecks of its film: the share
        # 1 - exp(-strain / CHAR_STRAIN) of what it still has, spread over the food of its layer
        give = np.where(self.burnt & has, (1.0 - self.char)*(1.0 - np.exp(-gdot*dt/CHAR_STRAIN)), 0.0)
        self.char = self.char + give
        food_kg = np.maximum(cap, 1e-12)/3900.0
        film_kg = P.HOB["film"]*1030.0*self.area                                  # the patch's film, as it was wet
        # ... as the share of that food's points that carry a fleck (the renderer draws that share of them dark):
        # eight times the film's mass share, so that a scraped patch reads as specks and not as a tint
        self._dchar.assign((give*np.minimum(1.0, 8.0*film_kg/food_kg)).astype(np.float32))
        # the food's surface: evaporation at the top layer's temperature (evaporation_kg_s_m2), taken from that layer
        top_c = float(b[7]/b[6]) if b[6] > 1e-9 else float(getattr(self, "mean_temp", AMBIENT_C))
        self.surface_c = top_c
        evap_kg_s = (math.pi*self.R**2)*evaporation_kg_s_m2(top_c) if b[6] > 1e-9 else 0.0
        q_evap = evap_kg_s*P.HOB["latent"]
        top_dt = q_evap*dt/b[6] if b[6] > 1e-9 else 0.0
        # the vapour made in the last step that did not leave condenses in the food below boiling, evenly by heat
        # capacity (the bubbles rise into it and collapse)
        cond_dt = self.cond_j/b[8] if b[8] > 1e-9 else 0.0
        back = self.cond_j if b[8] <= 1e-9 else 0.0                                # nowhere to condense: it leaves
        self.cond_j = 0.0
        self._lost.zero_(); self._lostp.zero_()
        wp.launch(k_patch_heat, dim=cont.capacity, inputs=[cont.x, cont.flags, cont.temp, vp, vr, self.R, self.layer,
                                                           N_RING, N_SECTOR, self._dtemp, BOIL_C, self._lost, cont.heatcap,
                                                           float(self.z_top), float(top_dt), self._dchar, cont.state,
                                                           int(M.ST_CHAR), float(cond_dt), self._lostp, int(M.ST_PEAK)],
                  device=cont.device)
        # vapour: what the food at 100 C could not take. Made in the layer on a patch it dries that patch's film ...
        vap_patch = self._lostp.numpy().astype(np.float64)                         # J this step, per patch
        vap_rest = float(self._lost.numpy()[0])
        wet = (~self.burnt) & has
        drying = wet & (vap_patch > 0.0)
        self.film = np.where(drying, self.film - vap_patch/(self.area*P.HOB["latent"]), self.film)
        newly = wet & (self.film <= 1e-9)
        if newly.any():
            self.film[newly] = 0.0
            self.burnt |= newly
            w.log_event(dict(type="burnt_on", tick=int(w.tick), patches=int(newly.sum()),
                             area_cm2=round(self.burnt_area()*1e4, 2)))
        # ... and it leaves the pan as steam by how near the whole pan is to the boil: none of it below simmer_c (the
        # bubbles collapse in the food above them), all of it at 100 C
        t_mean = b[1]/b[0] if b[0] > 0 else AMBIENT_C
        leave = float(np.clip((t_mean - P.HOB["simmer_c"])/(BOIL_C - P.HOB["simmer_c"]), 0.0, 1.0)) if b[8] > 1e-9 else 1.0
        made = float(vap_patch.sum()) + vap_rest
        left = leave*made + back
        self.cond_j = (1.0 - leave)*made
        e = self.energy
        e["electric_j"] += self.power_w*dt
        e["hob_j"] += self.pan_w*dt
        e["to_food_j"] += float((q_out*dt)[has].sum())
        e["wall_boiling_j"] += float(((q_boil + boil_share*stiff*dT_plate)*dt)[has].sum())
        e["boiled_j"] += leave*float(vap_patch.sum())
        e["vapour_j"] += leave*vap_rest + back
        e["condensed_j"] += self.cond_j
        e["loss_j"] += loss*dt
        e["evaporated_j"] += q_evap*dt
        # what a camera sees of this: steam (all vapour leaving, smoothed over 2 s) and the boil (vapour that leaves
        # the food as bubbles)
        a_ = 1.0 - math.exp(-dt/2.0)
        self.steam_g_s += a_*((q_evap*dt + left)/(P.HOB["latent"]*dt)*1e3 - self.steam_g_s)
        self.boiling_w += a_*(left/dt - self.boiling_w)
        # convection closure for liquids below the solver's viscosity floor
        if b[0] > 0:
            k_mean = b[2]/b[0]
            zm = b[3]/b[0]
            var = b[5]/b[0] - zm*zm
            dTdz = (b[4]/b[0] - zm*(b[1]/b[0]))/max(var, 1e-12)                   # K/m, least-squares vertical gradient
            depth = math.sqrt(max(12.0*var, 1e-8))
            self.z_top = zm + 0.5*depth - 2.0*cont.dx                              # the top two cells of the food
            dT = max(0.0, -dTdz*depth)                                             # hotter below
            if self.nusselt and k_mean < FLOOR_PA_S and dT > 0.05:
                nu_true = max(k_mean, 1.0e-3)/1030.0
                ra = P.GRAVITY*4.0e-4*dT*depth**3/(nu_true*k_food/rho_c)
                nu = max(1.0, 0.15*ra**(1/3))
                fade = min(1.0, (FLOOR_PA_S - k_mean)/(0.5*FLOOR_PA_S))
                self.conduction_gain = 1.0 + (nu - 1.0)*fade
            else:
                self.conduction_gain = 1.0
            self.mean_temp = b[1]/b[0]
            self.mean_k = k_mean
        self.record_food_peak(dt)

    def record_food_peak(self, dt=0.0):
        """The sauce's mean temperature as it is after this step, mass-weighted over the points of the mixture inside
        the pan (the set task.evaluate reads the mean in: Vessel.inside with 2 mm of margin, in the pan's frame) and
        summed in double precision, and the highest it has been (food_peak_c). The bulk sums of
        k_patch_gather are float32 atomics and read high (97.41 C for a sauce written to 97.00 C everywhere), so the
        record does not use them. Nothing physical reads it."""
        x_, t_, m_, f_ = self.w.cont.read("x", "temp", "mass", "flags")
        q = self.w.bq[self.pan.body]
        local = (x_ - q[:3].astype(np.float32)) @ Rotation.from_quat(q[3:]).as_matrix().astype(np.float32)
        sel = (f_ == 0) & self.pan.inside(local, margin=0.002)
        if sel.any():
            m64 = m_[sel].astype(np.float64)
            now = float(np.dot(t_[sel].astype(np.float64), m64)/m64.sum())
            self.food_peak_c = now if self.food_peak_c is None else max(self.food_peak_c, now)
            # the boil the cook sees (bubbles start at a mean of simmer_c), counted only while the hob heats, so that
            # the carry-over bubbling after the knob reaches 0 is not charged
            if now >= P.HOB["simmer_c"] and self.power_w > 0.0:
                self.boil_s += dt

    def state(self):
        return dict(level=int(self.knob.s_level), power_w=self.power_w, pan_on_zone=self.on_zone,
                    cut_out=bool(self.cut_out), sensor_c=round(float(self.sensor_c), 2),
                    plate_max_c=round(float(self.T.max()), 2), plate_mean_c=round(float(self.T.mean()), 2),
                    wall_c=round(float(self.T_wall), 2), film_w_m2k=round(float(getattr(self, "h_mean", 0.)), 1),
                    pan_w=round(float(getattr(self, "pan_w", 0.0)), 1),
                    food_mean_c=round(float(getattr(self, "mean_temp", 20.)), 2), burnt_cm2=round(self.burnt_area()*1e4, 3),
                    burnt_patches=int(self.burnt.sum()), film_min=round(float(self.film.min()/self.film_water0), 3)
                    if self.film is not None else 1.0, conduction_gain=round(float(self.conduction_gain), 2),
                    scrape_n=round(float(getattr(self, "scrape_n", 0.0)), 3),
                    surface_c=round(float(self.surface_c), 2), steam_g_s=round(float(self.steam_g_s), 4),
                    boiling_w=round(float(self.boiling_w), 1), burnt_scraped=round(float(self.char[self.burnt].mean()), 3)
                    if self.burnt.any() else 0.0,
                    energy_j={k: round(v, 1) for k, v in self.energy.items()})

    def snapshot(self):
        return dict(T=self.T.copy(), T_wall=self.T_wall, t_r=self.t_r.copy(), burnt=self.burnt.copy(),
                    film=None if self.film is None else self.film.copy(), energy=dict(self.energy),
                    gain=self.conduction_gain, sensor_c=self.sensor_c, cut_out=self.cut_out, char=self.char.copy(),
                    z_top=self.z_top, surface_c=self.surface_c, steam_g_s=self.steam_g_s, boiling_w=self.boiling_w,
                    cond_j=self.cond_j, food_peak_c=self.food_peak_c, boil_s=self.boil_s)

    def restore(self, s):
        self.T, self.T_wall, self.t_r, self.burnt = s["T"].copy(), s["T_wall"], s["t_r"].copy(), s["burnt"].copy()
        self.film = None if s["film"] is None else s["film"].copy()
        self.energy, self.conduction_gain = dict(s["energy"]), s["gain"]
        self.sensor_c, self.cut_out = float(s.get("sensor_c", 20.0)), bool(s.get("cut_out", False))
        self.char = s["char"].copy() if "char" in s else np.zeros(len(self.T))
        self.z_top, self.surface_c = float(s.get("z_top", 1.0)), float(s.get("surface_c", 20.0))
        self.steam_g_s, self.boiling_w = float(s.get("steam_g_s", 0.0)), float(s.get("boiling_w", 0.0))
        self.cond_j = float(s.get("cond_j", 0.0))
        self.food_peak_c = s.get("food_peak_c")
        self.boil_s = float(s.get("boil_s", 0.0))
        for k_ in ("electric_j", "wall_boiling_j", "condensed_j"):
            self.energy.setdefault(k_, 0.0)
