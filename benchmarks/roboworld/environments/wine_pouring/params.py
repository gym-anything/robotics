"""Parameter ledger: every physical constant of the wine-pouring environment with its provenance.

Labels: datasheet | standard | literature-read | search | literature | derived | estimate |
authored | measured (in this repository, probe named) | calibrated (fitted to a stated physical observable).
Probes and demos READ these values; they never override them on a command line.
`ledger()` returns the machine-readable form.
"""

import math

GRAVITY = 9.80665                      # m/s^2, standard gravity (literature)

# ------------------------------------------------------------------ liquids (SI)
# density kg/m^3, dynamic viscosity Pa s, surface tension N/m, contact angle on clean glass (deg), RGB decadic
# absorbance per cm (blue <- A420, green <- A520, red <- A620; OIV-MA-AS2-07B convention), refractive index
LIQUIDS = {
    "red_wine": dict(rho=992.0, mu=1.6e-3, sigma=0.047, theta_glass_deg=20.0, absorbance_rgb_per_cm=(0.8, 3.6, 3.3),
                     ior=1.34, source="rho: Hlavac et al. 2016 (literature-read); mu: AJEV 2019 Shiraz 1.49-1.70 mPa s "
                                      "at 20 C (search) scaled to 18 C; sigma, theta: estimate; "
                                      "absorbance: Cabernet/Merlot Glories values (search) [S26]"),
    "white_wine": dict(rho=992.0, mu=1.75e-3, sigma=0.047, theta_glass_deg=20.0, absorbance_rgb_per_cm=(0.01, 0.03, 0.10),
                       ior=1.34, source="as red; mu: Chardonnay 1.45-1.53 mPa s at 20 C (AJEV 2019, search) scaled to "
                                        "10 C; absorbance estimate (pale straw, A420 ~0.1/cm)"),
    "rose_wine": dict(rho=992.0, mu=1.6e-3, sigma=0.047, theta_glass_deg=20.0, absorbance_rgb_per_cm=(0.08, 0.6, 0.35),
                      ior=1.34, source="as red; absorbance estimate (A520 ~0.6/cm)"),
    "water": dict(rho=998.2, mu=1.002e-3, sigma=0.0727, theta_glass_deg=25.0, absorbance_rgb_per_cm=(0.0, 0.0, 0.0),
                  ior=1.333, source="IAPWS values (repo); contact angle 25 deg on "
                                    "cleaned silica (repo ledger)"),
}
LIQUID_IDS = {name: i for i, name in enumerate(LIQUIDS)}
IMPACT_ACCEL_M_S2 = 2*9.81             # vertical proper acceleration of a container above which the excess is handed to its
                                       # liquid as the exact uniform impulse (coupling.LumpedLiquid._impact_impulse);
                                       # handling stays below ~2 g, a bottle landing on its coaster reaches ~25 g
WALL_FOLLOW_TOL_M = 0.0002              # liquid in and against a container follows the container's position correction of
                                       # the last tick when it exceeds 0.2 mm at the container's extent (impacts; ordinary
                                       # handling: a dt^2 / 2 ~ 1e-5 m; coupling.LumpedLiquid._follow_walls)
IMPACT_MAX_TILT_DEG = 30.0             # ... only for a container within 30 deg of upright: the uniform vertical impulse is
                                       # the exact response of liquid with a level free surface over a floor that stops it;
                                       # applied to a glass landing on its side (E6 tip) or a bottle swung about (E10),
                                       # it throws the wine out at the 8 m/s cap
CONTACT_LINE_FRICTION_RATE = 100.0     # 1/s, sliding-velocity relaxation of liquid at its contact line on the bar top and
                                       # floor (closure for contact-angle hysteresis; with surface forces only). Chosen so
                                       # a 2 mL spill comes to rest at about the area the puddle-height law gives
                                       # (h = 2 l_c sin(theta/2) = 2.0 mm -> 9.9 cm2): 15.5 cm2 at 100/s, 105 cm2 at 30/s,
                                       # still spreading over 2400 cm2 at 10/s (V4 spill)
THETA_BAR_DEG = 55.0                   # wine/water on the polished engineered-quartz bar top and the floor: estimate between
                                       # clean silica (20-30 deg) and the polyester binder (~70 deg) of resin-bound quartz

# ------------------------------------------------------------------ SPH numerics
SPH_DX_M = 0.002                       # authored reference spacing (refinement 2.5/2.0/1.6 mm)
SPH_SUPPORT_RATIO = 2.0                # cubic spline support H = 2 dx (pressure, density)
SPH_CFL = 0.4                          # max particle displacement per substep, in units of dx
SPH_SPEED_CAP_M_S = 8.0                # numerical squirt cap applied every substep: liquid caught between a falling glass
                                       # rim and the bar top (a gap closing faster than particles can leave) is shot out
                                       # at 25 m/s and faults the episode without it (E6 tip). 8 m/s is above anything
                                       # the scene can produce physically (free fall from 3 m: 7.7 m/s)
SPH_MAX_SUBSTEPS = 24                  # per 1 ms rigid step (42 us): up to 19 m/s at CFL 0.4; liquid knocked off the
                                       # bar reaches the floor at ~5 m/s and splashes; a faster particle faults the episode
SPH_MIN_SUBSTEPS = 2
SPH_DENSITY_TOL_MEAN = 1e-4            # DFSPH mean density error per substep (SPlisHSPlasH default 0.01 %; at 0.1 % a
                                       # resting glass 'breathed': its load swung 0.6-2.7 N about the 1.46 N weight)
SPH_DENSITY_TOL_MAX = 1e-2             # ... and its maximum over particles (a mean alone hides local error)
SPH_DIVERGENCE_TOL = 1e-3              # mean |div v| * dt
SPH_MAX_ITERATIONS = 100
SPH_WARM_START_DENSITY = 1.0           # previous substep's total stiffness as the initial guess, everywhere (Bender &
SPH_WARM_START_DIVERGENCE = 0.5        # Koschier 2017; divergence 0.5 as SPlisHSPlasH); masked to compressed particles
                                       # (SPlisHSPlasH) it left holes in a deep column's pressure and a bottle blew up
SPH_JACOBI_OMEGA = 0.5                 # relaxed Jacobi: signed increments diverged at 1.0 (IISPH uses 0.5)
SPH_XSPH = 0.02                        # XSPH velocity filter (Monaghan 1989), numerical regulariser: physical viscosity
                                       # (1.6e-6 m^2/s) cannot damp 2 mm particle noise (viscous time dx^2/nu = 2.5 s); without
                                       # it a resting glass keeps 15-40 mm/s rms, with 0.02 it decays to ~4 mm/s mean
                                       # (E0); it damps sloshing
SPH_WALL_SHEAR = 1.0                   # scale of the no-slip wall shear term nu dv / d^2
SPH_CHECK_EVERY = 4                    # residual read back every 4th iteration: each readback is a host sync, and on the
                                       # shared GPU a sync waits ~1 ms for a time slice (70 per step cost 2/3 of the step)
WALL_CLEARANCE_RATIO = 0.3             # guard: particle centres kept >= 0.3 dx from solid surfaces (velocity level);
                                       # the boundary pressure rests them at 0.5 dx (samples dx/2 inside the material)
MAX_PARTICLE_SPEED_M_S = 20.0          # numerical-failure guard (barcode precedent: 3x a 2 m drop's speed)
ROOM = dict(x0=-3.4, x1=3.4, y0=-1.7, y1=3.6, h=3.1)   # the room's walls (m): drawn by dressing.py, and liquid
                                       # boundaries (scene.liquid_planes) so spilt liquid cannot leave the room
SQUEEZE_GAP_RATIO = 1.25               # squeeze closure: a particle between two facing solids closer together than 1.25 dx
                                       # (a gap one particle cannot fill) leaves along the gap, carried with the gap's
                                       # normal motion and pushing on neither solid (liquid.k_guard). A real film under a
                                       # glass set on a wet bar drains in milliseconds (Reynolds squeeze: 1 mm under a
                                       # 42 mm foot at 3.5 N in ~7 ms); without it the guard pushes each trapped particle
                                       # out of the bar at up to 2 m/s per substep and the vessel takes that momentum: a
                                       # glass set on a 5 mL puddle leaves the bar at 4.3 m/s.
                                       # 0 disables
SQUEEZE_ESCAPE_M_S = 0.2               # squeezed liquid leaves along the gap at the Reynolds squeeze-out speed r V / (2 h)
                                       # (r from the closing vessel's axis, V its closing speed, h the gap, >= dx/4),
                                       # bounded below by this (a film draining under its own weight: 1 mm under a 42 mm
                                       # foot at rest drains at ~0.2 m/s at the edge)...
SQUEEZE_ESCAPE_MAX_M_S = 0.5           # ...and above by this: the film is decelerated by its own lubrication pressure
                                       # (3 mu V R^2 / h^3) and leaves as a thin sheet that surface tension holds to the
                                       # bar, which a 2 mm particle cannot represent. Authored. Squeezed particles at
                                       # >= 0.5 m/s with faster ones kept and left in the pressure solve would throw 5 mL
                                       # under a glass set down at 5 cm/s at the 8 m/s cap, 0.9 m high

# ------------------------------------------------------------------ surface forces (Akinci, Akinci, Teschner 2013)
# Cohesion: the Akinci pair force, coefficient normalised so its planar surface energy equals sigma (liquid.py
# cohesion_gamma; titrate/newton15/cohesion2013.py). Adhesion: the same pair force integrated over a half-space of
# virtual liquid behind each wall, weighted (1 + cos theta)/2 by Young's law. Model normalisations from the measured
# sigma and contact angles; V4 tests the discrete behaviour.
ADHESION_SCALE = 1.0                   # factor on the Young-law wall adhesion weights (ablations only)
DUEZ_SEPARATION = True                 # inertial-capillary separation closure at container lips (Duez, Ybert, Clanet &
                                       # Bocquet 2010, PRL 104, 084503): within DUEZ_ZONE_M of a lip, wall adhesion cannot
                                       # hold liquid sliding along the glass faster than U_c = sqrt(We_c sigma / (rho e0)),
                                       # We_c = DUEZ_K (r_i + e0/2) r_i / e0^2 (1 + cos theta) (their Eq. 2), with r_i the
                                       # lip's own edge radius (liquid.lip_zone, from the geometry) and e0 the local liquid
                                       # thickness normal to the wall (at least one particle spacing). Beyond U_c the
                                       # attraction falls as (U_c / u_t)^2 (authored: the inertial load on the turning film
                                       # grows as rho u^2). With r_i = e0 = dx everywhere on glassware, U_c would move
                                       # with the spacing (0.156 / 0.174 / 0.195 m/s at 2.5 / 2.0 / 1.6 mm) and act on flat
                                       # walls too, outside Duez's edge mechanism. Without any closure the kernel-range
                                       # adhesion holds a steady 25-45 mL/s stream round the LEONOR's lip and back along
                                       # its neck
DUEZ_K = 0.44                          # slope of We_c / (1 + cos theta) against R r_i / e0^2 (Duez et al. 2010 Fig. 3b
                                       # inset, read from the fitted line: 30 at r_i / e0 = 8)
DUEZ_EDGE_MAX_M = 0.0015               # the LEONOR finish's lip round (geometry._bottle); thinner rims use half their wall
DUEZ_ZONE_M = 0.006                    # the zone round the lip's crest where the flow turns round the edge: the lip's
                                       # half-wall plus this (a lip round plus a thick stream's few millimetres). Authored
COHESION_SURFACE_S = 0.90              # cohesion acts only between pairs with a particle whose volume fraction s is below
                                       # this (the surface layer; s = 1 in the bulk). Everywhere (V2b): the tumbler's slosh
                                       # frequency +9 %, damping 9.3 %; surface layer only: +3.1 %, 5.2 %, and a 2 mL spill
                                       # still comes to rest at the puddle-law area (V4)
COHESION_SCALE = 1.0                   # factor on the sigma-normalised cohesion coefficient (V4 drop oscillation checks it)
SURFACE_FORCES = True                  # on by default. An Akinci curvature term with a dimensionally inconsistent
                                       # coefficient and an adhesion force cut off at its maximum at dx/2 would throw
                                       # meniscus particles out of a moving bottle's neck at 3-7 m/s; this model lets
                                       # none out (V4 moved bottle)

# ------------------------------------------------------------------ headspace gas
P_ATM = 101325.0                       # Pa (literature)
RHO_AIR = 1.204                        # kg/m^3 at 20 C (literature)
GAS_GAMMA = 1.4                        # adiabatic exponent (glug time scale 8-21 ms << thermal diffusion ~5 s)
GAS_SLAB_M = 0.002                     # neck slab length for seal detection (gas.py SLAB_M)

# ------------------------------------------------------------------ glass and contact (estimates)
SODA_LIME_DENSITY = 2500.0             # kg/m^3 (literature)
TRITAN_CRYSTAL_DENSITY = 2600.0        # kg/m^3 (estimate: barium-titanium crystal ~2.5-2.6)
GLASS_IOR = 1.52                       # ISO 3591 crystal >= 1.520 (standard); soda-lime 1.52 (literature)
# Coulomb friction. MuJoCo combines two equal-priority geoms by the MAXIMUM, so these are the values contacts really
# use (read back from mj_data.contact.friction); the pads have geom priority 1 and impose their own value.
#   glass on glass, quartz aggregate or the stainless coaster: 0.4 (Engineering ToolBox friction table: glass-glass
#     clean and dry 0.9-1.0 static / 0.4 sliding, greased 0.1-0.6 static; glass-metal clean 0.5-0.7, greased
#     0.2-0.3). A bar in service is wiped and often wet: one coefficient (MuJoCo has no static/kinetic split) at the
#     dry sliding value, inside the wet static range.
#   pad on glass: 0.8, the Franka Hand pad value qualified in barcode_scanning (rubber on smooth dry surfaces 0.5-0.85,
#     e.g. rubber-cardboard 0.5-0.8, rubber-dry concrete 0.6-0.85, same table); a wet pad would be lower
#     (rubber-wet concrete 0.45-0.75) - not modelled.
MU = dict(pad_glass=0.8, glass_quartz=0.4, glass_mat=0.8, glass_glass=0.4, bottle_coaster=0.4, default=0.5)
PAD_TORSIONAL_M = 0.0035               # torsional friction length of the finger pads (MuJoCo condim 4): a flat pad pressed
                                       # on a cylinder touches along a line of the collision pad's 17.5 mm extent along
                                       # the vessel axis (the rubber tip box, 17.5 x 15.2 x 18.5 mm, scene._fingertips);
                                       # spinning about the contact normal it resists mu N L / 4 = 0.8 N x 0.0175 m / 4 per
                                       # newton of grip (estimate from the pad geometry, uniform pressure; the real pad's
                                       # 21 mm height would be 20 % more than the pad the scene builds). With condim 3 the
                                       # pinch resists 0.045 N m about the pad line (V6)
CONTACT_GAP_M = 0.002
MESH_CONTACT_GAP_M = 0.002

# ------------------------------------------------------------------ bar and station ([S9, S10])
BAR_TOP_Z_M = 1.090                    # finished bar-top height (42-45 in range, authored 43 in)
BAR_TOP_THICK_M = 0.030                # 3 cm quartz top (search [S10])
BAR_DEPTH_M = 0.66                     # 26 in bar top (search [S10])
BAR_LENGTH_M = 3.00                    # authored
DELIVERY_SPOTS = {1: (-0.36, 0.42), 2: (-0.12, 0.42), 3: (0.12, 0.42), 4: (0.36, 0.42)}   # authored (Toni-style)
SPOT_RADIUS_M = 0.055
ARM_BASE_Z_M = 1.00                    # FR3 bases on the pedestal (authored)
ARM_BASE_XY = {"left_arm_mount": (-0.25, -0.18), "right_arm_mount": (0.25, -0.18)}
ARM_YAW = math.pi/2                    # bases face +y (the guests)

# ------------------------------------------------------------------ robot (datasheets via repo)
FR3_PAYLOAD_KG = 3.0                   # datasheet [S35]
FRANKA_HAND_FORCE_N = (30.0, 70.0)     # continuous grasp force range per finger pair (datasheet [S36])
REFLEX_TORQUE_NM = (10.0, 20.0)        # libfranka examples_common.cpp setDefaultBehavior upper thresholds: nominal,
                                       # accelerating (fetched 2026-10-02; repo barcode)
REFLEX_RECOVERY_S = 1.0                # authored (barcode precedent)
REFLEX_FILTER_S = 0.02                 # the reflex compares libfranka's *filtered* external torque estimate
                                       # (RobotState::tau_ext_hat_filtered, low-pass; time constant unpublished: estimate)
                                       # with the thresholds; unfiltered 1 kHz contact rows of a gripped bottle spiked to
                                       # 47 N m during grip micro-slip and stopped every pour
MAX_OBJECT_SPEED_M_S = 20.0
MAX_OBJECT_SPIN_RAD_S = 1000.0
STOP_SOLIMP = (0.99, 0.999, 0.5, 2.0)  # hard joint stops (barcode repo: d0, dmax, midpoint, power)
READY_POSE = (0.0, -1.2, 0.0, -2.8, 0.0, 1.6, 0.785, 0.04, 0.04)    # authored ready pose (arm joints + fingers): TCP
                                       # at (+-0.25, 0.03, 1.385) m, pointing down, 0.3 m above the bar's robot side (FK)

# ------------------------------------------------------------------ evaluation
SETTLE_STILL_S = 2.0                   # robot still before the evaluator reads (authored)
SETTLE_LIQUID_SPEED = 0.015            # m/s mean particle speed (the resting noise floor is ~4 mm/s mean, E0)
SPILL_TOL_ML = 0.2
DRIP_TOL_ML = 0.05
GLASS_UPRIGHT_DEG = 10.0
CONTACT_FORCE_TOL_N = 0.5


def ledger():
    """Every scalar/tuple constant of this module."""
    out = {}
    for k, v in globals().items():
        if k.isupper() and isinstance(v, (int, float, tuple, dict, bool)):
            out[k] = v
    return out
