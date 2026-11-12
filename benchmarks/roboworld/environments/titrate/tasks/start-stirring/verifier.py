"""Outcome of start-stirring: the stir bar turning with the motor for 5 s (mean bar speed at least 150 rpm and within
15 % of the motor's) while stirring stays safe (motor 200-800 rpm, the flask where it started and flat on the plate,
the stirrer not touched, no base in the flask), with nothing spilled and none of the aliquot lost.

track(ev, tick, m) keeps the last 5 s of bar and motor speeds on the task_evaluators.CatalogueEvaluator (ev.spin and
its running sums) before each evaluation; spin_ticks(ev, tick) gives the ticks a checkpoint's samples must cover.
verify(ev, m, **shared) is called at every tick by the CatalogueEvaluator (ev) with the tick's measurements (m) and the
values the tasks share as keywords (CatalogueEvaluator._evaluate); it returns (p, checks, terms, penalty): the raw
progress, the shared checks (spill, acid_retained) with the task's own added, the reward terms and the penalty taken
off p."""
import math

from titrate.newton15.task_evaluators import clip, grade


def track(ev, tick, m):
    ev.spin.append((tick, m['bar_rpm'], m['motor_rpm']))
    ev.spin_bar_sum += m['bar_rpm']
    ev.spin_motor_sum += m['motor_rpm']
    while ev.spin and (tick-ev.spin[0][0])*ev.dt_s > 5.+1.e-12:
        _, bar, motor = ev.spin.popleft()
        ev.spin_bar_sum -= bar
        ev.spin_motor_sum -= motor


def spin_ticks(ev, tick):
    return list(range(max(0, tick-math.floor((5.+1.e-12)/ev.dt_s)), tick+1))


def verify(ev, m, *, base, stir_s, checks, terms, penalty, **_):
    bar = ev.spin_bar_sum/len(ev.spin)
    motor = ev.spin_motor_sum/len(ev.spin)
    ratio = bar/motor if motor > 0 else 0.
    coupled = 150 <= bar and .85 <= ratio <= 1.15
    speed = clip(bar/200.)*grade(bar, 800., 1500.)
    coupling = clip((ratio-.5)/.35)*grade(ratio, 1.15, 1.5)
    p = speed*coupling*(1+clip(stir_s/5.))/2
    p -= clip((m['flask_start_distance_m']-.01)/.04)
    p -= max(0., base-.01)/.6
    checks.update(steady_stir=stir_s >= 5. and coupled, no_base=base <= .01)
    terms.update(mean_bar_rpm=bar, mean_motor_rpm=motor, speed=speed, coupling=coupling)
    return p, checks, terms, penalty
