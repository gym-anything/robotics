"""Read-only, clock-exact titration endpoint evaluation and progress reward.

Success is recomputed at every physics tick from the current flask and never
latched. All four task checks must hold at the evaluated tick:

* endpoint: the flask's own sodium (delivered titrant, from any source) has
  reached its own chloride (the analyte, which survives neutralization) and
  exceeds it by at most ``endpoint_overshoot_ml`` = 0.05 mL of the stated base.
  "The first faint pink" is the first drop that makes the colour permanent:
  "Once the addition of ONE drop of base changes the solution from colorless to
  pale pink, close the buret valve" (Moorpark College). A mixed flask turns pink
  0.4 uL past equivalence, so a correct stop lies at most one drop past it. One
  drop is about 0.05 mL, the environment's endpoint standard (endpoint detection
  to +/-1 drop, +/-0.05 mL) and the 50 mL Class A burette tolerance (ASTM E287:
  +/-0.05 mL).
  Titrant already on its way into the flask (task_readings pending_base_ml)
  counts against the upper bound as if landed. Both bounds include
  ``EDGE_TOLERANCE_ML`` for float rounding only. Judging against the flask's own
  chloride accepts back-titration with standard acid after an overshoot. A
  colour-intensity bound cannot replace the dose bound: one drop past and seven
  drops past look the same.
* indicator_hold: the colour has persisted throughout the solution for 30 s
  (the textbook and pharmacopoeia endpoint criterion) since titrant or acid
  last headed for the flask. The window starts when at least 90 % of the flask's
  indicator is in parcels at or above the phenolphthalein colour onset
  (task_readings.PINK_ONSET_PH) and the endpoint lower bound holds; 90 % (and
  85 % to continue) is a deliberate tolerance for the stir-bar wake, wall films
  and the last seconds of mixing, so the 30 s may begin while a tenth of the
  solution is still colourless. It restarts below 85 %, below equivalence, or
  when the sodium or chloride in the flask plus the liquid that can still reach
  it (burette tip, bore, ports and liquid arriving at the flask; task_readings
  loose_*) rises more than ``addition_restart_ml`` (a fifth of a drop) above
  its lowest value in the window. Titrant in the tip or falling into the flask
  restarts the window before it lands; a drip that cannot reach the flask does
  not. A sub-threshold addition still in transit is caught by the endpoint
  bound (pending titrant).
* spill: at most 0.05 mL of stray liquid outside every vessel cavity, from any
  source, moving or at rest (task_readings). Liquid falling or poured into any
  vessel's mouth is arriving, not spill, whichever tool delivered it; a drop at
  rest hanging from the burette tip is not spill. This is a deliberate
  workmanship standard: one drop of caustic or acid reagent
  left on the bench or glassware, or thrown about, fails success.
* acid_retained: at most 0.05 mL of the acid aliquot lost from the flask. The
  indicator was charged only into the flask's homogeneous starting solution and
  is a passive tracer, so the flask's share of it is the share of the original
  solution, and of the aliquot, still in the flask. Stock acid cannot restore
  it, so discarding sample and topping up with stock acid is a loss.

Global species conservation is a simulation-validity gate, not a task
requirement: a failed gate reports ``valid=False``, ``success=False`` and score
None, and the backend refuses to evaluate such a state.

Score: 1 for success, otherwise 0.9 x progress, where
progress = completion x accuracy x (1 + persistence) / 2 from the evaluated
state:

* completion: sodium in the flask over the flask's own equivalence, capped at
  1 (pending titrant is not credited before it lands, so an idle bench scores
  exactly 0);
* accuracy: 1 - E/W, floored at 0, where E (mL of the stated base) adds the
  overshoot beyond the endpoint band (pending titrant included), the aliquot
  lost beyond its 0.05 mL budget and the spill beyond its 0.05 mL budget (stray
  reagent shares the budget).
  W = ``error_width_ml`` =
  0.60 mL, the error at which assessment practice stops crediting a titre
  ("Award 1 mark for 0.40 cm3 < delta <= 0.60 cm3", Cambridge 9701/03 specimen
  mark scheme). It sets the width of partial credit only, not success;
* persistence: the indicator_hold window above as a fraction of 30 s.

The score is a state value of the evaluated tick, not an increment; an
invalid state's score is None (GAR's gym tuple renders a None reward as 0.0).
No term reads a tool, vessel of origin, stirring method or order.
Numerical/visual environment qualification is separate from task success.
"""

from dataclasses import asdict, dataclass
import copy
import math

FAIL_SCORE_SCALE = .9
EDGE_TOLERANCE_ML = 1.e-9
REWARD_CONTRACT = 'titration-progress-v1'
READINGS = ('pink_fraction', 'base_in_flask_ml', 'acid_in_flask_mol', 'spill_ml', 'indicator_retained',
            'loose_base_ml', 'loose_acid_mol', 'pending_base_ml', 'arriving_ml', 'stray_moving_ml')


@dataclass(frozen=True)
class TaskSpec:
    endpoint_overshoot_ml: float = .05
    spill_tolerance_ml: float = .05
    indicator_start_fraction: float = .90
    indicator_hold_fraction: float = .85
    hold_seconds: float = 30.
    addition_restart_ml: float = .01
    error_width_ml: float = .60

    def __post_init__(self):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('Positive finite task criteria required')
        if not 0 < self.indicator_hold_fraction <= self.indicator_start_fraction <= 1:
            raise ValueError('Invalid indicator fraction thresholds')


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class EndpointEvaluator:
    def __init__(self, dt_s, aliquot_ml, acid_molar, base_molar, spec=None):
        if any(not math.isfinite(v) or v <= 0 for v in (dt_s, aliquot_ml, acid_molar, base_molar)):
            raise ValueError('Endpoint evaluation requires a positive acid aliquot and concentrations')
        self.dt_s, self.spec = dt_s, spec or TaskSpec()
        self.aliquot_ml, self.acid_molar, self.base_molar = aliquot_ml, acid_molar, base_molar
        self.equivalence_ml = aliquot_ml*acid_molar/base_molar
        self.last_tick = -1
        self.pink_start_tick = None
        self.window_base_ml = self.window_acid_ml = None
        self.result = None

    def update(self, tick, *, pink_fraction, base_in_flask_ml, acid_in_flask_mol, spill_ml, invariants_ok,
               indicator_retained, loose_base_ml, loose_acid_mol, pending_base_ml, arriving_ml, stray_moving_ml):
        if type(tick) is not int or tick != self.last_tick+1:
            raise ValueError('Evaluator requires every consecutive physics tick, including tick zero')
        values = (pink_fraction, base_in_flask_ml, acid_in_flask_mol, spill_ml, indicator_retained,
                  loose_base_ml, loose_acid_mol, pending_base_ml, arriving_ml, stray_moving_ml)
        if any(not math.isfinite(v) or v < 0 for v in values) or pink_fraction > 1 or indicator_retained > 1+1.e-9:
            raise ValueError('Invalid measured endpoint state')
        if type(invariants_ok) is not bool:
            raise ValueError('Explicit conservation result required')
        s = self.spec
        base = base_in_flask_ml
        # Chloride measures the acid now in the flask even after neutralization,
        # including standard acid added in a back-titration.
        acid_eq_ml = acid_in_flask_mol/self.base_molar*1000.
        excess = base-acid_eq_ml
        reached = excess >= -EDGE_TOLERANCE_ML
        left_base = base+loose_base_ml
        left_acid = acid_eq_ml+loose_acid_mol/self.base_molar*1000.
        if self.pink_start_tick is not None and (pink_fraction < s.indicator_hold_fraction or not reached
                or left_base > self.window_base_ml+s.addition_restart_ml
                or left_acid > self.window_acid_ml+s.addition_restart_ml):
            self.pink_start_tick = None
        if self.pink_start_tick is None:
            self.window_base_ml, self.window_acid_ml = left_base, left_acid
            if reached and pink_fraction >= s.indicator_start_fraction:
                self.pink_start_tick = tick
        else:
            self.window_base_ml = min(self.window_base_ml, left_base)
            self.window_acid_ml = min(self.window_acid_ml, left_acid)
        hold = 0. if self.pink_start_tick is None else (tick-self.pink_start_tick)*self.dt_s
        # The indicator tracer, not chloride, measures how much of the original
        # aliquot is still in the flask.
        retained = min(1., indicator_retained)
        acid_loss_ml = self.aliquot_ml*(1.-retained)
        acid_in_flask_ml = acid_in_flask_mol/self.acid_molar*1000.
        landed_excess = excess+pending_base_ml
        checks = dict(endpoint=reached and landed_excess <= s.endpoint_overshoot_ml+EDGE_TOLERANCE_ML,
            indicator_hold=hold >= s.hold_seconds, spill=spill_ml <= s.spill_tolerance_ml,
            acid_retained=acid_loss_ml <= s.spill_tolerance_ml)
        success = invariants_ok and all(checks.values())
        reward = self.progress(base, acid_eq_ml, landed_excess, hold, spill_ml, acid_loss_ml)
        score = 1. if success else FAIL_SCORE_SCALE*reward['progress'] if invariants_ok else None
        self.last_tick = tick
        self.result = dict(tick=tick, success=success, score=score, valid=invariants_ok, checks=checks,
            progress=reward['progress'], reward=reward,
            flags=dict(overshoot_ml=max(0., landed_excess-s.endpoint_overshoot_ml),
                acid_added_ml=max(0., acid_in_flask_ml-self.aliquot_ml*retained)),
            stable_pink_seconds=hold, pink_fraction=pink_fraction,
            base_in_flask_ml=base_in_flask_ml, equivalence_ml=self.equivalence_ml,
            flask_equivalence_ml=acid_eq_ml, excess_base_ml=excess,
            acid_in_flask_mol=acid_in_flask_mol, indicator_retained=indicator_retained, acid_loss_ml=acid_loss_ml,
            spill_ml=spill_ml, loose_base_ml=loose_base_ml, loose_acid_mol=loose_acid_mol,
            pending_base_ml=pending_base_ml, arriving_ml=arriving_ml, stray_moving_ml=stray_moving_ml,
            qualification='Task criterion only; not environment physics acceptance')
        if not invariants_ok:
            self.result['invalid_reason'] = 'Global species conservation failed; not an evaluable task outcome'
        return copy.deepcopy(self.result)

    def progress(self, base, acid_eq_ml, excess, hold, spill_ml, acid_loss_ml):
        s = self.spec
        completion = min(1., base/acid_eq_ml) if acid_eq_ml > 0. else 0.
        errors = dict(overshoot_beyond_band_ml=max(0., excess-s.endpoint_overshoot_ml-EDGE_TOLERANCE_ML),
            aliquot_loss_beyond_budget_ml=max(0., acid_loss_ml-s.spill_tolerance_ml)*self.acid_molar/self.base_molar,
            spill_beyond_budget_ml=max(0., spill_ml-s.spill_tolerance_ml))
        accuracy = max(0., 1.-sum(errors.values())/s.error_width_ml)
        persistence = min(1., hold/s.hold_seconds)
        return dict(progress=completion*accuracy*(1.+persistence)/2., completion=completion, accuracy=accuracy,
            persistence=persistence, errors_ml=errors, contract=REWARD_CONTRACT,
            score_rule='1 if success else 0.9 x progress; None if not valid')

    def snapshot(self):
        return dict(schema='titration.endpoint.v4', dt_s=self.dt_s, spec=asdict(self.spec),
            aliquot_ml=self.aliquot_ml, acid_molar=self.acid_molar, base_molar=self.base_molar,
            last_tick=self.last_tick, pink_start_tick=self.pink_start_tick,
            window_base_ml=self.window_base_ml, window_acid_ml=self.window_acid_ml,
            result=copy.deepcopy(self.result))

    def validate(self, snapshot):
        current = self.snapshot()
        if set(snapshot) != set(current) or any(snapshot[k] != current[k] for k in
                ('schema', 'dt_s', 'spec', 'aliquot_ml', 'acid_molar', 'base_molar')):
            raise ValueError('Endpoint checkpoint configuration mismatch')
        tick, start, result = snapshot['last_tick'], snapshot['pink_start_tick'], snapshot['result']
        if type(tick) is not int or tick < 0 or (start is not None and (type(start) is not int or not 0 <= start <= tick)):
            raise ValueError('Invalid endpoint checkpoint clock')
        if not isinstance(result, dict) or result.get('tick') != tick:
            raise ValueError('Invalid endpoint checkpoint result')
        window = snapshot['window_base_ml'], snapshot['window_acid_ml']
        if not all(_finite_number(v) for v in window):
            raise ValueError('Invalid endpoint checkpoint window')
        # Recompute the result from its measurements, timer and window. A
        # serialized success bit is not authoritative, even in an owner-only
        # checkpoint.
        trial = EndpointEvaluator(self.dt_s, self.aliquot_ml, self.acid_molar, self.base_molar, self.spec)
        trial.last_tick, trial.pink_start_tick = tick-1, start
        trial.window_base_ml, trial.window_acid_ml = window
        try:
            expected = trial.update(tick, invariants_ok=result['valid'], **{k: result[k] for k in READINGS})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Invalid endpoint checkpoint measurements') from exc
        if (trial.pink_start_tick != start or (trial.window_base_ml, trial.window_acid_ml) != window
                or expected != result):
            raise ValueError('Endpoint checkpoint timer/result is inconsistent')

    def restore(self, snapshot):
        self.validate(snapshot)
        self.last_tick = snapshot['last_tick']
        self.pink_start_tick = snapshot['pink_start_tick']
        self.window_base_ml, self.window_acid_ml = snapshot['window_base_ml'], snapshot['window_acid_ml']
        self.result = copy.deepcopy(snapshot['result'])


def description(reagents):
    return (f'Titrate the {reagents.aliquot_ml:g} mL acid aliquot in the flask, diluted with '
        f'{reagents.dilution_ml:g} mL water and phenolphthalein, using '
        f'{reagents.base_molar:g} mol/L sodium hydroxide. Stop at the endpoint: the first '
        'faint pink that persists throughout the solution for 30 seconds.')
