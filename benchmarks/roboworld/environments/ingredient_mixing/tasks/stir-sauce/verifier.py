"""Outcome of stir-sauce: 95 % of the starch in the pan cooked with the sauce having reached 88 C, no more than 2 cm2
of the pan's bottom burnt on, no more than 180 s at the boil, none of it spilt (below the rim included), the pan
standing on the hob's glass and the hob turned off.

JUDGING holds the task's portion size, its criteria (task.TASKS; task.evaluate measures them) and its score rule
(reward="cook"): score(world, out, C, crit, layers, share) sets the evaluation's progress, reward and score from
cook_progress (task.evaluate calls it). verify(world, info, *, _progress=True) evaluates the state of the World with
the start record info and returns the evaluation dict (scoring.evaluate_criteria: task.evaluate; stir-sauce keeps its
own contract, stir-sauce-v1)."""
import math

from ingredient_mixing import params as P
from ingredient_mixing import scoring
from ingredient_mixing.task import FAIL_SCORE_SCALE

COOK_LAYERS = 3                  # cook_progress reads the least gelatinised of this many equal-mass layers of the sauce

JUDGING = dict(
    portion_ml=30.,
    criteria=dict(cooked=dict(fraction=0.95, temp_c=88.), scorch_cm2=2.0, contained=dict(limit=0.01, below_rim=True),
                  vessel_in_place=dict(on_hob=True), appliance_off="knob", boil_s=180.),
    reward="cook")


def score(world, out, C, crit, layers, share):
    bottom = world.heat.area.sum()*1e4 if world.heat is not None else math.pi*P.PAN["radius"]**2*1e4
    # the least layer, unless the whole pan already meets the verdict's share
    frac = C["cooked"]["value"]
    even = frac if frac >= crit["cooked"]["fraction"] else min(layers[COOK_LAYERS])
    # the hob's switch is credited over a sauce that is cooked and was not boiled on
    done = C["cooked"]["ok"] and C.get("not_boiled_on", dict(ok=True))["ok"]
    r = cook_progress(even, C["contained"]["value"], C["not_scorched"]["value"], bottom,
                      C["vessel_in_place"]["ok"], C["appliance_off"]["ok"], done, crit["cooked"]["fraction"])
    out["progress"] = r["progress"]
    out["reward"] = dict(r, score_rule=f"1 if success else {FAIL_SCORE_SCALE} x progress",
                         share_of_criteria_met=share, cooked_share_by_layer={str(n): s for n, s in layers.items()})
    out["score"] = 1.0 if out["success"] else round(FAIL_SCORE_SCALE*r["progress"], 4)


def cook_progress(cooked, spilt, burnt_cm2, bottom_cm2, pan_in_place, hob_off, cooked_ok, fraction=0.95):
    """stir-sauce's progress, from what the evaluator reads of the final state: how far the sauce has cooked, times the
    share of the batch still in the pan, times the share of the
    pan's bottom not burnt on, times the closing state. Nothing in it is of the motion, the tool, the order or the
    time taken; a pan left alone burns its whole bottom (unburnt 0), a knob turned off with nothing cooked has cook 0.

      cook      the gelatinised share of the starch over the criterion's 95 %, at most 1; evaluate() passes the least
                gelatinised of COOK_LAYERS equal-mass layers of the sauce in the pan (layer_shares): a layer cooked on
                an unstirred bottom under cool milk is not the sauce thickening
      keep      1 - the spilt share (contained's reading: on the counter, the floor, in the air, outside the pan below
                its rim)
      unburnt   1 - burnt-on area / the bottom's area (181 cm2): a ruined bottom is progress lost, as the burnt
                film is in the pan
      finish    0.75, + 0.10 with the pan on the hob, + 0.15 with the hob off once the sauce is cooked (the hob turned
                off over a raw sauce is not progress). The weights, and the linear burn factor, are authored.

    The four factors multiply: progress bought with a burnt pan or a lost batch is worth little, and no one factor
    stands in for the cooking. A state that meets every criterion scores 1 whatever its progress."""
    cook = min(1.0, max(0.0, float(cooked))/fraction)
    keep = min(1.0, max(0.0, 1.0 - float(spilt)))
    unburnt = min(1.0, max(0.0, 1.0 - float(burnt_cm2)/float(bottom_cm2)))
    finish = 0.75 + 0.10*bool(pan_in_place) + 0.15*bool(hob_off and cooked_ok)
    return dict(progress=round(cook*keep*unburnt*finish, 4), cook=round(cook, 4), keep=round(keep, 5),
                unburnt=round(unburnt, 4), finish=round(finish, 2))


def verify(world, info, *, _progress=True):
    return scoring.evaluate_criteria(world, info, _progress=_progress)
