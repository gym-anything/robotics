"""Start of open-bag: the boutique counter with its stack of five factory-flat rope-handle totes, the pile of three
bottom boards left of the mat and the boxed candle beside the boards (scene.counter); seeds move the stack, the boards
and the candle rigidly (variants.placement).

setup(spec, contract) gets the counter preset's scene spec (scene.SceneSpec) before the contact world is built.
scene.build calls it with contract None: the released start is the counter as scene.counter builds it, and setup changes
nothing. Under the task-set contracts open-bag runs on TASK_SET_WORLD's world and the env adapters (task_set_env,
task_set_v2_env) call setup with their contract, which moves the candle back from bag-purchase's place to open-bag's."""
import numpy as np

# The task-set contracts run open-bag on bag-purchase's world: bag-purchase's geometry/contact logic, but keep
# open-bag's candle placement (setup).
TASK_SET_WORLD = "bag-purchase"


def setup(spec, contract):
    if contract is None:
        return
    from gift_bags.scene import CANDLE_XY, CANDLE_XY_PURCHASE
    # The seed transform is around the candle's own centre, so its translation is unchanged.
    b = next(b for b in spec.bodies if b.name == "candle")
    b.p[:2] += np.asarray(CANDLE_XY) - CANDLE_XY_PURCHASE
