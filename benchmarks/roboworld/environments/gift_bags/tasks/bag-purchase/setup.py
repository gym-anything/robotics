"""Start of bag-purchase: the boutique counter with its stack of five factory-flat rope-handle totes, the pile of three
bottom boards left of the mat and the customer's boxed candle on the counter in front of the mat (scene.counter, which
places the candle at scene.CANDLE_XY_PURCHASE for bag-purchase's world, the world bag-candle and the task-set open-bag
also run on); seeds move the stack, the boards and the candle rigidly (variants.placement).

setup(spec, contract) gets the counter preset's scene spec (scene.SceneSpec) before the contact world is built;
scene.build calls it with contract None. Nothing in the start is bag-purchase's alone, so it changes nothing."""


def setup(spec, contract):
    pass
