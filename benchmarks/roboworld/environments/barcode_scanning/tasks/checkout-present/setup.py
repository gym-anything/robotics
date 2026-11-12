"""Start of checkout-present: the customer's hand basket on the counter's left with five to seven items dropped in at
random, the empty bagging tote on the right and the scanner standing in its cradle between them.

setup(scene) stocks the scene (scene.Scene calls it after the counter's fixtures) and returns nothing."""
from barcode_scanning import stock


def setup(scene):
    stock.stock_counter(scene)
