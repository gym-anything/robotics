"""Start of checkout-bulky: checkout-present's counter (the basket of items, the bagging tote, the scanner in its
cradle) and a 24-pack of water at the counter's far end beyond the tote, its barcode side turned to the robots.

setup(scene) stocks the scene (scene.Scene calls it after the counter's fixtures) and returns nothing; water_case(scene,
z) puts the case on the counter top at height z after the basket is filled."""
import math

from barcode_scanning import stock
from barcode_scanning.catalog import SKUS


def water_case(scene, z):
    rng = scene.rng
    # the 24-pack at the counter's end beyond the tote, its barcode side (catalogue "left") turned to the robots;
    # footprint 279 x 406 mm, 20 mm clear of the tote (0.14..0.54 m). The first layout put it at (0.35, 0.44), on
    # top of the tote: the settle threw the tote 1.2 m onto the floor (adversarial evaluator tests)
    name = "water_case"
    scene.add_item(name, "K11", (.70, .30, z + SKUS["K11"].dims[2]/2 + .0005), yaw=math.pi/2 + float(rng.normal(0, .02)))
    scene.items[name].update(container="counter")


def setup(scene):
    stock.stock_counter(scene, extra=water_case)
