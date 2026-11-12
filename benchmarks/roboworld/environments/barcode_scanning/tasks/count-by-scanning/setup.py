"""Start of count-by-scanning: the stockroom shelf with the counted slot (from the seed, stock.target_slot) holding one
or two products several units deep, faced to the aisle with their barcodes away from it; the scanner in its cradle.

setup(scene) stocks the scene (scene.Scene calls it after the stockroom's fixtures) and returns nothing; the counted
slot's stock is drawn by stock._target_entries."""
from barcode_scanning import stock


def setup(scene):
    stock.stock_stockroom(scene, counted=stock._target_entries)
