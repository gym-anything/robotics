"""Start of count-key-quantity: count-by-scanning's shelf, with every product in the counted slot at least three units
deep; the scanner in its cradle and the keypad on the cart tray.

setup(scene) stocks the scene (scene.Scene calls it after the stockroom's fixtures) and returns nothing; counted(deck,
rng) draws the counted slot's stock, [(sku, facings, depth)]."""
from barcode_scanning import stock


def counted(deck, rng):
    chosen = stock._target_entries(deck, rng)
    chosen = [(s, f, max(d, 3)) for s, f, d in chosen]
    return chosen


def setup(scene):
    stock.stock_stockroom(scene, counted=counted)
