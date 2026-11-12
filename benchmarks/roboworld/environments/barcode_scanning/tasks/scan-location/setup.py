"""Start of scan-location: the stockroom shelf as every stockroom task finds it, the scanner in its cradle on the cart
tray; the slot whose location label is to be read comes from the seed (stock.target_slot).

setup(scene) stocks the scene (scene.Scene calls it after the stockroom's fixtures) and returns nothing."""
from barcode_scanning import stock


def setup(scene):
    stock.stock_stockroom(scene)
