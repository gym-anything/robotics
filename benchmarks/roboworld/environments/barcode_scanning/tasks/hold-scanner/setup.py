"""Start of hold-scanner: the stockroom shelf as every stockroom task finds it, the scanner in its cradle on the cart
tray; the task names no shelf slot.

setup(scene) stocks the scene (scene.Scene calls it after the stockroom's fixtures) and returns nothing. NAMES_SLOT
False makes stock.target_slot give None: the scene's task_meta holds no counted or scanned slot."""
from barcode_scanning import stock

NAMES_SLOT = False


def setup(scene):
    stock.stock_stockroom(scene)
