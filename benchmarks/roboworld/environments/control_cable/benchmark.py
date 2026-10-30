"""Pure task recipe: discovery imports no physics engine or renderer."""
from .variants import resolve as select
from .instructions import instruction


def resolve(selection):
    item = select(selection.get("variant", "reference"), selection.get("seed", 0))
    port = item["config"]["sensor_port"]
    description = instruction(item["config"]) or (
        "Restore the inspection station's control connections using the wiring schedule. "
        f"Connect sensor {port} to B1, network LAN1 to PLC, brown SUPPLY_L to IO_L and blue SUPPLY_M to IO_M. "
        "Route W1 through G1, W2 through G2, W3 through G3 and W4 through G4. Press the physical COMMISSION/TEST "
        "button. Leave all cables released and stable for two seconds. Preserve the existing LAN2 uplink.")
    return dict(task_id=f"control-cable/v1/{item['id']}/seed-{item['seed']}",
                description=description,
                backend_options={"variant": item["id"], "config": item["config"]},
                selection=item, split=None)
