"""Named, deterministic physical/work-instruction variations, not texture seeds.

Harness variants change cut/assembly length of the same cable construction,
not connector dimensions or a claim to another manufacturer SKU. All preserve
the four circuits, actual keyed retention and protected uplink requirements.
"""
from copy import deepcopy
import hashlib,json,math

VARIANTS={
 "reference":dict(description="Reference station: X1 sensor and 1.0 m network harness",config={}),
 "lower-channel":dict(description="Use X2: the identical X1 socket is a wrong electrical destination",
     config=dict(sensor_port="X2",room_style="cool")),
 "raised-ports":dict(description="Sensor socket pair 20 mm higher within the module service face",
     config=dict(sensor_port_offset_m=.02)),
 "service-loop":dict(description="1.2 m network harness, quarter-turn storage orientation and warm workshop lighting",
     config=dict(network_length_m=1.2,network_coil_rotation_rad=math.pi/2,room_style="warm")),
 "offset-lower-service":dict(description="X2 assignment, lower ports, 1.2 m network cable and rotated storage coil",
     config=dict(sensor_port="X2",sensor_port_offset_m=-.02,network_length_m=1.2,
                 network_coil_rotation_rad=-math.pi/2,room_style="cool",task_revision=2)),
 "angled-sensor":dict(description="M12 sensor harness with perpendicular strain-relief exits and physical elbow boots",
     config=dict(sensor_connector_style="angled")),
 "wide-cabinet":dict(description="800 mm enclosure with correspondingly wider door, mounting rails and relocated wiring ducts",
     config=dict(cabinet_width_m=.80,room_style="warm")),
 "rotated-keyways":dict(description="Quarter-turn M12 key and contact orientation at both sensor destinations",
     config=dict(sensor_socket_roll_rad=math.pi/2)),
}

def resolve(name="reference",seed=0):
    if not isinstance(name,str) or name not in VARIANTS:raise ValueError("Unknown control-cable variation")
    if type(seed) is not int or seed<0:raise ValueError("Nonnegative integer seed required")
    record=deepcopy(VARIANTS[name]);record.update(id=name,seed=seed)
    record["config"]=dict(sensor_port="X1",sensor_port_offset_m=0.,network_length_m=1.,
        sensor_connector_style="straight",sensor_socket_roll_rad=0.,cabinet_width_m=.70,
        network_coil_rotation_rad=0.,room_style="neutral",**{}) | record["config"]
    record["config"]["seed"]=seed
    record["sha256"]=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return record

def contract():
    return dict(supported=True,selectors={"variant":list(VARIANTS),"seed":"nonnegative integer"},
        records=[resolve(n) for n in VARIANTS],
        dimensions=["sensor electrical destination","physical port offset","network length","coil pose","lighting",
                    "M12 boot/strain-relief geometry","cabinet/door/duct dimensions","M12 keyway orientation",
                    "task revision (2: revised instruction, port labels, bench-laid supply leads and evaluator; offset-lower-service only)"],
        preserved=["four circuits","keyed M12","RJ45 latch","spring terminals","LAN2 protection","robot-only actions"],
        limitations="Authored industrial assemblies, not exact commercial harness SKUs; cable construction remains fixed")
