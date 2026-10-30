"""Environment compatibility: explicit mounts and controller support, not auto-placement."""

from .catalog import for_environment


def deposit_setup(selected=None):
    setup = for_environment("deposit-cans", selected)
    data = setup.to_dict()
    if set(data["instances"]) != {"left", "right"}:
        raise ValueError("This deployment has exactly left/right arm mount slots")
    for name, item in data["instances"].items():
        if (
            item["model"].split("@")[0] not in ("fr3", "panda")
            or item["controller_profile"] != "deposit_cans"
        ):
            raise ValueError(
                "Deposit-cans supports the FR3 or Panda WSG50 controller profile"
            )
        if item.get("mount") != name + "_arm_mount" or "pose" in item:
            raise ValueError(
                "This deployment uses its authored arm mounts, not automatic placement"
            )
        if item.get("parameters") or item.get("initial_pose"):
            raise ValueError(
                "Deposit-cans uses its explicit environment drive settings and initial pose"
            )
    expected = {
        name + "_hand": dict(
            model="wsg50@1",
            parent=name + "/flange",
            mount_profile=data["instances"][name]["model"].split("@")[0]
            + "_wsg_adapter",
        )
        for name in ("left", "right")
    }
    expected.update(
        {
            name + "_camera": dict(
                model="d405@1", parent=name + "/wrist", mount_profile="deposit_wrist"
            )
            for name in ("left", "right")
        }
    )
    if data.get("attachments") != expected:
        raise ValueError(
            "Both arms require the declared physical WSG50 assembly and native belt mechanism"
        )
    if data.get("roles"):
        raise ValueError("Custom role aliases are not supported by this deployment")
    return setup
