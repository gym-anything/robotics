"""Lazy environment registry and public constructor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .backends.base import RoboticsBackend
from .env import RoboticsEnv


BackendFactory = Callable[..., RoboticsBackend]
_REGISTRY: dict[tuple[str, str], BackendFactory] = {}
_ALIASES = {
    "whiteboard-classroom": "whiteboard",
    "titrate": "titration",
    "pen-cap": "pencap",
    "lock-and-key": "lockkey",
    "lock-key": "lockkey",
    "laundry-folding": "laundry",
    "laundry-fold": "laundry",
    "football-kick": "football",
    "deposit-cans-into-container": "deposit-cans",
    "label_application": "label-application",
    "barcode_scanning": "barcode-scanning",
    "scan-barcode": "barcode-scanning",
    "apply-label": "label-application",
    "fillet": "filleting",
    "wine_pouring": "wine-pouring",
    "pour-wine": "wine-pouring",
    "pour-wine-glass": "wine-pouring",
    "fillet-fish": "filleting",
    "fish-filleting": "filleting",
    "steer-forklift": "forklift",
    "forklift-truck": "forklift",
    "drive-forklift": "forklift",
    "lift-truck": "forklift",
    "weld": "welding",
    "mig-welding": "welding",
    "gmaw": "welding",
    "weld-steel-joint": "welding",
    "sew": "sewing",
    "sewing-machine": "sewing",
    "lockstitch": "sewing",
    "sew-plain-seam": "sewing",


    "classroom_arrangement": "classroom-arrangement",
    "arrange-classroom-desks": "classroom-arrangement",
    "classroom-desks": "classroom-arrangement",
    "tap-percussion": "percussion",
    "tap-percussion-instrument": "percussion",
    "music-therapy-percussion": "percussion",
    "apply-paint": "painting",
    "paint-wall": "painting",
    "repaint-room": "painting",
    "surface_rinsing": "surface-rinsing",
    "rinse-surfaces": "surface-rinsing",
    "dish-rinsing": "surface-rinsing",
    "soil_excavation": "soil-excavation",
    "dig-soil": "soil-excavation",
    "dig": "soil-excavation",
    "soil-digging": "soil-excavation",


    "drill-hole": "drilling",
    "drill": "drilling",
    "gift_bags": "gift-bags",
    "ingredient_mixing": "ingredient-mixing",
    "syringe_handling": "syringe-handling",
    "depress-syringe-plunger": "syringe-handling",
    "mix-ingredients": "ingredient-mixing",
    "open-gift-bags": "gift-bags",
    "concrete_finishing": "concrete-finishing",
    "finish-concrete": "concrete-finishing",
    "concrete-finish": "concrete-finishing",
    "trowel-concrete": "concrete-finishing",
}


def register(environment_id: str, backend: str, factory: BackendFactory) -> None:
    key = (str(environment_id), str(backend))
    if key in _REGISTRY:
        raise ValueError(f"environment backend is already registered: {key}")
    _REGISTRY[key] = factory


def _newton_whiteboard_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_whiteboard import NewtonWhiteboardBackend

    return NewtonWhiteboardBackend(**kwargs)


register("whiteboard", "newton", _newton_whiteboard_factory)


def _newton_titration_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_titration import NewtonTitrationBackend

    return NewtonTitrationBackend(**kwargs)


register("titration", "newton", _newton_titration_factory)


def _newton_pencap_factory(**kwargs: Any) -> RoboticsBackend:
    from pencap.task_backend import NewtonPenCapTaskBackend
    return NewtonPenCapTaskBackend(**kwargs)


register("pencap", "newton", _newton_pencap_factory)


def _newton_lockkey_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_lockkey import NewtonLockKeyBackend
    return NewtonLockKeyBackend(**kwargs)


register("lockkey", "newton", _newton_lockkey_factory)


def _newton_laundry_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_laundry import NewtonLaundryBackend
    return NewtonLaundryBackend(**kwargs)


register("laundry", "newton", _newton_laundry_factory)


def _newton_football_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_football import NewtonFootballBackend
    return NewtonFootballBackend(**kwargs)


register("football", "newton", _newton_football_factory)


def _newton_deposit_cans_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_deposit_cans import NewtonDepositCansBackend
    return NewtonDepositCansBackend(**kwargs)

register("deposit-cans", "newton", _newton_deposit_cans_factory)


def _newton_tighten_codex_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_tighten_codex import NewtonTightenCodexBackend
    return NewtonTightenCodexBackend(**kwargs)


register("tighten-codex", "newton", _newton_tighten_codex_factory)


def available_environments() -> tuple[dict[str, str], ...]:
    return tuple(
        {"environment_id": environment_id, "backend": backend}
        for environment_id, backend in sorted(_REGISTRY)
    )


def create(
    environment_id: str,
    *,
    backend: str = "newton",
    robot: str | None = None,
    auto_reset: bool = True,
    **backend_options: Any,
) -> RoboticsEnv:
    """Create one interactive robotics environment.

    Every explicit option is passed through unchanged. Unsupported values fail
    before a simulator process is launched.
    """

    normalized_id = _ALIASES.get(str(environment_id), str(environment_id))
    key = (normalized_id, str(backend))
    try:
        factory = _REGISTRY[key]
    except KeyError as error:
        choices = ", ".join(
            f"{item['environment_id']}[{item['backend']}]"
            for item in available_environments()
        )
        raise ValueError(
            f"unknown environment/backend {normalized_id!r}/{backend!r}; "
            f"available: {choices}"
        ) from error
    from robots.catalog import default_robot, select_for_backend
    selected_robot = select_for_backend(normalized_id,default_robot(normalized_id) if robot is None else robot)
    instance = factory(robot=selected_robot, **backend_options)
    return RoboticsEnv(instance, auto_reset=auto_reset)


make = create


__all__ = ["available_environments", "create", "make", "register"]


def _newton_label_application_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_label_application import NewtonLabelApplicationBackend
    return NewtonLabelApplicationBackend(**kwargs)


register("label-application", "newton", _newton_label_application_factory)


def _newton_barcode_scanning_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_barcode_scanning import NewtonBarcodeScanningBackend

    return NewtonBarcodeScanningBackend(**kwargs)


register("barcode-scanning", "newton", _newton_barcode_scanning_factory)


def _newton_filleting_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_filleting_task_set import NewtonFilletingTaskSetBackend

    return NewtonFilletingTaskSetBackend(**kwargs)


register("filleting", "newton", _newton_filleting_factory)


def _newton_forklift_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_forklift import NewtonForkliftBackend

    return NewtonForkliftBackend(**kwargs)


register("forklift", "newton", _newton_forklift_factory)


def _newton_welding_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_welding import NewtonWeldingBackend

    return NewtonWeldingBackend(**kwargs)


register("welding", "newton", _newton_welding_factory)


def _newton_sewing_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_sewing import NewtonSewingBackend

    return NewtonSewingBackend(**kwargs)


register("sewing", "newton", _newton_sewing_factory)


def _newton_classroom_arrangement_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_classroom_arrangement import NewtonClassroomArrangementBackend

    return NewtonClassroomArrangementBackend(**kwargs)


register("classroom-arrangement", "newton", _newton_classroom_arrangement_factory)


def _newton_concrete_finishing_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_concrete_finishing_task_set import NewtonConcreteFinishingBackend

    return NewtonConcreteFinishingBackend(**kwargs)


register("concrete-finishing", "newton", _newton_concrete_finishing_factory)


def _newton_wine_pouring_factory(**kwargs: Any) -> RoboticsBackend:
    from wine_pouring.task_backend import NewtonWinePouringBackend

    return NewtonWinePouringBackend(**kwargs)


register("wine-pouring", "newton", _newton_wine_pouring_factory)


def _newton_percussion_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_percussion import NewtonPercussionBackend

    return NewtonPercussionBackend(**kwargs)


register("percussion", "newton", _newton_percussion_factory)


def _newton_painting_factory(**kwargs: Any) -> RoboticsBackend:
    from painting.robot_setup import validate
    from .errors import UnsupportedConfigurationError
    try:
        selected = validate(kwargs.get("robot", "mobile-fr3-duo"))
    except ValueError as exc:
        raise UnsupportedConfigurationError(str(exc)) from exc
    kwargs["robot"] = selected
    if selected == "g1":
        from .backends.newton_painting_g1 import NewtonPaintingG1Backend
        return NewtonPaintingG1Backend(**kwargs)
    from .backends.newton_painting import NewtonPaintingBackend
    return NewtonPaintingBackend(**kwargs)


register("painting", "newton", _newton_painting_factory)


def _newton_surface_rinsing_factory(**kwargs: Any) -> RoboticsBackend:
    from surface_rinsing.robot_setup import validate
    from .errors import UnsupportedConfigurationError
    try:
        kwargs["robot"] = validate(kwargs.get("robot", "mobile-fr3-duo"))
    except ValueError as exc:
        raise UnsupportedConfigurationError(str(exc)) from exc
    from .backends.newton_surface_rinsing_task_set import NewtonSurfaceRinsingBackend
    return NewtonSurfaceRinsingBackend(**kwargs)


register("surface-rinsing", "newton", _newton_surface_rinsing_factory)


def _newton_soil_excavation_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_soil_excavation_tasks import NewtonSoilExcavationTasksBackend

    return NewtonSoilExcavationTasksBackend(**kwargs)


register("soil-excavation", "newton", _newton_soil_excavation_factory)


def _newton_drilling_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_drilling_catalogue import NewtonDrillingCatalogueBackend

    return NewtonDrillingCatalogueBackend(**kwargs)


register("drilling", "newton", _newton_drilling_factory)


def _newton_ingredient_mixing_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_ingredient_mixing import NewtonIngredientMixingBackend

    return NewtonIngredientMixingBackend(**kwargs)


register("ingredient-mixing", "newton", _newton_ingredient_mixing_factory)


def _newton_syringe_handling_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_syringe_handling import NewtonSyringeHandlingBackend

    return NewtonSyringeHandlingBackend(**kwargs)


register("syringe-handling", "newton", _newton_syringe_handling_factory)


def _newton_gift_bags_factory(**kwargs: Any) -> RoboticsBackend:
    from .backends.newton_gift_bags import NewtonGiftBagsBackend

    return NewtonGiftBagsBackend(**kwargs)


register("gift-bags", "newton", _newton_gift_bags_factory)
