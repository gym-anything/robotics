"""Physics-first robotic laundry-folding environment.

The package root intentionally exports only backend-independent contracts.
Import :mod:`laundry.newton_backend` or :mod:`laundry.laundry_env` explicitly
when the optional Newton or Gymnasium dependencies are available.
"""

from .api import BackendDescriptor, LaundryPhysicsBackend, ReplayLaundryBackend
from .control import (
    BIMANUAL_CARTESIAN_ACTION_SCHEMA,
    FIXTURE_VELOCITY_ACTION_SCHEMA,
    ActionChannel,
    ActionSchema,
    make_bimanual_cartesian_action_schema,
)
from .episode import EpisodePhase, EpisodeStep, HalfFoldEpisodeTask
from .garment_asset import GarmentConstruction, PanelPattern, PathRef, PatternPath
from .garment_patterns import (
    GARMENT_CONSTRUCTIONS,
    construction_summaries,
    get_garment_construction,
)
from .garment_task import (
    GARMENT_TASK_SPECS,
    GarmentEpisodePhase,
    GarmentEpisodeState,
    GarmentEpisodeStep,
    GarmentEpisodeTask,
    GarmentTaskEvaluator,
    GarmentTaskResult,
    GarmentTaskSpec,
    get_garment_task_spec,
)
from .garments import (
    GARMENT_PROFILES,
    GarmentProfile,
    GarmentTopology,
    PhysicsSupport,
    get_garment_profile,
    implemented_garment_profiles,
)
from .observation import LaundryObservation, ToolState
from .specs import MATERIAL_PROFILES, PHYSICS_PROFILES, TowelSpec

__all__ = [
    "ActionChannel",
    "ActionSchema",
    "BackendDescriptor",
    "BIMANUAL_CARTESIAN_ACTION_SCHEMA",
    "EpisodePhase",
    "EpisodeStep",
    "FIXTURE_VELOCITY_ACTION_SCHEMA",
    "GARMENT_PROFILES",
    "GARMENT_CONSTRUCTIONS",
    "GARMENT_TASK_SPECS",
    "GarmentConstruction",
    "GarmentEpisodePhase",
    "GarmentEpisodeState",
    "GarmentEpisodeStep",
    "GarmentEpisodeTask",
    "GarmentProfile",
    "GarmentTopology",
    "GarmentTaskEvaluator",
    "GarmentTaskResult",
    "GarmentTaskSpec",
    "HalfFoldEpisodeTask",
    "LaundryObservation",
    "LaundryPhysicsBackend",
    "MATERIAL_PROFILES",
    "PHYSICS_PROFILES",
    "PhysicsSupport",
    "PanelPattern",
    "PathRef",
    "PatternPath",
    "ReplayLaundryBackend",
    "ToolState",
    "TowelSpec",
    "get_garment_profile",
    "get_garment_construction",
    "get_garment_task_spec",
    "implemented_garment_profiles",
    "construction_summaries",
    "make_bimanual_cartesian_action_schema",
]
