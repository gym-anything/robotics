"""Normalize results without inventing success thresholds or temporal history."""
import math
from .action_history import diagnostic_json


def normalize(result):
    base = {"schema": "gar.evaluation.v1", "scope": "task_success_not_physics_qualification"}
    if result is None:
        return {**base, "available": False, "success": None, "score": None,
                "reason": "This live configuration has no integrated task evaluator."}
    paths = []
    value = diagnostic_json(result, paths)
    if paths or not isinstance(value, dict):
        raise ValueError("Evaluator returned invalid/non-finite diagnostics")
    if type(value.get("success")) is not bool:
        raise ValueError("Evaluator must supply an explicit boolean success")
    score = value.get("score")
    if type(score) not in (int, float) or not math.isfinite(score):
        raise ValueError("Evaluator must supply a finite numeric score")
    return {**base, "available": True, "success": value["success"], "score": score,
            "details": value}
