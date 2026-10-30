"""Drilling evaluator score semantics (pure Python, no engine): the score counts goal checks only, and success needs
every goal and every constraint.
"""

from __future__ import annotations

import unittest


class DrillingScoreSemanticsTest(unittest.TestCase):
    """Pure-Python: score is the fraction of goal checks; success needs goals and constraints."""

    def test_score_counts_goals_only(self) -> None:
        from drilling import task as T

        class Stub(T.Evaluator):
            def __init__(self, checks):
                self.task, self.instruction, self._checks = "press-drill", "", checks

            def common(self):
                return []

            def _press_drill(self):
                return self._checks
        goals = [T._check("hole", False), T._check("entry", False)]
        cons = [T._check("spindle stopped", True, kind="constraint"), T._check("quill up", True, kind="constraint")]
        ev = Stub(goals + cons).evaluate()
        self.assertEqual(ev["score"], 0.)
        self.assertFalse(ev["success"])
        self.assertTrue(ev["constraints_ok"])
        ev = Stub([T._check("hole", True), T._check("entry", True),
                   T._check("spindle stopped", False, kind="constraint")]).evaluate()
        self.assertEqual(ev["score"], 1.)
        self.assertFalse(ev["success"])            # a broken constraint fails the task at full goal score


if __name__ == "__main__":
    unittest.main()
