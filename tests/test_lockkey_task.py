"""Lock/key task text: one job text in the catalogue and the policy instruction. No simulator."""
import re
import unittest

from gym_anything_robotics import tasks
from lockkey.newton15.task import DESCRIPTION, INSTRUCTION


class LockKeyTaskTextTest(unittest.TestCase):
    def test_catalogue_serves_the_policy_job_text(self):
        spec = tasks.resolve("lockkey", {})
        self.assertEqual(spec.task_id, "lockkey/unlock")
        self.assertEqual(spec.description, INSTRUCTION)
        self.assertTrue(DESCRIPTION.startswith(INSTRUCTION + " "))
        self.assertTrue(spec.evaluator["available"])

    def test_job_text_carries_no_scoring_tolerances_or_action_script(self):
        self.assertIsNone(re.search(r"\d", INSTRUCTION))
        for phrase in ("degree", "mm", "grasp", "raise the shackle", "cradle", "correct key"):
            self.assertIsNone(re.search(rf"\b{phrase}", DESCRIPTION), phrase)
        for phrase in ("near (-Y) stand", "swing the shackle open", "other key on its stand", "chest lid"):
            self.assertIn(phrase, INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
