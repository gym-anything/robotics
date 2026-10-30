import unittest
import numpy as np
from whiteboard.render_readback import capture_rgb


class ReadbackTests(unittest.TestCase):
    def test_single_synchronous_capture_returns_current_pixels(self):
        calls = []
        pixels, retries = capture_rgb(lambda: calls.append(1),
            lambda: np.full((2, 3, 4), 42), 3, 2, flush_attempts=1)
        np.testing.assert_array_equal(pixels, np.full((2, 3, 3), 42))
        self.assertEqual((len(calls), retries), (1, 0))

    def test_default_preserves_two_capture_contract(self):
        calls = []
        capture_rgb(lambda: calls.append(1), lambda: np.ones((2, 3, 4)), 3, 2)
        self.assertEqual(len(calls), 2)

    def test_invalid_flush_count_rejected_before_render(self):
        def unexpected():
            self.fail('Invalid configuration must not render')
        for value in (0, 9, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                capture_rgb(unexpected, unexpected, 3, 2, flush_attempts=value)

    def test_bad_readback_retried_and_latest_valid_pixels_returned(self):
        values = iter(
            (np.ones((2, 3, 4), dtype=np.uint8), np.zeros((2, 3, 4)), np.full((2, 3, 4), 42))
        )
        calls = []
        pixels, retries = capture_rgb(lambda: calls.append(1), lambda: next(values), 3, 2)
        np.testing.assert_array_equal(pixels, np.full((2, 3, 3), 42))
        self.assertEqual(len(calls), 3)
        self.assertEqual(retries, 1)

    def test_bad_shape_fails_after_bounded_attempts(self):
        calls = []
        with self.assertRaisesRegex(RuntimeError, "8 render-only attempts"):
            capture_rgb(lambda: calls.append(1), lambda: np.zeros((0,)), 3, 2)
        self.assertEqual(len(calls), 8)
