"""Narrow integer labels must produce the same metrics as int64 labels."""

import unittest
import numpy as np

from brtp.evaluation.native import native_metrics


class NativeIntegerLabelTests(unittest.TestCase):
    def test_confusion_matrix_and_macro_scores_do_not_overflow_integer_labels(self):
        logits = np.full((4, 30), -3.0)
        logits[np.arange(4), [9, 18, 4, 29]] = 3.0
        expected = native_metrics(logits, np.array([9, 18, 18, 29], dtype=np.int64))
        for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.uint64):
            with self.subTest(dtype=dtype):
                self.assertEqual(
                    native_metrics(logits, np.array([9, 18, 18, 29], dtype=dtype)), expected
                )

    def test_uint64_out_of_range_is_rejected_before_signed_conversion(self):
        for value in (30, 2**63, 2**64 - 1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "index native"):
                native_metrics(np.zeros((1, 30)), np.array([value], dtype=np.uint64))


if __name__ == "__main__":
    unittest.main()
