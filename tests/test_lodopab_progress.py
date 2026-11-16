"""Dependency-free tests for LoDoPaB per-slice checkpoints."""
import tempfile
import unittest
from pathlib import Path

from medreconlab import lodopab_progress as progress


class CheckpointTests(unittest.TestCase):
    def test_interruption_and_resume(self):
        calls = []

        def evaluate(index):
            calls.append(index)
            if calls == [0, 1]:
                raise InterruptedError
            return [{"index": index, "method": "fbp"}, {"index": index, "method": "cgls"}]

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "progress.sqlite"
            settings = {"indices": [0, 1], "iterations": [1, 2]}
            with self.assertRaises(InterruptedError):
                progress.collect_slices([0, 1], evaluate, checkpoint, settings)
            results = progress.collect_slices([0, 1], evaluate, checkpoint, settings)
            self.assertEqual(calls, [0, 1, 1])
            self.assertEqual([row["index"] for row in results], [0, 0, 1, 1])
            with self.assertRaisesRegex(ValueError, "Checkpoint settings differ"):
                progress.collect_slices(
                    [0, 1], evaluate, checkpoint, {**settings, "iterations": [2, 3]}
                )
            self.assertEqual(calls, [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
