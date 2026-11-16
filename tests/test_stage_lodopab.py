"""Dependency-free fixtures for release staging."""
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import stage_lodopab


class StageTests(unittest.TestCase):
    def test_verified_archive_and_refusal_to_replace_different_data(self):
        with tempfile.TemporaryDirectory() as temp:
            downloads = Path(temp) / "downloads"
            downloads.mkdir()
            output = Path(temp) / "output"
            for split in ("train", "test", "validation"):
                (downloads / f"patient_ids_rand_{split}.csv").write_text("fixture\n")
            for kind in ("observation", "ground_truth"):
                stem = f"{kind}_validation"
                with zipfile.ZipFile(downloads / f"{stem}.zip", "w") as archive:
                    for i in range(28):
                        archive.writestr(f"folder/{stem}_{i:03d}.hdf5", b"fixture")
            checksums = {
                p.name: hashlib.md5(p.read_bytes(), usedforsecurity=False).hexdigest()
                for p in downloads.iterdir()
            }
            broken = downloads / "patient_ids_rand_train.csv"
            with patch.object(stage_lodopab, "CHECKSUMS", checksums):
                broken.write_text("changed")
                with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                    stage_lodopab.stage(downloads, output, "validation")
                self.assertFalse(output.exists())
                broken.write_text("fixture\n")
                stage_lodopab.stage(downloads, output, "validation")
                self.assertEqual(len(list(output.iterdir())), 59)
                destination = output / "observation_validation_000.hdf5"
                destination.write_bytes(b"existing")
                with self.assertRaisesRegex(ValueError, "Existing file differs"):
                    stage_lodopab.stage(downloads, output, "validation")
                self.assertEqual(destination.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
