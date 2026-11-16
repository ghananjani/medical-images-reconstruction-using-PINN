"""Local response fixtures for resumable dataset downloads."""
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from scripts import download_lodopab


class Response(BytesIO):
    def __init__(self, content, status, headers):
        super().__init__(content)
        self.status = status
        self.headers = headers


class DownloadTests(unittest.TestCase):
    def test_interrupted_transfer_can_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.zip"
            first = Response(b"abc", 206, {"Content-Range": "bytes 0-5/6"})
            second = Response(b"def", 206, {"Content-Range": "bytes 3-5/6"})
            with patch.object(download_lodopab.urllib.request, "urlopen",
                              side_effect=[first, second]) as urlopen, patch.object(download_lodopab.time, "sleep") as sleep:
                download_lodopab.download_file(path, 6)
                sleep.assert_called_once_with(1)
                self.assertEqual(urlopen.call_args.args[0].get_header("Range"), "bytes=3-5")
                download_lodopab.download_file(path, 6)
                self.assertEqual(urlopen.call_count, 2)
            self.assertEqual(path.read_bytes(), b"abcdef")

    def test_unexpected_response_preserves_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.zip"
            path.write_bytes(b"abc")
            for response in [
                Response(b"abcdef", 200, {}),
                Response(b"def", 206, {"Content-Range": "bytes 1-3/6"}),
            ]:
                with patch.object(download_lodopab.urllib.request, "urlopen", return_value=response):
                    with self.assertRaises(ValueError):
                        download_lodopab.download_file(path, 6)
                self.assertEqual(path.read_bytes(), b"abc")


    def test_bounded_ranges_preserve_archive_order(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.zip"
            responses = [
                Response(b"abc", 206, {"Content-Range": "bytes 0-2/6"}),
                Response(b"def", 206, {"Content-Range": "bytes 3-5/6"}),
            ]
            with (
                patch.object(download_lodopab, "CHUNK_BYTES", 3),
                patch.object(download_lodopab.urllib.request, "urlopen",
                             side_effect=responses) as urlopen,
            ):
                download_lodopab.download_file(path, 6)
            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertEqual([c.args[0].get_header("Range") for c in urlopen.call_args_list],
                             ["bytes=0-2", "bytes=3-5"])


    def test_test_split_downloads_archives_and_required_mappings(self):
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch("sys.argv", ["download_lodopab", "--downloads", temp, "--split", "test"]),
                patch.object(download_lodopab, "download_file") as download,
            ):
                download_lodopab.main()
            self.assertEqual({call.args[0].name for call in download.call_args_list}, {
                "ground_truth_test.zip", "observation_test.zip", "patient_ids_rand_train.csv",
                "patient_ids_rand_validation.csv", "patient_ids_rand_test.csv",
            })
