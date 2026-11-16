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
                              side_effect=[first, second]) as urlopen:
                with self.assertRaisesRegex(OSError, "Incomplete"):
                    download_lodopab.download_file(path, 6)
                self.assertEqual(path.read_bytes(), b"abc")
                download_lodopab.download_file(path, 6)
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
