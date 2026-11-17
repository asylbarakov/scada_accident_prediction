"""Ensure a server ignoring Range cannot trigger a full ZIP download."""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from fetch_care_a import ARCHIVE_SIZE, RemoteZip, TRANSFER_LIMIT


class TestRemoteZip(unittest.TestCase):
    @patch("fetch_care_a.urlopen")
    def test_reject_full_response_without_reading_body(self, urlopen):
        response = MagicMock(status=200)
        urlopen.return_value.__enter__.return_value = response
        with RemoteZip() as remote:
            with self.assertRaisesRegex(OSError, "requested byte range"):
                remote.read(1)
            self.assertEqual(remote.transferred, 0)
        response.read.assert_not_called()

    @patch("fetch_care_a.urlopen")
    def test_validate_range_and_cache(self, urlopen):
        response = MagicMock(status=206, headers={"Content-Range": f"bytes 0-4095/{ARCHIVE_SIZE}"})
        response.read.return_value = b"abc" + bytes(4093)
        urlopen.return_value.__enter__.return_value = response
        with RemoteZip() as remote:
            self.assertEqual(remote.read(1), b"a")
            self.assertEqual(remote.read(2), b"bc")
            self.assertEqual(remote.transferred, 4096)
        response.read.assert_called_once_with(4097)

    @patch("fetch_care_a.urlopen")
    def test_transfer_limit(self, urlopen):
        with RemoteZip() as remote:
            with self.assertRaisesRegex(OSError, "transfer limit"):
                remote.read(TRANSFER_LIMIT + 1)
        urlopen.assert_not_called()
