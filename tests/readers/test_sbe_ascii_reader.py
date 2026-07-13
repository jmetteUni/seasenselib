"""Unit tests for SeaBird ASCII datetime extraction."""

import unittest
from datetime import datetime

from seasenselib.readers.sbe_ascii_reader import _extract_date


class TestExtractDate(unittest.TestCase):
    """Tests for supported datetime parsing patterns in `_extract_date`."""

    def test_extract_date_day_mon_year(self):
        result = _extract_date("30 Mar 2026 03:00:01")
        self.assertEqual(result, datetime(2026, 3, 30, 3, 0, 1))

    def test_extract_date_month_day_year_dash(self):
        result = _extract_date("03-30-2026 03:00:01")
        self.assertEqual(result, datetime(2026, 3, 30, 3, 0, 1))

    def test_extract_date_day_month_year_dash(self):
        result = _extract_date("30-03-2026 03:00:01")
        self.assertEqual(result, datetime(2026, 3, 30, 3, 0, 1))

    def test_extract_date_year_month_day_dash(self):
        result = _extract_date("2026-03-30 03:00:01")
        self.assertEqual(result, datetime(2026, 3, 30, 3, 0, 1))

    def test_extract_date_raises_for_invalid_string(self):
        with self.assertRaises(ValueError):
            _extract_date("not a valid datetime")
