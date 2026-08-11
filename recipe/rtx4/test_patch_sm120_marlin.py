#!/usr/bin/env python3
"""Tests for the fail-closed SM120 Marlin source patcher."""

from __future__ import annotations

import unittest

from recipe.rtx4.patch_sm120_marlin import REPLACEMENTS, source_state


class Sm120MarlinPatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original = "\n".join(old for old, _ in REPLACEMENTS)
        self.patched = "\n".join(new for _, new in REPLACEMENTS)

    def test_recognizes_exact_original_and_patched_sources(self) -> None:
        self.assertEqual(source_state(self.original), "original")
        self.assertEqual(source_state(self.patched), "patched")

    def test_rejects_partial_application(self) -> None:
        partial = self.original.replace(REPLACEMENTS[0][0], REPLACEMENTS[0][1])
        with self.assertRaisesRegex(RuntimeError, "partially applied"):
            source_state(partial)

    def test_rejects_source_drift(self) -> None:
        drifted = self.original.replace(REPLACEMENTS[0][0], "unexpected source")
        with self.assertRaisesRegex(RuntimeError, "source drift"):
            source_state(drifted)


if __name__ == "__main__":
    unittest.main()
