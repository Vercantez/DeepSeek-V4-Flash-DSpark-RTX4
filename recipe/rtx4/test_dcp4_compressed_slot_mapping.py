#!/usr/bin/env python3
"""Pure, deterministic checks for the DCP4 compressed-slot ownership patch."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import patch_dcp4_compressed_slot_mapping as patcher


DCP_WORLD_SIZE = 4
INTERLEAVE = 1
COMPRESS_RATIO = 4
COMPRESSED_BLOCK_SIZE = 64


def cp_owner(position: int, world_size: int, interleave: int) -> int:
    return (position // interleave) % world_size


def is_compression_boundary(position: int) -> bool:
    return (position + 1) % COMPRESS_RATIO == 0


def output_owner(position: int) -> int:
    assert is_compression_boundary(position)
    compressed_position = position // COMPRESS_RATIO
    return cp_owner(compressed_position, DCP_WORLD_SIZE, INTERLEAVE)


def output_block_index(position: int, *, dcp_aware: bool) -> int:
    assert is_compression_boundary(position)
    compressed_position = position // COMPRESS_RATIO
    virtual_block_size = COMPRESSED_BLOCK_SIZE
    if dcp_aware:
        virtual_block_size *= DCP_WORLD_SIZE
    return compressed_position // virtual_block_size


class CompressedSlotOwnershipTest(unittest.TestCase):
    def test_c4_source_owner_is_rank3_but_output_owner_rotates(self) -> None:
        boundaries = [3, 7, 11, 15]
        self.assertEqual(
            [cp_owner(pos, DCP_WORLD_SIZE, INTERLEAVE) for pos in boundaries],
            [3, 3, 3, 3],
        )
        self.assertEqual([output_owner(pos) for pos in boundaries], [0, 1, 2, 3])

    def test_fixed_mapping_has_one_boundary_per_rank(self) -> None:
        boundaries = [3, 7, 11, 15]
        valid_token_indices_by_rank = {
            rank: [pos for pos in boundaries if output_owner(pos) == rank]
            for rank in range(DCP_WORLD_SIZE)
        }
        self.assertEqual(
            valid_token_indices_by_rank,
            {0: [3], 1: [7], 2: [11], 3: [15]},
        )

    def test_old_default_mapping_advances_block_table_four_times_too_fast(self) -> None:
        # At the first compressed-page rollover, the old DCP-default call reads
        # block_table[1].  DCP4 still owns block_table[0] until position 1023.
        self.assertEqual(output_block_index(255, dcp_aware=False), 0)
        self.assertEqual(output_block_index(259, dcp_aware=False), 1)
        self.assertEqual(output_block_index(259, dcp_aware=True), 0)
        self.assertEqual(output_block_index(1023, dcp_aware=False), 3)
        self.assertEqual(output_block_index(1023, dcp_aware=True), 0)
        self.assertEqual(output_block_index(1027, dcp_aware=True), 1)


class DeterministicPatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_source = (
            "class DeepseekV4FlashMLAMetadataBuilder:\n"
            + patcher.INIT_OLD
            + "        slot_mapping = get_compressed_slot_mapping(\n"
            + patcher.CALL_OLD
        )

    def test_exact_old_source_is_patched_and_idempotent(self) -> None:
        patched, changed = patcher.patch_source(self.old_source)
        self.assertTrue(changed)
        self.assertIn(patcher.INIT_NEW, patched)
        self.assertIn(patcher.CALL_NEW, patched)

        patched_again, changed_again = patcher.patch_source(patched)
        self.assertFalse(changed_again)
        self.assertEqual(patched_again, patched)

    def test_partial_patch_fails_closed(self) -> None:
        partial = self.old_source.replace(patcher.INIT_OLD, patcher.INIT_NEW, 1)
        with self.assertRaisesRegex(RuntimeError, "partially applied"):
            patcher.patch_source(partial)

    def test_source_drift_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "builder-init"):
            patcher.patch_source("unrelated source\n")


if __name__ == "__main__":
    unittest.main()
