"""Deterministic tests for full-shape GEMM tiling and weighted measurements."""
from collections import Counter
import unittest

from pi05_runner import aggregate, dimension_tiles, operation_tiles


class Pi05TilingTests(unittest.TestCase):
    def test_exact_multiple(self):
        self.assertEqual(dimension_tiles(64, 32), Counter({32: 2}))

    def test_edge_tile(self):
        self.assertEqual(dimension_tiles(65, 32), Counter({32: 2, 1: 1}))
        self.assertEqual(dimension_tiles(17, 32), Counter({17: 1}))

    def test_invalid_dimensions(self):
        for length in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                dimension_tiles(length, 32)
        with self.assertRaises(ValueError):
            dimension_tiles(32, 0)

    def test_three_axes_and_repetition_preserve_macs(self):
        operation = dict(name='edge', stage='vision', m=65, n=33, k=17, count=3)
        tiles = operation_tiles(operation, 32)
        self.assertEqual(sum(tiles.values()), 18)
        self.assertEqual(len(tiles), 4)
        self.assertEqual(sum(m*n*k*count for (m,n,k), count in tiles.items()), 65*33*17*3)

    def test_invalid_repetitions(self):
        for count in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                operation_tiles(dict(m=32, n=32, k=32, count=count), 32)

    def test_weighted_aggregation(self):
        operation = dict(name='edge', stage='expert', m=65, n=33, k=17, count=3)
        shapes = sorted(operation_tiles(operation, 32))
        layers = [dict(replay=dict(requests=3, completed=3, cycles=11),
                       baseline=dict(compute_cycles_including_stalls=5),
                       coupled=dict(compute_cycles_including_stalls=7, stall_cycles=2,
                                    total_cycles_including_prefetch=9)) for _ in shapes]
        reports, stages, total = aggregate([operation], 32, shapes, {'layers': layers})
        self.assertEqual(total['macs'], 65*33*17*3)
        self.assertEqual(total['weighted_dram_requests'], 54)
        self.assertEqual(total['weighted_coupled_array_cycles'], 126)
        self.assertEqual(total['weighted_stall_cycles'], 36)
        self.assertEqual(stages['expert'], total)
        self.assertEqual(reports[0]['tile_count'], 18)

    def test_incomplete_requests_fail(self):
        operation = dict(name='one', stage='vision', m=1, n=1, k=1)
        with self.assertRaises(ValueError):
            aggregate([operation], 32, [(1,1,1)],
                      {'layers': [dict(replay=dict(requests=2, completed=1))]})


if __name__ == '__main__':
    unittest.main()
