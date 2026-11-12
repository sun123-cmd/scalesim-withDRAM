"""Proposal allocation and workload-conservation regressions."""
import unittest

from pi05_model import build_operations
from proposal_config import (HARDWARE_POINTS, Hardware, Schedule, allocate,
                             build_stage_workloads, macs, timestep_partition)


class TestProposalConfig(unittest.TestCase):
    def test_hardware_points(self):
        self.assertEqual(len(HARDWARE_POINTS), 16)
        self.assertEqual(len({point.point_id for point in HARDWARE_POINTS}), 16)
        self.assertEqual(Hardware(128, 8).bandwidth_gbps, 546.4)

    def test_timestep_partitions(self):
        self.assertEqual(timestep_partition(3), [4, 3, 3])
        self.assertEqual(timestep_partition(4), [3, 3, 2, 2])
        for stages in (1, 2, 3, 4, 5, 10):
            self.assertEqual(sum(timestep_partition(stages)), 10)

    def test_resources_conserved(self):
        data = build_stage_workloads()
        self.assertEqual(sum(stage['cores'] for stage in data['stages']), 32)
        self.assertAlmostEqual(sum(stage['bandwidth_gbps'] for stage in data['stages']), 136.6)
        self.assertEqual(data['kv_bytes_per_request'], 17842176)

    def test_invalid_stage_allocation(self):
        with self.assertRaises(ValueError):
            allocate(Hardware(8, 1), Schedule(kp=6, kd=10))
        with self.assertRaises(ValueError):
            allocate(Hardware(), Schedule(beta=0))

    def test_whole_model_mac_conservation(self):
        original = build_operations(action_horizon=16, denoise_steps=10)
        prefix = macs([op for op in original if op['stage'] != 'denoise'])
        denoise = macs([op for op in original if op['stage'] == 'denoise'])
        for kp in (1, 2, 3, 4, 6):
            for batch in (1, 2, 5, 10):
                data = build_stage_workloads(Hardware(128, 8), Schedule(kp=kp, kd=4, b=batch,
                                                                          dit_core_fraction_numerator=8))
                self.assertEqual(sum(macs(stage['operations']) for stage in data['prefix_stages']), prefix)
                self.assertEqual(sum(macs(stage['operations']) * stage['steps']
                                     for stage in data['dit_stages']), denoise * batch)
                boundaries = [(s['layer_start'], s['layer_end']) for s in data['prefix_stages']]
                self.assertEqual(boundaries[0][0], 0)
                self.assertEqual(boundaries[-1][1], 47)
                self.assertTrue(all(a[1] == b[0] for a, b in zip(boundaries, boundaries[1:])))

    def test_camera_batch_shares_dense_weights(self):
        data = build_stage_workloads(schedule=Schedule(kp=1))
        operations = data['prefix_stages'][0]['operations']
        patch = next(op for op in operations if op['name'].startswith('vision.patch_embedding'))
        self.assertEqual((patch['m'], patch['count'], patch['weight_copies']), (768, 1, 1))
        qkv = next(op for op in operations if op['name'].startswith('vision.blocks.qkv'))
        self.assertEqual((qkv['m'], qkv['count'], qkv['weight_copies']), (768, 3, 3))
        qk = next(op for op in operations if op['name'].startswith('vision.blocks.qk.layer'))
        self.assertEqual((qk['m'], qk['count'], qk['weight_copies']), (256, 48, 0))

    def test_proposal_pilot_invalid_schedule_is_documented(self):
        data = build_stage_workloads()
        self.assertEqual(data['pilot_validation_notes'][0]['status'], 'invalid')
        with self.assertRaises(ValueError):
            build_stage_workloads(Hardware(), Schedule(kp=4, kd=10, b=10))

    def test_batch_shares_dense_weights_preserves_attention(self):
        single = build_stage_workloads(schedule=Schedule(b=1))['dit_stages'][0]['operations']
        batched = build_stage_workloads(schedule=Schedule(b=5))['dit_stages'][0]['operations']
        for one, five in zip(single, batched):
            if one['name'].endswith(('.qk', '.av')):
                self.assertEqual(one['m'], five['m'])
                self.assertEqual(five['count'], 5 * one['count'])
            else:
                self.assertEqual(five['m'], 5 * one['m'])
                self.assertEqual(five['count'], one['count'])


if __name__ == '__main__':
    unittest.main()
