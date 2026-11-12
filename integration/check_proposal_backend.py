"""Proposal byte-accounting, batching, and roofline-cost regressions."""

import math
import unittest

from proposal_backend import operation_traffic, stage_cost
from proposal_config import Hardware, Schedule, build_stage_workloads


class TestProposalBackend(unittest.TestCase):
    def test_dense_traffic_counts_actual_distinct_weights(self):
        operation = {'name': 'denoise.blocks.mlp_gate_and_up',
                     'm': 16, 'n': 4096, 'k': 1024,
                     'count': 36, 'weight_copies': 36}
        traffic = operation_traffic(operation)
        self.assertEqual(traffic['weight_bytes'], 1024 * 4096 * 36 * 2)
        self.assertEqual(traffic['activation_bytes'], (16 * 1024 + 16 * 4096) * 36 * 2)
        self.assertEqual(traffic['total_bytes'], sum(traffic[key] for key in ('weight_bytes', 'activation_bytes')))

    def test_microbatch_preserves_weights_and_scales_activations(self):
        single = build_stage_workloads(schedule=Schedule(b=1))['dit_stages'][0]['operations']
        for batch in (2, 5, 10):
            batched = build_stage_workloads(schedule=Schedule(b=batch))['dit_stages'][0]['operations']
            for one, many in zip(single, batched):
                single_bytes = operation_traffic(one)
                batch_bytes = operation_traffic(many)
                self.assertEqual(batch_bytes['weight_bytes'], single_bytes['weight_bytes'])
                self.assertEqual(batch_bytes['activation_bytes'], batch * single_bytes['activation_bytes'])

    def test_mqa_reads_shared_kv_once_per_request(self):
        # Eighteen layers, eight Q heads, one shared K/V head, five requests.
        batch = 5
        for name, m, n, k in (('denoise.blocks.qk', 16, 984, 256),
                              ('denoise.blocks.av', 16, 256, 984)):
            operation = {'name': name, 'm': m, 'n': n, 'k': k,
                         'count': batch * 18 * 8, 'weight_copies': 0}
            traffic = operation_traffic(operation)
            lhs_and_output = (m * k + m * n) * batch * 18 * 8 * 2
            rhs = k * n * batch * 18 * 2
            self.assertEqual(traffic['weight_bytes'], 0)
            self.assertEqual(traffic['activation_bytes'], lhs_and_output + rhs)

    def test_vision_attention_preserves_independent_head_operands(self):
        operation = {'name': 'vision.blocks.qk.layer0', 'm': 256, 'n': 256,
                     'k': 72, 'count': 3 * 16, 'weight_copies': 0}
        traffic = operation_traffic(operation)
        self.assertEqual(traffic['activation_bytes'],
                         (256 * 72 + 256 * 256 + 72 * 256) * 48 * 2)

    def test_source_exact_flops_for_horizon_16(self):
        hardware = Hardware()
        plan = build_stage_workloads(hardware, Schedule(b=1))
        self.assertEqual(plan['workload']['action_horizon'], 16)
        prefix = sum(stage_cost(stage, hardware)['flops_per_service']
                     for stage in plan['prefix_stages'])
        dit_step = stage_cost(plan['dit_stages'][0], hardware)['flops_per_service']
        self.assertEqual(prefix, 4635481079808)
        self.assertEqual(dit_step, 12526288896)
        self.assertEqual(prefix + 10 * dit_step, 4760743968768)

    def test_stage_utilization_and_cost_are_finite_positive(self):
        hardware = Hardware()
        plan = build_stage_workloads(hardware)
        for stage in plan['stages']:
            result = stage_cost(stage, hardware)
            self.assertTrue(math.isfinite(result['seconds_per_service']))
            self.assertGreater(result['seconds_per_service'], 0)
            for operation in result['operators']:
                self.assertGreater(operation['utilization'], 0)
                self.assertLessEqual(operation['utilization'], 1)
                self.assertGreaterEqual(operation['duration_seconds'], operation['compute_seconds'])
                self.assertGreaterEqual(operation['duration_seconds'], operation['memory_seconds'])

    def test_bandwidth_increase_preserves_work_and_reduces_cost(self):
        hardware = Hardware()
        stage = build_stage_workloads(hardware)['dit_stages'][0]
        first = stage_cost(stage, hardware)
        faster = stage_cost(dict(stage, bandwidth_gbps=stage['bandwidth_gbps'] * 2), hardware)
        self.assertEqual(first['flops_per_service'], faster['flops_per_service'])
        self.assertEqual(first['total_bytes'], faster['total_bytes'])
        self.assertLessEqual(faster['seconds_per_service'], first['seconds_per_service'])

    def test_proposal_working_set_fits_dram(self):
        hardware = Hardware(cores=128, channels=8)
        plan = build_stage_workloads(hardware, Schedule(b=1))
        prefix_weights = sum(operation_traffic(op)['weight_bytes']
                             for stage in plan['prefix_stages'] for op in stage['operations'])
        dit_weights = sum(operation_traffic(op)['weight_bytes']
                          for op in plan['dit_stages'][0]['operations'])
        self.assertEqual(prefix_weights, 4791831552)
        self.assertEqual(dit_weights, 859963392)
        # Conservative accounting keeps KV for all 200 saturated requests.
        working_set = prefix_weights + dit_weights + 200 * plan['kv_bytes_per_request']
        self.assertLess(working_set, hardware.dram_capacity_bytes)
        chip_sram = hardware.cores * hardware.core_sram_bytes + hardware.shared_sram_bytes
        self.assertGreater(dit_weights, chip_sram)


class CalibrationInputs(unittest.TestCase):
    def test_one_mebibyte_is_total_operand_buffer_capacity(self):
        import configparser
        from pathlib import Path
        import tempfile
        from pi05_runner import write_inputs
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            write_inputs(root, path, [(16, 32, 32)], 2, array_size=32, total_sram_kib=1024)
            config = configparser.ConfigParser()
            config.read(path / 'scale.cfg')
            arch = config['architecture_presets']
            self.assertEqual(sum(int(arch[key]) for key in
                                 ('ifmapsramszkb', 'filtersramszkb', 'ofmapsramszkb')), 1024)
            self.assertEqual(arch['arrayheight'], '32')
            self.assertEqual(arch['wordsizebytes'], '2')


if __name__ == '__main__':
    unittest.main()
