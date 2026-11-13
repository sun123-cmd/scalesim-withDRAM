"""Online program layout, shared-address and measured-cost contract tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from online_workload import (KV_MATRIX_BYTES, KV_STRIDE, LOGICAL_CAPACITY,
    compile_online_program, read_compute_costs, target_memory_config, tile_shapes,
    unbatched, write_compute_costs, all_tile_shapes, isolated_config, maximum_m_tile, measure_compute_costs, prepare_cost_table)
from proposal_config import Hardware, Schedule, build_stage_workloads


def costs_for(hardware=Hardware(), schedule=Schedule()):
    result = {}
    for stage in build_stage_workloads(hardware, schedule)['stages']:
        for operation in unbatched(stage, schedule.b):
            for shape in tile_shapes(operation, 1 if stage['kind'] == 'prefix' else schedule.b):
                result[shape] = sum(shape)
    return result


class OnlineWorkload(unittest.TestCase):
    def compile(self, hardware=Hardware(), schedule=Schedule()):
        return compile_online_program(hardware, schedule, costs_for(hardware, schedule))

    def test_default_uses_complete_horizon16_workload_and_real_dma_limit(self):
        data = self.compile()
        self.assertEqual(data['workload']['workload']['action_horizon'], 16)
        self.assertEqual(data['online']['warmup_requests'], 0)
        self.assertEqual(data['online']['measure_requests'], 50)
        self.assertEqual(data['online']['max_inflight'], 64)
        self.assertEqual(sum(stage['timestep_count'] for stage in data['online']['stages']
                             if stage['kind'] == 'dit'), 10)

    def test_weights_identical_across_partition_batch_and_replicas(self):
        first = self.compile()['address_layout']['weights']
        other = self.compile(Hardware(cores=128), Schedule(kp=3, kd=5, b=10, replicas=2,
                                                         dit_core_fraction_numerator=8))
        self.assertEqual(first, other['address_layout']['weights'])
        for stage in other['online']['stages']:
            for operation in stage['operations']:
                if operation['b_shared']:
                    self.assertEqual(operation['b_base'], first[operation['weight_key']]['base'] + operation['weight_offset_bytes'])

    def test_weight_catalogue_cache_is_isolated_from_result_mutation(self):
        first = self.compile()
        key = next(iter(first['address_layout']['weights']))
        original = first['address_layout']['weights'][key]['base']
        first['address_layout']['weights'][key]['base'] += 128
        second = self.compile()
        self.assertEqual(second['address_layout']['weights'][key]['base'], original)

    def test_request_kv_slots_are_disjoint_and_within_capacity(self):
        data = self.compile()
        layout = data['address_layout']
        self.assertGreaterEqual(layout['kv_stride'], KV_STRIDE)
        self.assertGreater(layout['request_slots'], 70)
        self.assertLessEqual(layout['highest_address_exclusive'], LOGICAL_CAPACITY)
        self.assertLessEqual(layout['kv_base'] + layout['request_slots'] * layout['kv_stride'],
                             layout['activation_base'])

    def test_kv_producer_and_consumer_alias_per_layer(self):
        data = self.compile()
        operations = {operation['name']: operation for stage in data['online']['stages']
                      for operation in stage['operations']}
        kv = operations['prefix.blocks.kv.layer0']
        qk = operations['prefix.blocks.qk.layer0']
        av = operations['prefix.blocks.av.layer0']
        self.assertFalse(kv['c_stage_local'])
        self.assertEqual(kv['c_base'], qk['b_base'])
        self.assertEqual(kv['c_base'] + KV_MATRIX_BYTES, av['b_base'])
        self.assertTrue(qk['b_transposed'])
        self.assertEqual(qk['rhs_share'], 8)
        self.assertEqual(operations['denoise.blocks.qk.layer0']['b_base'], qk['b_base'])
        self.assertEqual(operations['denoise.blocks.kv.layer0']['c_base'],
                         kv['c_base'] + 968 * 256 * 2)

    def test_stage_scratch_disjoint_and_replica_stride_bounded(self):
        data = self.compile(Hardware(128), Schedule(kp=4, kd=10, b=10, replicas=2,
                                                   dit_core_fraction_numerator=8))
        starts = [stage['operations'][0]['a_base'] for stage in data['online']['stages']]
        self.assertEqual(len(starts), len(set(starts)))
        self.assertGreater(data['online']['request_slots'], 20)
        self.assertLessEqual(data['address_layout']['highest_address_exclusive'], LOGICAL_CAPACITY)

    def test_compact_program_keeps_counts_without_expanding_tiles(self):
        data = self.compile()
        operations = [operation for stage in data['online']['stages'] for operation in stage['operations']]
        self.assertLess(len(operations), 1000)
        self.assertTrue(all(operation['count'] > 0 for operation in operations))
        self.assertTrue(any(operation['count'] > 1 for operation in operations))

    def test_missing_or_invalid_costs_fail(self):
        with self.assertRaisesRegex(ValueError, 'misses'):
            compile_online_program(costs={})
        table = costs_for()
        table[next(iter(table))] = 0
        with self.assertRaisesRegex(ValueError, 'positive'):
            compile_online_program(costs=table)

    def test_cost_table_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'cost.json'
            original = {(16, 32, 32): 80, (32, 32, 32): 96}
            write_compute_costs(path, original)
            self.assertEqual(read_compute_costs(path), original)
            data = json.loads(path.read_text())
            data['tiles'].append(data['tiles'][0])
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_compute_costs(path)

    def test_cost_cache_serializes_concurrent_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'costs.json'
            table = {shape: sum(shape) for shape in all_tile_shapes()}
            with patch('online_workload.measure_compute_costs', return_value=table) as measure:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(prepare_cost_table, path) for _ in range(2)]
                    self.assertEqual([future.result() for future in futures], [table, table])
                measure.assert_called_once()
            self.assertEqual(read_compute_costs(path), table)

    def test_tiny_sram_rejected(self):
        with self.assertRaisesRegex(ValueError, 'SRAM'):
            self.compile(Hardware(core_sram_bytes=1024))

    def test_shared_sram_changes_tile_capacity_without_changing_work(self):
        table = {shape: sum(shape) for shape in all_tile_shapes()}
        schedule = Schedule(b=10)
        private = Hardware(32, core_sram_bytes=1 << 18, shared_sram_bytes=0)
        pooled = Hardware(32, core_sram_bytes=1 << 18, shared_sram_bytes=64 << 20)
        first = compile_online_program(private, schedule, table, mode='baseline')
        second = compile_online_program(pooled, schedule, table, mode='baseline')
        self.assertGreater(second['online']['stages'][0]['maximum_m_tile'],
                           first['online']['stages'][0]['maximum_m_tile'])
        for left, right in zip(first['online']['stages'], second['online']['stages']):
            self.assertEqual([(op['m'], op['n'], op['k'], op['count']) for op in left['operations']],
                             [(op['m'], op['n'], op['k'], op['count']) for op in right['operations']])

    def test_whole_chip_baseline_accepts_small_replica_core_partitions(self):
        table = {shape: sum(shape) for shape in all_tile_shapes()}
        data = compile_online_program(Hardware(8), Schedule(b=2, replicas=4),
                                      table, mode='baseline')
        self.assertEqual(data['online']['mode'], 'baseline')
        self.assertEqual([stage['cores'] for stage in data['online']['stages']], [2, 2])
        self.assertEqual(len(data['online']['stages']), 2)
        self.assertEqual(data['online']['stages'][1]['timestep_count'], 10)
        self.assertTrue(all(operation['request_stride'] == data['online']['request_stride']
                            for stage in data['online']['stages'] for operation in stage['operations']))

    def test_complete_cost_shape_universe_covers_every_batch(self):
        table = {shape: sum(shape) for shape in all_tile_shapes()}
        for batch in (1, 2, 5, 10):
            for mode in ('baseline', 'pipeline'):
                data = compile_online_program(Hardware(128), Schedule(b=batch), table, mode=mode)
                self.assertTrue(data['online']['stages'])

    def test_boolean_and_negative_measurement_inputs_fail(self):
        costs = costs_for()
        for kwargs in ({'warmup_requests': True}, {'measure_requests': False},
                       {'warmup_requests': -1}, {'measure_requests': -1},
                       {'max_cycles': -1}, {'max_cycles': True}):
            with self.assertRaises(ValueError):
                compile_online_program(costs=costs, **kwargs)

    def test_isolated_stage_retains_program_and_dma_contract(self):
        source = self.compile()
        stage_index = 3
        isolated = isolated_config(source, stage_index, {'memory_system': {'impl': 'test'}})
        online = isolated['online']
        self.assertEqual(online['mode'], 'isolated')
        self.assertEqual(online['replicas'], 1)
        self.assertEqual(len(online['stages']), 1)
        selected = online['stages'][0]
        self.assertEqual(selected['operations'], source['online']['stages'][stage_index]['operations'])
        self.assertEqual(selected['timestep_start'], 0)
        self.assertEqual(source['online']['stages'][stage_index]['timestep_start'], 5)
        self.assertEqual(selected['core_sram_bytes'], 1048576)
        self.assertEqual(selected['double_buffer_minimum_bytes_per_lane'], 12288)
        self.assertEqual(isolated['memory_system']['impl'], 'test')

    def test_isolated_prefix_single_job_is_valid(self):
        source = self.compile()
        isolated = isolated_config(source, 0, {'memory_system': {'impl': 'test'}},
                                   warmup_requests=0, measure_requests=1)
        self.assertEqual(isolated['online']['measure_requests'], 1)
        self.assertEqual(isolated['online']['stages'][0]['kind'], 'prefix')
        with self.assertRaises(ValueError):
            compile_online_program(costs=costs_for(), measure_requests=1)

    def test_compiled_mac_accounting_matches_unbatched_source(self):
        from proposal_config import macs
        schedule = Schedule(b=5, kp=3, kd=5)
        data = self.compile(Hardware(128), schedule)
        compiled_prefix = sum(op['m'] * op['n'] * op['k'] * op['count']
                              for stage in data['online']['stages'] if stage['kind'] == 'prefix'
                              for op in stage['operations'])
        compiled_dit = sum(op['m'] * op['n'] * op['k'] * op['count'] * stage['timestep_count']
                           for stage in data['online']['stages'] if stage['kind'] == 'dit'
                           for op in stage['operations'])
        self.assertEqual(compiled_prefix, 2317740539904)
        self.assertEqual(compiled_dit, 6263144448 * 10)
        self.assertEqual((compiled_prefix + compiled_dit) * 5, 2380371984384 * 5)

    def test_m_tile_uses_sram_and_preserves_batched_weight_reuse(self):
        self.assertEqual(maximum_m_tile(), 4064)
        table = {shape: sum(shape) for shape in all_tile_shapes()}
        data = compile_online_program(Hardware(128), Schedule(b=10), table)
        stage = next(stage for stage in data['online']['stages'] if stage['kind'] == 'dit')
        operation = next(op for op in stage['operations'] if op['name'] == 'denoise.action_input')
        self.assertEqual(operation['m_tile'], 160)
        self.assertTrue(any(row['m'] == 160 for row in operation['compute_costs']))
        for stage in data['online']['stages']:
            for op in stage['operations']:
                footprint = 2 * (op['m_tile'] * 32 + 32 * 32) * 2 + op['m_tile'] * 32 * 4
                self.assertLessEqual(footprint, stage.get('lane_sram_bytes', stage['core_sram_bytes']))

    def test_dit_executes_real_layer_order_and_weight_offsets(self):
        data = self.compile()
        stage = next(stage for stage in data['online']['stages'] if stage['kind'] == 'dit')
        ops = stage['operations']
        self.assertEqual([op['name'] for op in ops[:8]], [
            'denoise.action_input', 'denoise.time_mlp.input', 'denoise.time_mlp.output',
            'denoise.adarms_modulation.layer0.pre_attention', 'denoise.blocks.q.layer0',
            'denoise.blocks.kv.layer0', 'denoise.blocks.qk.layer0', 'denoise.blocks.av.layer0'])
        by_name = {op['name']: op for op in ops}
        self.assertEqual(by_name['denoise.blocks.q.layer1']['b_base'] -
                         by_name['denoise.blocks.q.layer0']['b_base'], 1024 * 2048 * 2)
        self.assertEqual(by_name['denoise.blocks.kv.layer1']['c_base'] -
                         by_name['denoise.blocks.kv.layer0']['c_base'], 2 * KV_MATRIX_BYTES)
        self.assertEqual([op['name'] for op in ops[-2:]],
                         ['denoise.adarms_modulation.final', 'denoise.action_output'])

    def test_attention_and_dense_consumer_tensor_views(self):
        data = self.compile()
        operations = {op['name']: op for stage in data['online']['stages'] for op in stage['operations']}
        for prefix in ('prefix', 'denoise'):
            q = operations[f'{prefix}.blocks.q.layer0']
            qk = operations[f'{prefix}.blocks.qk.layer0']
            av = operations[f'{prefix}.blocks.av.layer0']
            output = operations[f'{prefix}.blocks.attention_output.layer0']
            self.assertEqual(q['c_base'], qk['a_base'])
            self.assertEqual(qk['a_row_stride'], 4096)
            self.assertEqual(qk['c_base'], av['a_base'])
            self.assertEqual(av['c_base'], output['a_base'])
            self.assertEqual(av['c_row_stride'], 4096)
        qkv = operations['vision.blocks.qkv.layer0']
        qk = operations['vision.blocks.qk.layer0']
        av = operations['vision.blocks.av.layer0']
        self.assertEqual(qkv['a_instance_share'], 3)
        self.assertEqual(qkv['c_base'], qk['a_base'])
        self.assertEqual(qk['a_instance_groups'], 16)
        self.assertEqual(qk['a_row_stride'], 2304)
        self.assertEqual(qk['c_base'], av['a_base'])
        self.assertEqual(av['c_row_stride'], 2304)
        self.assertEqual(operations['denoise.time_mlp.input']['c_base'],
                         operations['denoise.time_mlp.output']['a_base'])
        self.assertEqual(operations['denoise.blocks.mlp_gate_and_up.layer0']['c_base'],
                         operations['denoise.blocks.mlp_down.layer0']['a_base'])

    def test_actual_scale_compute_entry_point(self):
        try:
            import numpy
            import scalesim
        except ImportError:
            self.skipTest('SCALE-Sim and NumPy are required for measured compute tests')
        shapes = {(4, 4, 4), (16, 32, 32), (968, 32, 32)}
        measured = measure_compute_costs(shapes)
        self.assertEqual(measured.keys(), shapes)
        self.assertTrue(all(type(cycles) is int and cycles > 0 for cycles in measured.values()))
        self.assertGreater(measured[968, 32, 32], measured[16, 32, 32])

    def test_target_clock_and_controller_counts(self):
        try:
            import ramulator
        except ImportError:
            self.skipTest('Ramulator Python configuration package required')
        for target, count in ((136.6, 2), (819.0, 16)):
            data = target_memory_config(target)
            metadata = data['memory_metadata']
            self.assertEqual(metadata['controller_count'], count)
            self.assertLess(metadata['relative_peak_error'], .002)
            self.assertGreaterEqual(metadata['physical_capacity_bytes'], LOGICAL_CAPACITY)
            self.assertEqual(len(data['memory_system']['controllers']), count)
            self.assertEqual(data['memory_system']['channel_mapper']['impl'], 'ModuloInterleave')
            controller = data['memory_system']['controllers'][0]
            protocol = 'HBM3' if target == 819.0 else 'LPDDR5'
            self.assertEqual(controller['dram']['impl'], protocol)
            self.assertEqual(metadata['protocol'], protocol)
            if protocol == 'HBM3':
                self.assertEqual(controller['impl'], 'HBM34')
                self.assertEqual(controller['refresh_manager']['impl'], 'HBM34PerBankRefresh')
                self.assertEqual(metadata['physical_capacity_bytes'], LOGICAL_CAPACITY)
                self.assertEqual(metadata['timing_overrides'], {})
                timings = dict(zip(ramulator.dram.HBM3.timing_params, controller['dram']['timing']))
                self.assertEqual(timings['tCK_ps'], metadata['tCK_ps'])
                self.assertAlmostEqual(metadata['effective_peak_gbps'], 16 * 2 * 32 * 1000 / (timings['nBL'] * timings['tCK_ps']))


if __name__ == '__main__':
    unittest.main()
