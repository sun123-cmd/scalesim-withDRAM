"""Acceptance tests using the real native shared-memory event loop.

Set ONLINE_BINARY to ramulator2-online. The suite reports an explicit skipped
class when a native executable is unavailable. Fixture compute costs are fixed
known test costs so callback ordering and traffic can be verified independently.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


def fixture_config(batch=2, replicas=2, target_gbps=136.6, max_inflight=4,
                   warmup=2, measure=6, event_log=None, mode='pipeline'):
    """A complete two-stage, ten-timestep program with real DRAM transactions."""
    from online_workload import target_memory_config
    config = target_memory_config(target_gbps)
    operation = {'m': 4, 'n': 8, 'k': 8, 'count': 1,
                 'a_base': 16 << 20, 'b_base': 0, 'c_base': 20 << 20,
                 'request_stride': 65536, 'tile_size': 4, 'm_tile': 8,
                 'word_bytes': 2, 'rhs_share': 1, 'b_shared': True,
                 'compute_costs': [{'m': 4, 'n': 4, 'k': 4, 'cycles': 12},
                                   {'m': 8, 'n': 4, 'k': 4, 'cycles': 18}]}
    expert = copy.deepcopy(operation)
    expert.update(a_base=20 << 20, b_base=4096, c_base=24 << 20)
    config['online'] = {'mode': mode, 'accelerator_ghz': 1.0,
                        'warmup_requests': warmup, 'measure_requests': measure,
                        'batch': batch, 'replicas': replicas, 'handoff_gbps': 256,
                        'max_inflight': max_inflight, 'issue_width': 16,
                        'max_cycles': 10000000, 'request_stride': 65536,
                        'request_slots': 64, 'queue_capacity': 2 * batch,
                        'activation_replica_stride': 2 << 20,
                        'dram_capacity_bytes': 16 << 30, 'handoff_bytes_per_request': 128,
                        'stages': [{'name': 'prefix', 'kind': 'prefix', 'cores': 2,
                                    'core_sram_bytes': 1 << 20, 'operations': [operation]},
                                   {'name': 'dit', 'kind': 'dit', 'cores': 2,
                                    'core_sram_bytes': 1 << 20, 'timestep_start': 0,
                                    'timestep_count': 10, 'operations': [expert]}]}
    if event_log is not None:
        config['online']['event_log'] = str(event_log)
    return config


class OnlineNativeAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ.get('ONLINE_BINARY', 'integration/build/ramulator2-online')).resolve()
        if not cls.binary.is_file():
            raise unittest.SkipTest('Build ramulator2-online or set ONLINE_BINARY for native acceptance tests')

    def run_config(self, config, success=True):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = copy.deepcopy(config)
            config['online']['event_log'] = str(directory / 'events.jsonl')
            source = directory / 'config.yaml'
            source.write_text(yaml.safe_dump(config, sort_keys=False))
            report = directory / 'report.yaml'
            process = subprocess.run([str(self.binary), str(source), str(report)],
                                     capture_output=True, text=True, timeout=60)
            if not success:
                self.assertNotEqual(process.returncode, 0, process.stdout)
                return process.stderr
            self.assertEqual(process.returncode, 0, process.stderr)
            result = yaml.safe_load(report.read_text())
            events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
            return result, events

    def test_default_measurement_and_macs(self):
        result, _ = self.run_config(fixture_config(warmup=20, measure=50))
        online = result['online']
        self.assertEqual(online['warmup_requests'], 20)
        self.assertEqual(online['measured_requests'], 50)
        self.assertEqual(len(online['requests']), 50)
        self.assertGreaterEqual(online['completed_requests'], 70)
        self.assertGreater(online['latency_mean_seconds'], 0)
        self.assertGreaterEqual(online['output_interval_seconds'], 0)
        for request in online['requests']:
            self.assertEqual(request['timesteps_completed'], 10)
            self.assertEqual(request['macs'], 4 * 8 * 8 * 11)
            self.assertGreater(request['completion_cycle'], request['admission_cycle'])
        latencies = sorted((row['completion_cycle'] - row['admission_cycle']) * online['tCK_ns'] * 1e-9
                           for row in online['requests'])
        import math
        position = .95 * (len(latencies) - 1)
        lower = math.floor(position)
        expected_p95 = latencies[lower] + (latencies[math.ceil(position)] - latencies[lower]) * (position - lower)
        self.assertAlmostEqual(online['latency_mean_seconds'], sum(latencies) / len(latencies), places=14)
        self.assertAlmostEqual(online['latency_p95_seconds'], expected_p95, places=14)
        for stage in online['stages']:
            self.assertLessEqual(stage['peak_inflight'], 4)
            self.assertEqual(stage['initial_outstanding_transactions'] + stage['reads'] + stage['writes'],
                             stage['completed_transactions'] + stage['outstanding_transactions'])
            self.assertGreater(stage['completed_transactions'], 0)
            self.assertGreater(stage['actual_bandwidth_gbps'], 0)

    def test_read_completion_gates_compute_and_writes_follow_accumulation(self):
        _, events = self.run_config(fixture_config())
        issued, completed, compute_starts = {}, {}, {}
        output_completions = {}
        checked_outputs = 0
        for event in events:
            key = (event['stage'], event['tile'])
            if event['event'] == 'read_issue':
                issued[key] = issued.get(key, 0) + 1
            elif event['event'] == 'read_complete':
                completed[key] = completed.get(key, 0) + 1
            elif event['event'] == 'compute_start':
                self.assertGreater(issued.get(key, 0), 0)
                self.assertEqual(issued[key], completed.get(key, 0))
                compute_starts[key] = event['cycle']
            elif event['event'] == 'compute_end':
                self.assertIn(key, compute_starts)
                self.assertGreater(event['cycle'], compute_starts[key])
                self.assertIn('output_tile', event)
                self.assertIn('k', event)
                output_key = (event['stage'], event['output_tile'])
                output_completions.setdefault(output_key, []).append((event['k'], event['cycle']))
            elif event['event'] == 'write_issue':
                self.assertIn(key, output_completions)
                completions = output_completions[key]
                self.assertEqual({offset for offset, _ in completions}, {0, 4})
                self.assertLessEqual(max(cycle for _, cycle in completions), event['cycle'])
                checked_outputs += 1
        self.assertGreater(len(compute_starts), 0)
        self.assertGreater(checked_outputs, 0)

    def test_timestep_progression_and_replica_weight_aliasing(self):
        _, events = self.run_config(fixture_config())
        per_request = {}
        replicas = set()
        for event in events:
            if event['event'] == 'compute_start' and event['timestep'] >= 0:
                for identifier in event['request_ids']:
                    per_request.setdefault(identifier, []).append(event['timestep'])
            if event['event'] == 'read_issue' and event['address'] < 8192:
                replicas.add(event['stage'] // 2)
        for steps in per_request.values():
            self.assertEqual(steps, sorted(steps))
            distinct = sorted(set(steps))
            self.assertEqual(distinct, list(range(distinct[-1] + 1)))
        self.assertEqual(replicas, {0, 1})

    def test_dma_quota_changes_execution_and_is_respected(self):
        serial, _ = self.run_config(fixture_config(max_inflight=1))
        parallel, _ = self.run_config(fixture_config(max_inflight=8))
        self.assertTrue(all(stage['peak_inflight'] <= 1 for stage in serial['online']['stages']))
        self.assertTrue(all(stage['peak_inflight'] <= 8 for stage in parallel['online']['stages']))
        self.assertNotEqual(serial['online']['dram_cycles'], parallel['online']['dram_cycles'])

    def test_dense_batch_shares_weight_reads(self):
        counts = []
        for batch in (1, 2):
            _, events = self.run_config(fixture_config(batch=batch, replicas=1))
            reads = 0
            for event in events:
                if event['stage'] != 1:
                    continue
                if event['event'] == 'task_complete':
                    break
                if event['event'] == 'read_issue' and 4096 <= event['address'] < 8192:
                    reads += 1
            self.assertGreater(reads, 0)
            counts.append(reads)
        self.assertEqual(counts[0], counts[1])

    def test_target_controller_counts(self):
        for bandwidth, expected in ((136.6, 2), (819.0, 16)):
            config = fixture_config(target_gbps=bandwidth, warmup=0, measure=2, replicas=1)
            metadata = config['memory_metadata']
            self.assertEqual(metadata['controller_count'], expected)
            self.assertLess(metadata['relative_peak_error'], .002)
            result, _ = self.run_config(config)
            controllers = result['memory_system']['controller']
            self.assertEqual(len(controllers), expected)
            if bandwidth == 819.0:
                self.assertEqual(config['memory_system']['controllers'][0]['dram']['impl'], 'HBM3')
                self.assertAlmostEqual(result['online']['tCK_ns'], metadata['tCK_ps'] / 1000)

    def test_baseline_serializes_stage_resources(self):
        _, events = self.run_config(fixture_config(mode='baseline', replicas=1))
        active_prefix = set()
        for event in events:
            key = (event['stage'], event['tile'])
            if event['stage'] == 0:
                if event['event'] == 'compute_start':
                    active_prefix.add(key)
                elif event['event'] == 'compute_end':
                    active_prefix.remove(key)
            if event['stage'] == 1 and event['event'] == 'compute_start':
                self.assertFalse(active_prefix)

    def test_small_input_queues_and_slot_backpressure_complete(self):
        config = fixture_config(batch=2, replicas=2, max_inflight=1)
        config['online']['queue_capacity'] = 2
        config['online']['request_slots'] = 4
        result, _ = self.run_config(config)
        self.assertEqual(result['online']['measured_requests'], 6)
        self.assertEqual(len(result['online']['requests']), 6)
        self.assertTrue(all(stage['peak_inflight'] <= 1 for stage in result['online']['stages']))
        self.assertTrue(all(row['timesteps_completed'] == 10 for row in result['online']['requests']))

    def test_baseline_replicas_preserve_per_replica_serialization(self):
        result, events = self.run_config(fixture_config(mode='baseline', replicas=2))
        prefix_active = {0: set(), 1: set()}
        for event in events:
            replica, position = divmod(event['stage'], 2)
            key = event['tile']
            if position == 0:
                if event['event'] == 'compute_start':
                    prefix_active[replica].add(key)
                elif event['event'] == 'compute_end':
                    prefix_active[replica].remove(key)
            if position == 1 and event['event'] == 'compute_start':
                self.assertFalse(prefix_active[replica])
        self.assertEqual({row['replica'] for row in result['online']['stages']}, {0, 1})
        self.assertTrue(all(row['macs'] == 2816 for row in result['online']['requests']))

    def test_negative_dimensions_and_address_capacity_fail(self):
        config = fixture_config()
        config['online']['stages'][0]['operations'][0]['m'] = -1
        self.assertIn('integer', self.run_config(config, success=False))
        config = fixture_config()
        config['online']['stages'][0]['operations'][0]['b_base'] = 16 << 30
        self.assertIn('capacity', self.run_config(config, success=False))
        config = fixture_config()
        config['online']['dram_capacity_bytes'] = 32768
        self.assertIn('capacity', self.run_config(config, success=False))

    def test_isolated_timestep_program(self):
        config = fixture_config(batch=1, replicas=1, warmup=0, measure=2)
        config['online'].update(mode='isolated', calibration_batch=1)
        config['online']['stages'] = [config['online']['stages'][1]]
        config['online']['stages'][0]['timestep_count'] = 5
        result, _ = self.run_config(config)
        self.assertEqual({row['timesteps_completed'] for row in result['online']['requests']}, {5})
        self.assertEqual({row['macs'] for row in result['online']['requests']}, {4 * 8 * 8 * 5})

    def test_missing_compute_cost_and_cycle_limit_fail(self):
        config = fixture_config()
        config['online']['stages'][1]['operations'][0]['compute_costs'] = [
            {'m': 4, 'n': 4, 'k': 4, 'cycles': 12}]
        self.assertIn('cost', self.run_config(config, success=False).lower())
        config = fixture_config()
        config['online']['max_cycles'] = 1
        self.assertIn('max_cycles', self.run_config(config, success=False))
        config = fixture_config(max_inflight=0)
        self.assertIn('limits', self.run_config(config, success=False))


if __name__ == '__main__':
    unittest.main()
