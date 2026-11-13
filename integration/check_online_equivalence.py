"""Differential native regression, including byte-identical event ordering.

Set ONLINE_BASELINE_BINARY and ONLINE_BINARY to independently built executables.
The small fixtures cover both memory protocols, batching, shared addresses,
replicas, edge tiles, transposition, and the measurement-window reset.
"""
import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml

from check_online_native import fixture_config


class OnlineNativeEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = Path(os.environ.get('ONLINE_BASELINE_BINARY', '')).resolve()
        cls.optimized = Path(os.environ.get('ONLINE_BINARY', '')).resolve()
        if not cls.baseline.is_file() or not cls.optimized.is_file():
            raise unittest.SkipTest('Provide baseline and optimized native executables')

    def compare(self, config):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = []
            for name, binary in (('baseline', self.baseline), ('optimized', self.optimized)):
                directory = root / name
                directory.mkdir()
                current = copy.deepcopy(config)
                current['online']['event_log'] = str(directory / 'events.jsonl')
                source = directory / 'program.yaml'
                source.write_text(yaml.safe_dump(current, sort_keys=False))
                report = directory / 'report.yaml'
                process = subprocess.run([str(binary), str(source), str(report)],
                                         capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr)
                results.append((yaml.safe_load(report.read_text()),
                                (directory / 'events.jsonl').read_bytes()))
            self.assertEqual(results[0][0], results[1][0], 'All native statistics must match')
            self.assertEqual(results[0][1], results[1][1], 'Every event and cycle must match')

    def test_protocol_batch_replica_and_measurement_boundary(self):
        for bandwidth in (136.6, 819.0):
            for batch, replicas in ((1, 1), (2, 2), (5, 2)):
                with self.subTest(bandwidth=bandwidth, batch=batch, replicas=replicas):
                    self.compare(fixture_config(target_gbps=bandwidth, batch=batch,
                                                replicas=replicas, warmup=3, measure=9))

    def test_baseline_resource_serialization(self):
        for bandwidth in (136.6, 819.0):
            with self.subTest(bandwidth=bandwidth):
                self.compare(fixture_config(target_gbps=bandwidth, mode='baseline',
                                            batch=2, replicas=2, warmup=3, measure=9))

    def test_edge_tiles_sample_boundaries_and_duplicate_bursts(self):
        for shared, transposed in ((True, False), (True, True), (False, False), (False, True)):
            with self.subTest(shared=shared, transposed=transposed):
                config = fixture_config(batch=5, replicas=2, warmup=3, measure=9)
                for stage in config['online']['stages']:
                    operation = stage['operations'][0]
                    operation.update(m=5, n=7, k=9, count=2, m_tile=8,
                                     b_shared=shared, b_transposed=transposed,
                                     b_base=operation['a_base'])
                    operation['compute_costs'] = [
                        {'m': m, 'n': n, 'k': k, 'cycles': m + n + k + 5}
                        for m in range(1, 9) for n in range(1, 5) for k in range(1, 5)]
                self.compare(config)


if __name__ == '__main__':
    unittest.main()
