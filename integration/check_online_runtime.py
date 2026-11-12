"""End-to-end search and cache tests backed by the native shared-memory loop."""
import copy
import itertools
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from check_online_native import fixture_config
from online_runtime import OnlineRuntime
from online_search import search


class FixtureRuntime(OnlineRuntime):
    def compile(self, parameters):
        config = fixture_config(batch=parameters['batch'], replicas=1,
                                max_inflight=parameters['dma'], warmup=self.warmup, measure=self.measure)
        for stage in config['online']['stages']:
            stage['cores'] = parameters['cores']
            stage.setdefault('timestep_start', 0)
            stage.setdefault('timestep_count', 1)
        return {'online': config['online'], 'address_layout': {'fixture': True}, 'workload': {}}


class OnlineRuntimeAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = Path(os.environ.get('ONLINE_BINARY', 'integration/build/ramulator2-online')).resolve()
        if not cls.binary.is_file():
            raise unittest.SkipTest('Native online engine required')

    def test_native_search_matches_exhaustive_and_reuses_completed_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = FixtureRuntime(self.binary, root, {}, warmup=20, measure=50)
            candidates = []
            for index, (batch, dma, cores) in enumerate(itertools.product((1, 2), (1, 4), (1, 2))):
                parameters = {'batch': batch, 'dma': dma, 'cores': cores}
                candidates.append({'id': f'actual_{index}', 'parameters': parameters, **engine.bounds(parameters)})
            result = search(candidates, engine.evaluate, workers=2, batch_size=2,
                            resume_path=root / 'checkpoint.json')
            all_results = [engine.evaluate(candidate) for candidate in candidates]
            metrics = [item['metrics'] for item in all_results]
            best_s = min(item['output_interval_seconds'] for item in metrics)
            best_l = min(item['latency_mean_seconds'] for item in metrics
                         if item['output_interval_seconds'] <= best_s * 1.1)
            self.assertEqual(result['throughput_optimum']['S'], best_s)
            self.assertEqual(result['latency_optimum_within_10_percent_S']['L'], best_l)
            self.assertTrue(result['pareto_frontier_complete'])
            self.assertTrue(result['equivalence_sets_complete'])
            self.assertEqual(result['bound_certificate'], 'exact')
            before = {str(path): path.stat().st_mtime_ns for path in root.rglob('native.yaml')}
            repeated = search(candidates, engine.evaluate, workers=2, batch_size=2,
                              resume_path=root / 'checkpoint.json')
            after = {str(path): path.stat().st_mtime_ns for path in root.rglob('native.yaml')}
            self.assertEqual(before, after)
            self.assertEqual(repeated['throughput_optimum']['id'], result['throughput_optimum']['id'])
            for candidate in result['evaluated']:
                for stage in candidate['result']['stage_timings']:
                    self.assertGreater(stage['uncontended_seconds'], 0)
                    self.assertGreater(stage['actual_over_uncontended'], 0)
            self.assertGreater(len(list((root / 'uncontended').rglob('reference.json'))), 0)

    def test_native_layer_pilot_measures_real_reference_jobs(self):
        from online_pilot import measure_pilot
        class PilotRuntime(FixtureRuntime):
            def compile(self, parameters):
                compiled = super().compile(parameters)
                for stage in compiled['online']['stages']:
                    first = stage['operations'][0]
                    first['name'] = f"{stage['kind']}.blocks.q.layer0"
                    second = copy.deepcopy(first)
                    second['name'] = f"{stage['kind']}.blocks.q.layer1"
                    stage['operations'].append(second)
                return compiled
        with tempfile.TemporaryDirectory() as temporary:
            engine = PilotRuntime(self.binary, temporary, {}, warmup=2, measure=6)
            result = measure_pilot(engine, {'batch': 1, 'dma': 4, 'cores': 2})
            self.assertEqual(result['execution_model'], engine.identity_digest)
            self.assertGreaterEqual(result['measured_error_margin'], 0)
            self.assertEqual(len(result['stages']), 2)
            for stage in result['stages']:
                self.assertEqual(len(stage['groups']), 2)
                self.assertGreater(stage['whole_stage_seconds'], 0)
                for group in stage['groups']:
                    self.assertTrue(Path(group['native_report']).is_file())

    def test_cli_pilot_checks_hardware_and_execution_identity(self):
        from online import parameters, resolve_error_margin
        args = SimpleNamespace(pilot_report=None, error_margin=None, cores=32,
                               kp=2, kd=2, batch=2, replicas=1, dit_core_numerator=4, mode='pipeline')
        engine = SimpleNamespace(identity_digest='current-model')
        self.assertEqual(resolve_error_margin(args, engine), (0, 'zero_default'))
        with tempfile.TemporaryDirectory() as temporary:
            args.pilot_report = Path(temporary) / 'pilot.json'
            report = {'execution_model': engine.identity_digest, 'parameters': parameters(args),
                      'measured_error_margin': .03}
            args.pilot_report.write_text(json.dumps(report))
            self.assertEqual(resolve_error_margin(args, engine)[0], .03)
            args.cores = 64
            with self.assertRaisesRegex(ValueError, 'hardware'):
                resolve_error_margin(args, engine)
            args.cores = 32
            engine.identity_digest = 'changed-model'
            with self.assertRaisesRegex(ValueError, 'simulator'):
                resolve_error_margin(args, engine)

    def test_uncontended_cache_tracks_execution_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = FixtureRuntime(self.binary, temporary, {}, warmup=2, measure=6)
            parameters = {'batch': 1, 'dma': 1, 'cores': 1}
            original = first.uncontended(first.compile(parameters), 0)
            changed = FixtureRuntime(self.binary, temporary, {(1, 1, 1): 2}, warmup=2, measure=6)
            replacement = changed.uncontended(changed.compile(parameters), 0)
            self.assertNotEqual(original['signature'], replacement['signature'])
            self.assertEqual(len(list((Path(temporary) / 'uncontended').rglob('reference.json'))), 2)

    def test_candidate_cache_rejects_changed_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = FixtureRuntime(self.binary, temporary, {}, warmup=2, measure=6)
            candidate = {'id': 'same', 'parameters': {'batch': 1, 'dma': 1, 'cores': 1}}
            engine.evaluate(candidate)
            changed = copy.deepcopy(candidate)
            changed['parameters']['dma'] = 4
            with self.assertRaisesRegex(ValueError, 'different model'):
                engine.evaluate(changed)


if __name__ == '__main__':
    unittest.main()
