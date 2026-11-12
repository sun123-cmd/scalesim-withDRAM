"""Pilot additivity accounting tests with explicit deterministic reference data."""
import copy
import math
import unittest

from online_pilot import measure_pilot, operation_groups


def op(name, weight=1):
    return {'name': name, 'm': 16, 'n': 32, 'k': 32, 'count': 1,
            'a_base': 1000, 'b_base': 2000, 'c_base': 3000, 'test_seconds': weight}


class ReferenceEngine:
    identity_digest = 'test-execution-identity'

    def __init__(self):
        self.program = {'online': {'batch': 5, 'replicas': 2, 'max_inflight': 64,
                                  'mode': 'pipeline', 'stages': [
            {'name': 'prefix_0', 'kind': 'prefix', 'cores': 12, 'timestep_count': 1,
             'operations': [op('vision.patch_embedding.layer0', 1),
                            op('vision.blocks.qkv.layer0', 2), op('vision.blocks.qk.layer0', 3),
                            op('prefix.blocks.q.layer0', 4)]},
            {'name': 'dit_0', 'kind': 'dit', 'cores': 4, 'timestep_count': 5,
             'operations': [op('denoise.action_input', 2), op('denoise.time_mlp.input', 2),
                            op('denoise.time_mlp.output', 3),
                            op('denoise.adarms_modulation.layer0.pre_attention', 3),
                            op('denoise.blocks.q.layer0', 4), op('denoise.blocks.mlp_down.layer0', 5),
                            op('denoise.adarms_modulation.final', 5)]}]}}
        self.calls = []
        self.invalid = None

    def compile(self, parameters):
        return copy.deepcopy(self.program)

    def uncontended(self, compiled, index):
        self.calls.append((copy.deepcopy(compiled), index))
        if self.invalid is not None:
            return {'seconds': self.invalid}
        stage = compiled['online']['stages'][index]
        if len(stage['operations']) == len(self.program['online']['stages'][index]['operations']):
            seconds = 8 if index == 0 else 20
        else:
            seconds = sum(operation['test_seconds'] for operation in stage['operations'])
        return {'seconds': seconds, 'signature': f'reference-{index}-{len(self.calls)}',
                'native_report': f'test-reference-{len(self.calls)}.yaml'}


class OnlinePilot(unittest.TestCase):
    def test_vision_and_prefix_layer_indices_are_separate(self):
        operations = [op('vision.blocks.qkv.layer0'), op('vision.blocks.av.layer0'),
                      op('prefix.blocks.q.layer0'), op('prefix.blocks.kv.layer0')]
        groups = operation_groups(operations)
        self.assertEqual([group['key'] for group in groups],
                         [('transformer_layer', 'vision', 0), ('transformer_layer', 'prefix', 0)])
        self.assertEqual([len(group['operations']) for group in groups], [2, 2])

    def test_dit_norms_and_dense_attention_share_their_layer(self):
        groups = operation_groups([op('denoise.adarms_modulation.layer7.pre_attention'),
                                   op('denoise.blocks.q.layer7'), op('denoise.blocks.kv.layer7'),
                                   op('denoise.adarms_modulation.layer7.pre_mlp'),
                                   op('denoise.blocks.mlp_down.layer7'), op('denoise.blocks.q.layer8')])
        self.assertEqual([group['key'] for group in groups],
                         [('transformer_layer', 'denoise', 7), ('transformer_layer', 'denoise', 8)])

    def test_pre_post_grouping_and_measured_maximum_error(self):
        engine = ReferenceEngine()
        parameters = {'hardware': {'cores': 32}, 'schedule': {'b': 5}}
        result = measure_pilot(engine, parameters)
        self.assertEqual(result['execution_model'], engine.identity_digest)
        self.assertEqual(result['parameters'], parameters)
        self.assertEqual(result['source'], 'native cold stage/layer job comparison')
        self.assertEqual(result['stages'][0]['layer_sum_seconds'], 10)
        self.assertEqual(result['stages'][0]['relative_additivity_error'], .25)
        self.assertEqual(result['stages'][1]['layer_sum_seconds'], 24)
        self.assertEqual(result['stages'][1]['relative_additivity_error'], .2)
        self.assertEqual(result['measured_error_margin'], .25)
        self.assertEqual(result['stages'][1]['groups'][1]['scope'], 'denoise.time_mlp')

    def test_calibrated_subprogram_retains_addresses_and_timestep_count(self):
        engine = ReferenceEngine()
        original = copy.deepcopy(engine.program)
        measure_pilot(engine, {})
        self.assertEqual(engine.program, original)
        for compiled, index in engine.calls:
            self.assertEqual(compiled['online']['batch'], 5)
            self.assertEqual(compiled['online']['replicas'], 2)
            self.assertEqual(compiled['online']['max_inflight'], 64)
            self.assertEqual(compiled['online']['stages'][index]['timestep_count'], 1 if index == 0 else 5)
            for operation in compiled['online']['stages'][index]['operations']:
                self.assertEqual(operation['a_base'], 1000)
                self.assertEqual(operation['b_base'], 2000)
                self.assertEqual(operation['c_base'], 3000)

    def test_noncontiguous_layer_visits_keep_original_order(self):
        groups = operation_groups([op('denoise.blocks.q.layer0'), op('denoise.blocks.q.layer1'),
                                   op('denoise.blocks.av.layer0')])
        self.assertEqual([group['key'][2] for group in groups], [0, 1, 0])

    def test_invalid_references_fail(self):
        for invalid in (False, '1', -1, 0, math.inf, math.nan):
            engine = ReferenceEngine()
            engine.invalid = invalid
            with self.assertRaises(ValueError):
                measure_pilot(engine, {})

    def test_invalid_group_names_and_empty_stage_fail(self):
        for operations in ([], [{'name': ''}], [{'name': None}], [{}]):
            with self.assertRaises(ValueError):
                operation_groups(operations)
        engine = ReferenceEngine()
        engine.program['online']['stages'] = []
        with self.assertRaises(ValueError):
            measure_pilot(engine, {})


if __name__ == '__main__':
    unittest.main()
