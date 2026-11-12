"""Measure native stage/layer additivity error for pilot-bound search."""
import copy
import math
import re

_LAYER = re.compile(r'\.layer(\d+)(?:\.|$)')


def _group_key(operation):
    name = operation.get('name')
    if not isinstance(name, str) or not name:
        raise ValueError('Every pilot operation needs a nonempty name')
    match = _LAYER.search(name)
    if match:
        if name.startswith('vision.blocks.'):
            return ('transformer_layer', 'vision', int(match.group(1)))
        if name.startswith('prefix.blocks.'):
            return ('transformer_layer', 'prefix', int(match.group(1)))
        if name.startswith('denoise.'):
            return ('transformer_layer', 'denoise', int(match.group(1)))
    if name.startswith('denoise.time_mlp.'):
        return ('operator_family', 'denoise.time_mlp', None)
    return ('operator_family', name, None)


def operation_groups(operations):
    """Partition the original sequence into contiguous, scope-aware layers."""
    if not operations:
        raise ValueError('A pilot stage must contain operations')
    groups = []
    for operation in operations:
        key = _group_key(operation)
        if groups and groups[-1]['key'] == key:
            groups[-1]['operations'].append(operation)
        else:
            groups.append({'key': key, 'operations': [operation]})
    return groups


def _seconds(reference):
    value = reference.get('seconds')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Native pilot references need numeric service seconds')
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Native pilot service seconds must be finite and positive')
    return float(value)


def measure_pilot(engine, parameters):
    """Compare native whole-stage jobs with sums of native isolated layer jobs.

    Each grouped program retains its original operation descriptors, addresses,
    stage resources, batch semantics, and assigned timestep count. References
    are obtained through the runtime's signed persistent native cache.
    """
    compiled = engine.compile(parameters)
    stages = compiled['online']['stages']
    if not stages:
        raise ValueError('A pilot workload needs at least one stage')
    rows = []
    maximum_error = 0.0
    for stage_index, stage in enumerate(stages):
        whole_reference = engine.uncontended(compiled, stage_index)
        whole_seconds = _seconds(whole_reference)
        groups = []
        for group_index, group in enumerate(operation_groups(stage['operations'])):
            grouped = copy.deepcopy(compiled)
            grouped['online']['stages'][stage_index]['operations'] = copy.deepcopy(group['operations'])
            reference = engine.uncontended(grouped, stage_index)
            seconds = _seconds(reference)
            kind, scope, layer = group['key']
            groups.append({'group_index': group_index, 'kind': kind, 'scope': scope,
                           'layer_index': layer,
                           'operation_names': [operation['name'] for operation in group['operations']],
                           'operation_count': len(group['operations']), 'seconds': seconds,
                           'reference_signature': reference.get('signature'),
                           'native_report': reference.get('native_report')})
        layer_sum = sum(group['seconds'] for group in groups)
        if not math.isfinite(layer_sum):
            raise ValueError('Native layer service sum must be finite')
        error = abs(layer_sum - whole_seconds) / whole_seconds
        maximum_error = max(maximum_error, error)
        rows.append({'stage_index': stage_index, 'name': stage.get('name', f'stage_{stage_index}'),
                     'kind': stage['kind'], 'cores': stage['cores'],
                     'timestep_count': stage.get('timestep_count', 1),
                     'whole_stage_seconds': whole_seconds, 'layer_sum_seconds': layer_sum,
                     'relative_additivity_error': error, 'groups': groups,
                     'whole_reference_signature': whole_reference.get('signature'),
                     'whole_native_report': whole_reference.get('native_report')})
    return {'source': 'native cold stage/layer job comparison',
            'method': 'absolute(sum of native contiguous layer services - native whole-stage service) / whole-stage service',
            'execution_model': engine.identity_digest,
            'parameters': copy.deepcopy(parameters),
            'grouping': 'contiguous transformer layers scoped by vision, prefix, and denoise; stable pre/post operator families',
            'stages': rows, 'measured_error_margin': maximum_error,
            'certificate_scope': 'empirical stage/layer additivity for the measured workload, hardware, and runtime identity'}
