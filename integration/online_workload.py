"""Compile pi0.5 programs, physical addresses, and measured tile compute costs."""
import argparse
import copy
import fcntl
import json
import itertools
import math
import os
import tempfile
from pathlib import Path
import sys
from functools import lru_cache

from proposal_config import Hardware, Schedule, build_stage_workloads
from pi05_runner import dimension_tiles

WORD_BYTES = 2
PREFIX_TOKENS = 968
HORIZON = 16
KV_HEAD_DIM = 256
LAYERS = 18
KV_MATRIX_BYTES = (PREFIX_TOKENS + HORIZON) * KV_HEAD_DIM * WORD_BYTES
KV_STRIDE = LAYERS * 2 * KV_MATRIX_BYTES
LOGICAL_CAPACITY = 16 << 30


def align(value, boundary=128):
    return (value + boundary - 1) // boundary * boundary


def attention(operation):
    return operation['weight_copies'] == 0


def unbatched(stage, batch):
    result = []
    for source in stage['operations']:
        operation = dict(source)
        if stage['kind'] == 'dit':
            key = 'count' if attention(operation) else 'm'
            if operation[key] % batch:
                raise ValueError('Batched operation dimensions must divide exactly')
            operation[key] //= batch
        operation['weight_key'] = operation['name']
        operation['weight_offset_bytes'] = 0
        result.append(operation)
    if stage['kind'] != 'dit':
        return result
    families = {operation['name']: operation for operation in result}
    ordered = []

    def append(key, index=0, count=1, layer=None, suffix=None):
        source = families[key]
        operation = dict(source)
        operation['count'] = count
        operation['weight_copies'] = 0 if attention(source) else count
        operation['weight_offset_bytes'] = index * source['k'] * source['n'] * WORD_BYTES
        operation['name'] = key + (suffix or (f'.layer{layer}' if layer is not None else ''))
        if layer is not None:
            operation['layer_index'] = layer
        ordered.append(operation)

    append('denoise.action_input')
    append('denoise.time_mlp', 0, suffix='.input')
    append('denoise.time_mlp', 1, suffix='.output')
    for layer in range(LAYERS):
        append('denoise.adarms_modulation', 2 * layer, layer=layer, suffix=f'.layer{layer}.pre_attention')
        append('denoise.blocks.q', layer, layer=layer)
        append('denoise.blocks.kv', 2 * layer, count=2, layer=layer)
        append('denoise.blocks.qk', count=8, layer=layer)
        append('denoise.blocks.av', count=8, layer=layer)
        append('denoise.blocks.attention_output', layer, layer=layer)
        append('denoise.adarms_modulation', 2 * layer + 1, layer=layer, suffix=f'.layer{layer}.pre_mlp')
        append('denoise.blocks.mlp_gate_and_up', 2 * layer, count=2, layer=layer)
        append('denoise.blocks.mlp_down', layer, layer=layer)
    append('denoise.adarms_modulation', 36, suffix='.final')
    append('denoise.action_output')
    return ordered


def maximum_m_tile(core_sram_bytes=1 << 20, tile_size=32, word_bytes=2):
    """Largest aligned M tile fitting two BF16 A/B buffers and FP32 C."""
    fixed_bytes = 2 * tile_size * tile_size * word_bytes
    bytes_per_row = 2 * tile_size * word_bytes + tile_size * 4
    rows = (core_sram_bytes - fixed_bytes) // bytes_per_row
    aligned = rows // tile_size * tile_size
    if aligned < tile_size:
        raise ValueError('Per-core SRAM cannot hold a complete double-buffered tile')
    return aligned


def tile_shapes(operation, batch=1, tile_size=32, m_tile=None):
    m = operation['m'] if attention(operation) else operation['m'] * batch
    m_tile = min(m, maximum_m_tile() if m_tile is None else m_tile)
    return {(mm, nn, kk) for mm in dimension_tiles(m, m_tile)
            for nn in dimension_tiles(operation['n'], tile_size)
            for kk in dimension_tiles(operation['k'], tile_size)}


def measure_compute_costs(shapes, array_size=32):
    """Measure pure compute demand cycles through SCALE-Sim's WS entry point."""
    import numpy as np
    from scalesim.compute.systolic_compute_ws import systolic_compute_ws
    from scalesim.scale_config import scale_config
    table = {}
    for m, n, k in sorted(shapes):
        config = scale_config()
        config.array_rows = config.array_cols = array_size
        config.word_size_bytes = WORD_BYTES
        config.df = "ws"
        config.valid_conf_flag = True
        a = np.arange(m * k).reshape(m, k)
        b = np.arange(k * n).reshape(k, n) + m * k
        c = np.arange(m * n).reshape(m, n) + m * k + k * n
        simulator = systolic_compute_ws()
        simulator.set_params(config_obj=config, ifmap_op_mat=a, filter_op_mat=b,
                             ofmap_op_mat=c, ifmap_op_mat_original=a)
        matrices = simulator.get_demand_matrices()
        cycles = matrices[2].shape[0]
        if cycles <= 0:
            raise ValueError('SCALE-Sim returned an empty compute demand')
        table[(m, n, k)] = int(cycles)
    return table


def read_compute_costs(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or data.get('array_size') != 32:
        raise ValueError('Online programs require the proposal 32x32 array cost table')
    result = {}
    if not isinstance(data.get('tiles'), list):
        raise ValueError('Cost table requires a list of measured tiles')
    for row in data['tiles']:
        if not isinstance(row, dict) or not isinstance(row.get('shape'), list):
            raise ValueError('Invalid tile measurement in cost table')
        shape = tuple(row['shape'])
        cycles = row.get('compute_cycles')
        if len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape):
            raise ValueError('Invalid tile shape in cost table')
        if type(cycles) is not int or cycles <= 0 or shape in result:
            raise ValueError('Invalid or duplicate tile cycle measurement')
        result[shape] = cycles
    return result


def write_compute_costs(path, table):
    data = {'array_size': 32, 'source': 'SCALE-Sim weight-stationary compute demand rows',
            'tiles': [{'shape': list(shape), 'compute_cycles': cycles}
                      for shape, cycles in sorted(table.items())]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(json.dumps(data, indent=2, allow_nan=False) + '\n')
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _canonical_weights():
    all_stages = build_stage_workloads(Hardware(), Schedule(kp=1, kd=1, b=1))['stages']
    catalogue = {}
    for stage in all_stages:
        for operation in stage['operations']:
            if operation['weight_copies']:
                catalogue[operation['name']] = operation
    cursor = 0
    weights = {}
    for name, operation in sorted(catalogue.items()):
        cursor = align(cursor)
        size = operation['k'] * operation['n'] * operation['weight_copies'] * WORD_BYTES
        weights[name] = {'base': cursor, 'bytes': size,
                         'copies': operation['weight_copies']}
        cursor += size
    return weights, align(cursor)


def _is_gemma_attention(operation):
    return attention(operation) and operation['name'].startswith(('prefix.', 'denoise.'))


def _kv_layer(operation):
    if 'layer_index' in operation:
        return operation['layer_index']
    if operation['name'].startswith('prefix.'):
        return int(operation['name'].rsplit('.layer', 1)[1])
    return 0


def activation_views(program):
    """Plan one layer-live scratch arena and tensor views for a stage job."""
    max_m = max(operation['m'] for operation in program)
    x_bytes = align(max_m * 2048 * WORD_BYTES)
    score_bytes = align(max((operation['m'] * operation['n'] * operation['count'] * WORD_BYTES
                             for operation in program if '.qk' in operation['name']), default=128))
    gate_bytes = align(max((operation['m'] * operation['n'] * WORD_BYTES for operation in program
                            if '.mlp_up' in operation['name'] or '.mlp_gate_and_up' in operation['name']), default=128))
    x0, x1, y, q = 0, x_bytes, 2 * x_bytes, 3 * x_bytes
    score, av, gate = 6 * x_bytes, 6 * x_bytes + score_bytes, 7 * x_bytes + score_bytes
    time_x = gate + 2 * gate_bytes
    time_y, modulation = time_x + 2048, time_x + 4096
    raw = align(modulation + 6144)
    raw_bytes = max((operation['m'] * operation['k'] * WORD_BYTES for operation in program
                     if 'patch_embedding' in operation['name'] or 'action_input' in operation['name']), default=128)
    current = x0
    result = []
    for operation in program:
        name = operation['name']
        view = {'a_offset': current, 'c_offset': y}
        if 'patch_embedding' in name or 'action_input' in name:
            view.update(a_offset=raw, c_offset=x0)
            current = x0
        elif 'multimodal_projector' in name:
            destination = x1 if current == x0 else x0
            view.update(a_offset=current, c_offset=destination)
            current = destination
        elif 'time_mlp' in name:
            if name.endswith('.input'):
                view.update(a_offset=time_x, c_offset=time_y)
            else:
                view.update(a_offset=time_y, c_offset=time_x)
        elif 'adarms_modulation' in name:
            view.update(a_offset=time_x, c_offset=modulation)
        elif '.qkv' in name or '.blocks.q.' in name:
            view.update(a_offset=current, c_offset=q, a_instance_share=operation['count'])
        elif '.blocks.kv' in name:
            view.update(a_offset=current, a_instance_share=2)
        elif '.qk' in name:
            view.update(a_offset=q, c_offset=score)
            if name.startswith('vision.'):
                matrix_bytes = operation['m'] * 3 * 1152 * WORD_BYTES
                view.update(b_offset=q + matrix_bytes, b_transposed=True,
                            a_row_stride=2304, a_instance_stride=144,
                            a_instance_groups=16, a_group_stride=operation['m'] * 2304,
                            b_row_stride=2304, b_instance_stride=144,
                            b_instance_groups=16, b_group_stride=operation['m'] * 2304)
            else:
                view.update(a_row_stride=4096, a_instance_stride=512)
        elif '.av' in name:
            view.update(a_offset=score, c_offset=av)
            if name.startswith('vision.'):
                matrix_bytes = operation['m'] * 3 * 1152 * WORD_BYTES
                view.update(b_offset=q + 2 * matrix_bytes, b_row_stride=2304,
                            b_instance_stride=144, b_instance_groups=16,
                            b_group_stride=operation['m'] * 2304, c_row_stride=2304,
                            c_instance_stride=144, c_instance_groups=16,
                            c_group_stride=operation['m'] * 2304)
            else:
                view.update(c_row_stride=4096, c_instance_stride=512)
        elif 'attention_output' in name:
            view.update(a_offset=av, c_offset=y)
        elif '.mlp_up' in name or '.mlp_gate_and_up' in name:
            view.update(a_offset=y, c_offset=gate, a_instance_share=operation['count'])
        elif '.mlp_down' in name:
            destination = x1 if current == x0 else x0
            view.update(a_offset=gate, c_offset=destination)
            current = destination
        elif 'action_output' in name:
            view.update(a_offset=current, c_offset=raw)
        result.append(view)
    return result, align(raw + raw_bytes)


def compile_online_program(hardware=Hardware(), schedule=Schedule(), costs=None,
                           warmup_requests=20, measure_requests=50, max_inflight=64,
                           tile_size=32, max_cycles=1000000000000, mode="pipeline"):
    """Compile compact lazy tile loops and a single shared physical address map."""
    for name, value in (('warmup_requests', warmup_requests), ('measure_requests', measure_requests),
                        ('max_cycles', max_cycles)):
        if type(value) is not int:
            raise ValueError(f'{name} must be an integer')
    if max_cycles <= 0:
        raise ValueError('max_cycles must be positive')
    if type(hardware.cores) is not int or hardware.cores <= 0:
        raise ValueError('cores must be a positive integer')
    if (hardware.array_height, hardware.array_width) != (32, 32):
        raise ValueError('Online measured costs require 32x32 arrays')
    if type(hardware.core_sram_bytes) is not int or hardware.core_sram_bytes <= 0:
        raise ValueError('core_sram_bytes must be a positive integer')
    if tile_size != 32:
        raise ValueError('Online programs currently use tile_size=32')
    if type(max_inflight) is not int or max_inflight <= 0:
        raise ValueError('max_inflight is a positive per-stage DMA request limit')
    if min(warmup_requests, measure_requests) < 0 or measure_requests < 2:
        raise ValueError('At least two measured requests and nonnegative warmup are required')
    minimum_lane_bytes = 2 * (tile_size * tile_size * 2 * WORD_BYTES) + tile_size * tile_size * 4
    if hardware.core_sram_bytes < minimum_lane_bytes:
        raise ValueError('Per-core SRAM cannot hold double-buffered A/B tiles and FP32 accumulators')
    if mode not in ('pipeline', 'baseline'):
        raise ValueError('mode must be pipeline or baseline')
    if mode == 'baseline':
        from dataclasses import asdict, replace
        if hardware.cores < schedule.replicas:
            raise ValueError('Every baseline replica needs at least one core')
        source_hardware = replace(hardware, cores=max(16, hardware.cores))
        source_schedule = replace(schedule, kp=1, kd=1, replicas=1, dit_core_fraction_numerator=8,
                                  prefix_stage_cores=None, dit_stage_cores=None)
        workload = build_stage_workloads(source_hardware, source_schedule)
        workload['hardware'] = dict(asdict(hardware), bandwidth_gbps=hardware.bandwidth_gbps,
                                    point_id=hardware.point_id, dram_target='LPDDR5X')
        workload['schedule'] = asdict(schedule)
        for stage in workload['stages']:
            stage['cores'] = hardware.cores // schedule.replicas
            stage['private_sram_bytes'] = stage['cores'] * hardware.core_sram_bytes
            stage['shared_sram_bytes'] = hardware.shared_sram_bytes // schedule.replicas
        workload['allocation'] = {'replica_cores': hardware.cores // schedule.replicas,
                                  'prefix_cores': [hardware.cores // schedule.replicas],
                                  'dit_cores': [hardware.cores // schedule.replicas], 'steps': [10]}
    else:
        workload = build_stage_workloads(hardware, schedule)
    operations = [unbatched(stage, schedule.b) for stage in workload['stages']]
    needed = set()
    lane_sram = hardware.core_sram_bytes + hardware.shared_sram_bytes // hardware.cores
    max_m = maximum_m_tile(lane_sram, tile_size, hardware.word_bytes)
    for stage, program in zip(workload['stages'], operations):
        batch = 1 if stage['kind'] == 'prefix' and mode == 'pipeline' else schedule.b
        for operation in program:
            needed.update(tile_shapes(operation, batch, tile_size, max_m))
    if costs is None:
        costs = measure_compute_costs(needed, hardware.array_height)
    if any(type(value) is not int or value <= 0 for value in costs.values()):
        raise ValueError('Compute costs must be positive measured integer cycle counts')
    missing = needed - costs.keys()
    if missing:
        raise ValueError(f'Compute cost table misses tile shapes: {sorted(missing)}')
    weights, weights_end = _canonical_weights()
    kv_base = weights_end
    # Active jobs share stage-private scratch; request slots carry persistent KV.
    scratch = []
    replica_scratch_bytes = 0
    view_programs = []
    for stage, program in zip(workload['stages'], operations):
        views, sample_bytes = activation_views(program)
        view_programs.append(views)
        batch = 1 if stage['kind'] == 'prefix' and mode == 'pipeline' else schedule.b
        scratch.append({'offset': replica_scratch_bytes, 'sample_stride': sample_bytes})
        replica_scratch_bytes += sample_bytes * batch
    replica_scratch_bytes = align(replica_scratch_bytes)
    scratch_bytes = replica_scratch_bytes * schedule.replicas
    available_kv_bytes = LOGICAL_CAPACITY - weights_end - scratch_bytes
    request_stride = align(KV_STRIDE)
    slots = available_kv_bytes // request_stride
    # A finite measurement needs this many requests at most, plus pipeline fill.
    queue_bound = schedule.replicas * (schedule.kp * (1 + 2 * schedule.b) + 3 * schedule.b * schedule.kd)
    slots = min(slots, max(warmup_requests + measure_requests + queue_bound, queue_bound))
    if slots < schedule.b * schedule.replicas:
        raise ValueError('The logical 16GiB DRAM budget cannot hold one batch per replica')
    activation_base = align(kv_base + slots * request_stride)
    if activation_base + scratch_bytes > LOGICAL_CAPACITY:
        raise ValueError('Physical address layout exceeds logical DRAM capacity')
    stages = []
    for stage_index, (stage, program) in enumerate(zip(workload['stages'], operations)):
        compiled = []
        scratch_base = activation_base + scratch[stage_index]['offset']
        batch = 1 if stage['kind'] == 'prefix' and mode == 'pipeline' else schedule.b
        for operation, view in zip(program, view_programs[stage_index]):
            m, n, k, count = (operation[key] for key in ('m', 'n', 'k', 'count'))
            is_attention = attention(operation)
            a_bytes = align(m * k * count * WORD_BYTES)
            c_bytes = align(m * n * count * WORD_BYTES)
            op = {'name': operation['name'], 'weight_key': operation['weight_key'],
                  'weight_offset_bytes': operation['weight_offset_bytes'], 'm': m, 'n': n, 'k': k, 'count': count,
                  'weight_copies': operation['weight_copies'], 'attention': is_attention,
                  'b_shared': not is_attention, 'rhs_share': 1,
                  'a_base': scratch_base, 'c_base': scratch_base + a_bytes,
                  'b_base': scratch_base + a_bytes + c_bytes,
                  'a_stage_local': True, 'c_stage_local': True, 'b_stage_local': is_attention,
                  'activation_sample_stride': scratch[stage_index]['sample_stride'],
                  'tile_size': tile_size, 'm_tile': min(max_m, m * batch if not is_attention else m),
                  'request_stride': request_stride}
            for field, value in view.items():
                if field.endswith('_offset'):
                    op[field[0] + '_base'] = scratch_base + value
                else:
                    op[field] = value
            name = operation['name']
            if not is_attention:
                op['b_base'] = weights[operation['weight_key']]['base'] + operation['weight_offset_bytes']
            if _is_gemma_attention(operation):
                op['b_stage_local'] = False
                op['rhs_share'] = 8
                op['b_transposed'] = '.qk' in name
                op['b_base'] = kv_base + _kv_layer(operation) * 2 * KV_MATRIX_BYTES
                if '.av' in name:
                    op['b_base'] += KV_MATRIX_BYTES
                op['b_instance_stride'] = 2 * KV_MATRIX_BYTES
            if '.kv' in name and not is_attention:
                op['c_stage_local'] = False
                op['c_base'] = kv_base + _kv_layer(operation) * 2 * KV_MATRIX_BYTES
                if name.startswith('denoise.'):
                    op['c_base'] += PREFIX_TOKENS * KV_HEAD_DIM * WORD_BYTES
                op['c_instance_stride'] = KV_MATRIX_BYTES
            batch = 1 if stage['kind'] == 'prefix' and mode == 'pipeline' else schedule.b
            op['compute_costs'] = [{'m': shape[0], 'n': shape[1], 'k': shape[2], 'cycles': costs[shape]}
                                   for shape in sorted(tile_shapes(operation, batch, tile_size, max_m))]
            compiled.append(op)
        stages.append({'name': stage['name'], 'kind': stage['kind'], 'cores': stage['cores'],
                       'timestep_start': stage.get('timestep_start', 0),
                       'timestep_count': stage['steps'], 'operations': compiled,
                       'core_sram_bytes': hardware.core_sram_bytes,
                       'lane_sram_bytes': lane_sram,
                       'shared_sram_bytes_per_lane': hardware.shared_sram_bytes // hardware.cores,
                       'double_buffer_minimum_bytes_per_lane': minimum_lane_bytes,
                       'maximum_m_tile': max_m})
    online = {'mode': mode, 'dram_capacity_bytes': LOGICAL_CAPACITY, 'accelerator_ghz': hardware.frequency_ghz, 'warmup_requests': warmup_requests,
              'measure_requests': measure_requests, 'batch': schedule.b, 'replicas': schedule.replicas,
              'handoff_gbps': hardware.handoff_gbps, 'max_inflight': max_inflight,
              'max_cycles': max_cycles, 'stages': stages, 'kv_base': kv_base,
              'kv_stride': request_stride, 'request_stride': request_stride, 'request_slots': slots,
              'activation_stride': replica_scratch_bytes,
              'activation_replica_stride': replica_scratch_bytes, 'word_bytes': WORD_BYTES}
    layout = {'logical_capacity_bytes': LOGICAL_CAPACITY, 'weights_bytes': weights_end,
              'weights': copy.deepcopy(weights), 'kv_base': kv_base, 'kv_bytes_per_request': KV_STRIDE,
              'kv_stride': request_stride, 'request_slots': slots,
              'kv_prefix_bytes_per_request': PREFIX_TOKENS * LAYERS * 2 * KV_HEAD_DIM * WORD_BYTES,
              'activation_base': activation_base, 'activation_bytes': scratch_bytes,
              'highest_address_exclusive': activation_base + scratch_bytes,
              'activation_storage': 'one scratch arena per active stage job and replica',
              'slot_lifecycle': 'persistent request KV slot recycled after request completion'}
    return {'online': online, 'address_layout': layout, 'workload': workload}


def isolated_config(compiled, stage_index, memory_config, warmup_requests=0, measure_requests=10):
    """Extract one real stage program for independent native calibration.

    The native engine's isolated mode executes the original prefix or DiT job
    program with one replica, retaining tile addresses, cores, DMA limits, and
    physical controller configuration. This gives a reproducible stage service
    measurement for contention slowdown reporting.
    """
    stages = compiled['online']['stages']
    if type(stage_index) is not int or not 0 <= stage_index < len(stages):
        raise ValueError('stage_index must select an existing stage')
    for name, value in (('warmup_requests', warmup_requests), ('measure_requests', measure_requests)):
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    if measure_requests < 1:
        raise ValueError('At least one isolated calibration request is required')
    result = copy.deepcopy(memory_config)
    result.update(copy.deepcopy(compiled))
    online = result['online']
    online['mode'] = 'isolated'
    online['replicas'] = 1
    online['warmup_requests'] = warmup_requests
    online['measure_requests'] = measure_requests
    selected = copy.deepcopy(stages[stage_index])
    if selected['kind'] == 'dit':
        selected['timestep_start'] = 0
    online['stages'] = [selected]
    online['handoff_gbps'] = compiled['online']['handoff_gbps']
    result['calibration_metadata'] = {'source_stage_index': stage_index, 'source_stage_name': selected['name'],
                                    'source_timestep_start': stages[stage_index]['timestep_start'],
                                    'job_timestep_count': selected['timestep_count'],
                                    'measurement': 'isolated native stage job service'}
    return result


def all_tile_shapes(tile_size=32):
    """Union of edge shapes for every supported batch, pipeline and baseline."""
    workload = build_stage_workloads(Hardware(), Schedule(kp=1, kd=1, b=1))
    needed = set()
    capacities = {int(private * (1 << 20)) + (shared << 20) // cores
                  for private in (.25, 1, 4) for shared in (0, 64, 256)
                  for cores in (8, 16, 32, 64, 128)}
    for batch, capacity in itertools.product((1, 2, 5, 10), sorted(capacities)):
        for stage in workload['stages']:
            for operation in stage['operations']:
                needed.update(tile_shapes(operation, batch, tile_size, maximum_m_tile(capacity)))
                if stage['kind'] == 'prefix':
                    needed.update(tile_shapes(operation, 1, tile_size))
    return needed


def prepare_cost_table(path=None):
    """Measure the complete supported shape universe once and cache it."""
    needed = all_tile_shapes()
    if path is None:
        return measure_compute_costs(needed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        table = read_compute_costs(path) if path.is_file() else {}
        missing = needed - table.keys()
        if missing:
            table.update(measure_compute_costs(missing))
            write_compute_costs(path, table)
        return table


def target_memory_config(target_gbps=136.6):
    """Resolve the LPDDR5-derived target or the upstream HBM3 protocol model.

    HBM3 uses the unmodified upstream speed bin, including its documented
    estimated core timings and integer-picosecond half-clock quantization.
    Neither configuration is a measured commercial-device timing calibration.
    """
    import ramulator
    if target_gbps not in (136.6, 819.0):
        raise ValueError('Target bandwidth must be 136.6 or 819 decimal GB/s')
    if target_gbps == 819.0:
        channels = 16
        organization = 'HBM3_16Gb_8hi'
        speed_bin = 'HBM3_6400Mbps'
        dram = ramulator.dram.HBM3(org_preset=organization, timing_preset=speed_bin)
        resolved = dram.to_config()
        timings = dict(zip(ramulator.dram.HBM3.timing_params, resolved['timing']))
        # One controller is a 64-bit channel with two 32-bit pseudochannels.
        # Each pseudochannel transfers one 32-byte transaction per nBL ticks.
        tck_ps = timings['tCK_ps']
        effective = channels * 2 * resolved['data_payload_bytes'] * 1000 / (timings['nBL'] * tck_ps)
        controller = ramulator.controller.HBM34(
            dram=dram, scheduler=ramulator.scheduler.FRFCFS(),
            refresh_manager=ramulator.refresh_manager.HBM34PerBankRefresh(),
            row_policy=ramulator.row_policy.Open(), addr_mapper=ramulator.addr_mapper.RoBaRaCoCh())
        memory = ramulator.memory_system.GenericDRAM(
            clock_ratio=1, controllers=[controller] * channels,
            channel_mapper={'impl': 'ModuloInterleave'})
        metadata = {
            'requested_bandwidth_gbps': target_gbps, 'protocol': 'HBM3',
            'controller_impl': 'HBM34', 'controller_count': channels,
            'channel_width_bits': 64, 'pseudochannels_per_controller': 2,
            'tCK_ps': tck_ps, 'nominal_ck_ps': 625, 'tick_multiplier': 2,
            'nominal_peak_gbps': 819.2, 'effective_peak_gbps': effective,
            'relative_peak_error': abs(effective / target_gbps - 1),
            'timing_model': 'HBM3 upstream preset with integer-picosecond half-CK ticks',
            'timing_provenance': 'ramulator2/python/ramulator/dram/hbm3.py HBM3_6400Mbps; core timings marked Ramulator Guesstimate upstream',
            'organization_preset': organization, 'timing_preset': speed_bin,
            'timing_overrides': {}, 'logical_capacity_bytes': LOGICAL_CAPACITY,
            'physical_capacity_bytes': channels * (1 << 30)}
        return {'frontend': {'impl': 'External', 'clock_ratio': 1},
                'memory_system': memory.to_config(), 'memory_metadata': metadata}
    channels = 2
    tck_ps = round(64000 * channels / target_gbps)
    base = ramulator.dram.LPDDR5.timing_presets['LPDDR5_6400']
    preserve_ns = ('nCL', 'nRCD', 'nRP', 'nRPab', 'nRAS', 'nRC', 'nWR',
                   'nCWL', 'nPPD', 'nWTRS', 'nWTRL', 'nCS')
    overrides = {name: math.ceil(base[name] * 1250 / tck_ps) for name in preserve_ns}
    overrides.update(tCK_ps=tck_ps, rate=round(8000000 / tck_ps))
    rank = 2 if channels == 2 else 1
    dram = ramulator.dram.LPDDR5(org_preset='LPDDR5_8Gb_x16', timing_preset='LPDDR5_6400',
                                channel_width=64, rank=rank, **overrides)
    controller = ramulator.controller.LPDDR5(
        dram=dram, scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(), row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh())
    memory = ramulator.memory_system.GenericDRAM(clock_ratio=1, controllers=[controller] * channels,
                                                channel_mapper={'impl': 'ModuloInterleave'})
    effective = channels * 64000 / tck_ps
    metadata = {'requested_bandwidth_gbps': target_gbps, 'protocol': 'LPDDR5',
                'target_label': 'LPDDR5X bandwidth target, LPDDR5-derived timing approximation',
                'controller_count': channels,
                'channel_width_bits': 64, 'tCK_ps': tck_ps,
                'effective_peak_gbps': effective, 'relative_peak_error': abs(effective / target_gbps - 1),
                'timing_model': 'LPDDR5-derived, bandwidth-calibrated, absolute-ns timing rescaled',
                'timing_provenance': 'LPDDR5_6400 constraints with explicitly reported custom timing overrides',
                'timing_overrides': overrides, 'logical_capacity_bytes': LOGICAL_CAPACITY,
                'physical_capacity_bytes': channels * rank * (4 << 30)}
    return {'frontend': {'impl': 'External', 'clock_ratio': 1},
            'memory_system': memory.to_config(), 'memory_metadata': metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cost-table', type=Path)
    parser.add_argument('--cores', type=int, default=32)
    parser.add_argument('--channels', type=int, default=2)
    parser.add_argument('--kp', type=int, default=2)
    parser.add_argument('--kd', type=int, default=2)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--replicas', type=int, default=1)
    parser.add_argument('--dit-core-numerator', type=int, default=4)
    parser.add_argument('--target-gbps', type=float, choices=(136.6, 819.0), default=136.6)
    parser.add_argument('--max-inflight', type=int, default=64)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--measure', type=int, default=50)
    args = parser.parse_args()
    costs = read_compute_costs(args.cost_table) if args.cost_table else None
    hardware = Hardware(args.cores, args.channels)
    schedule = Schedule(args.kp, args.kd, args.batch, args.replicas, args.dit_core_numerator)
    data = compile_online_program(hardware, schedule, costs, args.warmup, args.measure, args.max_inflight)
    memory = target_memory_config(args.target_gbps)
    memory.update(data)
    import yaml
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(memory, sort_keys=False))


if __name__ == '__main__':
    main()
