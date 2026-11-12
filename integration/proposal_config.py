"""Proposal hardware, deterministic stage allocation, and pi0.5 workloads."""

from dataclasses import asdict, dataclass
from functools import lru_cache
import math

from pi05_model import (EXCLUDED_OPERATIONS, MODEL_CONFIG, OPENPI_COMMIT,
                        SOURCE_URLS, build_operations)

CHANNEL_BANDWIDTH_GBPS = {1: 68.3, 2: 136.6, 4: 273.2, 8: 546.4}


@dataclass(frozen=True)
class Hardware:
    cores: int = 32
    channels: int = 2
    frequency_ghz: float = 1.0
    core_sram_bytes: int = 1 << 20
    shared_sram_bytes: int = 64 << 20
    dram_capacity_bytes: int = 16 << 30
    dram_latency_ns: float = 100.0
    handoff_gbps: float = 256.0
    array_height: int = 32
    array_width: int = 32
    word_bytes: int = 2

    @property
    def bandwidth_gbps(self):
        return CHANNEL_BANDWIDTH_GBPS[self.channels]

    @property
    def point_id(self):
        return f'c{self.cores}_ch{self.channels}'


HARDWARE_POINTS = tuple(Hardware(cores, channels) for cores, channels in (
    (8, 1), (8, 8), (16, 1), (16, 2),
    (32, 1), (32, 2), (32, 4), (32, 8),
    (64, 1), (64, 2), (64, 4), (64, 8),
    (128, 1), (128, 2), (128, 4), (128, 8)))


@dataclass(frozen=True)
class Schedule:
    kp: int = 2
    kd: int = 2
    b: int = 2
    replicas: int = 1
    dit_core_fraction_numerator: int = 4
    beta: float = 0.5


def distribute(total, groups):
    """Assign every integer resource deterministically, with earlier remainders."""
    if total < groups or groups < 1:
        raise ValueError('Every stage must receive at least one core')
    quotient, remainder = divmod(total, groups)
    return [quotient + (index < remainder) for index in range(groups)]


def timestep_partition(stages):
    if stages not in (1, 2, 3, 4, 5, 10):
        raise ValueError('DiT stages must be 1, 2, 3, 4, 5, or 10')
    return distribute(10, stages)


def allocate(hardware, schedule):
    if type(hardware.cores) is not int or hardware.cores < 1:
        raise ValueError('cores must be a positive integer')
    if hardware.channels not in CHANNEL_BANDWIDTH_GBPS:
        raise ValueError('channels must be 1, 2, 4, or 8')
    if hardware.array_height < 1 or hardware.array_width < 1 or hardware.word_bytes != 2:
        raise ValueError('Positive array dimensions and BF16 words are required')
    for value in (hardware.frequency_ghz, hardware.handoff_gbps):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Frequency and handoff bandwidth must be positive')
    if min(hardware.core_sram_bytes, hardware.shared_sram_bytes,
           hardware.dram_capacity_bytes) < 0:
        raise ValueError('Memory capacities must be nonnegative')
    if schedule.kp not in (1, 2, 3, 4, 6):
        raise ValueError('Prefix stages must be 1, 2, 3, 4, or 6')
    steps = timestep_partition(schedule.kd)
    if schedule.b not in (1, 2, 5, 10) or schedule.replicas not in (1, 2, 4):
        raise ValueError('Unsupported proposal microbatch or replica count')
    if type(schedule.dit_core_fraction_numerator) is not int or not 1 <= schedule.dit_core_fraction_numerator <= 15:
        raise ValueError('DiT core numerator must be an integer from 1 through 15')
    if not math.isfinite(schedule.beta) or not 0 < schedule.beta < 1:
        raise ValueError('beta must be strictly between zero and one')
    # Equal replica partitions leave the integer remainder explicitly idle.
    replica_cores = hardware.cores // schedule.replicas
    dit_cores = replica_cores * schedule.dit_core_fraction_numerator // 16
    prefix_cores = replica_cores - dit_cores
    return {'replica_cores': replica_cores,
            'idle_cores': hardware.cores - replica_cores * schedule.replicas,
            'prefix_cores': distribute(prefix_cores, schedule.kp),
            'dit_cores': distribute(dit_cores, schedule.kd), 'steps': steps}


def macs(operations):
    return sum(op['m'] * op['n'] * op['k'] * op['count'] for op in operations)


@lru_cache(maxsize=1)
def _prefix_layers():
    """Expand aggregated families into their original contiguous layer sequence."""
    families = build_operations(action_horizon=16, denoise_steps=1)
    layers = []
    for name, depth in (('vision.patch_embedding', 1), ('vision.blocks.', 27),
                        ('vision.multimodal_projector', 1), ('prefix.blocks.', 18)):
        selected = [op for op in families if op['name'].startswith(name)]
        for layer_index in range(depth):
            operations = []
            for source in selected:
                assert source['count'] % depth == 0
                operation = dict(source)
                operation['count'] //= depth
                if source['stage'] == 'vision' and not source['name'].endswith(('.qk', '.av')):
                    # Three camera views share each learned projection matrix.
                    operation['m'] *= 3
                    operation['count'] //= 3
                operation['weight_copies'] = (0 if source['name'].endswith(('.qk', '.av'))
                                              else operation['count'])
                operation['name'] = f'{source["name"]}.layer{layer_index}'
                operations.append(operation)
            layers.append(operations)
    return layers


def _balanced_contiguous(layers, groups):
    """Minimize the largest stage MAC count while preserving whole layers."""
    prefix = [0]
    for layer in layers:
        prefix.append(prefix[-1] + macs(layer))
    costs = {(0, 0): 0}
    cuts = {}
    for count in range(1, groups + 1):
        for end in range(count, len(layers) + 1):
            options = [(max(costs[count - 1, start], prefix[end] - prefix[start]), start)
                       for start in range(count - 1, end) if (count - 1, start) in costs]
            cost, start = min(options)
            costs[count, end] = cost
            cuts[count, end] = start
    boundaries = [len(layers)]
    for count in range(groups, 0, -1):
        boundaries.append(cuts[count, boundaries[-1]])
    boundaries.reverse()
    return [layers[start:end] for start, end in zip(boundaries, boundaries[1:])]


def _batched_timestep(batch):
    operations = []
    for source in build_operations(action_horizon=16, denoise_steps=1):
        if source['stage'] != 'denoise':
            continue
        operation = dict(source)
        # Attention has separate request-specific K/V tensors. Dense maps
        # concatenate request rows so each projection weight is loaded once.
        if source['name'].endswith(('.qk', '.av')):
            operation['count'] *= batch
        else:
            operation['m'] *= batch
        operation['weight_copies'] = (0 if source['name'].endswith(('.qk', '.av'))
                                      else operation['count'])
        operations.append(operation)
    return operations


def build_stage_workloads(hardware=Hardware(), schedule=Schedule()):
    allocation = allocate(hardware, schedule)
    replica_bandwidth = hardware.bandwidth_gbps / schedule.replicas
    prefix_groups = _balanced_contiguous(_prefix_layers(), schedule.kp)
    prefix_total = sum(allocation['prefix_cores'])
    dit_total = sum(allocation['dit_cores'])
    prefix_stages = []
    layer_start = 0
    for index, (layers, cores) in enumerate(zip(prefix_groups, allocation['prefix_cores'])):
        prefix_stages.append({'name': f'prefix_{index}', 'kind': 'prefix',
                              'cores': cores, 'steps': 1,
                              'operations': [dict(op) for layer in layers for op in layer],
                              'bandwidth_gbps': replica_bandwidth * (1 - schedule.beta) * cores / prefix_total,
                              'layer_start': layer_start, 'layer_end': layer_start + len(layers)})
        layer_start += len(layers)
    dit_stages = []
    timestep_start = 0
    for index, (steps, cores) in enumerate(zip(allocation['steps'], allocation['dit_cores'])):
        dit_stages.append({'name': f'dit_{index}', 'kind': 'dit', 'cores': cores,
                           'steps': steps, 'operations': _batched_timestep(schedule.b),
                           'bandwidth_gbps': replica_bandwidth * schedule.beta * cores / dit_total,
                           'timestep_start': timestep_start, 'timestep_end': timestep_start + steps})
        timestep_start += steps
    for stage in prefix_stages + dit_stages:
        stage['private_sram_bytes'] = stage['cores'] * hardware.core_sram_bytes
        stage['shared_sram_bytes'] = (hardware.shared_sram_bytes * stage['cores']
                                      // (schedule.replicas * allocation['replica_cores']))
    return {'hardware': dict(asdict(hardware), point_id=hardware.point_id,
                             bandwidth_gbps=hardware.bandwidth_gbps,
                             dram_target='LPDDR5X'),
            'schedule': asdict(schedule), 'allocation': allocation,
            'prefix_stages': prefix_stages, 'dit_stages': dit_stages,
            'stages': prefix_stages + dit_stages,
            'kv_bytes_per_request': 968 * 18 * 2 * 256 * hardware.word_bytes,
            'source_commit': OPENPI_COMMIT, 'sources': SOURCE_URLS,
            'camera_batching': 'Three camera views concatenate dense-projection rows and share each weight matrix',
            'operation_schema': {'weight_copies': 'Count of distinct learned matrices per operation family; attention products use zero'},
            'pilot_validation_notes': [{
                'hardware': {'cores': 32, 'channels': 2},
                'schedule': {'kp': 4, 'kd': 10, 'b': 10, 'replicas': 1,
                             'dit_core_fraction_numerator': 4},
                'status': 'invalid',
                'reason': 'Eight DiT cores cannot allocate at least one core to each of ten timestep stages'}],
            'excluded_operations': EXCLUDED_OPERATIONS,
            'workload': dict(MODEL_CONFIG, cameras=3, prompt_tokens=200,
                             prefix_tokens=968, action_horizon=16,
                             denoise_steps=10, batch_size=1)}
