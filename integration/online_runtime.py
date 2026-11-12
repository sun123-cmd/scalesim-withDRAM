"""Single-pass native execution, persistent uncontended tables, and safe bounds."""
from contextlib import contextmanager
from dataclasses import asdict
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import threading
import tempfile

from online_workload import (compile_online_program, target_memory_config,
                             isolated_config)
from proposal_config import Hardware, Schedule


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
            file.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_output(path, blocking=False):
    path.mkdir(parents=True, exist_ok=True)
    with (path / '.execution.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f'Another simulator owns output directory {path}') from error
        yield


class OnlineRuntime:
    def __init__(self, binary, output, costs, target_gbps=136.6, warmup=20,
                 measure=50, max_inflight=64, max_cycles=1000000000000,
                 timeout_seconds=None, events=False, reference_table=True):
        self.binary = Path(binary).resolve()
        if not self.binary.is_file():
            raise ValueError(f'Build the native engine first: {self.binary}')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.costs = costs
        self.memory = target_memory_config(target_gbps)
        self.target = target_gbps
        self.warmup, self.measure = warmup, measure
        self.max_inflight, self.max_cycles = max_inflight, max_cycles
        self.timeout = timeout_seconds
        self.events, self.reference_table = events, reference_table
        self._locks = {}
        self._mutex = threading.Lock()
        source_root = Path(__file__).resolve().parent
        model_sources = ('online_workload.py', 'proposal_config.py', 'pi05_model.py', 'online_runtime.py')
        compiler_hash = hashlib.sha256(b''.join((source_root / name).read_bytes() for name in model_sources)).hexdigest()
        library_root = self.binary.parents[2] / 'ramulator2'
        libraries = [path for path in (library_root / 'libramulator.so', library_root / 'libramulator.dylib') if path.is_file()]
        library_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries}
        self.identity = {'native_binary_sha256': hashlib.sha256(self.binary.read_bytes()).hexdigest(),
                         'ramulator_libraries': library_hashes, 'compiler_sources_sha256': compiler_hash,
                         'cost_table': sorted((list(shape), cycles) for shape, cycles in costs.items()),
                         'memory': self.memory, 'warmup': warmup, 'measure': measure,
                         'dma_limit': max_inflight, 'max_cycles': max_cycles,
                         'uncontended_references': reference_table}
        self.identity_digest = digest(self.identity)

    def compile(self, parameters):
        hardware = Hardware(**parameters['hardware'])
        schedule = Schedule(**parameters['schedule'])
        mode = parameters.get('mode', 'pipeline')
        if mode == 'baseline':
            schedule = Schedule(**dict(parameters['schedule'], b=parameters.get('tm_batch', schedule.b)))
        compiled = compile_online_program(hardware, schedule, self.costs,
                                           warmup_requests=self.warmup, measure_requests=self.measure,
                                           max_inflight=self.max_inflight, max_cycles=self.max_cycles,
                                           mode=mode)
        compiled['workload']['hardware']['target_bandwidth_gbps'] = self.target
        compiled['workload']['hardware']['bandwidth_gbps'] = self.target
        compiled['workload']['hardware']['physical_controller_count'] = self.memory['memory_metadata']['controller_count']
        compiled['workload']['hardware']['memory_execution'] = 'one shared target memory system'
        return compiled

    def config(self, compiled):
        return dict(copy.deepcopy(self.memory), **copy.deepcopy(compiled))

    def execute(self, config, directory):
        import yaml
        directory.mkdir(parents=True, exist_ok=True)
        config_path, report_path = directory / 'program.yaml', directory / 'native.yaml'
        if self.events:
            config = copy.deepcopy(config)
            config['online']['event_log'] = str(directory / 'events.jsonl')
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        report_path.unlink(missing_ok=True)
        with (directory / 'native.log').open('w') as log:
            subprocess.run([str(self.binary), str(config_path), str(report_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=self.timeout)
        report = yaml.safe_load(report_path.read_text())
        native = report['online']
        if native['measured_requests'] != config['online']['measure_requests']:
            raise ValueError('Native result did not finish its measurement window')
        if native['method'] != 'single-pass shared-memory closed-loop':
            raise ValueError('A candidate result must come from the native closed-loop engine')
        for request in native['requests']:
            if request['completion_cycle'] < request['admission_cycle']:
                raise ValueError('Completion precedes request admission')
        return report

    def uncontended(self, compiled, index):
        stage = compiled['online']['stages'][index]
        batch = compiled['online']['batch'] if (stage['kind'] == 'dit' or compiled['online']['mode'] == 'baseline') else 1
        config = isolated_config(compiled, index, copy.deepcopy(self.memory),
                                 warmup_requests=0, measure_requests=batch)
        config['online']['calibration_batch'] = batch
        config['online']['batch'] = batch
        config['online']['queue_capacity'] = 2 * batch
        config['online']['request_slots'] = batch
        key = digest({'stage_config': config['online'], 'memory_system': config['memory_system'],
                      'execution': self.identity_digest})
        directory = self.output / 'uncontended' / key
        with self._mutex:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock, exclusive_output(directory, blocking=True):
            cache = directory / 'reference.json'
            if cache.exists():
                result = json.loads(cache.read_text())
                if result.get('signature') != key:
                    raise ValueError('Uncontended table entry has a different identity')
                return result
            report = self.execute(config, directory)
            native = report['online']
            source = native['stages'][0]
            if source['jobs'] != 1:
                raise ValueError('A reference table entry must measure exactly one stage job')
            seconds = source['service_dram_cycles'] * native['tCK_ns'] * 1e-9
            if seconds <= 0:
                raise ValueError('Uncontended stage service time must be positive')
            result = {'signature': key, 'seconds': seconds, 'source': 'isolated native stage execution',
                      'batch': batch, 'native_report': str(directory / 'native.yaml')}
            atomic_json(cache, result)
            return result

    def evaluate(self, candidate):
        identifier = candidate['id']
        if not identifier or any(character not in '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_-' for character in identifier):
            raise ValueError('Candidate ID must be safe for a result directory')
        parameters = candidate['parameters']
        directory = self.output / 'candidates' / identifier
        signature = digest({'parameters': parameters, 'execution': self.identity})
        with exclusive_output(directory):
            cache = directory / 'result.json'
            if cache.exists():
                result = json.loads(cache.read_text())
                if result.get('execution_signature') != signature:
                    raise ValueError('Cached candidate uses a different model or measurement configuration')
                return result
            compiled = self.compile(parameters)
            references = [self.uncontended(compiled, index) for index in range(len(compiled['online']['stages']))] if self.reference_table else []
            report = self.execute(self.config(compiled), directory)
            native = report['online']
            stages = []
            unmeasured_stages = []
            controllers = report['memory_system']['controller']
            if isinstance(controllers, dict):
                controllers = [controllers]
            per_replica = len(compiled['online']['stages'])
            for index, source in enumerate(native['stages']):
                stage = dict(source)
                actual = source['service_dram_cycles'] * native['tCK_ns'] * 1e-9
                stage['sampled_jobs'] = source['jobs']
                stage['actual_seconds'] = actual / source['jobs'] if source['jobs'] else None
                stage['avg_read_latency_ns'] = source['avg_read_latency_cycles'] * native['tCK_ns']
                stage['dma_completion_bandwidth_gbps'] = source['actual_bandwidth_gbps']
                stage_id = source['id']
                hits = sum(controller.get(f'read_row_hits_core_{stage_id}', 0) for controller in controllers)
                misses = sum(controller.get(f'read_row_misses_core_{stage_id}', 0) for controller in controllers)
                conflicts = sum(controller.get(f'read_row_conflicts_core_{stage_id}', 0) for controller in controllers)
                stage['read_row_hit_rate'] = hits / (hits + misses + conflicts) if hits + misses + conflicts else None
                if not source['jobs']:
                    stage['sample_status'] = 'no complete stage job in measurement window'
                    unmeasured_stages.append(stage)
                    continue
                if references:
                    stage['uncontended_seconds'] = references[index % per_replica]['seconds']
                    stage['actual_over_uncontended'] = stage['actual_seconds'] / stage['uncontended_seconds']
                stages.append(stage)
            metrics = {key: native[key] for key in ('latency_mean_seconds', 'latency_p95_seconds', 'output_interval_seconds')}
            controller_bandwidth = sum(controller.get('read_throughput_MBps', 0)
                                       + controller.get('write_throughput_MBps', 0) for controller in controllers) / 1000

            result = {'execution_signature': signature, 'execution': 'single-pass native shared-memory closed loop',
                      'metrics': metrics, 'stage_timings': stages, 'unmeasured_stages': unmeasured_stages, 'native_report': str(directory / 'native.yaml'),
                      'memory_metadata': self.memory.get('memory_metadata', {}),
                      'memory_controller_bandwidth_gbps': controller_bandwidth,
                      'stage_service_scope': 'full service time of stage jobs completed during the measurement window',
                      'address_layout': compiled['address_layout'], 'parameters': parameters,
                      'warmup_requests': self.warmup, 'measured_requests': self.measure}
            atomic_json(cache, result)
            return result

    def stage_table_bounds(self, parameters):
        """Uncontended stage-table estimates with an explicit conditional certificate."""
        compiled = self.compile(parameters)
        online = compiled['online']
        references = [self.uncontended(compiled, index)['seconds']
                      for index in range(len(online['stages']))]
        latency = sum(references)
        batch, replicas = online['batch'], online['replicas']
        if online['mode'] == 'baseline':
            interval = latency / (batch * replicas)
        else:
            interval = max(value / (batch if stage['kind'] == 'dit' else 1)
                           for stage, value in zip(online['stages'], references)) / replicas
        steady_interval = interval
        intervals = self.measure - 1
        burst = batch * replicas
        interval = (intervals // burst) * burst * steady_interval / intervals
        return {'optimistic_l': latency, 'optimistic_s': interval, 'bounds_admissible': False,
                'steady_interval_estimate': steady_interval,
                'bound_method': 'isolated native stage table with finite-window batch adjustment',
                'certificate_scope': 'conditional on pilot additivity and finite-window error margin',
                'execution_model': self.identity_digest,
                'window': {'warmup': self.warmup, 'measure': self.measure}}

    def bounds(self, parameters):
        """Finite-window lower bounds with measured tile compute cycles.

        The last stage produces at most b completions per replica per service.
        This preserves admissibility for simultaneous batched completions.
        """
        compiled = self.compile(parameters)
        online = compiled['online']
        batch, replicas = online['batch'], online['replicas']
        service = []
        stages = online['stages']
        for stage in stages:
            job_batch = batch if stage['kind'] == 'dit' or online['mode'] == 'baseline' else 1
            cycles = 0
            for operation in stage['operations']:
                m = operation['m'] * job_batch if operation['b_shared'] else operation['m']
                repetitions = operation['count'] if operation['b_shared'] else operation['count'] * job_batch
                mt = operation.get('m_tile', operation['tile_size'])
                nt = kt = operation['tile_size']
                from pi05_runner import dimension_tiles
                cost = {(row['m'], row['n'], row['k']): row['cycles'] for row in operation['compute_costs']}
                for mm, mc in dimension_tiles(m, mt).items():
                    for nn, nc in dimension_tiles(operation['n'], nt).items():
                        for kk, kc in dimension_tiles(operation['k'], kt).items():
                            cycles += cost[(mm, nn, kk)] * mc * nc * kc * repetitions
            seconds = cycles / (stage['cores'] * online['accelerator_ghz'] * 1e9)
            service.append(seconds * (stage['timestep_count'] if stage['kind'] == 'dit' else 1))
        latency = sum(service)
        # Each baseline replica executes prefix and DiT sequentially.
        final_period = latency if online['mode'] == 'baseline' else service[-1]
        intervals = self.measure - 1
        bursts = batch * replicas
        interval = (intervals // bursts) * final_period / intervals
        return {'optimistic_l': latency, 'optimistic_s': interval, 'bounds_admissible': True,
                'bound_method': 'compute-only measured tile work / allocated lanes; finite completion-window burst bound',
                'execution_model': self.identity_digest, 'window': {'warmup': self.warmup, 'measure': self.measure}}
