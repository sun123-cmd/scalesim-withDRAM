"""Run single-pass closed-loop simulation and per-hardware bound search."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

from online_runtime import OnlineRuntime, atomic_json, exclusive_output
from online_search import enumerate_candidates, search
from online_workload import prepare_cost_table
from proposal_config import Hardware, Schedule


def parameters(args):
    schedule = Schedule(args.kp, args.kd, args.batch, args.replicas,
                        args.dit_core_numerator, .5, getattr(args, 'prefix_stage_cores', None),
                        getattr(args, 'dit_stage_cores', None))
    return {'hardware': asdict(hardware(args)), 'schedule': asdict(schedule),
            'mode': args.mode, 'tm_batch': args.batch}


def hardware(args):
    return Hardware(args.cores, 2, core_sram_bytes=int(getattr(args, 'core_sram_mib', 1) * (1 << 20)),
                    shared_sram_bytes=getattr(args, 'shared_sram_mib', 64) * (1 << 20))


def runtime(args, costs):
    return OnlineRuntime(args.binary, args.output, costs, args.target_gbps,
                         args.warmup, args.measure, args.max_inflight, args.max_cycles,
                         args.timeout_seconds, args.events, not args.skip_reference_calibration)


def resolve_error_margin(args, engine):
    if args.pilot_report is None:
        return (args.error_margin if args.error_margin is not None else 0.0,
                'user_supplied' if args.error_margin is not None else 'zero_default')
    pilot = json.loads(args.pilot_report.read_text())
    if not isinstance(pilot, dict):
        raise ValueError('Pilot report must contain a JSON object')
    if pilot.get('execution_model') != engine.identity_digest:
        raise ValueError('Pilot report uses a different simulator or measurement configuration')
    if pilot.get('parameters', {}).get('hardware') != parameters(args)['hardware']:
        raise ValueError('Pilot report belongs to a different hardware point')
    value = pilot.get('measured_error_margin')
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value < 1):
        raise ValueError('Pilot error margin must be finite and in [0,1)')
    return value, str(args.pilot_report.resolve())


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('simulate', 'search', 'pilot', 'prepare-costs', 'compile'))
    parser.add_argument('--cores', type=int, default=32)
    parser.add_argument('--target-gbps', type=float, choices=(136.6, 819.0), default=136.6)
    parser.add_argument('--kp', type=int, default=2)
    parser.add_argument('--kd', type=int, default=2)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--replicas', type=int, default=1)
    parser.add_argument('--dit-core-numerator', type=int, default=4)
    parser.add_argument('--prefix-stage-cores', type=int)
    parser.add_argument('--dit-stage-cores', type=int)
    parser.add_argument('--core-sram-mib', type=float, choices=(.25, 1, 4), default=1)
    parser.add_argument('--shared-sram-mib', type=int, choices=(0, 64, 256), default=64)
    parser.add_argument('--mode', choices=('pipeline', 'baseline'), default='pipeline')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--measure', type=int, default=50)
    parser.add_argument('--max-inflight', type=int, default=64,
                        help='Maximum outstanding DMA transactions per stage, across its lanes')
    parser.add_argument('--max-cycles', type=int, default=1000000000000)
    parser.add_argument('--timeout-seconds', type=float)
    parser.add_argument('--cost-table', type=Path, default=root / 'results/online/compute_costs.json')
    parser.add_argument('--binary', type=Path, default=root / 'integration/build/ramulator2-online')
    parser.add_argument('--output', type=Path, default=root / 'results/online')
    parser.add_argument('--events', action='store_true')
    parser.add_argument('--skip-reference-calibration', action='store_true',
                        help='Run a candidate without producing stage uncontended slowdown references')
    parser.add_argument('--workers', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=150)
    margin = parser.add_mutually_exclusive_group()
    margin.add_argument('--error-margin', type=float)
    margin.add_argument('--pilot-report', type=Path, help='Read measured e from a matching native pilot report')
    parser.add_argument('--objective', choices=('throughput', 'pareto'), default='pareto')
    parser.add_argument('--bound-source', choices=('certified', 'uncontended-stage-table'), default='certified')
    parser.add_argument('--include-uneven', action='store_true')
    parser.add_argument('--max-evaluations', type=int)
    parser.add_argument('--candidate-limit', type=int,
                        help='Explicit development subset, recorded as incomplete search space')
    args = parser.parse_args()
    if (args.cores < 1 or args.max_inflight < 1 or args.max_cycles < 1
            or args.warmup < 0 or args.measure < 2 or args.workers < 1 or args.batch_size < 1):
        parser.error('Invalid core, DMA, measurement, search worker, or error-margin settings')
    if args.error_margin is not None and (not math.isfinite(args.error_margin) or not 0 <= args.error_margin < 1):
        parser.error('Error margin must be finite and in [0,1)')
    if args.timeout_seconds is not None and (not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0):
        parser.error('Timeout must be finite and positive')
    if args.max_evaluations is not None and args.max_evaluations < 1:
        parser.error('Evaluation limit must be positive')
    if args.candidate_limit is not None and args.candidate_limit < 1:
        parser.error('Candidate limit must be positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with exclusive_output(args.output):
            if args.command in ('simulate', 'search', 'pilot'):
                summary_name = {'simulate': 'summary.json', 'search': 'search_summary.json', 'pilot': 'pilot.json'}[args.command]
                (args.output / summary_name).unlink(missing_ok=True)
            costs = prepare_cost_table(args.cost_table)
            if args.command == 'prepare-costs':
                print(f'Measured {len(costs)} SCALE-Sim compute shapes in {args.cost_table}')
                return
            engine = runtime(args, costs)
            if args.command == 'compile':
                import yaml
                compiled = engine.compile(parameters(args))
                (args.output / 'program.yaml').write_text(yaml.safe_dump(engine.config(compiled), sort_keys=False))
                print(f'Compiled complete horizon16 program at {args.output / "program.yaml"}')
            elif args.command == 'pilot':
                from online_pilot import measure_pilot
                result = measure_pilot(engine, parameters(args))
                atomic_json(args.output / 'pilot.json', result)
                print(f"Measured layer additivity e={result['measured_error_margin']:.8g} in {args.output / 'pilot.json'}")
            elif args.command == 'simulate':
                candidate = {'id': 'single', 'parameters': parameters(args)}
                result = engine.evaluate(candidate)
                atomic_json(args.output / 'summary.json', result)
                print(json.dumps(result['metrics'], indent=2))
            else:
                if args.skip_reference_calibration:
                    parser.error('Search requires uncontended stage references for the requested slowdown outputs')
                error_margin, margin_source = resolve_error_margin(args, engine)
                bound_fn = engine.bounds if args.bound_source == 'certified' else engine.stage_table_bounds
                candidates = enumerate_candidates(hardware(args), bound_fn, args.include_uneven)
                candidates.sort(key=lambda item: (item['optimistic_s'], item['optimistic_l'], item['id']))
                complete = args.candidate_limit is None or args.candidate_limit >= len(candidates)
                if args.candidate_limit is not None:
                    candidates = candidates[:args.candidate_limit]
                atomic_json(args.output / 'candidates.json', candidates)
                result = search(candidates, engine.evaluate, workers=args.workers, batch_size=args.batch_size,
                                error_margin=error_margin, mode=args.objective,
                                resume_path=args.output / 'checkpoint.json', max_evaluations=args.max_evaluations,
                                candidate_space_complete=complete)
                result['bound_source'] = args.bound_source
                result['error_margin_source'] = margin_source
                result['native_memory_configuration'] = engine.memory['memory_metadata']
                result['measurement'] = {'warmup': args.warmup, 'measure': args.measure}
                atomic_json(args.output / 'search_summary.json', result)
                print(f"Evaluated {result['evaluated_count']}/{result['candidate_count']}, certificate={result['bound_certificate']}")
    except (ValueError, RuntimeError, subprocess.SubprocessError, OSError) as error:
        parser.exit(1, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
