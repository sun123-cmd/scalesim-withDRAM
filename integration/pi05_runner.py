"""Simulate all GEMM shapes in a model manifest with cached cold-tile measurements."""

import argparse
from collections import Counter
import configparser
import itertools
import json
from pathlib import Path
import os
import subprocess
import sys


def dimension_tiles(length, tile_size):
    if type(length) is not int or length <= 0:
        raise ValueError('GEMM dimensions must be positive integers')
    if type(tile_size) is not int or tile_size <= 0:
        raise ValueError('tile_size must be a positive integer')
    full, remainder = divmod(length, tile_size)
    result = Counter()
    if full:
        result[tile_size] = full
    if remainder:
        result[remainder] += 1
    return result


def operation_tiles(operation, tile_size):
    count = operation.get('count', 1)
    if type(count) is not int or count <= 0:
        raise ValueError('Operation count must be a positive integer')
    axes = [dimension_tiles(operation[axis], tile_size) for axis in ('m', 'n', 'k')]
    result = {}
    for m, n, k in itertools.product(*axes):
        result[(m, n, k)] = axes[0][m] * axes[1][n] * axes[2][k] * count
    expected = operation['m'] * operation['n'] * operation['k'] * count
    if sum(m * n * k * multiplicity for (m, n, k), multiplicity in result.items()) != expected:
        raise AssertionError('Tiling must preserve every MAC')
    return result


def write_inputs(root, destination, shapes, word_bytes, array_size=16, sram_kib=64, total_sram_kib=None):
    config = configparser.ConfigParser()
    config.read(root / 'examples/scale.cfg')
    config['general']['run_name'] = 'pi05_tiles'
    for key, value in {'ArrayHeight': array_size, 'ArrayWidth': array_size, 'IfmapSramSzkB': sram_kib,
                       'FilterSramSzkB': sram_kib, 'OfmapSramSzkB': sram_kib, 'Bandwidth': 16,
                       'Dataflow': 'ws', 'WordSizeBytes': word_bytes}.items():
        config['architecture_presets'][key] = str(value)
    if total_sram_kib is not None:
        if total_sram_kib < 8:
            raise ValueError('Total SRAM must hold all three operand buffers')
        input_kib = total_sram_kib * 3 // 8
        for key, value in {'IfmapSramSzkB': input_kib, 'FilterSramSzkB': input_kib,
                           'OfmapSramSzkB': total_sram_kib - 2 * input_kib}.items():
            config['architecture_presets'][key] = str(value)
    config['run_presets']['InterfaceBandwidth'] = 'USER'
    config['run_presets']['UseRamulatorTrace'] = 'False'
    with (destination / 'scale.cfg').open('w') as file:
        config.write(file)
    with (destination / 'tiles.csv').open('w') as file:
        file.write('Layer,M,N,K,\n')
        for index, (m, n, k) in enumerate(shapes):
            file.write(f'tile_{index},{m},{n},{k},\n')
    layout = (root / 'examples/layout.csv').read_text().splitlines()
    # Custom layouts are disabled. SCALE-Sim still expects one layout row per layer.
    columns = layout[1].split(',')
    with (destination / 'layout.csv').open('w') as file:
        file.write(layout[0] + '\n')
        for index in range(len(shapes)):
            file.write(','.join([f'tile_{index}'] + columns[1:]) + '\n')


METRICS = ('macs', 'tile_count', 'weighted_baseline_array_cycles',
           'weighted_coupled_array_cycles', 'weighted_stall_cycles',
           'weighted_elapsed_cycles_including_prefetch', 'weighted_dram_cycles',
           'weighted_dram_requests', 'weighted_completed_requests')


def aggregate(operations, tile_size, shapes, simulation):
    if len(simulation['layers']) != len(shapes):
        raise ValueError('Every unique tile must have one simulated layer')
    measurements = {}
    for shape, layer in zip(shapes, simulation['layers']):
        if layer['replay']['requests'] != layer['replay']['completed']:
            raise ValueError('A simulated tile has incomplete DRAM requests')
        measurements[shape] = layer
    reports = []
    stages = {}
    totals = dict.fromkeys(METRICS, 0)
    for operation in operations:
        tiles = operation_tiles(operation, tile_size)
        report = {key: operation[key] for key in ('name', 'stage', 'm', 'n', 'k')}
        report['count'] = operation.get('count', 1)
        report.update(dict.fromkeys(METRICS, 0))
        report['macs'] = operation['m'] * operation['n'] * operation['k'] * report['count']
        for shape, multiplicity in tiles.items():
            layer = measurements[shape]
            report['tile_count'] += multiplicity
            report['weighted_baseline_array_cycles'] += multiplicity * layer['baseline']['compute_cycles_including_stalls']
            report['weighted_coupled_array_cycles'] += multiplicity * layer['coupled']['compute_cycles_including_stalls']
            report['weighted_stall_cycles'] += multiplicity * layer['coupled']['stall_cycles']
            report['weighted_elapsed_cycles_including_prefetch'] += multiplicity * layer['coupled']['total_cycles_including_prefetch']
            report['weighted_dram_cycles'] += multiplicity * layer['replay']['cycles']
            report['weighted_dram_requests'] += multiplicity * layer['replay']['requests']
            report['weighted_completed_requests'] += multiplicity * layer['replay']['completed']
        reports.append(report)
        stage = stages.setdefault(operation['stage'], dict.fromkeys(METRICS, 0))
        for metric in METRICS:
            stage[metric] += report[metric]
            totals[metric] += report[metric]
    return reports, stages, totals


def run(args):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(args.manifest.read_text())
    operations = manifest['operations']
    if not operations:
        raise ValueError('The model manifest contains no GEMM operations')
    # Each distinct edge shape is measured once. Multiplicities cover the full dimensions.
    shapes = sorted({shape for operation in operations for shape in operation_tiles(operation, args.tile_size)})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'summary.json'
    summary_path.unlink(missing_ok=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    write_inputs(root, output, shapes, args.word_bytes, args.array_size, args.sram_kib, args.total_sram_kib)
    simulation_dir = output / 'tile_simulation'
    command = [sys.executable, '-m', 'scalesim.ramulator2', '-c', str(output / 'scale.cfg'),
               '-t', str(output / 'tiles.csv'), '-l', str(output / 'layout.csv'), '-i', 'gemm',
               '-p', str(simulation_dir), '--replay', str(args.replay.resolve()),
               '--dram-config', str(args.dram_config.resolve()),
               '--accelerator-mhz', str(args.accelerator_mhz), '--word-bytes', str(args.word_bytes), '--quiet']
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join([str(root / 'SCALE-Sim'), str(root / 'ramulator2/python'),
                                               environment.get('PYTHONPATH', '')])
    with (output / 'simulation.log').open('w') as log:
        subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    simulation = json.loads((simulation_dir / 'summary.json').read_text())
    reports, stages, totals = aggregate(operations, args.tile_size, shapes, simulation)
    measured = {'unique_tile_shapes': len(shapes),
                'requests': sum(layer['replay']['requests'] for layer in simulation['layers']),
                'completed_requests': sum(layer['replay']['completed'] for layer in simulation['layers'])}
    result = {'model': manifest.get('model', 'pi0.5'),
              'method': 'full-shape GEMM tiling with cached independent cold-tile simulation',
              'scope': 'architecture and DRAM simulation of the manifest GEMMs',
              'limitations': ['GEMM-only accounting excludes nonlinear, normalization, softmax, embedding lookup and control overhead',
                              'Independent cold tiles have no cross-tile SRAM or weight reuse',
                              'Each K tile materializes an output tile; partial-sum read/add traffic is excluded',
                              'Two-pass DRAM latency feedback reuses first-pass queue timing',
                              'Weighted cycle sums are synthetic GEMM-cost estimates with no operator overlap',
                              'Model weights and numerical policy inference are outside this simulation'],
              'tile_size': args.tile_size, 'word_bytes': args.word_bytes,
              'accelerator_mhz': args.accelerator_mhz, 'dram_tck_ns': simulation['dram_tck_ns'],
              'measured': measured, 'totals': totals, 'stages': stages, 'operations': reports,
              'tile_shapes': [dict(zip(('m', 'n', 'k'), shape)) for shape in shapes],
              'tile_summary': str(simulation_dir / 'summary.json'),
              'manifest_metadata': {key: value for key, value in manifest.items() if key != 'operations'}}
    summary_path.write_text(json.dumps(result, indent=2) + '\n')
    print(f'π0.5 GEMM simulation complete: {len(operations)} operation groups, {len(shapes)} measured tile shapes')
    print(f'All {measured["requests"]} measured DRAM requests completed. Summary: {summary_path}')
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tile-size', type=int, default=32)
    parser.add_argument('--array-size', type=int, default=16)
    parser.add_argument('--sram-kib', type=int, default=64)
    parser.add_argument('--total-sram-kib', type=int)
    parser.add_argument('--word-bytes', type=int, default=2)
    parser.add_argument('--accelerator-mhz', type=float, default=1000)
    parser.add_argument('--replay', type=Path, default=root / 'integration/build/ramulator2-replay')
    parser.add_argument('--dram-config', type=Path, default=root / 'examples/ramulator2_ddr4.py')
    args = parser.parse_args()
    if min(args.tile_size, args.word_bytes, args.array_size, args.sram_kib, args.total_sram_kib) <= 0:
        parser.error('Tile size and word bytes must be positive')
    run(args)


if __name__ == '__main__':
    main()
