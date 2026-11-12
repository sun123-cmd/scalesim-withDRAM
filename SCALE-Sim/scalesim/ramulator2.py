"""Run SCALE-Sim, replay its traffic in Ramulator2, and feed latencies back."""

import argparse
import configparser
import csv
import json
import math
from pathlib import Path
import runpy
import subprocess

import numpy as np
import yaml

from scalesim.scale_sim import scalesim

STREAMS = (("ifmap", "IFMAP", "R", 1), ("filter", "FILTER", "R", 1),
           ("ofmap", "OFMAP", "W", 0))


def export_config(source, destination):
    if source.suffix == ".py":
        namespace = runpy.run_path(str(source))
        memory = namespace["memory_system"]
        memory = memory.to_config() if hasattr(memory, "to_config") else memory
        config = {"frontend": {"impl": "External", "clock_ratio": 1},
                  "memory_system": memory}
    else:
        config = yaml.safe_load(source.read_text())
        config["frontend"] = {"impl": "External", "clock_ratio": 1}
    destination.write_text(yaml.safe_dump(config, sort_keys=False))


def replay(executable, config, trace, responses, stats, max_cycles, issue_width):
    subprocess.run([str(executable), str(config), str(trace), str(responses),
                    str(stats), str(max_cycles), str(issue_width)], check=True)
    return yaml.safe_load(stats.read_text())


def prepare_requests(layer_path, output, tx_bytes, tck_ns, accelerator_mhz):
    matrices = {}
    events = []
    accelerator_ns = 1000.0 / accelerator_mhz
    for name, prefix, operation, default_latency in STREAMS:
        matrix = np.loadtxt(layer_path / f"{prefix}_DRAM_TRACE.csv", delimiter=",", ndmin=2)
        if matrix.shape[1] < 2 or not np.all(np.isfinite(matrix)):
            raise ValueError(f"Invalid {prefix} DRAM trace in {layer_path}")
        if np.any(matrix != np.floor(matrix)) or np.any(matrix[:, 1:] < -1):
            raise ValueError("DRAM traces must contain integer word addresses or -1 padding")
        # Upstream DRAM traces record completion times. The baseline read port
        # adds one cycle and the baseline write port adds zero cycles.
        arrivals = matrix[:, 0].astype(np.int64) - default_latency
        matrices[name] = (matrix[:, 1:].astype(np.int64), arrivals)
        for row, (addresses, arrival) in enumerate(zip(matrix[:, 1:], arrivals)):
            # SCALE-Sim currently models one-byte words. Merge only addresses
            # in the same row and DRAM transaction, retaining later refetches.
            bursts = sorted({int(address) // tx_bytes * tx_bytes
                             for address in addresses if address != -1})
            for address in bursts:
                events.append((int(arrival), name, row, operation, address))
    origin = min((event[0] for event in events), default=0)
    events.sort(key=lambda event: (event[0], event[1], event[2], event[4]))
    mapping = []
    with output.open("w") as trace:
        for request_id, (arrival, name, row, operation, address) in enumerate(events):
            cycle = math.ceil((arrival - origin) * accelerator_ns / tck_ns)
            trace.write(f"{request_id} {cycle} {operation} {address}\n")
            mapping.append((name, row, arrival, cycle, address))
    return matrices, mapping, accelerator_ns


def save_latencies(layer_path, matrices, mapping, response_path, tck_ns, accelerator_ns, tx_bytes=64):
    latencies = {}
    for name, _, _, default in STREAMS:
        requests, _ = matrices[name]
        latencies[name] = np.full(len(requests), default, dtype=np.int64)
    word_latencies = {}
    for name, _, _, default in STREAMS:
        requests, _ = matrices[name]
        word_latencies[name] = np.full(requests.shape, default, dtype=np.int64)
    origin = min((request[2] for request in mapping), default=0)
    seen = set()
    with response_path.open() as response_file:
        for response in csv.DictReader(response_file):
            request_id = int(response["id"])
            if request_id < 0 or request_id >= len(mapping) or request_id in seen:
                raise ValueError("Invalid or duplicate DRAM completion ID")
            name, row, arrival, requested, address = mapping[request_id]
            completion = int(response["completion_cycle"])
            if int(response["requested_cycle"]) != requested or completion < requested:
                raise ValueError("Invalid DRAM completion timestamp")
            latency = math.ceil(completion * tck_ns / accelerator_ns) - (arrival - origin)
            latencies[name][row] = max(latencies[name][row], latency)
            requests = matrices[name][0][row]
            burst_words = (requests >= address) & (requests < address + tx_bytes)
            word_latencies[name][row, burst_words] = latency
            seen.add(request_id)
    if len(seen) != len(mapping):
        raise ValueError("Ramulator2 did not complete every DRAM request")
    for name, (requests, _) in matrices.items():
        np.savez(layer_path / f"{name}.npz", requests=requests, latencies=latencies[name],
                 word_latencies=word_latencies[name])
    return {name: {"rows": len(values), "max_latency_cycles": int(values.max(initial=0))}
            for name, values in latencies.items()}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").unlink(missing_ok=True)
    source = configparser.ConfigParser()
    if not source.read(args.config):
        raise FileNotFoundError(args.config)
    # Both passes use identical USER-mode buffer scheduling and bandwidths.
    source["run_presets"]["InterfaceBandwidth"] = "USER"
    source["run_presets"]["UseRamulatorTrace"] = "False"
    source.remove_option("run_presets", "Ramulator2LatencyDir")
    scale_config = output / "scale.cfg"
    with scale_config.open("w") as config_file:
        source.write(config_file)
    run_name = source["general"]["run_name"]
    baseline = scalesim(config=str(scale_config), topology=str(args.topology.resolve()),
                        layout=str(args.layout.resolve()), input_type_gemm=args.input_type == "gemm",
                        verbose=not args.quiet)
    baseline.run_scale(str(output / "baseline"))

    latency_dir = output / "ramulator2"
    latency_dir.mkdir(exist_ok=True)
    dram_config = latency_dir / "config.yaml"
    export_config(args.dram_config.resolve(), dram_config)
    empty_trace = latency_dir / "empty.trace"
    empty_trace.write_text("")
    probe = replay(args.replay.resolve(), dram_config, empty_trace,
                   latency_dir / "probe.csv", latency_dir / "probe.yaml",
                   args.max_dram_cycles, args.issue_width)["replay"]
    tx_bytes = int(probe["transaction_bytes"])
    tck_ns = float(probe["tCK_ns"])
    if tx_bytes <= 0 or tck_ns <= 0:
        raise ValueError("Ramulator2 returned an invalid transaction size or clock period")
    layer_summaries = []
    for layer_id in range(baseline.topo.get_num_layers()):
        directory = latency_dir / f"layer{layer_id}"
        directory.mkdir(exist_ok=True)
        trace = directory / "requests.trace"
        responses = directory / "responses.csv"
        matrices, mapping, accelerator_ns = prepare_requests(
            output / "baseline" / run_name / f"layer{layer_id}", trace,
            tx_bytes, tck_ns, args.accelerator_mhz)
        stats = replay(args.replay.resolve(), dram_config, trace, responses,
                       directory / "stats.yaml", args.max_dram_cycles, args.issue_width)
        streams = save_latencies(directory, matrices, mapping, responses, tck_ns, accelerator_ns, tx_bytes)
        layer_summaries.append({"layer": layer_id, "streams": streams, "replay": stats["replay"]})

    source["run_presets"]["UseRamulatorTrace"] = "True"
    source["run_presets"]["Ramulator2LatencyDir"] = str(latency_dir)
    feedback_config = output / "feedback.cfg"
    with feedback_config.open("w") as config_file:
        source.write(config_file)
    feedback = scalesim(config=str(feedback_config), topology=str(args.topology.resolve()),
                        layout=str(args.layout.resolve()), input_type_gemm=args.input_type == "gemm",
                        verbose=not args.quiet)
    feedback.run_scale(str(output / "coupled"))
    for layer_summary, baseline_layer, coupled_layer in zip(
            layer_summaries, baseline.runner.single_layer_sim_object_list,
            feedback.runner.single_layer_sim_object_list):
        layer_summary["baseline"] = dict(zip(
            ("total_cycles_including_prefetch", "compute_cycles_including_stalls", "stall_cycles"),
            map(int, baseline_layer.get_compute_report_items()[:3])))
        layer_summary["coupled"] = dict(zip(
            ("total_cycles_including_prefetch", "compute_cycles_including_stalls", "stall_cycles"),
            map(int, coupled_layer.get_compute_report_items()[:3])))
    summary = {"method": "two-pass latency feedback", "accelerator_mhz": args.accelerator_mhz,
               "dram_tck_ns": tck_ns, "transaction_bytes": tx_bytes,
               "issue_width_per_dram_cycle": args.issue_width, "layers": layer_summaries,
               "baseline_report": str(output / "baseline" / run_name / "COMPUTE_REPORT.csv"),
               "coupled_report": str(output / "coupled" / run_name / "COMPUTE_REPORT.csv")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Ramulator2 latency feedback complete. Results in {output}")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-t", "--topology", type=Path, required=True)
    parser.add_argument("-l", "--layout", type=Path, required=True)
    parser.add_argument("-i", "--input-type", choices=("conv", "gemm"), default="conv")
    parser.add_argument("-p", "--output", type=Path, required=True)
    parser.add_argument("--dram-config", type=Path, default=root / "examples/ramulator2_ddr4.py")
    parser.add_argument("--replay", type=Path, default=root / "integration/build/ramulator2-replay")
    parser.add_argument("--accelerator-mhz", type=float, default=1000.0)
    parser.add_argument("--issue-width", type=int, default=4)
    parser.add_argument("--max-dram-cycles", type=int, default=100000000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.accelerator_mhz) or args.accelerator_mhz <= 0:
        parser.error("--accelerator-mhz must be finite and positive")
    if args.issue_width <= 0 or args.max_dram_cycles <= 0:
        parser.error("--issue-width and --max-dram-cycles must be positive")
    for path in (args.config, args.topology, args.layout, args.dram_config, args.replay):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    run(args)


if __name__ == "__main__":
    main()
