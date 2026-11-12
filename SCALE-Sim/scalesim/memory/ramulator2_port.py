"""Checked per-word latency feedback from the Ramulator2 replay."""

import numpy as np


class Ramulator2Port:
    def __init__(self, path, default_latency):
        with np.load(path, allow_pickle=False) as data:
            requests = data["requests"]
            latencies = data["latencies"]
            if requests.shape[0] != len(latencies):
                raise ValueError(f"Mismatched request and latency counts in {path}")
            word_latencies = (data["word_latencies"] if "word_latencies" in data
                              else np.broadcast_to(latencies.reshape(-1, 1), requests.shape))
            if word_latencies.shape != requests.shape:
                raise ValueError(f"Mismatched per-word latency shape in {path}")
            valid = requests != -1
            self.requests = requests[valid]
            self.latencies = word_latencies[valid]
        self.latency = default_latency
        self.count = 0
        self.path = str(path)

    def set_params(self, latency):
        self.latency = latency

    def get_latency(self):
        return self.latency

    def _service(self, requests, cycles):
        requests = np.asarray(requests)
        cycles = np.asarray(cycles).reshape(-1, 1)
        valid = requests != -1
        addresses = requests[valid]
        end = self.count + len(addresses)
        expected = self.requests[self.count:end]
        if not np.array_equal(addresses, expected):
            raise ValueError(f"DRAM address sequence changed at word {self.count} in {self.path}")
        if len(cycles) != len(requests):
            raise ValueError("Each DRAM request row must have one timestamp")
        # Write-buffer drains may repack partial rows when memory stalls change.
        # Match each address occurrence in order, then wait for the last word.
        latency_matrix = np.full(requests.shape, self.latency, dtype=np.int64)
        latency_matrix[valid] = self.latencies[self.count:end]
        result = cycles + latency_matrix.max(axis=1, keepdims=True)
        self.count = end
        return result

    def service_reads(self, incoming_requests_arr_np, incoming_cycles_arr):
        return self._service(incoming_requests_arr_np, incoming_cycles_arr)

    def service_writes(self, incoming_requests_arr_np, incoming_cycles_arr_np):
        return self._service(incoming_requests_arr_np, incoming_cycles_arr_np)

    def verify_complete(self):
        if self.count != len(self.latencies):
            raise ValueError(f"Unused DRAM words in {self.path}: {self.count}/{len(self.latencies)}")
