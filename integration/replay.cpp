#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

using namespace Ramulator;

namespace Ramulator {
// Controllers size their source statistics from this frontend before receiving traffic.
class StageReplayFrontend : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, StageReplayFrontend, "StageReplay")
  int m_sources = 1;
 public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio").required();
    RAMULATOR_PARSE_PARAM(m_sources, int, "sources").required();
  }
  void tick() override {}
  bool is_finished() override { return false; }
  int get_num_cores() override { return m_sources; }
  bool receive_external_requests(int type, Addr_t addr, int source,
      std::function<void(Request&)> callback, int bytes) override {
    Request request(addr, type, source, std::move(callback));
    request.size_bytes = bytes;
    return m_memory_system->send(request);
  }
};
}

struct TraceRequest {
  uint64_t id = 0;
  uint64_t cycle = 0;
  int type = 0;
  Addr_t address = 0;
  int stage = 0;
};

struct StageStats {
  uint64_t reads = 0, writes = 0, completed = 0, retries = 0;
  uint64_t latency = 0, read_latency = 0, completed_reads = 0;
  uint64_t first = std::numeric_limits<uint64_t>::max(), last = 0;
  double bandwidth = 0;
  long double next_issue_cycle = 0;
};

int main(int argc, char** argv) {
  try {
    if (argc < 7 || argc > 9) {
      throw std::runtime_error("Usage: ramulator2-replay CONFIG TRACE RESPONSES STATS MAX_CYCLES ISSUE_WIDTH [shared|static [ALLOCATIONS.csv]]");
    }
    const bool stage_mode = argc >= 8;
    const std::string policy = stage_mode ? argv[7] : "legacy";
    if (policy != "legacy" && policy != "shared" && policy != "static") {
      throw std::runtime_error("Policy must be shared or static");
    }
    if (policy == "static" && argc != 9) {
      throw std::runtime_error("Static policy requires stage_id,bandwidth_gbps allocations CSV");
    }
    const uint64_t max_cycles = std::stoull(argv[5]);
    const uint64_t issue_width = std::stoull(argv[6]);
    if (max_cycles == 0 || issue_width == 0) {
      throw std::runtime_error("MAX_CYCLES and ISSUE_WIDTH must be positive");
    }
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    std::ofstream stats(argv[4]);
    if (!input || !output || !stats) {
      throw std::runtime_error("Cannot open trace, response, or statistics file");
    }
    std::vector<TraceRequest> requests;
    std::map<int, StageStats> stage_stats;
    std::string line;
    while (std::getline(input, line)) {
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      std::istringstream record(line);
      int64_t id, timestamp;
      char operation;
      TraceRequest item;
      if (!(record >> id >> timestamp >> operation >> item.address)) {
        throw std::runtime_error("Malformed request trace");
      }
      record >> std::ws;
      if (!record.eof()) {
        if (!stage_mode || !(record >> item.stage)) throw std::runtime_error("Invalid stage ID");
        record >> std::ws;
      }
      if (!record.eof() || id < 0 || timestamp < 0 || item.address < 0 || item.stage < 0 ||
          item.stage == std::numeric_limits<int>::max() || (operation != 'R' && operation != 'W')) {
        throw std::runtime_error("Invalid request ID, timestamp, operation, or byte address");
      }
      item.id = static_cast<uint64_t>(id);
      item.cycle = static_cast<uint64_t>(timestamp);
      if (item.id != requests.size() || (!requests.empty() && item.cycle < requests.back().cycle)) {
        throw std::runtime_error("Trace requires consecutive IDs and sorted timestamps");
      }
      item.type = operation == 'R' ? Request::Type::Read : Request::Type::Write;
      stage_stats.try_emplace(item.stage);
      requests.push_back(item);
    }
    if (input.bad()) throw std::runtime_error("Failed to read request trace");
    if (policy == "static") {
      std::ifstream allocation(argv[8]);
      if (!allocation) throw std::runtime_error("Cannot open stage allocations");
      std::getline(allocation, line);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line != "stage_id,bandwidth_gbps") throw std::runtime_error("Invalid stage allocation header");
      std::map<int, double> rates;
      while (std::getline(allocation, line)) {
        if (line.empty()) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        int stage;
        double rate;
        if (!(row >> stage >> rate) || stage < 0 || !std::isfinite(rate) || rate <= 0 || rates.count(stage)) {
          throw std::runtime_error("Invalid or duplicate stage allocation");
        }
        row >> std::ws;
        if (!row.eof()) throw std::runtime_error("Invalid stage allocation fields");
        rates[stage] = rate;
      }
      for (auto& [stage, measurement] : stage_stats) {
        if (!rates.count(stage)) throw std::runtime_error("Missing bandwidth for active stage");
        measurement.bandwidth = rates.at(stage);
      }
    }

    auto config = Config::parse_config_file(argv[1]);
    if (stage_mode) {
      const int sources = stage_stats.empty() ? 1 : stage_stats.rbegin()->first + 1;
      config.set("frontend", ConfigNode::Map{{"impl", "StageReplay"}, {"clock_ratio", 1}, {"sources", sources}});
    }
    std::unique_ptr<IFrontEnd> frontend(Factory::create_frontend(config));
    std::unique_ptr<IMemorySystem> memory(Factory::create_memory_system(config));
    frontend->connect_memory_system(memory.get());
    memory->connect_frontend(frontend.get());
    const int transaction_bytes = memory->get_tx_bytes();
    const double tck_ns = memory->get_tCK();
    for (const auto& request : requests) {
      if (request.address % transaction_bytes != 0) throw std::runtime_error("Request address must be transaction-aligned");
    }
    uint64_t cycle = 0, completed = 0, rejected = 0;
    size_t next = 0;
    std::map<int, std::deque<size_t>> queues;
    std::vector<int> stages;
    for (const auto& [stage, measurement] : stage_stats) {
      stages.push_back(stage);
      queues[stage] = {};
    }
    size_t round_robin = 0;
    auto send = [&](const TraceRequest& request) {
      auto& measurement = stage_stats.at(request.stage);
      const bool accepted = frontend->receive_external_requests(
          request.type, request.address, request.stage,
          [&, request](Request&) {
            output << request.id << ',' << request.cycle << ',' << cycle << '\n';
            auto& source = stage_stats.at(request.stage);
            source.completed++;
            source.last = std::max(source.last, cycle);
            source.latency += cycle - request.cycle;
            if (request.type == Request::Type::Read) {
              source.completed_reads++;
              source.read_latency += cycle - request.cycle;
            }
            completed++;
          }, transaction_bytes);
      if (accepted) {
        measurement.first = std::min(measurement.first, cycle);
        if (request.type == Request::Type::Read) measurement.reads++;
        else measurement.writes++;
        if (policy == "static") {
          measurement.next_issue_cycle = std::max(measurement.next_issue_cycle, static_cast<long double>(cycle) - 1)
              + transaction_bytes / (measurement.bandwidth * tck_ns);
        }
      } else {
        rejected++;
        measurement.retries++;
      }
      return accepted;
    };
    output << "id,requested_cycle,completion_cycle\n";
    while (completed != requests.size()) {
      if (cycle >= max_cycles) throw std::runtime_error("DRAM replay exceeded MAX_CYCLES before draining all requests");
      uint64_t issued = 0;
      if (!stage_mode) {
        while (next < requests.size() && requests[next].cycle <= cycle && issued < issue_width) {
          if (!send(requests[next])) break;
          next++;
          issued++;
        }
      } else {
        while (next < requests.size() && requests[next].cycle <= cycle) {
          queues[requests[next].stage].push_back(next);
          next++;
        }
        size_t consecutive_skips = 0;
        while (issued < issue_width && consecutive_skips < stages.size()) {
          const int stage = stages[round_robin];
          round_robin = (round_robin + 1) % stages.size();
          auto& queue = queues.at(stage);
          if (queue.empty() || (policy == "static" && stage_stats.at(stage).next_issue_cycle > cycle + 1e-12L)) {
            consecutive_skips++;
            continue;
          }
          if (send(requests[queue.front()])) {
            queue.pop_front();
            issued++;
            consecutive_skips = 0;
          } else consecutive_skips++;
        }
      }
      cycle++;
      memory->tick();
    }
    frontend->finalize();
    memory->finalize();
    frontend->update_stats_recursive();
    memory->update_stats_recursive();
    frontend->print_stats(stats);
    memory->print_stats(stats);
    stats << "replay:\n  cycles: " << cycle << "\n  requests: " << requests.size()
          << "\n  completed: " << completed << "\n  queue_retries: " << rejected
          << "\n  transaction_bytes: " << transaction_bytes << "\n  tCK_ns: " << tck_ns << '\n';
    if (stage_mode) {
      stats << "  policy: " << policy << "\n  stages:";
      if (stage_stats.empty()) stats << " {}\n";
      else stats << '\n';
      for (const auto& [stage, measurement] : stage_stats) {
        const uint64_t bytes = (measurement.reads + measurement.writes) * transaction_bytes;
        const double span_ns = (measurement.last - measurement.first + 1) * tck_ns;
        stats << "    " << stage << ":\n      read_requests: " << measurement.reads
              << "\n      write_requests: " << measurement.writes
              << "\n      completed_requests: " << measurement.completed
              << "\n      read_bytes: " << measurement.reads * transaction_bytes
              << "\n      write_bytes: " << measurement.writes * transaction_bytes
              << "\n      first_issue_cycle: " << measurement.first
              << "\n      last_completion_cycle: " << measurement.last
              << "\n      latency_sum_cycles: " << measurement.latency
              << "\n      avg_read_latency_cycles: " << (measurement.completed_reads ? double(measurement.read_latency) / measurement.completed_reads : 0)
              << "\n      queue_retries: " << measurement.retries
              << "\n      allocated_bandwidth_gbps: " << measurement.bandwidth
              << "\n      actual_bandwidth_gbps: " << bytes / span_ns << '\n';
      }
    }
    if (!output || !stats) throw std::runtime_error("Failed to write replay results");
    std::cout << "Completed " << completed << " requests in " << cycle << " DRAM cycles, "
              << transaction_bytes << " bytes/transaction\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ramulator2-replay: " << error.what() << '\n';
    return 1;
  }
}
