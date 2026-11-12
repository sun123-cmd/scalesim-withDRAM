#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/i_memory_system.h"

using namespace Ramulator;

struct TraceRequest {
  uint64_t id;
  uint64_t cycle;
  int type;
  Addr_t address;
};

int main(int argc, char** argv) {
  try {
    if (argc != 7) {
      throw std::runtime_error(
          "Usage: ramulator2-replay CONFIG TRACE RESPONSES STATS MAX_CYCLES ISSUE_WIDTH");
    }
    const auto config = Config::parse_config_file(argv[1]);
    std::unique_ptr<IFrontEnd> frontend(Factory::create_frontend(config));
    std::unique_ptr<IMemorySystem> memory(Factory::create_memory_system(config));
    frontend->connect_memory_system(memory.get());
    memory->connect_frontend(frontend.get());
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
    std::string line;
    while (std::getline(input, line)) {
      if (line.find_first_not_of(" \t\r") == std::string::npos) {
        continue;
      }
      std::istringstream record(line);
      int64_t id, timestamp;
      char operation;
      TraceRequest item{};
      if (!(record >> id >> timestamp >> operation >> item.address)) {
        throw std::runtime_error("Malformed request trace");
      }
      record >> std::ws;
      if (!record.eof() || id < 0 || timestamp < 0 || item.address < 0 ||
          (operation != 'R' && operation != 'W')) {
        throw std::runtime_error("Invalid request ID, timestamp, operation, or byte address");
      }
      item.id = static_cast<uint64_t>(id);
      item.cycle = static_cast<uint64_t>(timestamp);
      if (item.id != requests.size() ||
          (!requests.empty() && item.cycle < requests.back().cycle)) {
        throw std::runtime_error("Trace requires consecutive IDs and sorted timestamps");
      }
      item.type = operation == 'R' ? Request::Type::Read : Request::Type::Write;
      requests.push_back(item);
    }
    if (input.bad()) {
      throw std::runtime_error("Failed to read request trace");
    }
    const int transaction_bytes = memory->get_tx_bytes();
    for (const auto& request : requests) {
      if (request.address % transaction_bytes != 0) {
        throw std::runtime_error("Request address must be transaction-aligned");
      }
    }
    uint64_t cycle = 0;
    uint64_t completed = 0;
    uint64_t rejected = 0;
    size_t next = 0;
    output << "id,requested_cycle,completion_cycle\n";
    while (completed != requests.size()) {
      if (cycle >= max_cycles) {
        throw std::runtime_error("DRAM replay exceeded MAX_CYCLES before draining all requests");
      }
      uint64_t issued = 0;
      while (next < requests.size() && requests[next].cycle <= cycle && issued < issue_width) {
        const auto request = requests[next];
        const bool accepted = frontend->receive_external_requests(
            request.type, request.address, 0,
            [&, request](Request&) {
              output << request.id << ',' << request.cycle << ',' << cycle << '\n';
              ++completed;
            },
            transaction_bytes);
        if (!accepted) {
          ++rejected;
          break;
        }
        ++next;
        ++issued;
      }
      ++cycle;
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
          << "\n  transaction_bytes: " << transaction_bytes
          << "\n  tCK_ns: " << memory->get_tCK() << '\n';
    if (!output || !stats) {
      throw std::runtime_error("Failed to write replay results");
    }
    std::cout << "Completed " << completed << " requests in " << cycle
              << " DRAM cycles, " << transaction_bytes << " bytes/transaction\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ramulator2-replay: " << error.what() << '\n';
    return 1;
  }
}
