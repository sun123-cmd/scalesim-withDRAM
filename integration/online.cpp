#include "ramulator/base/config.h"
#include "ramulator/base/factory.h"
#include "ramulator/base/param.h"
#include "ramulator/frontend/i_frontend.h"
#include "ramulator/memory_system/channel_mapper/i_channel_mapper.h"
#include "ramulator/memory_system/i_memory_system.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
using namespace Ramulator;
namespace Ramulator {
class OnlineFrontend : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, OnlineFrontend, "OnlineFrontend")
  int sources = 1;

public:
  void init() override {
    RAMULATOR_PARSE_PARAM(m_clock_ratio, unsigned int, "clock_ratio")
        .required();
    RAMULATOR_PARSE_PARAM(sources, int, "sources").required();
  }
  void tick() override {}
  bool is_finished() override { return false; }
  int get_num_cores() override { return sources; }
  bool receive_external_requests(int type, Addr_t addr, int source,
                                 std::function<void(Request &)> callback,
                                 int bytes) override {
    Request req(addr, type, source, std::move(callback));
    req.size_bytes = bytes;
    return m_memory_system->send(req);
  }
};
class ModuloInterleave : public IChannelMapper, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IChannelMapper, ModuloInterleave,
                                    "ModuloInterleave")
  int channels = 1,
      shift = 0;

public:
  void init() override {}
  void setup(int count, int bits) override {
    channels = count;
    shift = bits;
    if (count <= 0)
      throw std::runtime_error("Invalid channel count");
  }
  void apply(Request &req) const override {
    auto line = static_cast<uint64_t>(req.addr) >> shift;
    req.addr_vec.resize(1);
    req.addr_vec[0] = line % channels;
    req.intra_channel_addr = ((line / channels) << shift) |
                             (req.addr & ((uint64_t(1) << shift) - 1));
  }
};
} // namespace Ramulator
using U = uint64_t;
U add(U left, U right) {
  if (left > std::numeric_limits<U>::max() - right)
    throw std::runtime_error("Online address addition overflow");
  return left + right;
}
U integer(const ConfigNode &node, const std::string &key, U fallback = 0) {
  auto value = node[key];
  if (!value)
    return fallback;
  auto text = value.as<std::string>();
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("Expected nonnegative integer: " + key);
  return std::stoull(text);
}
U multiply(U left, U right) {
  if (right && left > std::numeric_limits<U>::max() / right)
    throw std::runtime_error("Online address multiplication overflow");
  return left * right;
}
struct Operation {
  U m, n, k, count, a, b, c, stride, ai, bi, ci, tile, mtile, word, share;
  U ar, br, cr, ashare, agroups, bgroups, cgroups, ags, bgs, cgs;
  bool shared, transposed, a_local, b_local, c_local;
  U sample_stride;
  std::map<std::array<U, 3>, U> costs;
};
struct Record {
  U slot, admission = 0, completion = 0, macs = 0;
  int step = 0;
};
struct Buffer {
  bool used = false, computing = false;
  U token = 0, k = 0, depth = 0, end = 0, pending = 0;
  size_t sent = 0;
  std::vector<U> reads;
};
struct Lane {
  bool active = false, writing = false;
  U last_stall = std::numeric_limits<U>::max(), token = 0, row = 0, col = 0,
    rep = 0, next_k = 0, consume_k = 0, write_pending = 0;
  size_t writes_sent = 0;
  std::array<Buffer, 2> buffers;
  std::vector<U> writes;
};
struct Stage {
  int replica = 0, position = 0;
  bool dit = false, active = false, finished = false;
  U cores = 1, step_start = 0, step_count = 1, current_step = 0, op_index = 0,
    next_output = 0, total_outputs = 0;
  U last_stall = std::numeric_limits<U>::max(), peak_inflight = 0, inflight = 0,
    initial_inflight = 0, read_requests = 0, write_requests = 0,
    completions = 0, completed_reads = 0, retries = 0, read_latency = 0,
    stall = 0, lane_stall = 0, compute_cycles = 0, jobs = 0;
  U started = 0, service_cycles = 0;
  std::string name;
  std::vector<Operation> operations;
  std::deque<U> queue;
  U reserved = 0;
  std::vector<U> ids;
  std::vector<Lane> lanes;
};
struct Transfer {
  U end;
  int destination;
  std::vector<U> ids;
};
class Engine {
  std::unique_ptr<IFrontEnd> frontend;
  std::unique_ptr<IMemorySystem> memory;
  std::vector<Stage> stages;
  std::vector<Record> records;
  std::vector<Transfer> transfers;
  std::deque<U> free_slots;
  std::vector<bool> replica_busy;
  bool baseline = false, isolated = false;
  U calibration_batch = 1;
  U cycle = 0, tx = 0, max_cycles = 0, warmup = 20, measured = 50, batch = 1,
    replicas = 1, max_inflight = 64, issue_width = 16, issued = 0,
    completed = 0, handoff_available = 0, handoff_bytes = 0, queue_capacity = 0;
  U admission_cursor = 0, request_stride = 0, capacity = U(16) << 30,
    next_token = 1, measurement_cycle = 0, replica_stride = 0;
  std::ofstream events;
  double tck = 0, accelerator_ghz = 1, handoff_gbps = 256;
  int per_replica = 0, prefix_count = 0;
  U ceil_cycles(double seconds) {
    return std::max<U>(1, static_cast<U>(std::ceil(seconds * 1e9 / tck)));
  }
  Operation parse_operation(const ConfigNode &node) {
    Operation op{};
    op.m = integer(node, "m");
    op.n = integer(node, "n");
    op.k = integer(node, "k");
    op.count = integer(node, "count", 1);
    op.a = integer(node, "a_base");
    op.b = integer(node, "b_base");
    op.c = integer(node, "c_base");
    op.stride = integer(node, "request_stride", request_stride);
    op.word = integer(node, "word_bytes", 2);
    op.tile = integer(node, "tile_size", 32);
    op.mtile = integer(node, "m_tile", op.tile);
    op.share = integer(node, "rhs_share", 1);
    op.a_local = node["a_stage_local"].as<bool>(false);
    op.b_local = node["b_stage_local"].as<bool>(false);
    op.c_local = node["c_stage_local"].as<bool>(false);
    op.sample_stride = integer(node, "activation_sample_stride");
    op.shared = node["b_shared"].as<bool>(true);
    op.transposed = node["b_transposed"].as<bool>(false);
    op.ai = integer(node, "a_instance_stride",
                    multiply(multiply(op.m, op.k), op.word));
    op.bi = integer(node, "b_instance_stride",
                    multiply(multiply(op.k, op.n), op.word));
    op.ci = integer(node, "c_instance_stride",
                    multiply(multiply(op.m, op.n), op.word));
    op.ar = integer(node, "a_row_stride", multiply(op.k, op.word));
    op.br = integer(node, "b_row_stride",
                    multiply(op.transposed ? op.k : op.n, op.word));
    op.cr = integer(node, "c_row_stride", multiply(op.n, op.word));
    op.ashare = integer(node, "a_instance_share", 1);
    op.agroups = integer(node, "a_instance_groups");
    op.bgroups = integer(node, "b_instance_groups");
    op.cgroups = integer(node, "c_instance_groups");
    op.ags = integer(node, "a_group_stride");
    op.bgs = integer(node, "b_group_stride");
    op.cgs = integer(node, "c_group_stride");
    if (!op.ashare)
      throw std::runtime_error("A instance sharing factor must be positive");
    if (!op.m || !op.n || !op.k || !op.count || !op.tile || !op.mtile ||
        (op.word != 1 && op.word != 2 && op.word != 4 && op.word != 8) ||
        !op.share || !op.stride)
      throw std::runtime_error("Invalid operation dimensions or strides");
    if (multiply(op.count, op.ai) > capacity ||
        multiply(op.count, op.bi) > capacity ||
        multiply(op.count, op.ci) > capacity)
      throw std::runtime_error(
          "Operation instances exceed DRAM address capacity");
    const auto cost_rows = node["compute_costs"];
    if (!cost_rows.is_sequence() || cost_rows.size() == 0)
      throw std::runtime_error("Missing measured compute costs");
    for (auto row : cost_rows.seq()) {
      U a = integer(row, "m"), b = integer(row, "n"), c = integer(row, "k"),
        cost = integer(row, "cycles");
      if (!a || !b || !c || !cost)
        throw std::runtime_error("Invalid tile cost");
      op.costs[{a, b, c}] = cost;
    }
    return op;
  }
  U instance_offset(U instance, U stride, U groups, U group_stride) {
    return groups ? add(multiply(instance / groups, group_stride),
                        multiply(instance % groups, stride))
                  : multiply(instance, stride);
  }
  U request_slot(const Stage &s, U sample) {
    if (sample >= s.ids.size())
      throw std::runtime_error("Tile sample out of range");
    return records[s.ids[sample]].slot;
  }
  U offset(const Stage &s, const Operation &op, U sample, bool local) {
    return local ? static_cast<U>(s.replica) * replica_stride +
                       sample * op.sample_stride
                 : request_slot(s, sample) * op.stride;
  }
  void event(const std::string &type, const Stage &s, U address = 0, U tile = 0,
             U output_tile = 0, U k = 0) {
    if (!events)
      return;
    events << "{\"event\":\"" << type << "\",\"stage\":" << (&s - stages.data())
           << ",\"cycle\":" << cycle << ",\"timestep\":"
           << (s.dit ? static_cast<long long>(s.current_step) : -1)
           << ",\"address\":" << address << ",\"tile\":" << tile
           << ",\"output_tile\":" << output_tile << ",\"k\":" << k
           << ",\"operation_index\":" << s.op_index << ",\"request_ids\":[";
    for (size_t n = 0; n < s.ids.size(); ++n) {
      if (n)
        events << ',';
      events << s.ids[n];
    }
    events << "]}\n";
  }
  void add_range(std::set<U> &addresses, U start, U bytes) {
    if (!bytes)
      return;
    if (start >= capacity || bytes > capacity - start)
      throw std::runtime_error("DMA address exceeds configured DRAM capacity");
    U last = (start + bytes - 1) / tx * tx;
    for (U addr = start / tx * tx;; addr += tx) {
      addresses.insert(addr);
      if (addr == last)
        break;
    }
  }
  std::vector<U> reads(Stage &s, Lane &lane, Buffer &buffer) {
    const auto &op = s.operations[s.op_index];
    std::set<U> addresses;
    U sample = op.shared ? 0 : lane.rep / op.count,
      instance = op.shared ? lane.rep : lane.rep % op.count;
    U height = std::min(op.mtile,
                        (op.shared ? op.m * s.ids.size() : op.m) - lane.row),
      width = std::min(op.tile, op.n - lane.col);
    for (U row = 0; row < height; ++row) {
      U logical = lane.row + row;
      U owner = op.shared ? logical / op.m : sample;
      U local = op.shared ? logical % op.m : logical;
      add_range(addresses,
                add(add(add(op.a, offset(s, op, owner, op.a_local)),
                        instance_offset(instance / op.ashare, op.ai, op.agroups,
                                        op.ags)),
                    add(multiply(local, op.ar), multiply(buffer.k, op.word))),
                buffer.depth * op.word);
    }
    U rhs = add(
        add(op.b, instance_offset(op.shared ? instance : instance / op.share,
                                  op.bi, op.bgroups, op.bgs)),
        op.shared ? 0 : offset(s, op, sample, op.b_local));
    if (op.transposed) {
      for (U column = 0; column < width; ++column)
        add_range(addresses,
                  add(rhs, add(multiply(lane.col + column, op.br),
                               multiply(buffer.k, op.word))),
                  buffer.depth * op.word);
    } else {
      for (U row = 0; row < buffer.depth; ++row)
        add_range(addresses,
                  add(rhs, add(multiply(buffer.k + row, op.br),
                               multiply(lane.col, op.word))),
                  width * op.word);
    }
    return {addresses.begin(), addresses.end()};
  }
  std::vector<U> writes(Stage &s, Lane &lane) {
    const auto &op = s.operations[s.op_index];
    std::set<U> addresses;
    U sample = op.shared ? 0 : lane.rep / op.count,
      instance = op.shared ? lane.rep : lane.rep % op.count;
    U height = std::min(op.mtile,
                        (op.shared ? op.m * s.ids.size() : op.m) - lane.row),
      width = std::min(op.tile, op.n - lane.col);
    for (U row = 0; row < height; ++row) {
      U logical = lane.row + row;
      U owner = op.shared ? logical / op.m : sample;
      U local = op.shared ? logical % op.m : logical;
      add_range(addresses,
                add(add(add(op.c, offset(s, op, owner, op.c_local)),
                        instance_offset(instance, op.ci, op.cgroups, op.cgs)),
                    add(multiply(local, op.cr), multiply(lane.col, op.word))),
                width * op.word);
    }
    return {addresses.begin(), addresses.end()};
  }
  void setup_operation(Stage &s) {
    const auto &op = s.operations[s.op_index];
    U m = op.shared ? op.m * s.ids.size() : op.m;
    U reps = op.shared ? op.count : op.count * s.ids.size();
    s.total_outputs = ((m + op.mtile - 1) / op.mtile) *
                      ((op.n + op.tile - 1) / op.tile) * reps;
    s.next_output = 0;
    for (auto &lane : s.lanes)
      lane = Lane{};
  }
  void assign_output(Stage &s, Lane &lane) {
    const auto &op = s.operations[s.op_index];
    U ntiles = (op.n + op.tile - 1) / op.tile,
      m = op.shared ? op.m * s.ids.size() : op.m,
      mtiles = (m + op.mtile - 1) / op.mtile;
    U index = s.next_output++;
    lane = Lane{};
    lane.active = true;
    lane.token = next_token++;
    lane.rep = index / (ntiles * mtiles);
    index %= ntiles * mtiles;
    lane.row = index / ntiles * op.mtile;
    lane.col = index % ntiles * op.tile;
  }
  void fill(Stage &s, Lane &lane) {
    const auto &op = s.operations[s.op_index];
    for (auto &buffer : lane.buffers)
      if (!buffer.used && lane.next_k < op.k) {
        buffer = Buffer{};
        buffer.used = true;
        buffer.token = next_token++;
        buffer.k = lane.next_k;
        buffer.depth = std::min(op.tile, op.k - buffer.k);
        lane.next_k += buffer.depth;
        buffer.reads = reads(s, lane, buffer);
      }
  }
  bool send(Stage &s, int stage_index, U address, bool write,
            std::function<void()> done, U token) {
    if (s.inflight >= max_inflight || issued >= issue_width)
      return false;
    U start = cycle;
    event(write ? "write_issue_attempt" : "read_issue_attempt", s, address,
          token);
    ++s.inflight;
    bool accepted = frontend->receive_external_requests(
        write ? Request::Type::Write : Request::Type::Read, address,
        stage_index,
        [this, &s, start, write, address, done, token](Request &) {
          event(write ? "write_complete" : "read_complete", s, address, token);
          --s.inflight;
          ++s.completions;
          if (!write) {
            s.read_latency += cycle - start;
            ++s.completed_reads;
          }
          done();
        },
        tx);
    if (!accepted) {
      --s.inflight;
      ++s.retries;
      return false;
    }
    event(write ? "write_issue" : "read_issue", s, address, token);
    s.peak_inflight = std::max(s.peak_inflight, s.inflight);
    ++issued;
    if (write)
      ++s.write_requests;
    else
      ++s.read_requests;
    return true;
  }
  void advance_stage(Stage &s, int index) {
    if (!s.active || s.finished)
      return;
    const auto &op = s.operations[s.op_index];
    bool all_finished = true, computing_any = false, waiting = false;
    for (auto &lane : s.lanes) {
      if (!lane.active && s.next_output < s.total_outputs)
        assign_output(s, lane);
      if (!lane.active)
        continue;
      all_finished = false;
      if (lane.writing) {
        while (lane.writes_sent < lane.writes.size() &&
               s.inflight < max_inflight && issued < issue_width) {
          ++lane.write_pending;
          bool ok = send(
              s, index, lane.writes[lane.writes_sent], true,
              [&lane]() { --lane.write_pending; }, lane.token);
          if (!ok) {
            --lane.write_pending;
            break;
          }
          ++lane.writes_sent;
        }
        if (lane.writes_sent < lane.writes.size() || lane.write_pending > 0) {
          waiting = true;
          if (lane.last_stall != cycle) {
            ++s.lane_stall;
            lane.last_stall = cycle;
          }
        }
        if (lane.writes_sent == lane.writes.size() && lane.write_pending == 0) {
          lane.active = false;
          lane.writing = false;
        }
        continue;
      }
      fill(s, lane);
      for (auto &buffer : lane.buffers)
        if (buffer.used && !buffer.computing) {
          while (buffer.sent < buffer.reads.size() &&
                 s.inflight < max_inflight && issued < issue_width) {
            ++buffer.pending;
            bool ok = send(
                s, index, buffer.reads[buffer.sent], false,
                [&buffer]() { --buffer.pending; }, buffer.token);
            if (!ok) {
              --buffer.pending;
              break;
            }
            ++buffer.sent;
          }
        }
      for (auto &buffer : lane.buffers)
        if (buffer.used && buffer.computing) {
          computing_any |= cycle < buffer.end;
          if (cycle >= buffer.end) {
            event("compute_end", s, 0, buffer.token, lane.token, buffer.k);
            lane.consume_k += buffer.depth;
            buffer = Buffer{};
            if (lane.consume_k == op.k) {
              lane.writing = true;
              lane.writes = writes(s, lane);
            }
          }
        }
      if (lane.writing)
        continue;
      bool busy = false;
      for (const auto &buffer : lane.buffers)
        busy |= buffer.computing;
      if (!busy) {
        for (auto &buffer : lane.buffers)
          if (buffer.used && buffer.k == lane.consume_k) {
            if (buffer.sent == buffer.reads.size() && buffer.pending == 0) {
              U height = std::min(op.mtile,
                                  (op.shared ? op.m * s.ids.size() : op.m) -
                                      lane.row),
                width = std::min(op.tile, op.n - lane.col);
              auto it = op.costs.find({height, width, buffer.depth});
              if (it == op.costs.end())
                throw std::runtime_error(
                    "Missing measured compute cost for edge tile");
              U duration = std::max<U>(
                  1, std::ceil(it->second / (accelerator_ghz * tck)));
              for (U row = 0; row < height; ++row) {
                U owner =
                    op.shared ? (lane.row + row) / op.m : lane.rep / op.count;
                records[s.ids[owner]].macs += width * buffer.depth;
              }
              event("compute_start", s, 0, buffer.token, lane.token, buffer.k);
              buffer.computing = true;
              buffer.end = cycle + duration;

              computing_any = true;
            } else {
              waiting = true;
              if (lane.last_stall != cycle) {
                ++s.lane_stall;
                lane.last_stall = cycle;
              }
            }
          }
      }
    }
    if (waiting && !computing_any && s.last_stall != cycle) {
      ++s.stall;
      s.last_stall = cycle;
    }
    if (all_finished && s.next_output == s.total_outputs) {
      ++s.op_index;
      if (s.op_index < s.operations.size()) {
        setup_operation(s);
        return;
      }
      if (s.dit) {
        for (U id : s.ids) {
          if (records[id].step != static_cast<int>(s.current_step))
            throw std::runtime_error("Timestep causal order violated");
          ++records[id].step;
        }
        ++s.current_step;
      }
      if (s.dit && s.current_step < s.step_start + s.step_count) {
        s.op_index = 0;
        setup_operation(s);
        return;
      }
      event("task_complete", s);
      s.finished = true;
      s.service_cycles += cycle - s.started;
      ++s.jobs;
    }
  }
  void start_jobs() {
    const auto start = (admission_cursor % replicas) * per_replica;
    for (size_t offset = 0; offset < stages.size(); ++offset) {
      const auto index = (start + offset) % stages.size();
      auto &s = stages[index];
      if (s.active)
        continue;
      if (s.position == 0 && s.queue.empty() &&
          (!baseline || !replica_busy[s.replica]) &&
          free_slots.size() >= (baseline || s.dit
                                    ? batch
                                    : (isolated ? calibration_batch : 1))) {
        for (U n = 0;
             n <
             (baseline || s.dit ? batch : (isolated ? calibration_batch : 1));
             ++n) {
          U id = records.size(), slot = free_slots.front();
          free_slots.pop_front();
          records.push_back(
              {slot, cycle, 0, 0,
               static_cast<int>(isolated && s.dit ? s.step_start : 0)});
          s.queue.push_back(id);
        }
        admission_cursor = (s.replica + 1) % replicas;
        if (baseline)
          replica_busy[s.replica] = true;
      }
      U needed =
          (s.dit || baseline) ? batch : (isolated ? calibration_batch : 1);
      if (s.queue.size() < needed)
        continue;
      s.ids.clear();
      for (U n = 0; n < needed; ++n) {
        s.ids.push_back(s.queue.front());
        s.queue.pop_front();
      }
      s.active = true;
      s.finished = false;
      s.op_index = 0;
      s.current_step = s.step_start;
      s.started = cycle;
      if (s.dit)
        for (U id : s.ids)
          if (records[id].step != static_cast<int>(s.step_start))
            throw std::runtime_error("DiT received an unready request");
      setup_operation(s);
    }
  }
  void route_jobs() {
    for (int index = 0; index < static_cast<int>(stages.size()); ++index) {
      auto &s = stages[index];
      if (!s.active || !s.finished)
        continue;
      if (s.position == per_replica - 1) {
        for (U id : s.ids) {
          if (records[id].step !=
              static_cast<int>(
                  isolated ? (s.dit ? s.step_start + s.step_count : 0) : 10))
            throw std::runtime_error(
                "Request finished with invalid timestep count");
          records[id].completion = cycle;
          free_slots.push_back(records[id].slot);
          ++completed;
        }
        s.active = false;
        if (baseline)
          replica_busy[s.replica] = false;
        continue;
      }
      int destination = index + 1;
      auto &next = stages[destination];
      if (next.queue.size() + next.reserved + s.ids.size() > queue_capacity)
        continue;
      U duration = baseline ? 0
                            : ceil_cycles(static_cast<double>(handoff_bytes) *
                                          s.ids.size() / (handoff_gbps * 1e9));
      event("handoff", s);
      U end = std::max(cycle, handoff_available) + duration;
      handoff_available = end;
      next.reserved += s.ids.size();
      transfers.push_back({end, destination, s.ids});
      s.active = false;
    }
  }

public:
  explicit Engine(ConfigNode config) {
    auto online = config["online"];
    capacity = integer(online, "dram_capacity_bytes", U(16) << 30);
    request_stride = integer(online, "request_stride");
    if (!capacity || capacity > std::numeric_limits<U>::max() / 16)
      throw std::runtime_error("Invalid DRAM capacity");
    const auto execution_mode = online["mode"].as<std::string>("pipeline");
    if (execution_mode != "pipeline" && execution_mode != "baseline" &&
        execution_mode != "isolated")
      throw std::runtime_error("Unknown online execution mode");
    baseline = execution_mode == "baseline";
    isolated = execution_mode == "isolated";
    calibration_batch = integer(online, "calibration_batch", 1);
    replica_stride = integer(online, "activation_replica_stride");
    if (online["event_log"]) {
      events.open(online["event_log"].as<std::string>());
      if (!events)
        throw std::runtime_error("Cannot open event log");
    }
    auto specs = online["stages"].seq();
    replicas = integer(online, "replicas", 1);
    batch = integer(online, "batch", 1);
    max_inflight = integer(online, "max_inflight", 64);
    issue_width = integer(online, "issue_width", 16);
    warmup = integer(online, "warmup_requests", 20);
    measured = integer(online, "measure_requests", 50);
    max_cycles = integer(online, "max_cycles", 100000000000ULL);
    accelerator_ghz = online["accelerator_ghz"].as<double>(1);
    handoff_gbps = online["handoff_gbps"].as<double>(256);
    handoff_bytes = integer(online, "handoff_bytes_per_request", 17842176);
    queue_capacity = integer(online, "queue_capacity", 2 * batch);
    U slots = integer(online, "request_slots", 1024);
    add(warmup, measured);
    if (!replicas || !batch || !max_inflight || !issue_width || !measured ||
        !max_cycles || queue_capacity < batch || slots < batch * replicas ||
        !std::isfinite(accelerator_ghz) || !std::isfinite(handoff_gbps) ||
        accelerator_ghz <= 0 || handoff_gbps <= 0)
      throw std::runtime_error("Invalid online limits");
    replica_busy.resize(replicas, false);
    if (multiply(replica_stride, replicas) > capacity)
      throw std::runtime_error("Replica scratch stride exceeds capacity");
    per_replica = specs.size();
    if (!per_replica)
      throw std::runtime_error("No online stages");
    bool seen_dit = false;
    U steps = 0;
    stages.reserve(replicas * per_replica);
    for (U replica = 0; replica < replicas; ++replica)
      for (int position = 0; position < per_replica; ++position) {
        auto spec = specs[position];
        Stage s;
        s.replica = replica;
        s.position = position;
        s.name = spec["name"].as<std::string>("stage");
        const auto kind = spec["kind"].as<std::string>();
        if (kind != "prefix" && kind != "dit")
          throw std::runtime_error("Unknown stage kind");
        s.dit = kind == "dit";
        s.cores = integer(spec, "cores", 1);
        s.step_start = integer(spec, "timestep_start", 0);
        s.step_count = integer(spec, "timestep_count", 1);
        if (!s.cores)
          throw std::runtime_error("Stage needs at least one core");
        if (replica == 0) {
          if (s.dit) {
            seen_dit = true;
            if (!isolated && s.step_start != steps)
              throw std::runtime_error("Noncontiguous timestep partition");
            steps = s.step_start + s.step_count;
          } else {
            if (seen_dit)
              throw std::runtime_error("Prefix stage follows DiT");
            ++prefix_count;
          }
        }
        const auto operation_rows = spec["operations"];
        for (auto operation : operation_rows.seq()) {
          auto parsed = parse_operation(operation);
          U sram = integer(spec, "lane_sram_bytes", integer(spec, "core_sram_bytes", 1048576));
          U required = add(multiply(2 * parsed.word,
                                    add(multiply(parsed.mtile, parsed.tile),
                                        multiply(parsed.tile, parsed.tile))),
                           multiply(4, multiply(parsed.mtile, parsed.tile)));
          if (required > sram)
            throw std::runtime_error(
                "Double buffers and FP32 accumulation exceed per-core SRAM");
          if (multiply(slots, parsed.stride) > capacity ||
              multiply(batch, parsed.sample_stride) > capacity)
            throw std::runtime_error(
                "Request or batch stride exceeds capacity");
          s.operations.push_back(std::move(parsed));
        }
        if (s.operations.empty())
          throw std::runtime_error("Empty stage program");
        s.lanes.resize(s.cores);
        stages.push_back(std::move(s));
      }
    if ((isolated && (per_replica != 1 || steps > 10 || !calibration_batch)) ||
        (!isolated && (!prefix_count || steps != 10)))
      throw std::runtime_error(
          "Require prefix stages and exactly ten timesteps");
    for (U slot = 0; slot < slots; ++slot)
      free_slots.push_back(slot);
    ConfigNode frontend_config(ConfigNode::Map{
        {"impl", ConfigNode("OnlineFrontend")},
        {"clock_ratio", ConfigNode(1)},
        {"sources", ConfigNode(static_cast<int>(stages.size()))}});
    config.set("frontend", frontend_config);
    frontend.reset(Factory::create_frontend(config));
    memory.reset(Factory::create_memory_system(config));
    frontend->connect_memory_system(memory.get());
    memory->connect_frontend(frontend.get());
    tx = memory->get_tx_bytes();
    tck = memory->get_tCK();
    if (tx == 0 || tck <= 0)
      throw std::runtime_error("Invalid native DRAM clock");
  }
  void run(const std::string &path) {
    bool measuring = warmup == 0;
    while (completed < warmup + measured) {
      if (cycle >= max_cycles)
        throw std::runtime_error("Online simulation exceeded max_cycles");
      issued = 0;
      for (auto it = transfers.begin(); it != transfers.end();) {
        if (it->end <= cycle) {
          auto &dst = stages[it->destination];
          for (U id : it->ids)
            dst.queue.push_back(id);
          dst.reserved -= it->ids.size();
          it = transfers.erase(it);
        } else
          ++it;
      }
      start_jobs();
      // Resolve retirement, operation barriers and dispatch in the same clock.
      // DMA acceptance remains limited by the single global issue budget.
      for (int pass = 0; pass < 3; ++pass)
        for (size_t offset = 0; offset < stages.size(); ++offset) {
          size_t index = (cycle + offset) % stages.size();
          advance_stage(stages[index], index);
        }
      route_jobs();
      if (!measuring && completed >= warmup) {
        measuring = true;
        measurement_cycle = cycle;
        memory->reset_stats_recursive();
        for (auto &s : stages) {
          s.initial_inflight = s.inflight;
          s.read_requests = 0;
          s.write_requests = 0;
          s.completions = 0;
          s.completed_reads = 0;
          s.retries = 0;
          s.read_latency = 0;
          s.stall = 0;
          s.lane_stall = 0;
          s.compute_cycles = 0;
          s.jobs = 0;
          s.service_cycles = 0;
          s.peak_inflight = s.inflight;
        }
      }
      if (completed >= warmup + measured)
        break;
      for (auto &s : stages)
        for (const auto &lane : s.lanes)
          for (const auto &buffer : lane.buffers)
            if (buffer.used && buffer.computing && buffer.end > cycle)
              ++s.compute_cycles;
      ++cycle;
      memory->tick();
    }
    std::vector<const Record *> done;
    for (const auto &record : records)
      if (record.completion)
        done.push_back(&record);
    std::stable_sort(done.begin(), done.end(), [](auto a, auto b) {
      return a->completion < b->completion;
    });
    std::vector<double> latencies;
    for (U n = warmup; n < warmup + measured; ++n)
      latencies.push_back((done[n]->completion - done[n]->admission) * tck *
                          1e-9);
    double mean = 0;
    for (double latency : latencies)
      mean += latency;
    mean /= latencies.size();
    auto sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    double p = .95 * (sorted.size() - 1);
    size_t lower = std::floor(p), upper = std::ceil(p);
    double p95 = sorted[lower] + (sorted[upper] - sorted[lower]) * (p - lower);
    double interval = measured > 1 ? (done[warmup + measured - 1]->completion -
                                      done[warmup]->completion) *
                                         tck * 1e-9 / (measured - 1)
                                   : 0;
    frontend->finalize();
    memory->finalize();
    memory->update_stats_recursive();
    frontend->update_stats_recursive();
    std::ofstream output(path);
    if (!output)
      throw std::runtime_error("Cannot open online report");
    output << std::setprecision(17);
    output
        << "online:\n  mode: "
        << (baseline   ? "baseline"
            : isolated ? "isolated"
                       : "pipeline")
        << "\n  method: single-pass shared-memory closed-loop\n  dram_cycles: "
        << cycle << "\n  measurement_start_cycle: " << measurement_cycle
        << "\n  measurement_dram_cycles: " << (cycle - measurement_cycle)
        << "\n  tCK_ns: " << tck << "\n  admitted_requests: " << records.size()
        << "\n  completed_requests: " << completed
        << "\n  warmup_requests: " << warmup
        << "\n  measured_requests: " << measured
        << "\n  latency_mean_seconds: " << mean
        << "\n  latency_p95_seconds: " << p95
        << "\n  output_interval_seconds: " << interval
        << "\n  max_inflight_per_stage: " << max_inflight << "\n  stages:\n";
    for (size_t index = 0; index < stages.size(); ++index) {
      auto &s = stages[index];
      double bandwidth =
          s.completions * tx / ((cycle - measurement_cycle) * tck);
      output << "    - id: " << index << "\n      replica: " << s.replica
             << "\n      name: " << s.name
             << "\n      peak_inflight: " << s.peak_inflight
             << "\n      reads: " << s.read_requests
             << "\n      writes: " << s.write_requests
             << "\n      completed_transactions: " << s.completions
             << "\n      initial_outstanding_transactions: "
             << s.initial_inflight
             << "\n      outstanding_transactions: " << s.inflight
             << "\n      queue_retries: " << s.retries
             << "\n      avg_read_latency_cycles: "
             << (s.completed_reads ? double(s.read_latency) / s.completed_reads
                                   : 0)
             << "\n      actual_bandwidth_gbps: " << bandwidth
             << "\n      memory_stall_dram_cycles: " << s.stall
             << "\n      memory_stall_lane_dram_cycles: " << s.lane_stall
             << "\n      compute_lane_dram_cycles: " << s.compute_cycles
             << "\n      jobs: " << s.jobs
             << "\n      service_dram_cycles: " << s.service_cycles << "\n";
    }
    output << "  requests:\n";
    for (U n = warmup; n < warmup + measured; ++n) {
      auto *r = done[n];
      output << "    - request_id: " << (r - records.data())
             << "\n      macs: " << r->macs
             << "\n      admission_cycle: " << r->admission
             << "\n      completion_cycle: " << r->completion
             << "\n      timesteps_completed: " << r->step << "\n";
    }
    frontend->print_stats(output);
    memory->print_stats(output);
    if (!output)
      throw std::runtime_error("Online report write failed");
    std::cout << "Online completed " << measured << " measured requests after "
              << warmup << " warmup requests in " << cycle << " DRAM cycles\n";
  }
};
int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("Usage: ramulator2-online CONFIG REPORT.yaml");
    Engine engine(Config::parse_config_file(argv[1]));
    engine.run(argv[2]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ramulator2-online: " << error.what() << '\n';
    return 1;
  }
}
