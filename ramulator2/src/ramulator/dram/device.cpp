#include "ramulator/dram/device.h"
#include <algorithm>

namespace Ramulator {

void DRAMDevice::init(std::unique_ptr<DRAMSpec> spec) {
  m_spec_owner = std::move(spec);
  m_spec = m_spec_owner.get();
  m_bank_level = m_spec->get_level_id("Bank");
  m_row_level = m_spec->get_level_id("Row");
  m_root = std::make_unique<DRAMNode>(m_spec, nullptr, 0, 0);
  m_root->for_each_at_level(m_bank_level, [&](DRAMNode* bank) { m_bank_nodes.push_back(bank); });
  m_prerequisite_cache.resize(m_bank_nodes.size());
  m_timing_cache.resize(m_bank_nodes.size() * m_spec->command_count);
}

void DRAMDevice::set_channel_id(int channel_id) {
  m_root->m_node_id = channel_id;
}

void DRAMDevice::issue_command(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  // Every issued command can change rank/channel/sibling constraints, including
  // commands with no bank-state action. Invalidate all cached timing paths.
  ++m_timing_generation;
  m_root->update_timing(command, addr_vec, clk);
  if (m_spec->bank_targets[command] == BankTarget::Pair) {
    // Partner path from the pair level down; the shared nodes above it are updated once
    AddrVec_t partner = partner_addr_vec(command, addr_vec);
    node_at(partner, m_spec->pair_rules[command].level)->update_timing(command, partner, clk);
  }
  apply_action(command, addr_vec, clk);
}

bool DRAMDevice::check_timing(int command, const AddrVec_t& addr_vec, Clk_t clk,
                            int flat_bank_id) {
  // A concrete single-bank leaf address follows exactly one root-to-bank path.
  // Its timing is the maximum absolute ready clock on that path. Between DRAM
  // commands only clk changes, so comparing against that maximum is exact even
  // for future-clock lookahead queries. Wildcards, bank pairs and subarrays use
  // the original traversal below.
  if (flat_bank_id >= 0 && m_spec->bank_targets[command] == BankTarget::Single &&
      m_bank_nodes[flat_bank_id]->m_child_nodes.empty()) {
    auto& cached = m_timing_cache[flat_bank_id * m_spec->command_count + command];
    if (cached.generation != m_timing_generation) {
      cached.ready = -1;
      for (auto* node = m_bank_nodes[flat_bank_id]; node; node = node->m_parent_node) {
        cached.ready = std::max(cached.ready, node->m_cmd_ready_clk[command]);
      }
      cached.generation = m_timing_generation;
    }
    return cached.ready == -1 || clk >= cached.ready;
  }
  if (!m_root->check_timing(command, addr_vec, clk)) {
    return false;
  }
  return m_spec->bank_targets[command] != BankTarget::Pair ||
         m_root->check_timing(command, partner_addr_vec(command, addr_vec), clk);
}

int DRAMDevice::get_preq_command(int command, const AddrVec_t& addr_vec, Clk_t clk,
                               int flat_bank_id) {
  auto preq_fn = m_spec->funcs.preqs[command];
  if (!preq_fn) return command;

  // RD/WR prerequisites only change after an action on their bank. Keep timing
  // checks and scheduler eligibility checks live on every cycle; only reuse the
  // state-derived command, with exact bank/row/final-command identity.
  if (m_spec->command_meta[command].preq_bank_row_only &&
      m_spec->bank_targets[command] == BankTarget::Single) {
    const int bank_id = flat_bank_id >= 0 ? flat_bank_id : get_flat_bank_id(addr_vec);
    auto& cached = m_prerequisite_cache[bank_id];
    const int row = addr_vec[m_row_level];
    if (cached.command != command || cached.row != row) {
      cached = {command, row, preq_fn(m_bank_nodes[bank_id], command, addr_vec, clk)};
    }
    return cached.resolved;
  }

  int resolved = command;
  for_each_target_bank_while(command, addr_vec, [&](int flat_bank_id) {
    int preq = preq_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
    if (preq != command) { resolved = preq; return false; }
    return true;
  });
  return resolved;
}

AddrVec_t DRAMDevice::preq_addr_vec(int command, int preq, const AddrVec_t& addr_vec, Clk_t clk) const {
  if (preq == command || m_spec->bank_targets[preq] != BankTarget::Single) {
    return addr_vec;
  }
  AddrVec_t preq_addr = addr_vec;
  for_each_target_bank_while(command, addr_vec, [&](int bank_id) {
    DRAMNode* bank = m_bank_nodes[bank_id];
    if (m_spec->funcs.preqs[command](bank, command, addr_vec, clk) == command) {
      return true;
    }
    for (DRAMNode* n = bank; n != nullptr; n = n->m_parent_node) {
      preq_addr[n->m_level] = n->m_node_id;
    }
    return false;
  });
  return preq_addr;
}

bool DRAMDevice::check_rowbuffer_hit(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto rowhit_fn = m_spec->funcs.rowhits[command];
  if (!rowhit_fn) {
    return false;
  }
  int flat_bank_id = get_flat_bank_id(addr_vec);
  return rowhit_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
}

bool DRAMDevice::check_node_open(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto rowopen_fn = m_spec->funcs.rowopens[command];
  if (!rowopen_fn) {
    return false;
  }
  int flat_bank_id = get_flat_bank_id(addr_vec);
  return rowopen_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
}

int DRAMDevice::get_flat_bank_id(const AddrVec_t& addr_vec) const {
  int id = 0;
  for (int lvl = 1; lvl <= m_bank_level; lvl++) {
    id = id * m_spec->organization.level_sizes[lvl] + addr_vec[lvl];
  }
  return id;
}

bool DRAMDevice::bank_matches(DRAMNode* bank, const AddrVec_t& addr_vec) {
  for (auto* n = bank; n != nullptr; n = n->m_parent_node) {
    if (addr_vec[n->m_level] != -1 && addr_vec[n->m_level] != n->m_node_id) {
      return false;
    }
  }
  return true;
}

std::vector<int> DRAMDevice::get_target_banks(int command, const AddrVec_t& addr_vec) const {
  std::vector<int> ids;
  for_each_target_bank(command, addr_vec, [&](int id) { ids.push_back(id); });
  return ids;
}

AddrVec_t DRAMDevice::partner_addr_vec(int command, const AddrVec_t& addr_vec) const {
  const PairRule& rule = m_spec->pair_rules[command];
  AddrVec_t partner = addr_vec;
  partner[rule.level] ^= rule.xor_mask;
  return partner;
}

DRAMNode* DRAMDevice::node_at(const AddrVec_t& addr_vec, int level) const {
  DRAMNode* node = m_root.get();
  for (int lvl = 1; lvl <= level; lvl++) {
    node = node->m_child_nodes[addr_vec[lvl]].get();
  }
  return node;
}

void DRAMDevice::apply_action(int command, const AddrVec_t& addr_vec, Clk_t clk) {
  auto action_fn = m_spec->funcs.actions[command];
  if (!action_fn) return;
  for_each_target_bank(command, addr_vec, [&](int flat_bank_id) {
    action_fn(m_bank_nodes[flat_bank_id], command, addr_vec, clk);
    // Includes refresh, split activation, precharge and auto-precharge actions.
    m_prerequisite_cache[flat_bank_id].command = -1;
  });
}

}  // namespace Ramulator
