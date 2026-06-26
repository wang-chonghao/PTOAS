// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/VFcostmodel/VfSimulator/ParamDB.h"

#include "PTO/VFcostmodel/VfSimulator/Json.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <utility>

namespace vfsim {
namespace {

using JsonValue = json::Value;

const JsonValue *findKey(const JsonValue::Object &object, const std::string &key) {
  auto it = object.find(key);
  if (it == object.end())
    return nullptr;
  return &it->second;
}

std::string readStringField(const JsonValue::Object &object, const char *key,
                            const std::string &defaultValue = {}) {
  const JsonValue *value = findKey(object, key);
  return value ? value->asString(defaultValue) : defaultValue;
}

std::string makeFormKey(const std::string &op, const std::string &form) {
  return op + "." + form;
}

int64_t readIntField(const JsonValue::Object &object, const char *key,
                     int64_t defaultValue = 0) {
  const JsonValue *value = findKey(object, key);
  return value ? value->asInt(defaultValue) : defaultValue;
}

bool readBoolField(const JsonValue::Object &object, const char *key,
                   bool defaultValue = false) {
  const JsonValue *value = findKey(object, key);
  return value ? value->asBool(defaultValue) : defaultValue;
}

std::vector<std::string> readStringArrayField(const JsonValue::Object &object,
                                              const char *key) {
  std::vector<std::string> out;
  const JsonValue *value = findKey(object, key);
  if (!value || !value->isArray())
    return out;
  for (const JsonValue &item : value->asArray())
    out.push_back(item.asString());
  return out;
}

void overlayInstConfig(InstConfig &cfg, const JsonValue::Object &object) {
  if (findKey(object, "pipeline_startup_cost"))
    cfg.pipelineStartupCost = readIntField(object, "pipeline_startup_cost");
  if (findKey(object, "latency"))
    cfg.latency = readIntField(object, "latency");
  if (findKey(object, "throughput"))
    cfg.throughput = readIntField(object, "throughput");
  if (findKey(object, "pipeline_drain_cost"))
    cfg.pipelineDrainCost = readIntField(object, "pipeline_drain_cost");
  if (findKey(object, "data_load_cost"))
    cfg.dataLoadCost = readIntField(object, "data_load_cost");
  if (findKey(object, "data_store_cost"))
    cfg.dataStoreCost = readIntField(object, "data_store_cost");
  if (findKey(object, "EXU"))
    cfg.exu = readStringField(object, "EXU");
  if (findKey(object, "dispatch_exu"))
    cfg.dispatchExu = readStringField(object, "dispatch_exu");
  if (findKey(object, "op_class"))
    cfg.opClass = readStringField(object, "op_class");
  if (cfg.opClass.empty())
    cfg.opClass = readStringField(object, "class", readStringField(object, "category", cfg.opClass));
  if (findKey(object, "dtype"))
    cfg.dtype = readStringField(object, "dtype");
  if (findKey(object, "src_dtypes"))
    cfg.srcDtypes = readStringArrayField(object, "src_dtypes");
  if (findKey(object, "dst_dtypes"))
    cfg.dstDtypes = readStringArrayField(object, "dst_dtypes");
}

InstConfig readInstConfig(const JsonValue::Object &object) {
  InstConfig cfg;
  overlayInstConfig(cfg, object);
  return cfg;
}

InstConfig readInstFormConfig(const JsonValue::Object &opObject,
                              const std::string &formName,
                              const JsonValue::Object &formObject) {
  InstConfig cfg;
  overlayInstConfig(cfg, opObject);
  overlayInstConfig(cfg, formObject);
  cfg.form = formName;
  if (cfg.dtype.empty())
    cfg.dtype = formName;
  return cfg;
}

std::unordered_map<std::string, JsonValue> loadObjectFile(const std::filesystem::path &path) {
  JsonValue value = json::parseFile(path);
  if (!value.isObject())
    throw std::runtime_error("JSON root must be an object: " + path.string());
  return value.asObject();
}

std::filesystem::path pickPath(std::filesystem::path baseDir, const char *envKey,
                               std::initializer_list<std::filesystem::path> candidates) {
  if (const char *envPath = std::getenv(envKey); envPath && *envPath) {
    std::filesystem::path path(envPath);
    if (!std::filesystem::exists(path))
      throw std::runtime_error(std::string(envKey) + ": path not found: " + path.string());
    return std::filesystem::absolute(path);
  }

  for (const auto &candidate : candidates) {
    std::filesystem::path path = baseDir / candidate;
    if (std::filesystem::exists(path))
      return std::filesystem::absolute(path);
  }

  std::string message = std::string("Could not locate ") + envKey + ". Tried:";
  for (const auto &candidate : candidates)
    message += "\n  " + (baseDir / candidate).string();
  throw std::runtime_error(message);
}

std::optional<std::filesystem::path>
pickPathOptional(std::filesystem::path baseDir, const char *envKey,
                 std::initializer_list<std::filesystem::path> candidates) {
  if (const char *envPath = std::getenv(envKey); envPath && *envPath) {
    std::filesystem::path path(envPath);
    if (!std::filesystem::exists(path))
      throw std::runtime_error(std::string(envKey) + ": path not found: " + path.string());
    return std::filesystem::absolute(path);
  }

  for (const auto &candidate : candidates) {
    std::filesystem::path path = baseDir / candidate;
    if (std::filesystem::exists(path))
      return std::filesystem::absolute(path);
  }
  return std::nullopt;
}

} // namespace

ParamDB::ParamDB(std::filesystem::path baseDir)
    : baseDir_(resolveBaseDir(std::move(baseDir))) {
  const std::filesystem::path isaPath =
      pickPath(baseDir_, "ISA_JSON_PATH", {"configs/isa.json", "isa.json"});
  const std::filesystem::path uarchPath =
      pickPath(baseDir_, "UARCH_JSON_PATH", {"configs/uarch.json", "uarch.json"});
  const std::optional<std::filesystem::path> forwardingPath = pickPathOptional(
      baseDir_, "FORWARDING_JSON_PATH", {"configs/forwarding.json", "forwarding.json"});
  const std::optional<std::filesystem::path> iiPath = pickPathOptional(
      baseDir_, "II_JSON_PATH",
      {"configs/InitiationInterval.json", "Initiation_Interval.json"});

  const auto isaRoot = loadObjectFile(isaPath);
  const auto uarchRoot = loadObjectFile(uarchPath);
  const auto fwdRoot = forwardingPath ? loadObjectFile(*forwardingPath)
                                      : std::unordered_map<std::string, JsonValue>{};
  const auto iiRoot = iiPath ? loadObjectFile(*iiPath)
                             : std::unordered_map<std::string, JsonValue>{};

  if (const JsonValue *defaults = findKey(isaRoot, "defaults")) {
    if (!defaults->isObject())
      throw std::runtime_error("isa.json.defaults must be an object");
    const auto &obj = defaults->asObject();
    bundle_.isaDefaults.vfStartupCost = readIntField(obj, "vf_startup_cost");
    bundle_.isaDefaults.vfDrainCost = readIntField(obj, "vf_drain_cost");
  }

  if (const JsonValue *instructions = findKey(isaRoot, "instructions")) {
    if (!instructions->isObject())
      throw std::runtime_error("isa.json.instructions must be an object");
    for (const auto &[opName, opValue] : instructions->asObject()) {
      if (!opValue.isObject())
        throw std::runtime_error("isa.json.instructions." + opName +
                                 " must be an object");
      auto &dtypeMap = bundle_.isa[opName];
      const auto &opObject = opValue.asObject();
      if (const JsonValue *forms = findKey(opObject, "forms")) {
        if (!forms->isObject())
          throw std::runtime_error("isa.json.instructions." + opName +
                                   ".forms must be an object");
        for (const auto &[formName, formValue] : forms->asObject()) {
          if (!formValue.isObject())
            throw std::runtime_error("isa.json.instructions." + opName +
                                     ".forms." + formName +
                                     " must be an object");
          dtypeMap.emplace(formName,
                           readInstFormConfig(opObject, formName,
                                              formValue.asObject()));
        }
      } else {
        for (const auto &[dtypeName, dtypeValue] : opObject) {
          if (!dtypeValue.isObject())
            throw std::runtime_error("isa.json.instructions." + opName + "." +
                                     dtypeName + " must be an object");
          InstConfig cfg = readInstConfig(dtypeValue.asObject());
          cfg.form = dtypeName;
          if (cfg.dtype.empty())
            cfg.dtype = dtypeName;
          dtypeMap.emplace(dtypeName, std::move(cfg));
        }
      }
    }
  }

  if (!uarchRoot.empty()) {
    const auto &obj = uarchRoot;
    bundle_.uarch.issuePorts = readIntField(obj, "issue_ports");
    bundle_.uarch.loadPorts = readIntField(obj, "load_ports");
    bundle_.uarch.storePorts = readIntField(obj, "store_ports");
    bundle_.uarch.iduWindowWidth = readIntField(obj, "IDU_window_width");
    bundle_.uarch.iduIssueWidth = readIntField(obj, "IDU_issue_width");
    bundle_.uarch.ldqWidth = readIntField(obj, "LDQ_width");
    bundle_.uarch.vregNum = readIntField(obj, "vreg_num");
    bundle_.uarch.enableIsuQueueModel = readBoolField(obj, "enable_isu_queue_model");
    bundle_.uarch.shqDepth = readIntField(obj, "shq_depth");
    bundle_.uarch.exqDepth = readIntField(obj, "exq_depth");
    bundle_.uarch.admitBlockedToExq = readBoolField(obj, "admit_blocked_to_exq");
    bundle_.uarch.enableShqCreditModel = readBoolField(obj, "enable_shq_credit_model");
    bundle_.uarch.shqReleaseDelay = readIntField(obj, "shq_release_delay");
    bundle_.uarch.enableCreditVisibilityDelay =
        readBoolField(obj, "enable_credit_visibility_delay");
    bundle_.uarch.iduVisiblePregDelay = readIntField(obj, "idu_visible_preg_delay");
    bundle_.uarch.iduVisibleShqDelay = readIntField(obj, "idu_visible_shq_delay");
    bundle_.uarch.globalShqPregGate = readBoolField(obj, "global_shq_preg_gate");
    bundle_.uarch.useExplicitIduCreditBank = readBoolField(obj, "use_explicit_idu_credit_bank");
    bundle_.uarch.iduToOooDelay = readIntField(obj, "idu_to_ooo_delay");
    bundle_.uarch.vloopToDispatchDelay = readIntField(obj, "vloop_to_dispatch_delay");
    bundle_.uarch.iduDispatchStartAdvance = readIntField(obj, "idu_dispatch_start_advance");
    bundle_.uarch.initialTopBlockVloopStartCycle =
        readIntField(obj, "initial_top_block_vloop_start_cycle");
    bundle_.uarch.nestedVloopInitialStartGap = readIntField(obj, "nested_vloop_initial_start_gap");
    bundle_.uarch.loop1MinFeedbackGap = readIntField(obj, "loop1_min_feedback_gap");
    bundle_.uarch.innermostIterDispatchStride =
        readIntField(obj, "innermost_iter_dispatch_stride");
    bundle_.uarch.consumerReleaseStartOffset = readIntField(obj, "consumer_release_start_offset");
    bundle_.uarch.loadDoneLatency = readIntField(obj, "load_done_latency");
    bundle_.uarch.oooToShqDelay = readIntField(obj, "ooo_to_shq_delay");
    bundle_.uarch.oooToLsqDelay = readIntField(obj, "ooo_to_lsq_delay");
    bundle_.uarch.exqRecvDelay = readIntField(obj, "exq_recv_delay");
    bundle_.uarch.shqToExqPortPerCycle = readIntField(obj, "shq_to_exq_port_per_cycle");
    bundle_.uarch.computeInflightCap = readIntField(obj, "compute_inflight_cap");
    bundle_.uarch.exqIssueInflightCapPerPort =
        readIntField(obj, "exq_issue_inflight_cap_per_port");
    bundle_.uarch.exqCapacityCountsInflight =
        readBoolField(obj, "exq_capacity_counts_inflight");
    bundle_.uarch.memBarMode = readStringField(obj, "mem_bar_mode");
    bundle_.uarch.enforceSameCycleSrcHazard =
        readBoolField(obj, "enforce_same_cycle_src_hazard");
    bundle_.uarch.enableCrossFuIi = readBoolField(obj, "enable_cross_fu_ii");
  }

  if (const JsonValue *forwarding = findKey(fwdRoot, "forwarding")) {
    if (!forwarding->isObject())
      throw std::runtime_error("forwarding.json.forwarding must be an object");
    const bool v2Forwarding =
        findKey(fwdRoot, "schema_version") &&
        findKey(fwdRoot, "schema_version")->asInt(1) >= 2;
    if (v2Forwarding) {
      auto &prodMap = bundle_.forwarding["__forms"];
      for (const auto &[prodKey, prodValue] : forwarding->asObject()) {
        if (!prodValue.isObject())
          throw std::runtime_error("forwarding.json.forwarding." + prodKey +
                                   " must be an object");
        auto &consMap = prodMap[prodKey];
        for (const auto &[consKey, consValue] : prodValue.asObject())
          consMap.emplace(consKey, consValue.asInt());
      }
    } else {
      for (const auto &[dtypeName, dtypeValue] : forwarding->asObject()) {
        if (!dtypeValue.isObject())
          throw std::runtime_error("forwarding.json.forwarding." + dtypeName +
                                   " must be an object");
        auto &prodMap = bundle_.forwarding[dtypeName];
        for (const auto &[prodName, prodValue] : dtypeValue.asObject()) {
          if (!prodValue.isObject())
            throw std::runtime_error("forwarding.json.forwarding." + dtypeName +
                                     "." + prodName + " must be an object");
          auto &consMap = prodMap[prodName];
          for (const auto &[consName, consValue] : prodValue.asObject())
            consMap.emplace(consName, consValue.asInt());
        }
      }
    }
  }

  if (const JsonValue *ii = findKey(iiRoot, "InitiationInterval")) {
    if (!ii->isObject())
      throw std::runtime_error("InitiationInterval.json.InitiationInterval must be an object");
    const bool v2Ii =
        findKey(iiRoot, "schema_version") &&
        findKey(iiRoot, "schema_version")->asInt(1) >= 2;
    if (v2Ii) {
      auto &prevMap = bundle_.initiationInterval["__forms"];
      for (const auto &[prevKey, prevValue] : ii->asObject()) {
        if (!prevValue.isObject())
          throw std::runtime_error("InitiationInterval.json.InitiationInterval." +
                                   prevKey + " must be an object");
        auto &curMap = prevMap[prevKey];
        for (const auto &[curKey, curValue] : prevValue.asObject())
          curMap.emplace(curKey, curValue.asInt());
      }
    } else {
      for (const auto &[dtypeName, dtypeValue] : ii->asObject()) {
        if (!dtypeValue.isObject())
          throw std::runtime_error("InitiationInterval.json.InitiationInterval." +
                                   dtypeName + " must be an object");
        auto &prevMap = bundle_.initiationInterval[dtypeName];
        for (const auto &[prevName, prevValue] : dtypeValue.asObject()) {
          if (!prevValue.isObject())
            throw std::runtime_error("InitiationInterval.json.InitiationInterval." +
                                     dtypeName + "." + prevName + " must be an object");
          auto &curMap = prevMap[prevName];
          for (const auto &[curName, curValue] : prevValue.asObject())
            curMap.emplace(curName, curValue.asInt());
        }
      }
    }
  }
}

std::filesystem::path ParamDB::resolveBaseDir(std::filesystem::path baseDir) {
  if (baseDir.empty())
    return std::filesystem::absolute(std::filesystem::current_path());
  return std::filesystem::absolute(std::move(baseDir));
}

bool ParamDB::hasInst(const std::string &op, const std::string &dtype) const {
  return hasInstForm(op, dtype);
}

const InstConfig &ParamDB::inst(const std::string &op, const std::string &dtype) const {
  return instForm(op, dtype);
}

bool ParamDB::hasInstForm(const std::string &op, const std::string &form) const {
  const auto opIt = bundle_.isa.find(op);
  if (opIt == bundle_.isa.end())
    return false;
  return opIt->second.find(form) != opIt->second.end();
}

const InstConfig &ParamDB::instForm(const std::string &op, const std::string &form) const {
  const auto opIt = bundle_.isa.find(op);
  if (opIt == bundle_.isa.end())
    throw std::runtime_error("Instruction not found: op=" + op + ", form=" + form);
  const auto formIt = opIt->second.find(form);
  if (formIt != opIt->second.end())
    return formIt->second;
  if (form != "fp32") {
    const auto fp32It = opIt->second.find("fp32");
    if (fp32It != opIt->second.end())
      return fp32It->second;
  }
  throw std::runtime_error("Instruction not found: op=" + op + ", form=" + form);
}

int64_t ParamDB::forwardingCycles(const std::string &prodForm,
                                  const std::string &prod,
                                  const std::string &consForm,
                                  const std::string &cons) const {
  const auto formsIt = bundle_.forwarding.find("__forms");
  if (formsIt != bundle_.forwarding.end()) {
    const auto prodIt = formsIt->second.find(makeFormKey(prod, prodForm));
    if (prodIt != formsIt->second.end()) {
      const auto consIt = prodIt->second.find(makeFormKey(cons, consForm));
      if (consIt != prodIt->second.end())
        return std::max<int64_t>(0, consIt->second);
    }
  }
  const auto dtypeIt = bundle_.forwarding.find(prodForm);
  if (dtypeIt != bundle_.forwarding.end()) {
    const auto prodIt = dtypeIt->second.find(prod);
    if (prodIt != dtypeIt->second.end()) {
      const auto consIt = prodIt->second.find(cons);
      if (consIt != prodIt->second.end())
        return std::max<int64_t>(0, consIt->second);
    }
  }
  const int64_t latency = hasInstForm(prod, prodForm) ? instForm(prod, prodForm).latency : 0;
  return std::max<int64_t>(0, latency - 3);
}

int64_t ParamDB::initiationInterval(const std::string &prevForm,
                                    const std::string &prev,
                                    const std::string &curForm,
                                    const std::string &cur) const {
  const auto formsIt = bundle_.initiationInterval.find("__forms");
  if (formsIt != bundle_.initiationInterval.end()) {
    const auto prevIt = formsIt->second.find(makeFormKey(prev, prevForm));
    if (prevIt != formsIt->second.end()) {
      const auto curIt = prevIt->second.find(makeFormKey(cur, curForm));
      if (curIt != prevIt->second.end())
        return std::max<int64_t>(1, curIt->second);
    }
  }
  const auto dtypeIt = bundle_.initiationInterval.find(curForm);
  if (dtypeIt != bundle_.initiationInterval.end()) {
    const auto prevIt = dtypeIt->second.find(prev);
    if (prevIt != dtypeIt->second.end()) {
      const auto curIt = prevIt->second.find(cur);
      if (curIt != prevIt->second.end())
        return std::max<int64_t>(1, curIt->second);
    }
  }
  return 1;
}

} // namespace vfsim
