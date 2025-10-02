#include "search/path_integral/config.h"
#include <algorithm>

namespace lczero {

bool PathIntegralConfig::IsValid() const {
  return lambda >= 0.001f && lambda <= 10.0f &&
         samples >= 1 && samples <= 100000;
}

void PathIntegralConfig::SetDefaults() {
  lambda = 0.1f;
  samples = 50;
  reward_mode = PathIntegralRewardMode::kHybrid;
  sampling_mode = PathIntegralSamplingMode::kCompetitive;
  enabled = false;
  debug_logging = false;
  export_format = "none";
}

PathIntegralRewardMode PathIntegralConfig::ParseRewardMode(const std::string& mode_str) {
  if (mode_str == "policy") return PathIntegralRewardMode::kPolicy;
  if (mode_str == "cp_score") return PathIntegralRewardMode::kCpScore;
  if (mode_str == "hybrid") return PathIntegralRewardMode::kHybrid;
  return PathIntegralRewardMode::kHybrid; // default
}

PathIntegralSamplingMode PathIntegralConfig::ParseSamplingMode(const std::string& mode_str) {
  if (mode_str == "competitive") return PathIntegralSamplingMode::kCompetitive;
  if (mode_str == "quantum_limit") return PathIntegralSamplingMode::kQuantumLimit;
  return PathIntegralSamplingMode::kCompetitive; // default
}

std::string PathIntegralConfig::RewardModeToString(PathIntegralRewardMode mode) {
  switch (mode) {
    case PathIntegralRewardMode::kPolicy: return "policy";
    case PathIntegralRewardMode::kCpScore: return "cp_score";
    case PathIntegralRewardMode::kHybrid: return "hybrid";
    default: return "hybrid";
  }
}

std::string PathIntegralConfig::SamplingModeToString(PathIntegralSamplingMode mode) {
  switch (mode) {
    case PathIntegralSamplingMode::kCompetitive: return "competitive";
    case PathIntegralSamplingMode::kQuantumLimit: return "quantum_limit";
    default: return "competitive";
  }
}

} // namespace lczero