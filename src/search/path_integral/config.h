#pragma once

#include <string>

namespace lczero {

// Enum for Path Integral reward modes
enum class PathIntegralRewardMode {
  kPolicy,
  kCpScore,
  kHybrid
};

// Enum for Path Integral sampling modes
enum class PathIntegralSamplingMode {
  kCompetitive,
  kQuantumLimit
};

// Configuration structure for Path Integral sampling
struct PathIntegralConfig {
  // UCI option: PathIntegralLambda (0.001-10.0, default 0.1)
  float lambda = 0.1f;
  
  // UCI option: PathIntegralSamples (1-100000, default 50)
  int samples = 50;
  
  // UCI option: PathIntegralRewardMode (policy/cp_score/hybrid, default hybrid)
  PathIntegralRewardMode reward_mode = PathIntegralRewardMode::kHybrid;
  
  // UCI option: PathIntegralMode (competitive/quantum_limit, default competitive)
  PathIntegralSamplingMode sampling_mode = PathIntegralSamplingMode::kCompetitive;
  
  // Internal flags
  bool enabled = false;
  bool debug_logging = false;
  
  // Export configuration
  std::string export_format = "none";  // "json", "csv", "none"
  
  // Validation methods
  bool IsValid() const;
  void SetDefaults();
  
  // String conversion helpers for UCI
  static PathIntegralRewardMode ParseRewardMode(const std::string& mode_str);
  static PathIntegralSamplingMode ParseSamplingMode(const std::string& mode_str);
  static std::string RewardModeToString(PathIntegralRewardMode mode);
  static std::string SamplingModeToString(PathIntegralSamplingMode mode);
};

}  // namespace lczero