#pragma once

#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include "chess/board.h"
#include "chess/position.h"
#include "chess/types.h"
#include "search/path_integral/config.h"

namespace lczero {

// Forward declarations are now replaced with actual includes

// Search limits structure for Path Integral sampling
struct SearchLimits {
  int nodes = -1;           // Maximum nodes to search (-1 = unlimited)
  int time_ms = -1;         // Maximum time in milliseconds (-1 = unlimited)  
  int depth = -1;           // Maximum depth (-1 = unlimited)
  int64_t visits = -1;      // Maximum visits (-1 = unlimited)
  int64_t playouts = -1;    // Maximum playouts (-1 = unlimited)
  
  SearchLimits() = default;
};

// MCTS state snapshot for preservation during sampling
struct MCTSStateSnapshot {
  // Placeholder for MCTS state data
  // In actual implementation, this would contain:
  // - Search tree nodes and statistics
  // - Cached neural network evaluations
  // - Search parameters and state
  
  bool is_valid = false;
  std::string state_info; // For debugging
  
  MCTSStateSnapshot() = default;
};

// Base sample structure
struct PathIntegralSample {
  std::vector<Move> path;
  float probability = 0.0f;
  float evaluation = 0.0f;
  std::chrono::microseconds computation_time{0};
  std::string metadata;
  
  PathIntegralSample() = default;
  virtual ~PathIntegralSample() = default;
};

// Competitive mode sample with search-specific data
struct CompetitiveSample : public PathIntegralSample {
  float search_score = 0.0f;
  int nodes_searched = 0;
  float engine_temperature = 0.0f;
  Move sampled_move; // The move that was sampled for this competitive sample
};

// Quantum limit mode sample with neural network data
struct QuantumSample : public PathIntegralSample {
  std::vector<float> policy_probs;
  float value_head_score = 0.0f;
  float cp_score = 0.0f;
  PathIntegralRewardMode reward_applied = PathIntegralRewardMode::kHybrid;
};

// Results container for sampling operations
struct SamplingResults {
  std::vector<std::unique_ptr<PathIntegralSample>> samples;
  Move selected_move;
  float total_computation_time_ms = 0.0f;
  int successful_samples = 0;
  int failed_samples = 0;
  std::string error_message;
  
  bool IsValid() const {
    return !samples.empty() && !selected_move.is_null();
  }
};

// Interface for softmax calculation
class SoftmaxCalculatorInterface {
public:
  virtual ~SoftmaxCalculatorInterface() = default;
  
  // Main softmax calculation using log-sum-exp
  virtual std::vector<float> CalculateSoftmax(const std::vector<float>& scores, 
                                              float lambda) = 0;
  
  // Fallback for invalid inputs
  virtual std::vector<float> UniformProbabilities(size_t count) = 0;
  
  // Validation helpers
  virtual bool IsValidInput(const std::vector<float>& scores) = 0;
};

// Interface for sampling engine
class SamplingEngineInterface {
public:
  virtual ~SamplingEngineInterface() = default;
  
  // Generate samples based on configuration
  virtual SamplingResults GenerateSamples(const Position& position,
                                           const PathIntegralConfig& config,
                                           const SearchLimits& limits) = 0;
  
  // Adaptive depth to nodes conversion
  virtual int ConvertDepthToNodes(const Position& position, int target_depth) = 0;
};

// Interface for mode handlers
class ModeHandlerInterface {
public:
  virtual ~ModeHandlerInterface() = default;
  
  // Handle sampling for specific mode
  virtual SamplingResults HandleSampling(const Position& position,
                                         const PathIntegralConfig& config,
                                         const SearchLimits& limits) = 0;
  
  // Check if mode is supported
  virtual bool IsSupported(PathIntegralSamplingMode mode) const = 0;
};

// Interface for result export
class ResultExporterInterface {
public:
  virtual ~ResultExporterInterface() = default;
  
  // Export results in specified format
  virtual bool ExportResults(const SamplingResults& results,
                             const std::string& format,
                             const std::string& filename = "") = 0;
  
  // Check if format is supported
  virtual bool SupportsFormat(const std::string& format) const = 0;
};

}  // namespace lczero