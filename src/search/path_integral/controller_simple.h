#pragma once

#include "search/path_integral/config.h"
#include "search/path_integral/softmax.h"
#include "search/path_integral/performance_monitor.h"
#include "utils/optionsdict.h"
#include "chess/position.h"
#include "chess/types.h"
#include "chess/board.h"

namespace lczero {

// Forward declarations
struct SearchLimits;
class Backend;

// Simple Path Integral controller for basic functionality
class SimplePathIntegralController {
public:
  explicit SimplePathIntegralController(const OptionsDict& options, Backend* backend = nullptr);
  ~SimplePathIntegralController();
  
  // Main interface - returns selected move or null move if disabled/failed
  Move SelectMove(const Position& position, const SearchLimits& limits);
  
  // Overload for direct move selection from MCTS data
  Move SelectMove(const std::vector<Move>& legal_moves, 
                  const std::vector<float>& move_scores, 
                  const Position& position);
  
  // Configuration management
  void UpdateOptions(const OptionsDict& options);
  void SetConfig(const PathIntegralConfig& config);
  const PathIntegralConfig& GetConfig() const { return config_; }
  bool IsEnabled() const { return config_.enabled; }
  
  // Performance monitoring access
  PathIntegralPerformanceMonitor::SamplingMetrics GetLastSamplingMetrics() const;
  void ExportPerformanceMetrics(const std::string& filename) const;
  
private:
  // Update configuration from UCI options
  void UpdateConfigFromOptions(const OptionsDict& options);
  
  // Mode implementations
  Move HandleCompetitiveMode(const Position& position, const SearchLimits& limits);
  Move HandleQuantumLimitMode(const Position& position, const SearchLimits& limits);
  
  // Sampling functionality
  struct SampleResult {
    Move move;
    float score;
    float probability;
  };
  
  std::vector<SampleResult> PerformRootNodeSampling(const Position& position, const MoveList& legal_moves);
  std::vector<SampleResult> PerformQuantumLimitSampling(const Position& position, const MoveList& legal_moves);
  Move SelectMoveFromSampling(const std::vector<SampleResult>& results, const MoveList& legal_moves);
  float EvaluateMove(const Position& position, const Move& move);
  float EvaluateMovePolicy(const Position& position, const Move& move);
  
  // Sample count verification
  bool ValidateSampleCountIntegrity(int requested_samples, int legal_move_count) const;
  
  // Backend availability verification
  bool VerifyBackendAvailability() const;
  
  // Configuration and components
  PathIntegralConfig config_;
  std::unique_ptr<SoftmaxCalculator> softmax_calculator_;
  std::unique_ptr<PathIntegralPerformanceMonitor> performance_monitor_;
  Backend* backend_;  // Neural network backend for move evaluation
};

} // namespace lczero