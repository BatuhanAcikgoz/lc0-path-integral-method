#pragma once

#include "search/path_integral/config.h"
#include "search/path_integral/softmax.h"
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
  
  // Configuration management
  void UpdateOptions(const OptionsDict& options);
  bool IsEnabled() const { return config_.enabled; }
  
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
  
  // Configuration and components
  PathIntegralConfig config_;
  std::unique_ptr<SoftmaxCalculator> softmax_calculator_;
  Backend* backend_;  // Neural network backend for move evaluation
};

} // namespace lczero