#include "search/path_integral/controller_simple.h"
#include "utils/logging.h"
#include "chess/position.h"
#include "utils/optionsparser.h"
#include <random>
#include <cmath>

#include "search/path_integral/options.h"

namespace lczero {


SimplePathIntegralController::SimplePathIntegralController(const OptionsDict& options) {
  softmax_calculator_ = std::make_unique<SoftmaxCalculator>();
  UpdateConfigFromOptions(options);
}

SimplePathIntegralController::~SimplePathIntegralController() = default;

Move SimplePathIntegralController::SelectMove(const Position& position, const SearchLimits& limits) {
  // If Path Integral is not enabled, return null move (fallback to standard LC0)
  if (!config_.enabled) {
    return Move();
  }
  
  try {
    // For now, only implement competitive mode
    if (config_.sampling_mode == PathIntegralSamplingMode::kCompetitive) {
      return HandleCompetitiveMode(position, limits);
    }
    
    // Quantum limit mode not implemented yet, fallback
    return Move();
    
  } catch (const std::exception& e) {
    CERR << "Path Integral error: " << e.what() << ", falling back to standard LC0";
    return Move();
  }
}

void SimplePathIntegralController::UpdateOptions(const OptionsDict& options) {
  UpdateConfigFromOptions(options);
}

void SimplePathIntegralController::UpdateConfigFromOptions(const OptionsDict& options) {
  try {
    // Get Path Integral options from UCI
    config_.lambda = options.Get<float>(kPathIntegralLambdaId);
    config_.samples = options.Get<int>(kPathIntegralSamplesId);
    
    std::string reward_mode_str = options.Get<std::string>(kPathIntegralRewardModeId);
    config_.reward_mode = PathIntegralConfig::ParseRewardMode(reward_mode_str);
    
    std::string sampling_mode_str = options.Get<std::string>(kPathIntegralModeId);
    config_.sampling_mode = PathIntegralConfig::ParseSamplingMode(sampling_mode_str);
    
    // Enable Path Integral if lambda > 0 and samples > 0
    config_.enabled = (config_.lambda > 0.0f && config_.samples > 0);
    
    if (config_.enabled) {
      CERR << "Path Integral enabled: lambda=" << config_.lambda 
           << " samples=" << config_.samples 
           << " mode=" << PathIntegralConfig::SamplingModeToString(config_.sampling_mode);
    }
    
  } catch (const std::exception& e) {
    CERR << "Failed to update Path Integral options: " << e.what();
    config_.SetDefaults();
    config_.enabled = false;
  }
}

Move SimplePathIntegralController::HandleCompetitiveMode(const Position& position, const SearchLimits& limits) {
  try {
    // Generate legal moves for the current position
    auto legal_moves = position.GetBoard().GenerateLegalMoves();
    if (legal_moves.empty()) {
      return Move(); // No legal moves, fallback
    }
    
    // Perform basic root node sampling
    auto sampling_results = PerformRootNodeSampling(position, legal_moves);
    if (sampling_results.empty()) {
      return Move(); // Sampling failed, fallback
    }
    
    // Apply softmax to move scores and select best move
    return SelectMoveFromSampling(sampling_results, legal_moves);
    
  } catch (const std::exception& e) {
    CERR << "Path Integral competitive mode error: " << e.what();
    return Move(); // Fallback on any error
  }
}

std::vector<SimplePathIntegralController::SampleResult> 
SimplePathIntegralController::PerformRootNodeSampling(const Position& position, const MoveList& legal_moves) {
  std::vector<SampleResult> results;
  results.reserve(legal_moves.size());
  
  if (config_.debug_logging) {
    CERR << "Path Integral: Sampling " << legal_moves.size() << " legal moves with " 
         << config_.samples << " samples, lambda=" << config_.lambda;
  }
  
  // For each legal move, evaluate it multiple times (simple sampling)
  for (const auto& move : legal_moves) {
    float total_score = 0.0f;
    int valid_samples = 0;
    
    // Generate multiple samples for this move
    for (int sample = 0; sample < config_.samples; ++sample) {
      try {
        float score = EvaluateMove(position, move);
        if (std::isfinite(score)) {
          total_score += score;
          valid_samples++;
        }
      } catch (const std::exception& e) {
        // Skip invalid samples
        if (config_.debug_logging) {
          CERR << "Path Integral: Sample failed for move " << move.ToString(false) << ": " << e.what();
        }
      }
    }
    
    if (valid_samples > 0) {
      float avg_score = total_score / valid_samples;
      results.push_back({move, avg_score, 0.0f}); // probability will be calculated later
    }
  }
  
  return results;
}

Move SimplePathIntegralController::SelectMoveFromSampling(const std::vector<SampleResult>& results, const MoveList& legal_moves) {
  if (results.empty()) {
    return Move();
  }
  
  // Extract scores for softmax calculation
  std::vector<float> scores;
  scores.reserve(results.size());
  for (const auto& result : results) {
    scores.push_back(result.score);
  }
  
  // Apply softmax to get probabilities
  auto probabilities = softmax_calculator_->CalculateSoftmax(scores, config_.lambda);
  if (probabilities.size() != results.size()) {
    CERR << "Path Integral: Softmax calculation failed, using fallback";
    return Move();
  }
  
  // Find the move with highest probability
  size_t best_idx = 0;
  float best_prob = probabilities[0];
  for (size_t i = 1; i < probabilities.size(); ++i) {
    if (probabilities[i] > best_prob) {
      best_prob = probabilities[i];
      best_idx = i;
    }
  }
  
  if (config_.debug_logging) {
    CERR << "Path Integral: Selected move " << results[best_idx].move.ToString(false)
         << " with probability " << best_prob << " (score: " << results[best_idx].score << ")";
    
    // Log all move probabilities
    for (size_t i = 0; i < results.size(); ++i) {
      CERR << "  " << results[i].move.ToString(false) << ": score=" << results[i].score
           << " prob=" << probabilities[i];
    }
  }
  
  return results[best_idx].move;
}

float SimplePathIntegralController::EvaluateMove(const Position& position, const Move& move) {
  // Simple evaluation: for now, just return a basic heuristic score
  // In a full implementation, this would use the neural network backend
  // to get proper policy and value evaluations
  
  // Basic heuristic: prefer captures and central moves
  float score = 0.0f;
  
  // Bonus for captures
  const ChessBoard& board = position.GetBoard();
  bool is_capture = board.theirs().get(move.to()) || move.is_en_passant();
  if (is_capture) {
    score += 1.0f;
  }
  
  // Bonus for central squares (e4, e5, d4, d5)
  Square to_square = move.to();
  int file = to_square.file() - kFileA;
  int rank = to_square.rank() - kRank1;
  if ((file == 3 || file == 4) && (rank == 3 || rank == 4)) {
    score += 0.5f;
  }
  
  // Add some randomness to simulate sampling variation
  // This is a placeholder - real implementation would use neural network
  static thread_local std::random_device rd;
  static thread_local std::mt19937 gen(rd());
  std::normal_distribution<float> noise(0.0f, 0.1f);
  score += noise(gen);
  
  return score;
}

} // namespace lczero