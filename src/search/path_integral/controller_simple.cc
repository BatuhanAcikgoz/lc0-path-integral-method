#include "search/path_integral/controller_simple.h"
#include "utils/logging.h"
#include "chess/position.h"
#include "utils/optionsparser.h"
#include "neural/backend.h"
#include <random>
#include <cmath>

#include "search/path_integral/options.h"

namespace lczero {


SimplePathIntegralController::SimplePathIntegralController(const OptionsDict& options, Backend* backend)
    : backend_(backend) {
  softmax_calculator_ = std::make_unique<SoftmaxCalculator>();
  UpdateConfigFromOptions(options);
}

SimplePathIntegralController::~SimplePathIntegralController() = default;

Move SimplePathIntegralController::SelectMove(const Position& position, const SearchLimits& limits) {
  // If Path Integral is not enabled, return null move (fallback to standard LC0)
  if (!config_.enabled) {
    return Move();
  }
  
  if (config_.debug_logging) {
    CERR << "Path Integral: Starting move selection with Lambda=" << config_.lambda 
         << " Samples=" << config_.samples 
         << " Mode=" << PathIntegralConfig::SamplingModeToString(config_.sampling_mode);
  }
  
  try {
    // Implement both competitive and quantum limit modes
    if (config_.sampling_mode == PathIntegralSamplingMode::kCompetitive) {
      return HandleCompetitiveMode(position, limits);
    } else if (config_.sampling_mode == PathIntegralSamplingMode::kQuantumLimit) {
      return HandleQuantumLimitMode(position, limits);
    }
    
    // Unknown mode, fallback
    if (config_.debug_logging) {
      CERR << "Path Integral: Unknown sampling mode, falling back to standard LC0";
    }
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
  // Use neural network backend for proper evaluation if available
  if (backend_) {
    try {
      // Create position after the move
      Position new_position(position, move);
      
      // Get legal moves for the new position
      MoveList legal_moves = new_position.GetBoard().GenerateLegalMoves();
      
      // Create evaluation position
      EvalPosition eval_pos;
      eval_pos.pos = std::span<const Position>(&new_position, 1);
      eval_pos.legal_moves = std::span<const Move>(legal_moves.begin(), legal_moves.size());
      
      // Try to get cached evaluation first
      auto cached_result = backend_->GetCachedEvaluation(eval_pos);
      if (cached_result.has_value()) {
        // Return Q value (from white's perspective)
        return cached_result->q;
      }
      
      // If not cached, evaluate using backend
      std::vector<EvalPosition> positions = {eval_pos};
      auto results = backend_->EvaluateBatch(positions);
      
      if (!results.empty()) {
        // Return Q value (from white's perspective)
        return results[0].q;
      }
      
    } catch (const std::exception& e) {
      if (config_.debug_logging) {
        CERR << "Path Integral: Neural network evaluation failed for move " 
             << move.ToString(false) << ": " << e.what();
      }
      // Fall through to heuristic evaluation
    }
  }
  
  // Fallback to basic heuristic evaluation
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
  static thread_local std::random_device rd;
  static thread_local std::mt19937 gen(rd());
  std::normal_distribution<float> noise(0.0f, 0.1f);
  score += noise(gen);
  
  return score;
}

Move SimplePathIntegralController::HandleQuantumLimitMode(const Position& position, const SearchLimits& limits) {
  try {
    // Generate legal moves for the current position
    auto legal_moves = position.GetBoard().GenerateLegalMoves();
    if (legal_moves.empty()) {
      return Move(); // No legal moves, fallback
    }
    
    if (config_.debug_logging) {
      CERR << "Path Integral: Quantum Limit mode with reward mode " 
           << PathIntegralConfig::RewardModeToString(config_.reward_mode);
    }
    
    // Perform quantum limit sampling with reward modes
    auto sampling_results = PerformQuantumLimitSampling(position, legal_moves);
    if (sampling_results.empty()) {
      return Move(); // Sampling failed, fallback
    }
    
    // Apply softmax to move scores and select best move
    return SelectMoveFromSampling(sampling_results, legal_moves);
    
  } catch (const std::exception& e) {
    CERR << "Path Integral quantum limit mode error: " << e.what();
    return Move(); // Fallback on any error
  }
}

std::vector<SimplePathIntegralController::SampleResult> 
SimplePathIntegralController::PerformQuantumLimitSampling(const Position& position, const MoveList& legal_moves) {
  std::vector<SampleResult> results;
  results.reserve(legal_moves.size());
  
  if (config_.debug_logging) {
    CERR << "Path Integral: Quantum Limit sampling " << legal_moves.size() << " legal moves with " 
         << config_.samples << " samples, reward mode=" 
         << PathIntegralConfig::RewardModeToString(config_.reward_mode);
  }
  
  // For each legal move, evaluate using the specified reward mode
  for (const auto& move : legal_moves) {
    float total_score = 0.0f;
    int valid_samples = 0;
    
    // Generate multiple samples for this move
    for (int sample = 0; sample < config_.samples; ++sample) {
      try {
        float score = 0.0f;
        
        // Apply reward mode
        switch (config_.reward_mode) {
          case PathIntegralRewardMode::kPolicy:
            score = EvaluateMovePolicy(position, move);
            break;
          case PathIntegralRewardMode::kCpScore:
            score = EvaluateMove(position, move);  // Use Q-value as CP score
            break;
          case PathIntegralRewardMode::kHybrid:
            {
              float policy_score = EvaluateMovePolicy(position, move);
              float cp_score = EvaluateMove(position, move);
              score = policy_score * cp_score;  // Hybrid: policy * cp_score
            }
            break;
        }
        
        if (std::isfinite(score)) {
          total_score += score;
          valid_samples++;
        }
      } catch (const std::exception& e) {
        // Skip invalid samples
        if (config_.debug_logging) {
          CERR << "Path Integral: Quantum sample failed for move " << move.ToString(false) << ": " << e.what();
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

float SimplePathIntegralController::EvaluateMovePolicy(const Position& position, const Move& move) {
  // Use neural network backend for policy evaluation if available
  if (backend_) {
    try {
      // Get legal moves for current position
      MoveList legal_moves = position.GetBoard().GenerateLegalMoves();
      
      // Create evaluation position
      EvalPosition eval_pos;
      eval_pos.pos = std::span<const Position>(&position, 1);
      eval_pos.legal_moves = std::span<const Move>(legal_moves.begin(), legal_moves.size());
      
      // Try to get cached evaluation first
      auto cached_result = backend_->GetCachedEvaluation(eval_pos);
      if (cached_result.has_value()) {
        // Find the policy probability for this move
        for (size_t i = 0; i < legal_moves.size(); ++i) {
          if (legal_moves[i] == move && i < cached_result->p.size()) {
            return cached_result->p[i];
          }
        }
      }
      
      // If not cached, evaluate using backend
      std::vector<EvalPosition> positions = {eval_pos};
      auto results = backend_->EvaluateBatch(positions);
      
      if (!results.empty()) {
        // Find the policy probability for this move
        for (size_t i = 0; i < legal_moves.size(); ++i) {
          if (legal_moves[i] == move && i < results[0].p.size()) {
            return results[0].p[i];
          }
        }
      }
      
    } catch (const std::exception& e) {
      if (config_.debug_logging) {
        CERR << "Path Integral: Neural network policy evaluation failed for move " 
             << move.ToString(false) << ": " << e.what();
      }
    }
  }
  
  // Fallback to uniform policy
  MoveList legal_moves = position.GetBoard().GenerateLegalMoves();
  return 1.0f / std::max(1, static_cast<int>(legal_moves.size()));
}

} // namespace lczero