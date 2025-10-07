#include "search/path_integral/controller_simple.h"
#include "search/path_integral/debug_logger.h"
#include "utils/logging.h"
#include "chess/position.h"
#include "utils/optionsparser.h"
#include "neural/backend.h"
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>

#include "search/path_integral/options.h"

namespace lczero {


SimplePathIntegralController::SimplePathIntegralController(const OptionsDict& options, Backend* backend)
    : backend_(backend) {
  softmax_calculator_ = std::make_unique<SoftmaxCalculator>();
  performance_monitor_ = std::make_unique<PathIntegralPerformanceMonitor>();
  UpdateConfigFromOptions(options);
}

SimplePathIntegralController::~SimplePathIntegralController() = default;

Move SimplePathIntegralController::SelectMove(const Position& position, const SearchLimits& limits) {
  // If Path Integral is not enabled, return null move (fallback to standard LC0)
  if (!config_.enabled) {
    return Move();
  }
  
  // Start debug session
  if (config_.debug_logging) {
    PathIntegralDebugLogger::Instance().StartSession(PositionToFen(position));
    CERR << "Path Integral: Starting move selection with Lambda=" << config_.lambda 
         << " Samples=" << config_.samples 
         << " Mode=" << PathIntegralConfig::SamplingModeToString(config_.sampling_mode);
  }
  
  try {
    Move selected_move;
    
    // Implement both competitive and quantum limit modes
    if (config_.sampling_mode == PathIntegralSamplingMode::kCompetitive) {
      selected_move = HandleCompetitiveMode(position, limits);
    } else if (config_.sampling_mode == PathIntegralSamplingMode::kQuantumLimit) {
      selected_move = HandleQuantumLimitMode(position, limits);
    } else {
      // Unknown mode, fallback
      if (config_.debug_logging) {
        PI_DEBUG_LOG_WARNING("Unknown sampling mode, falling back to standard LC0");
      }
      selected_move = Move();
    }
    
    // End debug session
    if (config_.debug_logging) {
      PathIntegralDebugLogger::Instance().EndSession();
    }
    
    return selected_move;
    
  } catch (const std::exception& e) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_ERROR("Path Integral error: " + std::string(e.what()) + ", falling back to standard LC0");
      PathIntegralDebugLogger::Instance().EndSession();
    }
    CERR << "Path Integral error: " << e.what() << ", falling back to standard LC0";
    return Move();
  }
}

void SimplePathIntegralController::UpdateOptions(const OptionsDict& options) {
  UpdateConfigFromOptions(options);
}

void SimplePathIntegralController::SetConfig(const PathIntegralConfig& config) {
  config_ = config;

  // Update debug logger settings
  PathIntegralDebugLogger::Instance().SetEnabled(config_.debug_logging);
  if (!config_.metrics_file.empty()) {
    PathIntegralDebugLogger::Instance().SetOutputFile(config_.metrics_file);
  }
  PathIntegralDebugLogger::Instance().SetOutputToStderr(config_.debug_logging);

  if (config_.enabled) {
    CERR << "Path Integral config updated: lambda=" << config_.lambda
         << " samples=" << config_.samples
         << " mode=" << PathIntegralConfig::SamplingModeToString(config_.sampling_mode);
  }
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
    
    // Get debug options
    config_.debug_logging = options.Get<bool>(kPathIntegralDebugModeId);
    config_.metrics_file = options.Get<std::string>(kPathIntegralMetricsFileId);
    
    // Configure debug logger
    PathIntegralDebugLogger::Instance().SetEnabled(config_.debug_logging);
    if (!config_.metrics_file.empty()) {
      PathIntegralDebugLogger::Instance().SetOutputFile(config_.metrics_file);
    }
    PathIntegralDebugLogger::Instance().SetOutputToStderr(config_.debug_logging);
    
    // Enable Path Integral if lambda > 0 and samples > 0
    config_.enabled = (config_.lambda > 0.0f && config_.samples > 0);
    
    if (config_.enabled) {
      CERR << "Path Integral enabled: lambda=" << config_.lambda 
           << " samples=" << config_.samples 
           << " mode=" << PathIntegralConfig::SamplingModeToString(config_.sampling_mode)
           << " debug=" << (config_.debug_logging ? "on" : "off");
    }
    
  } catch (const std::exception& e) {
    CERR << "Failed to update Path Integral options: " << e.what();
    config_.SetDefaults();
    config_.enabled = false;
  }
}

Move SimplePathIntegralController::HandleCompetitiveMode(const Position& position, const SearchLimits&) {
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
  
  // Perform sample count integrity checks
  if (!ValidateSampleCountIntegrity(config_.samples, legal_moves.size())) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_ERROR("Sample count integrity check failed, aborting sampling");
    }
    return results; // Return empty results on integrity check failure
  }
  
  auto sampling_start_time = std::chrono::high_resolution_clock::now();
  
  // Start performance monitoring
  int total_requested_samples = config_.samples * legal_moves.size();
  performance_monitor_->StartSampling(total_requested_samples);
  
  // Log sampling start
  if (config_.debug_logging) {
    PI_DEBUG_LOG_SAMPLING_START(config_.samples, legal_moves.size(), config_.lambda, 
                                PathIntegralConfig::SamplingModeToString(config_.sampling_mode), "");
    CERR << "Path Integral: Sampling " << legal_moves.size() << " legal moves with " 
         << config_.samples << " samples, lambda=" << config_.lambda;
  }
  
  // Sample count verification variables
  int total_samples_performed = 0;
  int total_samples_requested = config_.samples * legal_moves.size();
  int neural_net_evaluations = 0;
  int cached_evaluations = 0;
  int heuristic_evaluations = 0;
  std::vector<std::pair<Move, int>> per_move_sample_counts; // Track samples per move
  per_move_sample_counts.reserve(legal_moves.size());
  
  // For each legal move, evaluate it multiple times (simple sampling)
  for (const auto& move : legal_moves) {
    float total_score = 0.0f;
    int valid_samples = 0;
    int attempted_samples = 0;
    
    // Generate multiple samples for this move
    for (int sample = 0; sample < config_.samples; ++sample) {
      attempted_samples++;
      try {
        auto eval_start_time = std::chrono::high_resolution_clock::now();
        
        float score = EvaluateMove(position, move);
        
        auto eval_end_time = std::chrono::high_resolution_clock::now();
        double eval_time_ms = std::chrono::duration<double, std::milli>(eval_end_time - eval_start_time).count();
        
        if (std::isfinite(score)) {
          total_score += score;
          valid_samples++;
          total_samples_performed++;
          
          // Determine evaluation method based on actual backend usage
          std::string eval_method;
          if (VerifyBackendAvailability()) {
            // Check if we actually used neural network or fell back to heuristic
            // This will be logged by the EvaluateMove function itself
            eval_method = "neural_network_attempted";
            neural_net_evaluations++;
          } else {
            eval_method = "heuristic_backend_unavailable";
            heuristic_evaluations++;
          }
          
          // Record sample in performance monitor
          performance_monitor_->RecordSample(eval_method, eval_time_ms);
          
          if (config_.debug_logging) {
            PI_DEBUG_LOG_SAMPLE_EVAL(move, sample + 1, score, eval_method, eval_time_ms);
          }
        }
      } catch (const std::exception& e) {
        // Skip invalid samples
        if (config_.debug_logging) {
          PI_DEBUG_LOG_WARNING("Sample failed for move " + move.ToString(false) + ": " + std::string(e.what()));
        }
      }
    }
    
    // Record sample count for this move
    per_move_sample_counts.emplace_back(move, valid_samples);
    
    // Sample count verification for this move
    if (valid_samples != config_.samples) {
      if (config_.debug_logging) {
        PI_DEBUG_LOG_WARNING("Sample count discrepancy for move " + move.ToString(false) + 
                             ": requested=" + std::to_string(config_.samples) + 
                             ", actual=" + std::to_string(valid_samples) + 
                             ", attempted=" + std::to_string(attempted_samples));
      }
      CERR << "Path Integral Warning: Move " << move.ToString(false) 
           << " completed " << valid_samples << "/" << config_.samples << " samples";
    } else if (config_.debug_logging) {
      PI_DEBUG_LOG_INFO("Move " + move.ToString(false) + " completed all " + 
                        std::to_string(config_.samples) + " samples successfully");
    }
    
    if (valid_samples > 0) {
      float avg_score = total_score / valid_samples;
      results.push_back({move, avg_score, 0.0f}); // probability will be calculated later
    }
  }
  
  // Overall sample count integrity check
  if (total_samples_performed != total_samples_requested) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_WARNING("Total sample count discrepancy: requested=" + 
                           std::to_string(total_samples_requested) + 
                           ", actual=" + std::to_string(total_samples_performed));
    }
    CERR << "Path Integral Warning: Total samples performed (" << total_samples_performed 
         << ") differs from requested (" << total_samples_requested << ")";
  } else if (config_.debug_logging) {
    PI_DEBUG_LOG_INFO("Sample count verification passed: " + std::to_string(total_samples_performed) + 
                      " samples performed as requested");
  }
  
  // End performance monitoring
  performance_monitor_->EndSampling();
  
  // Log sampling completion
  if (config_.debug_logging) {
    auto sampling_end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(sampling_end_time - sampling_start_time).count();
    
    PI_DEBUG_LOG_SAMPLING_COMPLETE(total_samples_performed, total_time_ms, 
                                   neural_net_evaluations, cached_evaluations, heuristic_evaluations);
    
    // Export performance metrics if enabled
    auto metrics = performance_monitor_->GetMetrics();
    CERR << "Path Integral Performance: " << metrics.actual_samples << "/" << metrics.requested_samples 
         << " samples, " << std::fixed << std::setprecision(2) << metrics.total_time_ms << "ms, "
         << std::fixed << std::setprecision(1) << metrics.samples_per_second << " samples/sec";
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
    if (config_.debug_logging) {
      PI_DEBUG_LOG_ERROR("Softmax calculation failed, using fallback");
    }
    CERR << "Path Integral: Softmax calculation failed, using fallback";
    return Move();
  }
  
  // Log softmax calculation
  if (config_.debug_logging) {
    PI_DEBUG_LOG_SOFTMAX(scores, config_.lambda, probabilities);
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
  
  // Prepare all probabilities for logging
  std::vector<std::pair<Move, float>> all_probabilities;
  all_probabilities.reserve(results.size());
  for (size_t i = 0; i < results.size(); ++i) {
    all_probabilities.emplace_back(results[i].move, probabilities[i]);
  }
  
  // Log move selection
  if (config_.debug_logging) {
    PI_DEBUG_LOG_MOVE_SELECTION(results[best_idx].move, best_prob, results[best_idx].score, all_probabilities);
    
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
  auto eval_start_time = std::chrono::high_resolution_clock::now();
  
  // Backend availability verification
  bool backend_available = VerifyBackendAvailability();
  
  // Use neural network backend for proper evaluation if available
  if (backend_available) {
    try {
      auto nn_start_time = std::chrono::high_resolution_clock::now();
      
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
        auto nn_end_time = std::chrono::high_resolution_clock::now();
        double nn_time_ms = std::chrono::duration<double, std::milli>(nn_end_time - nn_start_time).count();
        
        if (config_.debug_logging) {
          PI_DEBUG_LOG_NN_CALL(true, nn_time_ms, "cached evaluation - Q value retrieved from cache");
        }
        
        // Log detailed evaluation source
        if (config_.debug_logging) {
          PI_DEBUG_LOG_INFO("Move " + move.ToString(false) + " evaluated using CACHED neural network result: Q=" + 
                            std::to_string(cached_result->q) + " in " + std::to_string(nn_time_ms) + "ms");
        }
        
        // Return Q value (from white's perspective)
        return cached_result->q;
      }
      
      // If not cached, evaluate using backend
      std::vector<EvalPosition> positions = {eval_pos};
      auto results = backend_->EvaluateBatch(positions);
      
      auto nn_end_time = std::chrono::high_resolution_clock::now();
      double nn_time_ms = std::chrono::duration<double, std::milli>(nn_end_time - nn_start_time).count();
      
      if (!results.empty()) {
        if (config_.debug_logging) {
          PI_DEBUG_LOG_NN_CALL(false, nn_time_ms, "fresh neural network evaluation - Q value computed by backend");
        }
        
        // Log detailed evaluation source
        if (config_.debug_logging) {
          PI_DEBUG_LOG_INFO("Move " + move.ToString(false) + " evaluated using FRESH neural network computation: Q=" + 
                            std::to_string(results[0].q) + " in " + std::to_string(nn_time_ms) + "ms");
        }
        
        // Return Q value (from white's perspective)
        return results[0].q;
      } else {
        if (config_.debug_logging) {
          PI_DEBUG_LOG_NN_CALL(false, nn_time_ms, "neural network evaluation returned empty results - falling back to heuristic");
          PI_DEBUG_LOG_WARNING("Neural network evaluation returned empty results for move " + move.ToString(false) + 
                               " after " + std::to_string(nn_time_ms) + "ms - using heuristic fallback");
        }
      }
      
    } catch (const std::exception& e) {
      auto nn_end_time = std::chrono::high_resolution_clock::now();
      double nn_time_ms = std::chrono::duration<double, std::milli>(eval_start_time - nn_end_time).count();
      
      if (config_.debug_logging) {
        PI_DEBUG_LOG_ERROR("Neural network evaluation failed for move " + move.ToString(false) + 
                          " after " + std::to_string(nn_time_ms) + "ms: " + std::string(e.what()) + 
                          " - falling back to heuristic");
      }
      // Fall through to heuristic evaluation
    }
  } else {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_WARNING("Backend not available for move " + move.ToString(false) + " - using heuristic evaluation");
    }
  }
  
  // Fallback to basic heuristic evaluation
  auto heuristic_start_time = std::chrono::high_resolution_clock::now();
  
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
  
  auto heuristic_end_time = std::chrono::high_resolution_clock::now();
  double heuristic_time_ms = std::chrono::duration<double, std::milli>(heuristic_end_time - heuristic_start_time).count();
  
  if (config_.debug_logging) {
    PI_DEBUG_LOG_INFO("Move " + move.ToString(false) + " evaluated using HEURISTIC method: score=" + 
                      std::to_string(score) + " in " + std::to_string(heuristic_time_ms) + "ms" +
                      (is_capture ? " (capture bonus)" : "") + 
                      ((file == 3 || file == 4) && (rank == 3 || rank == 4) ? " (center bonus)" : ""));
  }
  
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
  
  // Perform sample count integrity checks
  if (!ValidateSampleCountIntegrity(config_.samples, legal_moves.size())) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_ERROR("Sample count integrity check failed, aborting quantum limit sampling");
    }
    return results; // Return empty results on integrity check failure
  }
  
  auto sampling_start_time = std::chrono::high_resolution_clock::now();
  
  // Start performance monitoring
  int total_requested_samples = config_.samples * legal_moves.size();
  performance_monitor_->StartSampling(total_requested_samples);
  
  // Log sampling start
  if (config_.debug_logging) {
    PI_DEBUG_LOG_SAMPLING_START(config_.samples, legal_moves.size(), config_.lambda, 
                                PathIntegralConfig::SamplingModeToString(config_.sampling_mode),
                                PathIntegralConfig::RewardModeToString(config_.reward_mode));
    CERR << "Path Integral: Quantum Limit sampling " << legal_moves.size() << " legal moves with " 
         << config_.samples << " samples, reward mode=" 
         << PathIntegralConfig::RewardModeToString(config_.reward_mode);
  }
  
  // Sample count verification variables
  int total_samples_performed = 0;
  int total_samples_requested = config_.samples * legal_moves.size();
  int neural_net_evaluations = 0;
  int cached_evaluations = 0;
  int heuristic_evaluations = 0;
  std::vector<std::pair<Move, int>> per_move_sample_counts; // Track samples per move
  per_move_sample_counts.reserve(legal_moves.size());
  
  // For each legal move, evaluate using the specified reward mode
  for (const auto& move : legal_moves) {
    float total_score = 0.0f;
    int valid_samples = 0;
    int attempted_samples = 0;
    
    // Generate multiple samples for this move
    for (int sample = 0; sample < config_.samples; ++sample) {
      attempted_samples++;
      try {
        auto eval_start_time = std::chrono::high_resolution_clock::now();
        
        float score = 0.0f;
        std::string eval_method;
        
        // Apply reward mode
        switch (config_.reward_mode) {
          case PathIntegralRewardMode::kPolicy:
            score = EvaluateMovePolicy(position, move);
            eval_method = VerifyBackendAvailability() ? "policy_neural_network_attempted" : "policy_heuristic_backend_unavailable";
            break;
          case PathIntegralRewardMode::kCpScore:
            score = EvaluateMove(position, move);  // Use Q-value as CP score
            eval_method = VerifyBackendAvailability() ? "cp_score_neural_network_attempted" : "cp_score_heuristic_backend_unavailable";
            break;
          case PathIntegralRewardMode::kHybrid:
            {
              float policy_score = EvaluateMovePolicy(position, move);
              float cp_score = EvaluateMove(position, move);
              score = policy_score * cp_score;  // Hybrid: policy * cp_score
              eval_method = VerifyBackendAvailability() ? "hybrid_neural_network_attempted" : "hybrid_heuristic_backend_unavailable";
            }
            break;
        }
        
        auto eval_end_time = std::chrono::high_resolution_clock::now();
        double eval_time_ms = std::chrono::duration<double, std::milli>(eval_end_time - eval_start_time).count();
        
        if (std::isfinite(score)) {
          total_score += score;
          valid_samples++;
          total_samples_performed++;
          
          // Count evaluation types based on actual backend usage
          if (VerifyBackendAvailability()) {
            neural_net_evaluations++;
          } else {
            heuristic_evaluations++;
          }
          
          // Record sample in performance monitor
          performance_monitor_->RecordSample(eval_method, eval_time_ms);
          
          if (config_.debug_logging) {
            PI_DEBUG_LOG_SAMPLE_EVAL(move, sample + 1, score, eval_method, eval_time_ms);
          }
        }
      } catch (const std::exception& e) {
        // Skip invalid samples
        if (config_.debug_logging) {
          PI_DEBUG_LOG_WARNING("Quantum sample failed for move " + move.ToString(false) + ": " + std::string(e.what()));
        }
      }
    }
    
    // Record sample count for this move
    per_move_sample_counts.emplace_back(move, valid_samples);
    
    // Sample count verification for this move
    if (valid_samples != config_.samples) {
      if (config_.debug_logging) {
        PI_DEBUG_LOG_WARNING("Sample count discrepancy for move " + move.ToString(false) + 
                             ": requested=" + std::to_string(config_.samples) + 
                             ", actual=" + std::to_string(valid_samples) + 
                             ", attempted=" + std::to_string(attempted_samples));
      }
      CERR << "Path Integral Warning: Move " << move.ToString(false) 
           << " completed " << valid_samples << "/" << config_.samples << " samples";
    } else if (config_.debug_logging) {
      PI_DEBUG_LOG_INFO("Move " + move.ToString(false) + " completed all " + 
                        std::to_string(config_.samples) + " samples successfully");
    }
    
    if (valid_samples > 0) {
      float avg_score = total_score / valid_samples;
      results.push_back({move, avg_score, 0.0f}); // probability will be calculated later
    }
  }
  
  // Overall sample count integrity check
  if (total_samples_performed != total_samples_requested) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_WARNING("Total sample count discrepancy: requested=" + 
                           std::to_string(total_samples_requested) + 
                           ", actual=" + std::to_string(total_samples_performed));
    }
    CERR << "Path Integral Warning: Total samples performed (" << total_samples_performed 
         << ") differs from requested (" << total_samples_requested << ")";
  } else if (config_.debug_logging) {
    PI_DEBUG_LOG_INFO("Sample count verification passed: " + std::to_string(total_samples_performed) + 
                      " samples performed as requested");
  }
  
  // End performance monitoring
  performance_monitor_->EndSampling();
  
  // Log sampling completion
  if (config_.debug_logging) {
    auto sampling_end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(sampling_end_time - sampling_start_time).count();
    
    PI_DEBUG_LOG_SAMPLING_COMPLETE(total_samples_performed, total_time_ms, 
                                   neural_net_evaluations, cached_evaluations, heuristic_evaluations);
    
    // Export performance metrics if enabled
    auto metrics = performance_monitor_->GetMetrics();
    CERR << "Path Integral Performance: " << metrics.actual_samples << "/" << metrics.requested_samples 
         << " samples, " << std::fixed << std::setprecision(2) << metrics.total_time_ms << "ms, "
         << std::fixed << std::setprecision(1) << metrics.samples_per_second << " samples/sec";
  }
  
  return results;
}

float SimplePathIntegralController::EvaluateMovePolicy(const Position& position, const Move& move) {
  auto eval_start_time = std::chrono::high_resolution_clock::now();
  
  // Backend availability verification
  bool backend_available = VerifyBackendAvailability();
  
  // Use neural network backend for policy evaluation if available
  if (backend_available) {
    try {
      auto nn_start_time = std::chrono::high_resolution_clock::now();
      
      // Get legal moves for current position
      MoveList legal_moves = position.GetBoard().GenerateLegalMoves();
      
      // Create evaluation position
      EvalPosition eval_pos;
      eval_pos.pos = std::span<const Position>(&position, 1);
      eval_pos.legal_moves = std::span<const Move>(legal_moves.begin(), legal_moves.size());
      
      // Try to get cached evaluation first
      auto cached_result = backend_->GetCachedEvaluation(eval_pos);
      if (cached_result.has_value()) {
        auto nn_end_time = std::chrono::high_resolution_clock::now();
        double nn_time_ms = std::chrono::duration<double, std::milli>(nn_end_time - nn_start_time).count();
        
        // Find the policy probability for this move
        for (size_t i = 0; i < legal_moves.size(); ++i) {
          if (legal_moves[i] == move && i < cached_result->p.size()) {
            if (config_.debug_logging) {
              PI_DEBUG_LOG_NN_CALL(true, nn_time_ms, "cached policy evaluation - policy probability retrieved from cache");
              PI_DEBUG_LOG_INFO("Move " + move.ToString(false) + " policy evaluated using CACHED neural network result: P=" + 
                                std::to_string(cached_result->p[i]) + " in " + std::to_string(nn_time_ms) + "ms");
            }
            return cached_result->p[i];
          }
        }
        
        if (config_.debug_logging) {
          PI_DEBUG_LOG_WARNING("Move " + move.ToString(false) + " not found in cached policy results - falling back to fresh evaluation");
        }
      }
      
      // If not cached, evaluate using backend
      std::vector<EvalPosition> positions = {eval_pos};
      auto results = backend_->EvaluateBatch(positions);
      
      auto nn_end_time = std::chrono::high_resolution_clock::now();
      double nn_time_ms = std::chrono::duration<double, std::milli>(nn_end_time - nn_start_time).count();
      
      if (!results.empty()) {
        // Find the policy probability for this move
        for (size_t i = 0; i < legal_moves.size(); ++i) {
          if (legal_moves[i] == move && i < results[0].p.size()) {
            if (config_.debug_logging) {
              PI_DEBUG_LOG_NN_CALL(false, nn_time_ms, "fresh policy evaluation - policy probability computed by backend");
              PI_DEBUG_LOG_INFO("Move " + move.ToString(false) + " policy evaluated using FRESH neural network computation: P=" + 
                                std::to_string(results[0].p[i]) + " in " + std::to_string(nn_time_ms) + "ms");
            }
            return results[0].p[i];
          }
        }
        
        if (config_.debug_logging) {
          PI_DEBUG_LOG_WARNING("Move " + move.ToString(false) + " not found in fresh policy results - using uniform fallback");
        }
      } else {
        if (config_.debug_logging) {
          PI_DEBUG_LOG_NN_CALL(false, nn_time_ms, "neural network policy evaluation returned empty results - falling back to uniform");
          PI_DEBUG_LOG_WARNING("Neural network policy evaluation returned empty results for move " + move.ToString(false) + 
                               " after " + std::to_string(nn_time_ms) + "ms - using uniform fallback");
        }
      }
      
    } catch (const std::exception& e) {
      auto nn_end_time = std::chrono::high_resolution_clock::now();
      double nn_time_ms = std::chrono::duration<double, std::milli>(nn_end_time - eval_start_time).count();
      
      if (config_.debug_logging) {
        PI_DEBUG_LOG_ERROR("Neural network policy evaluation failed for move " + move.ToString(false) + 
                          " after " + std::to_string(nn_time_ms) + "ms: " + std::string(e.what()) + 
                          " - falling back to uniform policy");
      }
    }
  } else {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_WARNING("Backend not available for policy evaluation of move " + move.ToString(false) + " - using uniform policy");
    }
  }
  
  // Fallback to uniform policy
  auto heuristic_start_time = std::chrono::high_resolution_clock::now();
  
  MoveList legal_moves = position.GetBoard().GenerateLegalMoves();
  float uniform_prob = 1.0f / std::max(1, static_cast<int>(legal_moves.size()));
  
  auto heuristic_end_time = std::chrono::high_resolution_clock::now();
  double heuristic_time_ms = std::chrono::duration<double, std::milli>(heuristic_end_time - heuristic_start_time).count();
  
  if (config_.debug_logging) {
    PI_DEBUG_LOG_INFO("Move " + move.ToString(false) + " policy evaluated using UNIFORM HEURISTIC: P=" + 
                      std::to_string(uniform_prob) + " (1/" + std::to_string(legal_moves.size()) + 
                      " legal moves) in " + std::to_string(heuristic_time_ms) + "ms");
  }
  
  return uniform_prob;
}

PathIntegralPerformanceMonitor::SamplingMetrics SimplePathIntegralController::GetLastSamplingMetrics() const {
  return performance_monitor_->GetMetrics();
}

void SimplePathIntegralController::ExportPerformanceMetrics(const std::string& filename) const {
  performance_monitor_->ExportMetrics(filename);
}

bool SimplePathIntegralController::ValidateSampleCountIntegrity(int requested_samples, int legal_move_count) const {
  // Check if sample count is reasonable
  if (requested_samples <= 0) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_ERROR("Invalid sample count: " + std::to_string(requested_samples) + " (must be > 0)");
    }
    CERR << "Path Integral Error: Invalid sample count " << requested_samples << " (must be > 0)";
    return false;
  }
  
  // Check if sample count is within reasonable limits (prevent excessive computation)
  const int MAX_SAMPLES_PER_MOVE = 10000;
  if (requested_samples > MAX_SAMPLES_PER_MOVE) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_WARNING("High sample count: " + std::to_string(requested_samples) + 
                           " (max recommended: " + std::to_string(MAX_SAMPLES_PER_MOVE) + ")");
    }
    CERR << "Path Integral Warning: High sample count " << requested_samples 
         << " (max recommended: " << MAX_SAMPLES_PER_MOVE << ")";
  }
  
  // Check if we have legal moves to sample
  if (legal_move_count <= 0) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_ERROR("No legal moves available for sampling");
    }
    CERR << "Path Integral Error: No legal moves available for sampling";
    return false;
  }
  
  // Calculate total samples and check for reasonable limits
  const int MAX_TOTAL_SAMPLES = 100000;
  int total_samples = requested_samples * legal_move_count;
  if (total_samples > MAX_TOTAL_SAMPLES) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_WARNING("High total sample count: " + std::to_string(total_samples) + 
                           " (" + std::to_string(requested_samples) + " samples × " + 
                           std::to_string(legal_move_count) + " moves, max recommended: " + 
                           std::to_string(MAX_TOTAL_SAMPLES) + ")");
    }
    CERR << "Path Integral Warning: High total sample count " << total_samples 
         << " (" << requested_samples << " samples × " << legal_move_count 
         << " moves, max recommended: " << MAX_TOTAL_SAMPLES << ")";
  }
  
  if (config_.debug_logging) {
    PI_DEBUG_LOG_INFO("Sample count integrity check passed: " + std::to_string(requested_samples) + 
                      " samples per move, " + std::to_string(legal_move_count) + " legal moves, " +
                      std::to_string(total_samples) + " total samples");
  }
  
  return true;
}

bool SimplePathIntegralController::VerifyBackendAvailability() const {
  // Check if backend pointer is valid
  if (!backend_) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_WARNING("Backend verification failed: backend pointer is null");
    }
    return false;
  }
  
  try {
    // Try to get backend attributes to verify it's properly initialized
    auto attributes = backend_->GetAttributes();
    
    if (config_.debug_logging) {
      PI_DEBUG_LOG_INFO("Backend verification passed: backend is available and initialized");
    }
    
    return true;
    
  } catch (const std::exception& e) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_ERROR("Backend verification failed: " + std::string(e.what()));
    }
    return false;
  } catch (...) {
    if (config_.debug_logging) {
      PI_DEBUG_LOG_ERROR("Backend verification failed: unknown exception occurred");
    }
    return false;
  }
}

Move SimplePathIntegralController::SelectMove(const std::vector<Move>& legal_moves, 
                                            const std::vector<float>& move_scores, 
                                            const Position& position) {
  // If Path Integral is not enabled, return null move (fallback)
  if (!config_.enabled || legal_moves.empty() || move_scores.empty()) {
    return Move();
  }
  
  if (legal_moves.size() != move_scores.size()) {
    if (config_.debug_logging) {
      PathIntegralDebugLogger::Instance().LogError("Move count and score count mismatch");
    }
    return Move();
  }
  
  auto& debug_logger = PathIntegralDebugLogger::Instance();
  
  try {
    // Log sampling start
    if (debug_logger.IsEnabled()) {
      debug_logger.LogSamplingStart(
        config_.samples, 
        legal_moves.size(), 
        config_.lambda,
        PathIntegralConfig::SamplingModeToString(config_.sampling_mode),
        PathIntegralConfig::RewardModeToString(config_.reward_mode)
      );
    }
    
    // Apply softmax to move scores
    std::vector<float> probabilities = softmax_calculator_->CalculateSoftmax(move_scores, config_.lambda);
    
    // Log softmax calculation
    if (debug_logger.IsEnabled()) {
      debug_logger.LogSoftmaxCalculation(move_scores, config_.lambda, probabilities);
    }
    
    // Select move based on probabilities
    Move selected_move;
    
    if (config_.sampling_mode == PathIntegralSamplingMode::kCompetitive) {
      // For competitive mode, use probabilistic selection
      std::random_device rd;
      std::mt19937 gen(rd());
      std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
      
      int selected_idx = dist(gen);
      selected_move = legal_moves[selected_idx];
      
      // Log move selection
      if (debug_logger.IsEnabled()) {
        std::vector<std::pair<Move, float>> all_probs;
        for (size_t i = 0; i < legal_moves.size(); ++i) {
          all_probs.emplace_back(legal_moves[i], probabilities[i]);
        }
        
        debug_logger.LogMoveSelection(
          selected_move, 
          probabilities[selected_idx], 
          move_scores[selected_idx], 
          all_probs
        );
      }
      
    } else {
      // For quantum limit mode, select highest probability move
      auto max_it = std::max_element(probabilities.begin(), probabilities.end());
      int selected_idx = std::distance(probabilities.begin(), max_it);
      selected_move = legal_moves[selected_idx];
      
      // Log move selection
      if (debug_logger.IsEnabled()) {
        std::vector<std::pair<Move, float>> all_probs;
        for (size_t i = 0; i < legal_moves.size(); ++i) {
          all_probs.emplace_back(legal_moves[i], probabilities[i]);
        }
        
        debug_logger.LogMoveSelection(
          selected_move, 
          probabilities[selected_idx], 
          move_scores[selected_idx], 
          all_probs
        );
      }
    }
    
    // Log sampling complete
    if (debug_logger.IsEnabled()) {
      debug_logger.LogSamplingComplete(
        legal_moves.size(),  // total_samples (using available moves)
        0.0,  // total_time_ms (not measured here)
        0,    // neural_net_evals (not applicable)
        legal_moves.size(),  // cached_evals (using existing MCTS data)
        0     // heuristic_evals
      );
    }
    
    return selected_move;
    
  } catch (const std::exception& e) {
    if (debug_logger.IsEnabled()) {
      debug_logger.LogError("Path Integral SelectMove error: " + std::string(e.what()));
    }
    return Move();
  }
}

} // namespace lczero

