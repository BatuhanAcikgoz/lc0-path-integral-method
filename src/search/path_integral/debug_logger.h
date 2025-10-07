#pragma once

#include <string>
#include <fstream>
#include <memory>
#include <chrono>
#include <vector>
#include "chess/types.h"

namespace lczero {

// Enhanced debug logging system for Path Integral sampling verification
class PathIntegralDebugLogger {
public:
    // Singleton access
    static PathIntegralDebugLogger& Instance();
    
    // Configuration
    void SetEnabled(bool enabled);
    void SetOutputFile(const std::string& filename);
    void SetOutputToStderr(bool to_stderr);
    
    // Session management
    void StartSession(const std::string& position_fen);
    void EndSession();
    
    // Sampling lifecycle logging
    void LogSamplingStart(int requested_samples, int legal_moves, float lambda, 
                         const std::string& sampling_mode, const std::string& reward_mode = "");
    void LogSampleEvaluation(const Move& move, int sample_num, float score, 
                           const std::string& eval_method, double eval_time_ms);
    void LogSamplingComplete(int total_samples, double total_time_ms, 
                           int neural_net_evals, int cached_evals, int heuristic_evals);
    void LogMoveSelection(const Move& selected_move, float probability, float score,
                         const std::vector<std::pair<Move, float>>& all_probabilities);
    
    // Neural network evaluation logging
    void LogNeuralNetworkCall(bool cache_hit, double eval_time_ms, const std::string& details = "");
    
    // Softmax calculation logging
    void LogSoftmaxCalculation(const std::vector<float>& input_scores, float lambda,
                             const std::vector<float>& output_probabilities);
    
    // Error and warning logging
    void LogWarning(const std::string& message);
    void LogError(const std::string& message);
    void LogInfo(const std::string& message);
    
    // Configuration getters
    bool IsEnabled() const { return enabled_; }
    
private:
    PathIntegralDebugLogger() = default;
    ~PathIntegralDebugLogger();
    
    // Internal logging methods
    void WriteLogEntry(const std::string& event_type, const std::string& data);
    std::string GetTimestamp() const;
    std::string GenerateSessionId() const;
    std::string EscapeJsonString(const std::string& str) const;
    
    // Configuration
    bool enabled_ = false;
    bool output_to_stderr_ = true;
    std::string output_filename_;
    std::unique_ptr<std::ofstream> output_file_;
    
    // Session state
    std::string current_session_id_;
    std::string current_position_fen_;
    std::chrono::high_resolution_clock::time_point session_start_time_;
    bool session_active_ = false;
    
    // Prevent copying
    PathIntegralDebugLogger(const PathIntegralDebugLogger&) = delete;
    PathIntegralDebugLogger& operator=(const PathIntegralDebugLogger&) = delete;
};

// Convenience macros for debug logging
#define PI_DEBUG_LOG_SAMPLING_START(samples, moves, lambda, mode, reward) \
    PathIntegralDebugLogger::Instance().LogSamplingStart(samples, moves, lambda, mode, reward)

#define PI_DEBUG_LOG_SAMPLE_EVAL(move, sample_num, score, method, time_ms) \
    PathIntegralDebugLogger::Instance().LogSampleEvaluation(move, sample_num, score, method, time_ms)

#define PI_DEBUG_LOG_SAMPLING_COMPLETE(total_samples, total_time, nn_evals, cached_evals, heuristic_evals) \
    PathIntegralDebugLogger::Instance().LogSamplingComplete(total_samples, total_time, nn_evals, cached_evals, heuristic_evals)

#define PI_DEBUG_LOG_MOVE_SELECTION(move, prob, score, all_probs) \
    PathIntegralDebugLogger::Instance().LogMoveSelection(move, prob, score, all_probs)

#define PI_DEBUG_LOG_NN_CALL(cache_hit, time_ms, details) \
    PathIntegralDebugLogger::Instance().LogNeuralNetworkCall(cache_hit, time_ms, details)

#define PI_DEBUG_LOG_SOFTMAX(input_scores, lambda, output_probs) \
    PathIntegralDebugLogger::Instance().LogSoftmaxCalculation(input_scores, lambda, output_probs)

#define PI_DEBUG_LOG_WARNING(message) \
    PathIntegralDebugLogger::Instance().LogWarning(message)

#define PI_DEBUG_LOG_ERROR(message) \
    PathIntegralDebugLogger::Instance().LogError(message)

#define PI_DEBUG_LOG_INFO(message) \
    PathIntegralDebugLogger::Instance().LogInfo(message)

} // namespace lczero