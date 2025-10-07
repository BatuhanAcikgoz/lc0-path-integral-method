#include "search/path_integral/debug_logger.h"
#include "utils/logging.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <algorithm>

namespace lczero {

PathIntegralDebugLogger& PathIntegralDebugLogger::Instance() {
    static PathIntegralDebugLogger instance;
    return instance;
}

PathIntegralDebugLogger::~PathIntegralDebugLogger() {
    if (session_active_) {
        EndSession();
    }
    if (output_file_ && output_file_->is_open()) {
        output_file_->close();
    }
}

void PathIntegralDebugLogger::SetEnabled(bool enabled) {
    enabled_ = enabled;
}

void PathIntegralDebugLogger::SetOutputFile(const std::string& filename) {
    output_filename_ = filename;
    if (!filename.empty() && enabled_) {
        output_file_ = std::make_unique<std::ofstream>(filename, std::ios::app);
        if (!output_file_->is_open()) {
            CERR << "PathIntegralDebugLogger: Failed to open output file: " << filename;
            output_file_.reset();
        }
    }
}

void PathIntegralDebugLogger::SetOutputToStderr(bool to_stderr) {
    output_to_stderr_ = to_stderr;
}

void PathIntegralDebugLogger::StartSession(const std::string& position_fen) {
    if (!enabled_) return;
    
    if (session_active_) {
        EndSession();
    }
    
    current_session_id_ = GenerateSessionId();
    current_position_fen_ = position_fen;
    session_start_time_ = std::chrono::high_resolution_clock::now();
    session_active_ = true;
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << current_session_id_ << "\","
         << "\"position_fen\":\"" << EscapeJsonString(position_fen) << "\""
         << "}";
    
    WriteLogEntry("session_start", data.str());
}

void PathIntegralDebugLogger::EndSession() {
    if (!enabled_ || !session_active_) return;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - session_start_time_);
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << current_session_id_ << "\","
         << "\"total_session_time_ms\":" << duration.count()
         << "}";
    
    WriteLogEntry("session_end", data.str());
    
    session_active_ = false;
    current_session_id_.clear();
    current_position_fen_.clear();
}

void PathIntegralDebugLogger::LogSamplingStart(int requested_samples, int legal_moves, float lambda,
                                             const std::string& sampling_mode, const std::string& reward_mode) {
    if (!enabled_ || !session_active_) return;
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << current_session_id_ << "\","
         << "\"position_fen\":\"" << EscapeJsonString(current_position_fen_) << "\","
         << "\"requested_samples\":" << requested_samples << ","
         << "\"legal_moves\":" << legal_moves << ","
         << "\"lambda\":" << std::fixed << std::setprecision(6) << lambda << ","
         << "\"sampling_mode\":\"" << EscapeJsonString(sampling_mode) << "\"";
    
    if (!reward_mode.empty()) {
        data << ",\"reward_mode\":\"" << EscapeJsonString(reward_mode) << "\"";
    }
    
    data << "}";
    
    WriteLogEntry("sampling_start", data.str());
}

void PathIntegralDebugLogger::LogSampleEvaluation(const Move& move, int sample_num, float score,
                                                const std::string& eval_method, double eval_time_ms) {
    if (!enabled_ || !session_active_) return;
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << current_session_id_ << "\","
         << "\"move\":\"" << EscapeJsonString(move.ToString(false)) << "\","
         << "\"sample_number\":" << sample_num << ","
         << "\"score\":" << std::fixed << std::setprecision(6) << score << ","
         << "\"evaluation_method\":\"" << EscapeJsonString(eval_method) << "\","
         << "\"evaluation_time_ms\":" << std::fixed << std::setprecision(3) << eval_time_ms
         << "}";
    
    WriteLogEntry("sample_evaluation", data.str());
}

void PathIntegralDebugLogger::LogSamplingComplete(int total_samples, double total_time_ms,
                                                int neural_net_evals, int cached_evals, int heuristic_evals) {
    if (!enabled_ || !session_active_) return;
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << current_session_id_ << "\","
         << "\"total_samples\":" << total_samples << ","
         << "\"total_time_ms\":" << std::fixed << std::setprecision(3) << total_time_ms << ","
         << "\"neural_net_evaluations\":" << neural_net_evals << ","
         << "\"cached_evaluations\":" << cached_evals << ","
         << "\"heuristic_evaluations\":" << heuristic_evals << ","
         << "\"avg_time_per_sample_ms\":" << std::fixed << std::setprecision(3) 
         << (total_samples > 0 ? total_time_ms / total_samples : 0.0)
         << "}";
    
    WriteLogEntry("sampling_complete", data.str());
}

void PathIntegralDebugLogger::LogMoveSelection(const Move& selected_move, float probability, float score,
                                             const std::vector<std::pair<Move, float>>& all_probabilities) {
    if (!enabled_ || !session_active_) return;
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << current_session_id_ << "\","
         << "\"selected_move\":\"" << EscapeJsonString(selected_move.ToString(false)) << "\","
         << "\"probability\":" << std::fixed << std::setprecision(6) << probability << ","
         << "\"score\":" << std::fixed << std::setprecision(6) << score << ","
         << "\"all_probabilities\":[";
    
    for (size_t i = 0; i < all_probabilities.size(); ++i) {
        if (i > 0) data << ",";
        data << "{"
             << "\"move\":\"" << EscapeJsonString(all_probabilities[i].first.ToString(false)) << "\","
             << "\"probability\":" << std::fixed << std::setprecision(6) << all_probabilities[i].second
             << "}";
    }
    
    data << "]}";
    
    WriteLogEntry("move_selection", data.str());
}

void PathIntegralDebugLogger::LogNeuralNetworkCall(bool cache_hit, double eval_time_ms, const std::string& details) {
    if (!enabled_ || !session_active_) return;
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << current_session_id_ << "\","
         << "\"cache_hit\":" << (cache_hit ? "true" : "false") << ","
         << "\"evaluation_time_ms\":" << std::fixed << std::setprecision(3) << eval_time_ms;
    
    if (!details.empty()) {
        data << ",\"details\":\"" << EscapeJsonString(details) << "\"";
    }
    
    data << "}";
    
    WriteLogEntry("neural_network_call", data.str());
}

void PathIntegralDebugLogger::LogSoftmaxCalculation(const std::vector<float>& input_scores, float lambda,
                                                  const std::vector<float>& output_probabilities) {
    if (!enabled_ || !session_active_) return;
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << current_session_id_ << "\","
         << "\"lambda\":" << std::fixed << std::setprecision(6) << lambda << ","
         << "\"input_scores\":[";
    
    for (size_t i = 0; i < input_scores.size(); ++i) {
        if (i > 0) data << ",";
        data << std::fixed << std::setprecision(6) << input_scores[i];
    }
    
    data << "],\"output_probabilities\":[";
    
    for (size_t i = 0; i < output_probabilities.size(); ++i) {
        if (i > 0) data << ",";
        data << std::fixed << std::setprecision(6) << output_probabilities[i];
    }
    
    data << "]}";
    
    WriteLogEntry("softmax_calculation", data.str());
}

void PathIntegralDebugLogger::LogWarning(const std::string& message) {
    if (!enabled_) return;
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << (session_active_ ? current_session_id_ : "none") << "\","
         << "\"message\":\"" << EscapeJsonString(message) << "\""
         << "}";
    
    WriteLogEntry("warning", data.str());
}

void PathIntegralDebugLogger::LogError(const std::string& message) {
    if (!enabled_) return;
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << (session_active_ ? current_session_id_ : "none") << "\","
         << "\"message\":\"" << EscapeJsonString(message) << "\""
         << "}";
    
    WriteLogEntry("error", data.str());
}

void PathIntegralDebugLogger::LogInfo(const std::string& message) {
    if (!enabled_) return;
    
    std::ostringstream data;
    data << "{"
         << "\"session_id\":\"" << (session_active_ ? current_session_id_ : "none") << "\","
         << "\"message\":\"" << EscapeJsonString(message) << "\""
         << "}";
    
    WriteLogEntry("info", data.str());
}

void PathIntegralDebugLogger::WriteLogEntry(const std::string& event_type, const std::string& data) {
    std::ostringstream log_entry;
    log_entry << "{"
              << "\"timestamp\":\"" << GetTimestamp() << "\","
              << "\"event_type\":\"" << event_type << "\","
              << "\"data\":" << data
              << "}";
    
    std::string entry = log_entry.str();
    
    // Output to stderr if enabled
    if (output_to_stderr_) {
        std::cerr << "PI_DEBUG: " << entry << std::endl;
    }
    
    // Output to file if available
    if (output_file_ && output_file_->is_open()) {
        *output_file_ << entry << std::endl;
        output_file_->flush();
    }
}

std::string PathIntegralDebugLogger::GetTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    return oss.str();
}

std::string PathIntegralDebugLogger::GenerateSessionId() const {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    
    std::ostringstream oss;
    for (int i = 0; i < 32; ++i) {
        if (i == 8 || i == 12 || i == 16 || i == 20) {
            oss << '-';
        }
        oss << std::hex << dis(gen);
    }
    return oss.str();
}

std::string PathIntegralDebugLogger::EscapeJsonString(const std::string& str) const {
    std::ostringstream oss;
    for (char c : str) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c >= 0 && c < 32) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
                } else {
                    oss << c;
                }
                break;
        }
    }
    return oss.str();
}

} // namespace lczero