#include "search/path_integral/performance_monitor.h"
#include "utils/logging.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace lczero {

void PathIntegralPerformanceMonitor::SamplingMetrics::Reset() {
    requested_samples = 0;
    actual_samples = 0;
    neural_net_evaluations = 0;
    cached_evaluations = 0;
    heuristic_evaluations = 0;
    total_time_ms = 0.0;
    avg_time_per_sample_ms = 0.0;
    neural_net_time_ms = 0.0;
    samples_per_second = 0.0;
}

void PathIntegralPerformanceMonitor::SamplingMetrics::CalculateDerivedMetrics() {
    if (actual_samples > 0) {
        avg_time_per_sample_ms = total_time_ms / actual_samples;
    } else {
        avg_time_per_sample_ms = 0.0;
    }
    
    if (total_time_ms > 0.0) {
        samples_per_second = (actual_samples * 1000.0) / total_time_ms;
    } else {
        samples_per_second = 0.0;
    }
}

PathIntegralPerformanceMonitor::PathIntegralPerformanceMonitor() {
    current_metrics_.Reset();
}

void PathIntegralPerformanceMonitor::StartSampling(int requested_samples) {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    // Reset metrics for new sampling session
    current_metrics_.Reset();
    current_metrics_.requested_samples = requested_samples;
    
    // Record start time
    start_time_ = std::chrono::high_resolution_clock::now();
    sampling_active_ = true;
    
    LOGFILE << "PathIntegralPerformanceMonitor: Started sampling session with " 
            << requested_samples << " requested samples";
}

void PathIntegralPerformanceMonitor::RecordSample(const std::string& eval_method, double time_ms) {
    if (!enabled_ || !sampling_active_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    current_metrics_.actual_samples++;
    
    // Categorize evaluation method
    if (eval_method == "neural_network" || eval_method == "neural_net") {
        current_metrics_.neural_net_evaluations++;
        current_metrics_.neural_net_time_ms += time_ms;
    } else if (eval_method == "cached" || eval_method == "cache") {
        current_metrics_.cached_evaluations++;
    } else if (eval_method == "heuristic") {
        current_metrics_.heuristic_evaluations++;
    } else {
        // Default to neural network if method is unclear
        current_metrics_.neural_net_evaluations++;
        current_metrics_.neural_net_time_ms += time_ms;
        LOGFILE << "PathIntegralPerformanceMonitor: Unknown evaluation method '" 
                << eval_method << "', categorizing as neural_network";
    }
}

void PathIntegralPerformanceMonitor::EndSampling() {
    if (!enabled_ || !sampling_active_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    UpdateTotalTime();
    current_metrics_.CalculateDerivedMetrics();
    sampling_active_ = false;
    
    LOGFILE << "PathIntegralPerformanceMonitor: Sampling session completed - "
            << "Requested: " << current_metrics_.requested_samples
            << ", Actual: " << current_metrics_.actual_samples
            << ", Time: " << std::fixed << std::setprecision(2) << current_metrics_.total_time_ms << "ms"
            << ", Rate: " << std::fixed << std::setprecision(1) << current_metrics_.samples_per_second << " samples/sec";
}

PathIntegralPerformanceMonitor::SamplingMetrics PathIntegralPerformanceMonitor::GetMetrics() const {
    if (!enabled_) return SamplingMetrics{};
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    SamplingMetrics metrics = current_metrics_;
    
    // Update timing if sampling is still active
    if (sampling_active_) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time_);
        metrics.total_time_ms = elapsed.count() / 1000.0;
        metrics.CalculateDerivedMetrics();
    }
    
    return metrics;
}

void PathIntegralPerformanceMonitor::ExportMetrics(const std::string& filename) const {
    if (!enabled_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    try {
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) {
            LOGFILE << "PathIntegralPerformanceMonitor: Failed to open metrics file: " << filename;
            return;
        }
        
        file << FormatMetricsAsJson() << std::endl;
        file.close();
        
        LOGFILE << "PathIntegralPerformanceMonitor: Exported metrics to " << filename;
    } catch (const std::exception& e) {
        LOGFILE << "PathIntegralPerformanceMonitor: Error exporting metrics: " << e.what();
    }
}

void PathIntegralPerformanceMonitor::RecordNeuralNetEvaluation(double time_ms) {
    if (!enabled_ || !sampling_active_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_.neural_net_evaluations++;
    current_metrics_.neural_net_time_ms += time_ms;
}

void PathIntegralPerformanceMonitor::RecordCachedEvaluation() {
    if (!enabled_ || !sampling_active_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_.cached_evaluations++;
}

void PathIntegralPerformanceMonitor::RecordHeuristicEvaluation() {
    if (!enabled_ || !sampling_active_) return;
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    current_metrics_.heuristic_evaluations++;
}

void PathIntegralPerformanceMonitor::UpdateTotalTime() {
    if (sampling_active_) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time_);
        current_metrics_.total_time_ms = elapsed.count() / 1000.0;
    }
}

std::string PathIntegralPerformanceMonitor::FormatMetricsAsJson() const {
    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    
    json << "{\n";
    json << "  \"timestamp\": \"" << std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() << "\",\n";
    json << "  \"metrics\": {\n";
    json << "    \"requested_samples\": " << current_metrics_.requested_samples << ",\n";
    json << "    \"actual_samples\": " << current_metrics_.actual_samples << ",\n";
    json << "    \"neural_net_evaluations\": " << current_metrics_.neural_net_evaluations << ",\n";
    json << "    \"cached_evaluations\": " << current_metrics_.cached_evaluations << ",\n";
    json << "    \"heuristic_evaluations\": " << current_metrics_.heuristic_evaluations << ",\n";
    json << "    \"total_time_ms\": " << current_metrics_.total_time_ms << ",\n";
    json << "    \"avg_time_per_sample_ms\": " << current_metrics_.avg_time_per_sample_ms << ",\n";
    json << "    \"neural_net_time_ms\": " << current_metrics_.neural_net_time_ms << ",\n";
    json << "    \"samples_per_second\": " << current_metrics_.samples_per_second << "\n";
    json << "  }\n";
    json << "}";
    
    return json.str();
}

} // namespace lczero