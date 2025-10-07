#pragma once

#include <chrono>
#include <string>
#include <atomic>
#include <mutex>

namespace lczero {

// Performance monitoring for Path Integral sampling operations
class PathIntegralPerformanceMonitor {
public:
  struct SamplingMetrics {
    int requested_samples = 0;
    int actual_samples = 0;
    int neural_net_evaluations = 0;
    int cached_evaluations = 0;
    int heuristic_evaluations = 0;
    double total_time_ms = 0.0;
    double avg_time_per_sample_ms = 0.0;
    double neural_net_time_ms = 0.0;
    double samples_per_second = 0.0;

    // Reset all metrics to zero
    void Reset();

    // Calculate derived metrics
    void CalculateDerivedMetrics();
  };

  PathIntegralPerformanceMonitor();
  ~PathIntegralPerformanceMonitor() = default;

  // Sampling session management
  void StartSampling(int requested_samples);
  void RecordSample(const std::string& eval_method, double time_ms);
  void EndSampling();

  // Metrics access
  SamplingMetrics GetMetrics() const;
  void ExportMetrics(const std::string& filename) const;

  // Thread-safe operation tracking
  void RecordNeuralNetEvaluation(double time_ms);
  void RecordCachedEvaluation();
  void RecordHeuristicEvaluation();

  // Enable/disable monitoring
  void SetEnabled(bool enabled) { enabled_ = enabled; }
  bool IsEnabled() const { return enabled_; }

private:
  mutable std::mutex metrics_mutex_;
  SamplingMetrics current_metrics_;
  std::chrono::high_resolution_clock::time_point start_time_;
  std::atomic<bool> enabled_{true};
  std::atomic<bool> sampling_active_{false};

  // Internal helper methods
  void UpdateTotalTime();
  std::string FormatMetricsAsJson() const;
};

} // namespace lczero