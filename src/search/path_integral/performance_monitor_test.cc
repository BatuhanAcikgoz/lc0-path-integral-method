#include "search/path_integral/performance_monitor.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

namespace lczero {

class PathIntegralPerformanceMonitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        monitor_ = std::make_unique<PathIntegralPerformanceMonitor>();
    }
    
    std::unique_ptr<PathIntegralPerformanceMonitor> monitor_;
};

TEST_F(PathIntegralPerformanceMonitorTest, BasicSamplingMetrics) {
    // Test basic sampling session
    monitor_->StartSampling(10);
    
    // Record some samples
    monitor_->RecordSample("neural_network", 5.0);
    monitor_->RecordSample("neural_network", 3.0);
    monitor_->RecordSample("cached", 0.1);
    monitor_->RecordSample("heuristic", 1.0);
    
    monitor_->EndSampling();
    
    auto metrics = monitor_->GetMetrics();
    
    EXPECT_EQ(metrics.requested_samples, 10);
    EXPECT_EQ(metrics.actual_samples, 4);
    EXPECT_EQ(metrics.neural_net_evaluations, 2);
    EXPECT_EQ(metrics.cached_evaluations, 1);
    EXPECT_EQ(metrics.heuristic_evaluations, 1);
    EXPECT_NEAR(metrics.neural_net_time_ms, 8.0, 0.1);
    EXPECT_GT(metrics.total_time_ms, 0.0);
    EXPECT_GT(metrics.samples_per_second, 0.0);
}

TEST_F(PathIntegralPerformanceMonitorTest, DirectEvaluationRecording) {
    monitor_->StartSampling(5);
    
    // Test direct evaluation recording methods
    monitor_->RecordNeuralNetEvaluation(10.0);
    monitor_->RecordCachedEvaluation();
    monitor_->RecordHeuristicEvaluation();
    
    monitor_->EndSampling();
    
    auto metrics = monitor_->GetMetrics();
    
    EXPECT_EQ(metrics.requested_samples, 5);
    EXPECT_EQ(metrics.neural_net_evaluations, 1);
    EXPECT_EQ(metrics.cached_evaluations, 1);
    EXPECT_EQ(metrics.heuristic_evaluations, 1);
    EXPECT_NEAR(metrics.neural_net_time_ms, 10.0, 0.1);
}

TEST_F(PathIntegralPerformanceMonitorTest, DisabledMonitoring) {
    monitor_->SetEnabled(false);
    
    monitor_->StartSampling(5);
    monitor_->RecordSample("neural_network", 5.0);
    monitor_->EndSampling();
    
    auto metrics = monitor_->GetMetrics();
    
    // When disabled, should return empty metrics
    EXPECT_EQ(metrics.requested_samples, 0);
    EXPECT_EQ(metrics.actual_samples, 0);
}

TEST_F(PathIntegralPerformanceMonitorTest, MetricsCalculation) {
    monitor_->StartSampling(3);
    
    // Add a small delay to ensure measurable time
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    monitor_->RecordSample("neural_network", 5.0);
    monitor_->RecordSample("neural_network", 3.0);
    
    monitor_->EndSampling();
    
    auto metrics = monitor_->GetMetrics();
    
    EXPECT_EQ(metrics.actual_samples, 2);
    EXPECT_NEAR(metrics.avg_time_per_sample_ms, metrics.total_time_ms / 2.0, 0.1);
    EXPECT_GT(metrics.samples_per_second, 0.0);
    EXPECT_LT(metrics.samples_per_second, 1000000.0); // Reasonable upper bound
}

TEST_F(PathIntegralPerformanceMonitorTest, UnknownEvaluationMethod) {
    monitor_->StartSampling(2);
    
    // Test unknown evaluation method - should default to neural_network
    monitor_->RecordSample("unknown_method", 2.5);
    
    monitor_->EndSampling();
    
    auto metrics = monitor_->GetMetrics();
    
    EXPECT_EQ(metrics.actual_samples, 1);
    EXPECT_EQ(metrics.neural_net_evaluations, 1);
    EXPECT_NEAR(metrics.neural_net_time_ms, 2.5, 0.1);
}

} // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
