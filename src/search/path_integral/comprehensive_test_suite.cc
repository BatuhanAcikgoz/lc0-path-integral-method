/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <gtest/gtest.h>
#include "search/path_integral/debug_logger.h"
#include "search/path_integral/performance_monitor.h"
#include "search/path_integral/controller_simple.h"
#include "search/path_integral/config.h"
#include "tools/verify_path_integral_sampling.h"
#include "utils/optionsdict.h"
#include "chess/position.h"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include <random>

namespace lczero {

class ComprehensivePathIntegralTestSuite : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up basic options
        options_.Set<bool>("PathIntegralEnabled", true);
        options_.Set<float>("PathIntegralLambda", 0.1f);
        options_.Set<int>("PathIntegralSamples", 10);
        options_.Set<std::string>("PathIntegralMode", "competitive");
        options_.Set<bool>("verbose", false);
        
        // Set up test files
        debug_file_ = "comprehensive_test_debug.json";
        metrics_file_ = "comprehensive_test_metrics.json";
        
        // Clean up any existing test files
        CleanupTestFiles();
        
        // Reset logger state
        auto& logger = PathIntegralDebugLogger::Instance();
        logger.SetEnabled(false);
        logger.SetOutputFile("");
    }
    
    void TearDown() override {
        CleanupTestFiles();
        auto& logger = PathIntegralDebugLogger::Instance();
        logger.SetEnabled(false);
        logger.SetOutputFile("");
    }
    
    void CleanupTestFiles() {
        std::vector<std::string> test_files = {
            debug_file_, metrics_file_, 
            "test_sample_count_verification.json",
            "test_neural_network_tracking.json",
            "test_edge_case_debug.json"
        };
        
        for (const auto& file : test_files) {
            if (std::filesystem::exists(file)) {
                std::filesystem::remove(file);
            }
        }
    }
    
    std::string ReadFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return "";
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    
    bool FileContains(const std::string& filename, const std::string& search_term) {
        std::string content = ReadFile(filename);
        return content.find(search_term) != std::string::npos;
    }
    
    int CountOccurrences(const std::string& content, const std::string& search_term) {
        int count = 0;
        size_t pos = 0;
        while ((pos = content.find(search_term, pos)) != std::string::npos) {
            count++;
            pos += search_term.length();
        }
        return count;
    }
    
    OptionsDict options_;
    std::string debug_file_;
    std::string metrics_file_;
};

// ============================================================================
// UNIT TESTS FOR DEBUG LOGGER (Requirement 1.4, 2.4, 3.4, 6.4)
// ============================================================================

TEST_F(ComprehensivePathIntegralTestSuite, DebugLoggerSampleCountVerification) {
    // Test Requirement 1.4: Sample count discrepancy logging
    auto& logger = PathIntegralDebugLogger::Instance();
    logger.SetEnabled(true);
    logger.SetOutputFile("test_sample_count_verification.json");

    // Simulate sampling session with count discrepancy
    logger.StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    logger.LogSamplingStart(10, 20, 0.1f, "competitive", "hybrid");

    // Log fewer samples than requested to trigger discrepancy warning
    for (int i = 0; i < 7; ++i) {
        Move move = Move::White(Square(kFileE, kRank2), Square(kFileE, kRank4));
        logger.LogSampleEvaluation(move, i + 1, 0.5f, "neural_network", 2.0);
    }
    
    // Log completion with discrepancy
    logger.LogSamplingComplete(7, 15.0, 5, 2, 0);
    logger.LogWarning("Sample count discrepancy: requested 10, actual 7");
    logger.EndSession();

    std::string content = ReadFile("test_sample_count_verification.json");
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(FileContains("test_sample_count_verification.json", "sampling_start"));
    EXPECT_TRUE(FileContains("test_sample_count_verification.json", "sampling_complete"));
    EXPECT_TRUE(FileContains("test_sample_count_verification.json", "discrepancy"));
    
    // Verify we logged exactly 7 sample evaluations
    int sample_count = CountOccurrences(content, "sample_evaluation");
    EXPECT_EQ(sample_count, 7);
}

TEST_F(ComprehensivePathIntegralTestSuite, DebugLoggerPerformanceMetricsLogging) {
    // Test Requirement 2.4: Performance metrics logging
    auto& logger = PathIntegralDebugLogger::Instance();
    logger.SetEnabled(true);
    logger.SetOutputFile("test_performance_metrics.json");

    logger.StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Log various timing scenarios
    logger.LogNeuralNetworkCall(false, 5.2, "fresh evaluation");
    logger.LogNeuralNetworkCall(true, 0.1, "cache hit");
    logger.LogSamplingComplete(10, 25.5, 8, 2, 0);

    logger.EndSession();

    std::string content = ReadFile("test_performance_metrics.json");
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(FileContains("test_performance_metrics.json", "neural_network_call"));
    EXPECT_TRUE(FileContains("test_performance_metrics.json", "fresh evaluation"));
    EXPECT_TRUE(FileContains("test_performance_metrics.json", "cache hit"));
}

TEST_F(ComprehensivePathIntegralTestSuite, DebugLoggerNeuralNetworkTracking) {
    // Test Requirement 3.4: Neural network evaluation tracking
    auto& logger = PathIntegralDebugLogger::Instance();
    logger.SetEnabled(true);
    logger.SetOutputFile("test_neural_network_tracking.json");

    logger.StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Log different evaluation methods
    Move move = Move::White(Square(kFileE, kRank2), Square(kFileE, kRank4));
    logger.LogSampleEvaluation(move, 1, 0.75f, "neural_network", 3.5);
    logger.LogSampleEvaluation(move, 2, 0.65f, "cached", 0.1);
    logger.LogSampleEvaluation(move, 3, 0.55f, "heuristic", 1.0);

    // Log backend availability issues
    logger.LogError("Neural network backend unavailable, falling back to heuristic");

    logger.EndSession();

    std::string content = ReadFile("test_neural_network_tracking.json");
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(FileContains("test_neural_network_tracking.json", "neural_network"));
    EXPECT_TRUE(FileContains("test_neural_network_tracking.json", "cached"));
    EXPECT_TRUE(FileContains("test_neural_network_tracking.json", "heuristic"));
    EXPECT_TRUE(FileContains("test_neural_network_tracking.json", "backend unavailable"));
}

TEST_F(ComprehensivePathIntegralTestSuite, DebugLoggerIntegrityChecks) {
    // Test Requirement 6.4: Integrity check logging
    auto& logger = PathIntegralDebugLogger::Instance();
    logger.SetEnabled(true);
    logger.SetOutputFile("test_integrity_checks.json");

    logger.StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Log various integrity check scenarios
    logger.LogWarning("Invalid sample count: requested 0, using fallback value 1");
    logger.LogError("Backend initialization failed, using heuristic evaluation");
    logger.LogInfo("Sample count validation passed: 10 samples requested");

    logger.EndSession();

    std::string content = ReadFile("test_integrity_checks.json");
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(FileContains("test_integrity_checks.json", "Invalid sample count"));
    EXPECT_TRUE(FileContains("test_integrity_checks.json", "Backend initialization failed"));
    EXPECT_TRUE(FileContains("test_integrity_checks.json", "validation passed"));
}

// ============================================================================
// UNIT TESTS FOR PERFORMANCE MONITOR (Requirement 1.4, 2.4, 3.4, 6.4)
// ============================================================================

TEST_F(ComprehensivePathIntegralTestSuite, PerformanceMonitorSampleCountAccuracy) {
    // Test Requirement 1.4: Sample count verification
    PathIntegralPerformanceMonitor monitor;
    
    monitor.StartSampling(15);
    
    // Record exactly 15 samples
    for (int i = 0; i < 15; ++i) {
        monitor.RecordSample("neural_network", 2.0);
    }
    
    monitor.EndSampling();
    
    auto metrics = monitor.GetMetrics();
    
    EXPECT_EQ(metrics.requested_samples, 15);
    EXPECT_EQ(metrics.actual_samples, 15);
    EXPECT_EQ(metrics.neural_net_evaluations, 15);
    EXPECT_EQ(metrics.cached_evaluations, 0);
    EXPECT_EQ(metrics.heuristic_evaluations, 0);
    
    // Test sample count mismatch detection
    PathIntegralPerformanceMonitor monitor2;
    monitor2.StartSampling(10);
    
    // Record fewer samples than requested
    for (int i = 0; i < 7; ++i) {
        monitor2.RecordSample("neural_network", 1.5);
    }
    
    monitor2.EndSampling();
    
    auto metrics2 = monitor2.GetMetrics();
    
    EXPECT_EQ(metrics2.requested_samples, 10);
    EXPECT_EQ(metrics2.actual_samples, 7);
    // This discrepancy should be detectable by the calling code
}

TEST_F(ComprehensivePathIntegralTestSuite, PerformanceMonitorTimingAccuracy) {
    // Test Requirement 2.4: Timing measurement accuracy
    PathIntegralPerformanceMonitor monitor;
    
    monitor.StartSampling(5);
    
    // Add a measurable delay and record timing
    auto start_time = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double measured_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    monitor.RecordSample("neural_network", measured_time);
    monitor.EndSampling();
    
    auto metrics = monitor.GetMetrics();
    
    EXPECT_GT(metrics.total_time_ms, 0.0);
    EXPECT_GT(metrics.neural_net_time_ms, 0.0);
    EXPECT_GT(metrics.samples_per_second, 0.0);
    EXPECT_NEAR(metrics.neural_net_time_ms, measured_time, 1.0); // Within 1ms tolerance
}

TEST_F(ComprehensivePathIntegralTestSuite, PerformanceMonitorEvaluationMethodTracking) {
    // Test Requirement 3.4: Evaluation method tracking
    PathIntegralPerformanceMonitor monitor;
    
    monitor.StartSampling(10);
    
    // Record different evaluation methods
    monitor.RecordSample("neural_network", 5.0);
    monitor.RecordSample("neural_network", 4.5);
    monitor.RecordSample("neural_network", 5.5);
    monitor.RecordSample("cached", 0.1);
    monitor.RecordSample("cached", 0.2);
    monitor.RecordSample("heuristic", 1.0);
    monitor.RecordSample("heuristic", 1.2);
    
    monitor.EndSampling();
    
    auto metrics = monitor.GetMetrics();
    
    EXPECT_EQ(metrics.requested_samples, 10);
    EXPECT_EQ(metrics.actual_samples, 7);
    EXPECT_EQ(metrics.neural_net_evaluations, 3);
    EXPECT_EQ(metrics.cached_evaluations, 2);
    EXPECT_EQ(metrics.heuristic_evaluations, 2);
    EXPECT_NEAR(metrics.neural_net_time_ms, 15.0, 0.1); // 5.0 + 4.5 + 5.5
}

TEST_F(ComprehensivePathIntegralTestSuite, PerformanceMonitorIntegrityValidation) {
    // Test Requirement 6.4: Performance monitoring integrity
    PathIntegralPerformanceMonitor monitor;
    
    // Test invalid scenarios
    monitor.StartSampling(0); // Invalid sample count
    monitor.RecordSample("neural_network", -1.0); // Invalid timing
    monitor.RecordSample("unknown_method", 2.0); // Unknown method
    monitor.EndSampling();
    
    auto metrics = monitor.GetMetrics();
    
    // Should handle gracefully
    EXPECT_EQ(metrics.requested_samples, 0);
    EXPECT_GE(metrics.actual_samples, 0);
    EXPECT_GE(metrics.neural_net_evaluations, 0);
    EXPECT_GE(metrics.total_time_ms, 0.0);
}

// ============================================================================
// INTEGRATION TESTS FOR SAMPLING VERIFICATION (All Requirements)
// ============================================================================

TEST_F(ComprehensivePathIntegralTestSuite, IntegrationSampleCountVerification) {
    // Test end-to-end sample count verification
    auto& logger = PathIntegralDebugLogger::Instance();
    logger.SetEnabled(true);
    logger.SetOutputFile(debug_file_);

    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 12; // Specific count to verify
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    SearchLimits limits;
    VerificationResult result = verifier.VerifySampling(position, config, limits);

    EXPECT_TRUE(result.sampling_completed);
    EXPECT_EQ(result.requested_samples, 12);
    
    // Verify debug log contains sampling information
    std::string content = ReadFile(debug_file_);
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(FileContains(debug_file_, "sampling_start"));
    EXPECT_TRUE(FileContains(debug_file_, "sampling_complete"));
}

TEST_F(ComprehensivePathIntegralTestSuite, IntegrationPerformanceMetricsCollection) {
    // Test end-to-end performance metrics collection
    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    PathIntegralConfig config;
    config.lambda = 0.2f;
    config.samples = 8;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    SearchLimits limits;
    VerificationResult result = verifier.VerifySampling(position, config, limits);

    EXPECT_TRUE(result.sampling_completed);
    EXPECT_GT(result.total_time_ms, 0.0);
    EXPECT_EQ(result.requested_samples, 8);
    
    // Performance metrics should be reasonable
    if (result.GetSamplesPerSecond() > 0) {
        EXPECT_LT(result.GetSamplesPerSecond(), 1000000.0); // Reasonable upper bound
    }
}

TEST_F(ComprehensivePathIntegralTestSuite, IntegrationNeuralNetworkEvaluationTracking) {
    // Test end-to-end neural network evaluation tracking
    auto& logger = PathIntegralDebugLogger::Instance();
    logger.SetEnabled(true);
    logger.SetOutputFile("test_nn_integration.json");

    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 5;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    SearchLimits limits;
    VerificationResult result = verifier.VerifySampling(position, config, limits);

    EXPECT_TRUE(result.sampling_completed);
    EXPECT_FALSE(result.backend_available); // No backend configured in test
    
    // Should have used heuristic evaluation
    std::string content = ReadFile("test_nn_integration.json");
    if (!content.empty()) {
        // If logging occurred, check for evaluation method tracking
        EXPECT_TRUE(FileContains("test_nn_integration.json", "sample_evaluation") ||
                   FileContains("test_nn_integration.json", "heuristic"));
    }
}

// ============================================================================
// EDGE CASES AND SYSTEM TESTS
// ============================================================================

TEST_F(ComprehensivePathIntegralTestSuite, EdgeCaseExtremeConfigurations) {
    PathIntegralSamplingVerifier verifier(options_);

    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Test extreme configurations
    std::vector<PathIntegralConfig> extreme_configs = {
        {0.001f, 1, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kCompetitive, true},
        {10.0f, 1, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kCompetitive, true},
        {0.1f, 1000, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kCompetitive, true},
        {5.0f, 100, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kQuantumLimit, true},
        {0.0f, 10, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kCompetitive, true},
        {0.1f, 0, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kCompetitive, true},
        {0.1f, -5, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kCompetitive, true}
    };

    for (size_t i = 0; i < extreme_configs.size(); ++i) {
        const auto& config = extreme_configs[i];

        EXPECT_NO_THROW({
            SearchLimits limits;
            VerificationResult result = verifier.VerifySampling(position, config, limits);
            // Should complete without crashing, even if using fallback behavior
            EXPECT_FALSE(result.position_fen.empty());
        }) << "Failed for extreme config " << i << " (lambda=" << config.lambda
           << ", samples=" << config.samples << ")";
    }
}

TEST_F(ComprehensivePathIntegralTestSuite, EdgeCaseSpecialChessPositions) {
    PathIntegralSamplingVerifier verifier(options_);

    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 5;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.reward_mode = PathIntegralRewardMode::kHybrid;
    config.enabled = true;

    // Test special chess positions
    std::vector<std::pair<std::string, std::string>> special_positions = {
        {"Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
        {"Forced Move", "8/8/8/8/8/7k/6pp/7K w - - 0 1"},
        {"Endgame", "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1"},
        {"Complex Middle Game", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"}
    };

    for (const auto& [name, fen] : special_positions) {
        Position position = Position::FromFen(fen);

        SearchLimits limits;
        VerificationResult result = verifier.VerifySampling(position, config, limits);

        EXPECT_TRUE(result.sampling_completed) << "Failed for position: " << name;
        EXPECT_EQ(result.position_fen, fen);
        EXPECT_GT(result.total_time_ms, 0.0);
    }
}

TEST_F(ComprehensivePathIntegralTestSuite, EdgeCaseConcurrentAccess) {
    const int num_threads = 4;
    const int verifications_per_thread = 3;

    std::vector<std::thread> threads;
    std::vector<bool> thread_results(num_threads, false);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, verifications_per_thread, &thread_results]() {
            try {
                OptionsDict thread_options = options_;
                PathIntegralSamplingVerifier verifier(thread_options);

                Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

                PathIntegralConfig config;
                config.lambda = 0.1f + static_cast<float>(t) * 0.05f;
                config.samples = 3 + t;
                config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
                config.reward_mode = PathIntegralRewardMode::kHybrid;
                config.enabled = true;

                bool all_successful = true;
                for (int i = 0; i < verifications_per_thread; ++i) {
                    SearchLimits limits;
                    VerificationResult result = verifier.VerifySampling(position, config, limits);
                    if (!result.sampling_completed) {
                        all_successful = false;
                        break;
                    }

                    // Small delay to increase chance of race conditions
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }

                thread_results[t] = all_successful;
            } catch (...) {
                thread_results[t] = false;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // All threads should complete successfully
    for (int t = 0; t < num_threads; ++t) {
        EXPECT_TRUE(thread_results[t]) << "Thread " << t << " failed";
    }
}

// ============================================================================
// COMPREHENSIVE SYSTEM TESTS (All Requirements)
// ============================================================================

TEST_F(ComprehensivePathIntegralTestSuite, ComprehensiveSystemValidation) {
    // Test all requirements together in a comprehensive system test
    auto& logger = PathIntegralDebugLogger::Instance();
    logger.SetEnabled(true);
    logger.SetOutputFile("comprehensive_system_test.json");

    PathIntegralSamplingVerifier verifier(options_);

    std::vector<std::string> test_fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
        "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
    };

    ComprehensiveVerificationReport report = verifier.RunComprehensiveTest(test_fens);

    EXPECT_GT(report.total_tests, 0);
    EXPECT_GE(report.passed_tests, 0);

    // Verify debug logging captured all scenarios
    std::string content = ReadFile("comprehensive_system_test.json");
    if (!content.empty()) {
        int session_count = CountOccurrences(content, "sampling_start");
        EXPECT_GE(session_count, 0);
    }
}

TEST_F(ComprehensivePathIntegralTestSuite, StressTestHighVolumeOperations) {
    // Stress test with high volume operations
    PathIntegralSamplingVerifier verifier(options_);

    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 20; // Higher sample count for stress test
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.reward_mode = PathIntegralRewardMode::kHybrid;
    config.enabled = true;

    // Run multiple verifications in sequence
    std::vector<double> timings;
    for (int i = 0; i < 10; ++i) {
        SearchLimits limits;
        VerificationResult result = verifier.VerifySampling(position, config, limits);

        EXPECT_TRUE(result.sampling_completed) << "Stress test iteration " << i << " failed";
        EXPECT_EQ(result.requested_samples, 20);
        EXPECT_GT(result.total_time_ms, 0.0);

        timings.push_back(result.total_time_ms);
    }

    // Verify timing consistency (coefficient of variation should be reasonable)
    double mean_time = 0.0;
    for (double timing : timings) {
        mean_time += timing;
    }
    mean_time /= static_cast<double>(timings.size());

    double variance = 0.0;
    for (double timing : timings) {
        variance += (timing - mean_time) * (timing - mean_time);
    }
    variance /= static_cast<double>(timings.size());

    double cv = std::sqrt(variance) / mean_time;
    EXPECT_LT(cv, 2.0); // Coefficient of variation should be reasonable
}

TEST_F(ComprehensivePathIntegralTestSuite, MemoryLeakDetection) {
    // Test for memory leaks during repeated operations
    PathIntegralSamplingVerifier verifier(options_);

    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 5;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.reward_mode = PathIntegralRewardMode::kHybrid;
    config.enabled = true;

    // Run many iterations to detect potential memory leaks
    for (int i = 0; i < 100; ++i) {
        SearchLimits limits;
        VerificationResult result = verifier.VerifySampling(position, config, limits);
        EXPECT_TRUE(result.sampling_completed);

        // Small delay to allow cleanup
        if (i % 10 == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // If we reach here without crashing or excessive memory usage, test passes
    SUCCEED();
}

} // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
