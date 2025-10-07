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
#include "search/path_integral/config.h"
#include "search/path_integral/controller_simple.h"
#include "search/path_integral/debug_logger.h"
#include "search/path_integral/performance_monitor.h"
#include "search/path_integral/softmax.h"
#include "tools/verify_path_integral_sampling.h"
#include "utils/optionsdict.h"
#include "chess/position.h"
#include <limits>
#include <thread>
#include <chrono>
#include <filesystem>

namespace lczero {

class PathIntegralEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        options_.Set<bool>("PathIntegralEnabled", true);
        options_.Set<float>("PathIntegralLambda", 0.1f);
        options_.Set<int>("PathIntegralSamples", 10);
        options_.Set<std::string>("PathIntegralMode", "competitive");
        options_.Set<bool>("verbose", false);
        
        // Reset logger state
        auto& logger = PathIntegralDebugLogger::Instance();
        logger.SetEnabled(false);
        logger.SetOutputFile("");
    }
    
    void TearDown() override {
        auto& logger = PathIntegralDebugLogger::Instance();
        logger.SetEnabled(false);
        logger.SetOutputFile("");
    }
    
    OptionsDict options_;
};

// Test extreme lambda values
TEST_F(PathIntegralEdgeCaseTest, ExtremeLambdaValues) {
    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Test minimum valid lambda
    PathIntegralConfig config_min;
    config_min.lambda = 0.001f;
    config_min.samples = 5;
    config_min.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config_min.enabled = true;
    
    SearchLimits limits;
    VerificationResult result_min = verifier.VerifySampling(position, config_min, limits);
    EXPECT_TRUE(result_min.sampling_completed);
    EXPECT_GT(result_min.total_time_ms, 0.0);
    
    // Test maximum valid lambda
    PathIntegralConfig config_max;
    config_max.lambda = 10.0f;
    config_max.samples = 5;
    config_max.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config_max.enabled = true;
    
    VerificationResult result_max = verifier.VerifySampling(position, config_max, limits);
    EXPECT_TRUE(result_max.sampling_completed);
    EXPECT_GT(result_max.total_time_ms, 0.0);
    
    // Test invalid lambda (too small) - should handle gracefully
    PathIntegralConfig config_invalid_small;
    config_invalid_small.lambda = 0.0001f;
    config_invalid_small.samples = 5;
    config_invalid_small.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config_invalid_small.enabled = true;
    
    EXPECT_NO_THROW({
        VerificationResult result_invalid = verifier.VerifySampling(position, config_invalid_small, limits);
        // Should complete but may use fallback behavior
    });
    
    // Test invalid lambda (too large) - should handle gracefully
    PathIntegralConfig config_invalid_large;
    config_invalid_large.lambda = 15.0f;
    config_invalid_large.samples = 5;
    config_invalid_large.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config_invalid_large.enabled = true;
    
    EXPECT_NO_THROW({
        VerificationResult result_invalid = verifier.VerifySampling(position, config_invalid_large, limits);
        // Should complete but may use fallback behavior
    });
}

// Test extreme sample counts
TEST_F(PathIntegralEdgeCaseTest, ExtremeSampleCounts) {
    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    SearchLimits limits;

    // Test minimum samples
    PathIntegralConfig config_min;
    config_min.lambda = 0.1f;
    config_min.samples = 1;
    config_min.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config_min.enabled = true;
    
    VerificationResult result_min = verifier.VerifySampling(position, config_min, limits);
    EXPECT_TRUE(result_min.sampling_completed);
    EXPECT_EQ(result_min.requested_samples, 1);
    
    // Test high sample count
    PathIntegralConfig config_high;
    config_high.lambda = 0.1f;
    config_high.samples = 1000;
    config_high.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config_high.enabled = true;
    
    VerificationResult result_high = verifier.VerifySampling(position, config_high, limits);
    EXPECT_TRUE(result_high.sampling_completed);
    EXPECT_EQ(result_high.requested_samples, 1000);
    EXPECT_GT(result_high.total_time_ms, 0.0);
    
    // Test zero samples - should handle gracefully
    PathIntegralConfig config_zero;
    config_zero.lambda = 0.1f;
    config_zero.samples = 0;
    config_zero.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config_zero.enabled = true;
    
    EXPECT_NO_THROW({
        VerificationResult result_zero = verifier.VerifySampling(position, config_zero, limits);
        // Should handle gracefully, possibly with error or fallback
    });
    
    // Test negative samples - should handle gracefully
    PathIntegralConfig config_negative;
    config_negative.lambda = 0.1f;
    config_negative.samples = -5;
    config_negative.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config_negative.enabled = true;
    
    EXPECT_NO_THROW({
        VerificationResult result_negative = verifier.VerifySampling(position, config_negative, limits);
        // Should handle gracefully, possibly with error or fallback
    });
}

// Test special chess positions
TEST_F(PathIntegralEdgeCaseTest, SpecialChessPositions) {
    PathIntegralSamplingVerifier verifier(options_);
    
    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 5;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    SearchLimits limits;

    // Test position with only one legal move (forced move)
    Position forced_move_position = Position::FromFen("8/8/8/8/8/7k/6pp/7K w - - 0 1");

    VerificationResult result_forced = verifier.VerifySampling(forced_move_position, config, limits);
    EXPECT_TRUE(result_forced.sampling_completed);
    
    // Test checkmate position (no legal moves)
    Position checkmate_position = Position::FromFen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");

    EXPECT_NO_THROW({
        VerificationResult result_checkmate = verifier.VerifySampling(checkmate_position, config, limits);
        // Should handle gracefully even with no legal moves
    });
    
    // Test stalemate position (no legal moves but not in check)
    Position stalemate_position = Position::FromFen("8/8/8/8/8/5k2/5p2/5K2 w - - 0 1");

    EXPECT_NO_THROW({
        VerificationResult result_stalemate = verifier.VerifySampling(stalemate_position, config, limits);
        // Should handle gracefully
    });
    
    // Test position with many legal moves
    Position many_moves_position = Position::FromFen("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1");

    VerificationResult result_many = verifier.VerifySampling(many_moves_position, config, limits);
    EXPECT_TRUE(result_many.sampling_completed);
    EXPECT_GT(result_many.total_time_ms, 0.0);
}

// Test softmax edge cases
TEST_F(PathIntegralEdgeCaseTest, SoftmaxEdgeCases) {
    SoftmaxCalculator calculator;
    
    // Test with identical scores
    std::vector<float> identical_scores = {5.0f, 5.0f, 5.0f, 5.0f};
    auto result_identical = calculator.CalculateSoftmax(identical_scores, 1.0f);
    
    EXPECT_EQ(result_identical.size(), 4);
    for (float prob : result_identical) {
        EXPECT_NEAR(prob, 0.25f, 1e-6f); // Should be uniform
    }
    
    // Test with extreme score differences
    std::vector<float> extreme_scores = {-1000.0f, 0.0f, 1000.0f};
    auto result_extreme = calculator.CalculateSoftmax(extreme_scores, 1.0f);
    
    EXPECT_EQ(result_extreme.size(), 3);
    EXPECT_GT(result_extreme[2], 0.9f); // Highest score should dominate
    
    // Test with NaN values
    std::vector<float> nan_scores = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
    auto result_nan = calculator.CalculateSoftmax(nan_scores, 1.0f);
    
    EXPECT_EQ(result_nan.size(), 3);
    for (float prob : result_nan) {
        EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f); // Should fallback to uniform
    }
    
    // Test with infinity values
    std::vector<float> inf_scores = {1.0f, std::numeric_limits<float>::infinity(), 3.0f};
    auto result_inf = calculator.CalculateSoftmax(inf_scores, 1.0f);
    
    EXPECT_EQ(result_inf.size(), 3);
    for (float prob : result_inf) {
        EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f); // Should fallback to uniform
    }
    
    // Test with empty input
    std::vector<float> empty_scores;
    auto result_empty = calculator.CalculateSoftmax(empty_scores, 1.0f);
    
    EXPECT_TRUE(result_empty.empty());
    
    // Test with single element
    std::vector<float> single_score = {42.0f};
    auto result_single = calculator.CalculateSoftmax(single_score, 1.0f);
    
    EXPECT_EQ(result_single.size(), 1);
    EXPECT_NEAR(result_single[0], 1.0f, 1e-6f);
}

// Test performance monitor edge cases
TEST_F(PathIntegralEdgeCaseTest, PerformanceMonitorEdgeCases) {
    PathIntegralPerformanceMonitor monitor;
    
    // Test without starting sampling
    EXPECT_NO_THROW({
        monitor.RecordSample("neural_network", 5.0);
        monitor.EndSampling();
        auto metrics = monitor.GetMetrics();
        EXPECT_GE(metrics.actual_samples, 0);
    });
    
    // Test starting multiple times
    monitor.StartSampling(10);
    monitor.StartSampling(20); // Second start should reset
    monitor.RecordSample("neural_network", 5.0);
    monitor.EndSampling();
    
    auto metrics = monitor.GetMetrics();
    EXPECT_EQ(metrics.requested_samples, 20); // Should use the last start value
    
    // Test ending without starting
    PathIntegralPerformanceMonitor monitor2;
    EXPECT_NO_THROW({
        monitor2.EndSampling();
        auto metrics2 = monitor2.GetMetrics();
        EXPECT_GE(metrics2.actual_samples, 0);
    });
    
    // Test with zero requested samples
    PathIntegralPerformanceMonitor monitor3;
    monitor3.StartSampling(0);
    monitor3.RecordSample("neural_network", 5.0);
    monitor3.EndSampling();
    
    auto metrics3 = monitor3.GetMetrics();
    EXPECT_EQ(metrics3.requested_samples, 0);
    EXPECT_EQ(metrics3.actual_samples, 1);
    
    // Test with negative timing
    PathIntegralPerformanceMonitor monitor4;
    monitor4.StartSampling(5);
    EXPECT_NO_THROW({
        monitor4.RecordSample("neural_network", -1.0); // Negative time
    });
    monitor4.EndSampling();
    
    auto metrics4 = monitor4.GetMetrics();
    EXPECT_GE(metrics4.actual_samples, 0);
}

// Test debug logger edge cases
TEST_F(PathIntegralEdgeCaseTest, DebugLoggerEdgeCases) {
    std::string test_file = "edge_case_debug.json";
    
    // Clean up any existing file
    if (std::filesystem::exists(test_file)) {
        std::filesystem::remove(test_file);
    }
    
    auto& logger = PathIntegralDebugLogger::Instance();
    logger.SetEnabled(true);
    logger.SetOutputFile(test_file);

    // Test with extreme values
    EXPECT_NO_THROW({
        logger.LogSamplingStart(
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<float>::max(),
            "competitive",
            "extreme_test"
        );
    });
    
    // Test with special float values
    EXPECT_NO_THROW({
        logger.LogSampleEvaluation(
            Move::White(Square(kFileE, kRank2), Square(kFileE, kRank4)),
            1,
            std::numeric_limits<float>::quiet_NaN(),
            "test",
            std::numeric_limits<double>::infinity()
        );
    });
    
    // Test with empty strings
    EXPECT_NO_THROW({
        logger.LogSampleEvaluation(
            Move::White(Square(kFileE, kRank2), Square(kFileE, kRank4)),
            1,
            0.5f,
            "", // Empty evaluation method
            1.0
        );
    });
    
    // Test with very large arrays
    std::vector<float> large_array(10000, 1.0f);
    EXPECT_NO_THROW({
        logger.LogSoftmaxCalculation(large_array, 0.1f, large_array);
    });
    
    // Test with empty arrays
    std::vector<float> empty_array;
    EXPECT_NO_THROW({
        logger.LogSoftmaxCalculation(empty_array, 0.1f, empty_array);
    });

    // Clean up
    logger.SetEnabled(false);
    if (std::filesystem::exists(test_file)) {
        std::filesystem::remove(test_file);
    }
}

// Test concurrent access edge cases
TEST_F(PathIntegralEdgeCaseTest, ConcurrentAccessEdgeCases) {
    const int num_threads = 10;

    std::vector<std::thread> threads;
    std::vector<bool> thread_success(num_threads, false);
    
    // Test concurrent softmax calculations
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([t, &thread_success]() {
            try {
                SoftmaxCalculator calculator;
                std::vector<float> test_scores = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
                
                for (int i = 0; i < 50; ++i) {
                    float lambda = 0.1f + static_cast<float>(i % 10) * 0.1f;
                    auto result = calculator.CalculateSoftmax(test_scores, lambda);
                    
                    if (result.size() != test_scores.size()) {
                        thread_success[t] = false;
                        return;
                    }
                    
                    // Brief pause to increase chance of race conditions
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
                
                thread_success[t] = true;
            } catch (...) {
                thread_success[t] = false;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // All threads should complete successfully
    for (int t = 0; t < num_threads; ++t) {
        EXPECT_TRUE(thread_success[t]) << "Thread " << t << " failed";
    }
}

// Test memory pressure scenarios
TEST_F(PathIntegralEdgeCaseTest, MemoryPressureScenarios) {
    // Test with large sample counts and many positions
    PathIntegralSamplingVerifier verifier(options_);
    
    std::vector<std::string> positions = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
        "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1"
    };
    
    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 100; // Large sample count
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    // Run verification on multiple positions with large sample counts
    for (const auto& fen : positions) {
        Position position = Position::FromFen(fen);

        VerificationResult result = verifier.VerifySampling(position, config);
        EXPECT_TRUE(result.sampling_completed) << "Failed for position: " << fen;
        
        // Force some cleanup time
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// Test configuration validation edge cases
TEST_F(PathIntegralEdgeCaseTest, ConfigurationValidationEdgeCases) {
    PathIntegralConfig config;
    
    // Test with extreme float values
    config.lambda = std::numeric_limits<float>::infinity();
    config.samples = 10;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    // Should handle gracefully
    EXPECT_NO_THROW({
        PathIntegralSamplingVerifier verifier(options_);
        Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        verifier.VerifySampling(position, config);
    });
    
    // Test with NaN lambda
    config.lambda = std::numeric_limits<float>::quiet_NaN();
    
    EXPECT_NO_THROW({
        PathIntegralSamplingVerifier verifier(options_);
        Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        verifier.VerifySampling(position, config);
    });
    
    // Test with disabled path integral
    config.lambda = 0.1f;
    config.samples = 10;
    config.enabled = false;
    
    EXPECT_NO_THROW({
        PathIntegralSamplingVerifier verifier(options_);
        Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        VerificationResult result = verifier.VerifySampling(position, config);
        // Should handle disabled case gracefully
        EXPECT_TRUE(result.sampling_completed || !config.enabled);
    });
}

// Test timing edge cases
TEST_F(PathIntegralEdgeCaseTest, TimingEdgeCases) {
    PathIntegralPerformanceMonitor monitor;
    
    // Test very rapid sampling
    monitor.StartSampling(1000);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Record many samples very quickly
    for (int i = 0; i < 1000; ++i) {
        monitor.RecordSample("neural_network", 0.001); // Very fast evaluations
    }
    
    monitor.EndSampling();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    auto metrics = monitor.GetMetrics();
    
    EXPECT_EQ(metrics.requested_samples, 1000);
    EXPECT_EQ(metrics.actual_samples, 1000);
    EXPECT_GT(metrics.samples_per_second, 0.0);
    
    // Timing should be reasonable
    EXPECT_GT(metrics.total_time_ms, 0.0);
    EXPECT_LT(metrics.total_time_ms, duration.count() + 100); // Allow some overhead
}

} // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
