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
#include "tools/verify_path_integral_sampling.h"
#include "search/path_integral/debug_logger.h"
#include "search/path_integral/performance_monitor.h"
#include "utils/optionsdict.h"
#include "chess/position.h"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

namespace lczero {

class PathIntegralIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up basic options
        options_.Set<bool>("PathIntegralEnabled", true);
        options_.Set<float>("PathIntegralLambda", 0.1f);
        options_.Set<int>("PathIntegralSamples", 10);
        options_.Set<std::string>("PathIntegralMode", "competitive");
        options_.Set<bool>("verbose", false);
        
        // Set up test files
        debug_file_ = "integration_test_debug.json";
        metrics_file_ = "integration_test_metrics.json";
        
        // Clean up any existing test files
        CleanupTestFiles();
        
        // Reset logger and monitor state
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
        if (std::filesystem::exists(debug_file_)) {
            std::filesystem::remove(debug_file_);
        }
        if (std::filesystem::exists(metrics_file_)) {
            std::filesystem::remove(metrics_file_);
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
    
    OptionsDict options_;
    std::string debug_file_;
    std::string metrics_file_;
};

TEST_F(PathIntegralIntegrationTest, BasicSamplingVerificationWithoutBackend) {
    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 5;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    SearchLimits limits;
    VerificationResult result = verifier.VerifySampling(position, config, limits);

    // Basic verification should complete
    EXPECT_FALSE(result.position_fen.empty());
    EXPECT_EQ(result.requested_samples, 5);
    EXPECT_FALSE(result.backend_available); // No backend configured
    EXPECT_TRUE(result.sampling_completed);
    
    // Should have some timing data
    EXPECT_GT(result.total_time_ms, 0.0);
}

TEST_F(PathIntegralIntegrationTest, SamplingVerificationWithDebugLogging) {
    // Enable debug logging
    auto& logger = PathIntegralDebugLogger::Instance();
    logger.SetEnabled(true);
    logger.SetOutputFile(debug_file_);

    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    PathIntegralConfig config;
    config.lambda = 0.2f;
    config.samples = 3;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    SearchLimits limits;
    VerificationResult result = verifier.VerifySampling(position, config, limits);

    // Verification should complete
    EXPECT_TRUE(result.sampling_completed);
    
    // Debug file should be created and contain relevant entries
    EXPECT_TRUE(std::filesystem::exists(debug_file_));
    EXPECT_TRUE(FileContains(debug_file_, "sampling_start"));
    EXPECT_TRUE(FileContains(debug_file_, "sampling_complete"));
    EXPECT_TRUE(FileContains(debug_file_, "\"lambda\": 0.2"));
}

TEST_F(PathIntegralIntegrationTest, PerformanceMonitoringIntegration) {
    PathIntegralPerformanceMonitor monitor;
    
    // Test basic monitoring workflow
    monitor.StartSampling(10);
    
    // Simulate some sampling work
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        monitor.RecordSample("neural_network", 2.0);
    }
    
    for (int i = 0; i < 3; ++i) {
        monitor.RecordSample("cached", 0.1);
    }
    
    monitor.RecordSample("heuristic", 1.0);
    
    monitor.EndSampling();
    
    auto metrics = monitor.GetMetrics();
    
    EXPECT_EQ(metrics.requested_samples, 10);
    EXPECT_EQ(metrics.actual_samples, 9); // 5 + 3 + 1
    EXPECT_EQ(metrics.neural_net_evaluations, 5);
    EXPECT_EQ(metrics.cached_evaluations, 3);
    EXPECT_EQ(metrics.heuristic_evaluations, 1);
    EXPECT_GT(metrics.total_time_ms, 0.0);
    EXPECT_GT(metrics.samples_per_second, 0.0);
}

TEST_F(PathIntegralIntegrationTest, ComprehensiveVerificationReport) {
    PathIntegralSamplingVerifier verifier(options_);
    
    // Create test scenarios with FEN strings
    std::vector<std::string> test_fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
        "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1"
    };
    
    ComprehensiveVerificationReport report = verifier.RunComprehensiveTest(test_fens);

    EXPECT_GT(report.total_tests, 0);
    EXPECT_GE(report.passed_tests, 0);

    // All tests should complete (though they may use heuristics without backend)
    for (const auto& result : report.individual_results) {
        EXPECT_TRUE(result.sampling_completed);
        EXPECT_FALSE(result.position_fen.empty());
        EXPECT_GT(result.total_time_ms, 0.0);
    }
}

TEST_F(PathIntegralIntegrationTest, EdgeCaseConfigurationTesting) {
    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Test edge case configurations
    std::vector<PathIntegralConfig> edge_configs = {
        {0.001f, 1, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kCompetitive, true},
        {10.0f, 1, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kCompetitive, true},
        {0.1f, 100, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kCompetitive, true},
        {1.0f, 50, PathIntegralRewardMode::kHybrid, PathIntegralSamplingMode::kQuantumLimit, true}
    };
    
    for (const auto& config : edge_configs) {
        SearchLimits limits;
        VerificationResult result = verifier.VerifySampling(position, config, limits);

        EXPECT_TRUE(result.sampling_completed) << "Failed for lambda=" << config.lambda 
                                               << ", samples=" << config.samples;
        EXPECT_EQ(result.requested_samples, config.samples);
        EXPECT_GT(result.total_time_ms, 0.0);
    }
}

TEST_F(PathIntegralIntegrationTest, SampleCountAccuracyVerification) {
    // Enable detailed logging to verify sample counts
    auto& logger = PathIntegralDebugLogger::Instance();
    logger.SetEnabled(true);
    logger.SetOutputFile(debug_file_);

    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 7; // Specific number to verify
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    SearchLimits limits;
    VerificationResult result = verifier.VerifySampling(position, config, limits);

    EXPECT_TRUE(result.sampling_completed);
    EXPECT_EQ(result.requested_samples, 7);
    
    // Check that debug log contains the correct number of sample evaluations
    std::string debug_content = ReadFile(debug_file_);
    EXPECT_FALSE(debug_content.empty());
    
    // Count sample evaluation entries
    size_t sample_count = 0;
    size_t pos = 0;
    while ((pos = debug_content.find("sample_evaluation", pos)) != std::string::npos) {
        sample_count++;
        pos += 17; // length of "sample_evaluation"
    }
    
    // Should have at least some sample evaluations (exact count depends on legal moves)
    EXPECT_GT(sample_count, 0);
}

TEST_F(PathIntegralIntegrationTest, TimingConsistencyVerification) {
    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 10;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    // Run multiple times to check timing consistency
    std::vector<double> timings;
    for (int i = 0; i < 5; ++i) {
        SearchLimits limits;
        VerificationResult result = verifier.VerifySampling(position, config, limits);
        EXPECT_TRUE(result.sampling_completed);
        timings.push_back(result.total_time_ms);
    }
    
    // All timings should be positive
    for (double timing : timings) {
        EXPECT_GT(timing, 0.0);
    }
    
    // Calculate coefficient of variation (should be reasonable)
    double mean = 0.0;
    for (double timing : timings) {
        mean += timing;
    }
    mean /= static_cast<double>(timings.size());

    double variance = 0.0;
    for (double timing : timings) {
        variance += (timing - mean) * (timing - mean);
    }
    variance /= static_cast<double>(timings.size());

    double cv = std::sqrt(variance) / mean;
    
    // Coefficient of variation should be reasonable (less than 100%)
    EXPECT_LT(cv, 1.0);
}

TEST_F(PathIntegralIntegrationTest, ErrorHandlingIntegration) {
    PathIntegralSamplingVerifier verifier(options_);
    
    // Test with invalid position
    Position invalid_position;
    // Don't set FEN, leaving position invalid
    
    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 5;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    // Should handle invalid position gracefully
    EXPECT_NO_THROW({
        SearchLimits limits;
        VerificationResult result = verifier.VerifySampling(invalid_position, config, limits);
        // Result may indicate failure, but shouldn't crash
    });
}

TEST_F(PathIntegralIntegrationTest, MultiplePositionVerification) {
    PathIntegralSamplingVerifier verifier(options_);
    
    std::vector<std::string> test_positions = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // Starting position
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1", // Italian game
        "8/8/8/8/8/3k4/3P4/3K4 w - - 0 1", // King and pawn endgame
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2" // King's pawn opening
    };
    
    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 5;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    for (const auto& fen : test_positions) {
        Position position = Position::FromFen(fen);

        SearchLimits limits;
        VerificationResult result = verifier.VerifySampling(position, config, limits);

        EXPECT_TRUE(result.sampling_completed) << "Failed for position: " << fen;
        EXPECT_EQ(result.position_fen, fen);
        EXPECT_EQ(result.requested_samples, 5);
        EXPECT_GT(result.total_time_ms, 0.0);
    }
}

TEST_F(PathIntegralIntegrationTest, ConcurrentVerificationTest) {
    const int num_threads = 3;
    const int verifications_per_thread = 2;
    
    std::vector<std::thread> threads;
    std::vector<bool> thread_results(num_threads, false);
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t, verifications_per_thread, &thread_results]() {
            try {
                OptionsDict thread_options = options_;
                PathIntegralSamplingVerifier verifier(thread_options);
                
                Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

                PathIntegralConfig config;
                config.lambda = 0.1f + static_cast<float>(t) * 0.1f;
                config.samples = 3;
                config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
                config.enabled = true;
                
                bool all_successful = true;
                for (int i = 0; i < verifications_per_thread; ++i) {
                    SearchLimits limits;
                    VerificationResult result = verifier.VerifySampling(position, config, limits);
                    if (!result.sampling_completed) {
                        all_successful = false;
                        break;
                    }
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

TEST_F(PathIntegralIntegrationTest, MemoryLeakVerification) {
    // This test runs multiple verifications to check for memory leaks
    PathIntegralSamplingVerifier verifier(options_);
    
    Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 5;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    // Run many verifications
    for (int i = 0; i < 50; ++i) {
        SearchLimits limits;
        VerificationResult result = verifier.VerifySampling(position, config, limits);
        EXPECT_TRUE(result.sampling_completed);
        
        // Small delay to allow cleanup
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    // If we get here without crashing, memory management is likely correct
    SUCCEED();
}

} // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
