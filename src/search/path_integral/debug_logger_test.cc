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

#include "search/path_integral/debug_logger.h"
#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include "chess/position.h"

namespace lczero {

class PathIntegralDebugLoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary test file
        test_file_ = "test_debug_output.json";
        
        // Clean up any existing test file
        if (std::filesystem::exists(test_file_)) {
            std::filesystem::remove(test_file_);
        }
        
        // Reset logger state
        logger_ = &PathIntegralDebugLogger::Instance();
        logger_->SetEnabled(false);
        logger_->SetOutputFile("");
    }
    
    void TearDown() override {
        // Clean up test file
        if (std::filesystem::exists(test_file_)) {
            std::filesystem::remove(test_file_);
        }
        
        // Reset logger state
        logger_->SetEnabled(false);
        logger_->SetOutputFile("");
    }
    
    std::string ReadTestFile() {
        std::ifstream file(test_file_);
        if (!file.is_open()) {
            return "";
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    
    bool FileContains(const std::string& content, const std::string& search_term) {
        return content.find(search_term) != std::string::npos;
    }
    
    std::string test_file_;
    PathIntegralDebugLogger* logger_;
};

TEST_F(PathIntegralDebugLoggerTest, EnableDisableLogging) {
    // Test enabling
    logger_->SetEnabled(true);
    
    // Test disabling
    logger_->SetEnabled(false);
}

TEST_F(PathIntegralDebugLoggerTest, OutputFileConfiguration) {
    // Test setting output file
    logger_->SetOutputFile(test_file_);
    
    // Test clearing output file
    logger_->SetOutputFile("");
}

TEST_F(PathIntegralDebugLoggerTest, BasicLoggingWhenDisabled) {
    // When disabled, should not create any output
    logger_->SetEnabled(false);
    logger_->SetOutputFile(test_file_);
    
    logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    logger_->LogSamplingStart(10, 20, 0.1f, "competitive");
    logger_->EndSession();
    
    // File should not exist or be empty
    EXPECT_FALSE(std::filesystem::exists(test_file_));
}

TEST_F(PathIntegralDebugLoggerTest, BasicLoggingWhenEnabled) {
    logger_->SetEnabled(true);
    logger_->SetOutputFile(test_file_);
    
    logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    logger_->LogSamplingStart(10, 20, 0.1f, "competitive");
    logger_->EndSession();
    
    std::string content = ReadTestFile();
    EXPECT_FALSE(content.empty());
}

TEST_F(PathIntegralDebugLoggerTest, SampleEvaluationLogging) {
    logger_->SetEnabled(true);
    logger_->SetOutputFile(test_file_);
    
    Move move = Move::White(Square(kFileE, kRank2), Square(kFileE, kRank4));

    logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    logger_->LogSampleEvaluation(move, 1, 0.75f, "neural_network", 2.5);
    logger_->EndSession();
    
    std::string content = ReadTestFile();
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(FileContains(content, "sample_evaluation"));
}

TEST_F(PathIntegralDebugLoggerTest, SamplingCompleteLogging) {
    logger_->SetEnabled(true);
    logger_->SetOutputFile(test_file_);
    
    logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    logger_->LogSamplingComplete(50, 125.5, 45, 5, 0);
    logger_->EndSession();
    
    std::string content = ReadTestFile();
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(FileContains(content, "sampling_complete"));
}

TEST_F(PathIntegralDebugLoggerTest, NeuralNetworkCallLogging) {
    logger_->SetEnabled(true);
    logger_->SetOutputFile(test_file_);
    
    logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    // Test cache hit
    logger_->LogNeuralNetworkCall(true, 0.1, "cache hit");
    
    // Test cache miss
    logger_->LogNeuralNetworkCall(false, 5.2, "fresh evaluation");
    
    logger_->EndSession();
    
    std::string content = ReadTestFile();
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(FileContains(content, "neural_network_call"));
}

TEST_F(PathIntegralDebugLoggerTest, SoftmaxCalculationLogging) {
    logger_->SetEnabled(true);
    logger_->SetOutputFile(test_file_);
    
    std::vector<float> input_scores = {1.0f, 2.0f, 3.0f};
    std::vector<float> output_probabilities = {0.09f, 0.24f, 0.67f};
    
    logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    logger_->LogSoftmaxCalculation(input_scores, 0.1f, output_probabilities);
    logger_->EndSession();
    
    std::string content = ReadTestFile();
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(FileContains(content, "softmax"));
}

TEST_F(PathIntegralDebugLoggerTest, WarningAndErrorLogging) {
    logger_->SetEnabled(true);
    logger_->SetOutputFile(test_file_);
    
    logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    logger_->LogWarning("Test warning message");
    logger_->LogError("Test error message");
    logger_->LogInfo("Test info message");
    logger_->EndSession();
    
    std::string content = ReadTestFile();
    EXPECT_FALSE(content.empty());
    EXPECT_TRUE(FileContains(content, "warning") || FileContains(content, "error") || FileContains(content, "info"));
}

TEST_F(PathIntegralDebugLoggerTest, SessionManagement) {
    logger_->SetEnabled(true);
    logger_->SetOutputFile(test_file_);
    
    // First session
    logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    logger_->LogInfo("First session");
    logger_->EndSession();
    
    // Second session
    logger_->StartSession("r1bqkb1r/pppp1ppp/2n2nn2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1");
    logger_->LogInfo("Second session");
    logger_->EndSession();
    
    std::string content = ReadTestFile();
    EXPECT_FALSE(content.empty());
}

TEST_F(PathIntegralDebugLoggerTest, ErrorHandlingInvalidFile) {
    logger_->SetEnabled(true);
    
    // Try to set an invalid file path (directory that doesn't exist)
    std::string invalid_path = "/nonexistent/directory/test.json";
    logger_->SetOutputFile(invalid_path);
    
    // Should not crash when trying to log
    EXPECT_NO_THROW({
        logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        logger_->LogInfo("Test message");
        logger_->EndSession();
    });
}

TEST_F(PathIntegralDebugLoggerTest, LargeDataHandling) {
    logger_->SetEnabled(true);
    logger_->SetOutputFile(test_file_);
    
    // Test with large arrays
    std::vector<float> large_scores(1000);
    std::vector<float> large_probabilities(1000);
    
    for (size_t i = 0; i < 1000; ++i) {
        large_scores[i] = static_cast<float>(i) / 100.0f;
        large_probabilities[i] = 1.0f / 1000.0f;
    }
    
    EXPECT_NO_THROW({
        logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        logger_->LogSoftmaxCalculation(large_scores, 0.1f, large_probabilities);
        logger_->EndSession();
    });
    
    std::string content = ReadTestFile();
    EXPECT_FALSE(content.empty());
}

TEST_F(PathIntegralDebugLoggerTest, EmptyDataHandling) {
    logger_->SetEnabled(true);
    logger_->SetOutputFile(test_file_);
    
    // Test with empty vectors
    std::vector<float> empty_scores;
    std::vector<float> empty_probabilities;
    std::vector<std::pair<Move, float>> empty_move_probs;
    
    EXPECT_NO_THROW({
        logger_->StartSession("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        logger_->LogSoftmaxCalculation(empty_scores, 0.1f, empty_probabilities);
        logger_->EndSession();
    });
    
    std::string content = ReadTestFile();
    EXPECT_FALSE(content.empty());
}

} // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
