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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include <gtest/gtest.h>
#include "tools/verify_path_integral_sampling.h"
#include "utils/optionsdict.h"
#include "chess/position.h"

namespace lczero {

class PathIntegralSamplingVerifierTest : public ::testing::Test {
protected:
  void SetUp() override {
    options_.Set<bool>("PathIntegralEnabled", true);
    options_.Set<float>("PathIntegralLambda", 0.1f);
    options_.Set<int>("PathIntegralSamples", 100);
    options_.Set<std::string>("PathIntegralMode", "competitive");
    options_.Set<bool>("verbose", false);
  }

  OptionsDict options_;
};

TEST_F(PathIntegralSamplingVerifierTest, ConstructorTest) {
  EXPECT_NO_THROW({
    PathIntegralSamplingVerifier verifier(options_);
  });
}

TEST_F(PathIntegralSamplingVerifierTest, GetDefaultTestPositions) {
  auto positions = PathIntegralSamplingVerifier::GetDefaultTestPositions();
  EXPECT_FALSE(positions.empty());
  EXPECT_GE(positions.size(), 5);
  
  // Check that the starting position is included
  bool has_starting_position = false;
  for (const auto& fen : positions) {
    if (fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") {
      has_starting_position = true;
      break;
    }
  }
  EXPECT_TRUE(has_starting_position);
}

TEST_F(PathIntegralSamplingVerifierTest, CreateStandardTestScenarios) {
  auto scenarios = PathIntegralSamplingVerifier::CreateStandardTestScenarios();
  EXPECT_FALSE(scenarios.empty());
  EXPECT_GE(scenarios.size(), 3);
  
  // Check that we have different lambda values
  bool has_different_lambdas = false;
  float first_lambda = scenarios[0].config.lambda;
  for (size_t i = 1; i < scenarios.size(); ++i) {
    if (scenarios[i].config.lambda != first_lambda) {
      has_different_lambdas = true;
      break;
    }
  }
  EXPECT_TRUE(has_different_lambdas);
}

TEST_F(PathIntegralSamplingVerifierTest, CreatePerformanceTestScenarios) {
  auto scenarios = PathIntegralSamplingVerifier::CreatePerformanceTestScenarios();
  EXPECT_FALSE(scenarios.empty());
  
  // Performance tests should have higher sample counts
  bool has_high_samples = false;
  for (const auto& scenario : scenarios) {
    if (scenario.config.samples >= 100) {
      has_high_samples = true;
      break;
    }
  }
  EXPECT_TRUE(has_high_samples);
}

TEST_F(PathIntegralSamplingVerifierTest, CreateEdgeCaseTestScenarios) {
  auto scenarios = PathIntegralSamplingVerifier::CreateEdgeCaseTestScenarios();
  EXPECT_FALSE(scenarios.empty());
  
  // Edge cases should include extreme values
  bool has_extreme_lambda = false;
  bool has_min_samples = false;
  
  for (const auto& scenario : scenarios) {
    if (scenario.config.lambda <= 0.001f || scenario.config.lambda >= 10.0f) {
      has_extreme_lambda = true;
    }
    if (scenario.config.samples == 1) {
      has_min_samples = true;
    }
  }
  
  EXPECT_TRUE(has_extreme_lambda);
  EXPECT_TRUE(has_min_samples);
}

TEST_F(PathIntegralSamplingVerifierTest, VerificationResultStructure) {
  VerificationResult result;
  
  // Test default values
  EXPECT_FALSE(result.IsValid());
  EXPECT_EQ(result.GetSamplesPerSecond(), 0.0);
  
  // Test with valid data
  result.samples_match_requested = true;
  result.sampling_completed = true;
  result.actual_samples = 10;
  result.total_time_ms = 100.0;
  
  EXPECT_TRUE(result.IsValid());
  EXPECT_DOUBLE_EQ(result.GetSamplesPerSecond(), 100.0); // 10 samples / 0.1 seconds
}

TEST_F(PathIntegralSamplingVerifierTest, BasicVerificationWithoutBackend) {
  PathIntegralSamplingVerifier verifier(options_);
  
  Position position = Position::FromFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  PathIntegralConfig config;
  config.lambda = 0.1f;
  config.samples = 5;
  config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
  config.enabled = true;
  
  SearchLimits limits;

  // This should work even without a neural network backend
  VerificationResult result = verifier.VerifySampling(position, config, limits);

  // Basic checks - the verification should complete even if it uses heuristics
  EXPECT_FALSE(result.position_fen.empty());
  EXPECT_EQ(result.requested_samples, 5);
  EXPECT_FALSE(result.backend_available); // No backend was set
}

TEST_F(PathIntegralSamplingVerifierTest, ComprehensiveReportStructure) {
  ComprehensiveVerificationReport report;
  
  // Test default values
  EXPECT_TRUE(report.IsOverallSuccess()); // No failures yet
  EXPECT_EQ(report.total_tests, 0);
  EXPECT_EQ(report.passed_tests, 0);
  EXPECT_EQ(report.failed_tests, 0);
  
  // Add some mock results
  VerificationResult result1;
  result1.samples_match_requested = true;
  result1.sampling_completed = true;
  
  VerificationResult result2;
  result2.samples_match_requested = false;
  result2.sampling_completed = true;
  result2.errors.push_back("Test error");
  
  report.individual_results.push_back(result1);
  report.individual_results.push_back(result2);
  
  // The report should reflect the mixed results
  EXPECT_EQ(report.individual_results.size(), 2);
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
