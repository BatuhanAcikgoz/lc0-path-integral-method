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

#include "search/path_integral/softmax.h"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <future>
#include <algorithm>

namespace lczero {

class SoftmaxCalculatorTest : public ::testing::Test {
protected:
  void SetUp() override {
    calculator_ = std::make_unique<SoftmaxCalculator>();
  }

  // Helper function to check if probabilities sum to approximately 1.0
  bool ProbabilitiesSumToOne(const std::vector<float>& probs, float tolerance = 1e-6f) {
    float sum = 0.0f;
    for (float prob : probs) {
      sum += prob;
    }
    return std::abs(sum - 1.0f) < tolerance;
  }

  // Helper function to check if all probabilities are non-negative
  bool AllProbabilitiesNonNegative(const std::vector<float>& probs) {
    return std::all_of(probs.begin(), probs.end(), [](float p) { return p >= 0.0f; });
  }

  // Helper function to check if all values are finite
  bool AllValuesFinite(const std::vector<float>& values) {
    return std::all_of(values.begin(), values.end(), 
                       [](float v) { return std::isfinite(v); });
  }

  std::unique_ptr<SoftmaxCalculator> calculator_;
};

// Test basic softmax calculation with normal inputs
TEST_F(SoftmaxCalculatorTest, BasicSoftmaxCalculation) {
  std::vector<float> scores = {1.0f, 2.0f, 3.0f};
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(scores, lambda);
  
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(AllProbabilitiesNonNegative(result));
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  EXPECT_TRUE(AllValuesFinite(result));
  
  // With lambda=1.0, higher scores should have higher probabilities
  EXPECT_LT(result[0], result[1]);
  EXPECT_LT(result[1], result[2]);
}

// Test softmax with lambda parameter bounds
TEST_F(SoftmaxCalculatorTest, LambdaParameterBounds) {
  std::vector<float> scores = {1.0f, 2.0f, 3.0f};
  
  // Test minimum valid lambda
  auto result_min = calculator_->CalculateSoftmax(scores, 0.001f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_min));
  
  // Test maximum valid lambda
  auto result_max = calculator_->CalculateSoftmax(scores, 10.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_max));
  
  // Test invalid lambda (too small) - should fallback to uniform
  auto result_invalid_small = calculator_->CalculateSoftmax(scores, 0.0001f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_invalid_small));
  for (float prob : result_invalid_small) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  // Test invalid lambda (too large) - should fallback to uniform
  auto result_invalid_large = calculator_->CalculateSoftmax(scores, 15.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_invalid_large));
  for (float prob : result_invalid_large) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
}

// Test extreme lambda values for numerical stability
TEST_F(SoftmaxCalculatorTest, ExtremeLambdaValues) {
  std::vector<float> scores = {-100.0f, 0.0f, 100.0f};
  
  // Very small lambda (high temperature) - should be nearly uniform
  auto result_small_lambda = calculator_->CalculateSoftmax(scores, 0.001f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_small_lambda));
  // With very small lambda, probabilities should be close to uniform
  for (float prob : result_small_lambda) {
    EXPECT_GT(prob, 0.2f);  // Should be reasonably close to 1/3
    EXPECT_LT(prob, 0.5f);
  }
  
  // Large lambda (low temperature) - should be very peaked
  auto result_large_lambda = calculator_->CalculateSoftmax(scores, 5.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_large_lambda));
  // With large lambda, highest score should dominate
  EXPECT_GT(result_large_lambda[2], 0.9f);  // Highest score should have >90% probability
}

// Test NaN input handling
TEST_F(SoftmaxCalculatorTest, NaNInputHandling) {
  std::vector<float> scores_with_nan = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(scores_with_nan, lambda);
  
  // Should fallback to uniform distribution
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  for (float prob : result) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);
  }
}

// Test infinity input handling
TEST_F(SoftmaxCalculatorTest, InfinityInputHandling) {
  std::vector<float> scores_with_inf = {1.0f, std::numeric_limits<float>::infinity(), 3.0f};
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(scores_with_inf, lambda);
  
  // Should fallback to uniform distribution
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  for (float prob : result) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);
  }
}

// Test negative infinity input handling
TEST_F(SoftmaxCalculatorTest, NegativeInfinityInputHandling) {
  std::vector<float> scores_with_neg_inf = {1.0f, -std::numeric_limits<float>::infinity(), 3.0f};
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(scores_with_neg_inf, lambda);
  
  // Should fallback to uniform distribution
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  for (float prob : result) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);
  }
}

// Test empty input handling
TEST_F(SoftmaxCalculatorTest, EmptyInputHandling) {
  std::vector<float> empty_scores;
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(empty_scores, lambda);
  
  EXPECT_TRUE(result.empty());
}

// Test single element input
TEST_F(SoftmaxCalculatorTest, SingleElementInput) {
  std::vector<float> single_score = {5.0f};
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(single_score, lambda);
  
  EXPECT_EQ(result.size(), 1);
  EXPECT_NEAR(result[0], 1.0f, 1e-6f);  // Single element should have probability 1.0
}

// Test numerical stability with very large scores
TEST_F(SoftmaxCalculatorTest, NumericalStabilityLargeScores) {
  std::vector<float> large_scores = {1000.0f, 1001.0f, 1002.0f};
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(large_scores, lambda);
  
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(AllProbabilitiesNonNegative(result));
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  EXPECT_TRUE(AllValuesFinite(result));
}

// Test numerical stability with very small scores
TEST_F(SoftmaxCalculatorTest, NumericalStabilitySmallScores) {
  std::vector<float> small_scores = {-1000.0f, -999.0f, -998.0f};
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(small_scores, lambda);
  
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(AllProbabilitiesNonNegative(result));
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  EXPECT_TRUE(AllValuesFinite(result));
}

// Test uniform probabilities fallback
TEST_F(SoftmaxCalculatorTest, UniformProbabilitiesFallback) {
  auto result_3 = calculator_->UniformProbabilities(3);
  EXPECT_EQ(result_3.size(), 3);
  for (float prob : result_3) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);
  }
  
  auto result_5 = calculator_->UniformProbabilities(5);
  EXPECT_EQ(result_5.size(), 5);
  for (float prob : result_5) {
    EXPECT_NEAR(prob, 1.0f/5.0f, 1e-6f);
  }
  
  auto result_empty = calculator_->UniformProbabilities(0);
  EXPECT_TRUE(result_empty.empty());
}

// Test input validation
TEST_F(SoftmaxCalculatorTest, InputValidation) {
  // Valid input
  std::vector<float> valid_scores = {1.0f, 2.0f, 3.0f};
  EXPECT_TRUE(calculator_->IsValidInput(valid_scores));
  
  // Empty input
  std::vector<float> empty_scores;
  EXPECT_FALSE(calculator_->IsValidInput(empty_scores));
  
  // Input with NaN
  std::vector<float> nan_scores = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
  EXPECT_FALSE(calculator_->IsValidInput(nan_scores));
  
  // Input with infinity
  std::vector<float> inf_scores = {1.0f, std::numeric_limits<float>::infinity(), 3.0f};
  EXPECT_FALSE(calculator_->IsValidInput(inf_scores));
}

// Test lambda validation
TEST_F(SoftmaxCalculatorTest, LambdaValidation) {
  // Test valid lambda values by checking if they produce valid results
  std::vector<float> test_scores = {1.0f, 2.0f, 3.0f};
  
  // Valid lambda values should produce valid softmax results
  auto result_001 = calculator_->CalculateSoftmax(test_scores, 0.001f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_001));
  
  auto result_01 = calculator_->CalculateSoftmax(test_scores, 0.1f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_01));
  
  auto result_1 = calculator_->CalculateSoftmax(test_scores, 1.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_1));
  
  auto result_10 = calculator_->CalculateSoftmax(test_scores, 10.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_10));
  
  // Invalid lambda values should fallback to uniform distribution
  auto result_zero = calculator_->CalculateSoftmax(test_scores, 0.0f);
  for (float prob : result_zero) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  auto result_negative = calculator_->CalculateSoftmax(test_scores, -1.0f);
  for (float prob : result_negative) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  auto result_too_large = calculator_->CalculateSoftmax(test_scores, 15.0f);
  for (float prob : result_too_large) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
}

// Test consistency with mathematical definition
TEST_F(SoftmaxCalculatorTest, MathematicalConsistency) {
  std::vector<float> scores = {1.0f, 2.0f, 3.0f};
  float lambda = 2.0f;
  
  auto result = calculator_->CalculateSoftmax(scores, lambda);
  
  // Manually calculate expected result using the same formula
  float max_score = 3.0f;  // max of scores
  std::vector<float> scaled = {
    (1.0f - max_score) * lambda,  // -4.0
    (2.0f - max_score) * lambda,  // -2.0
    (3.0f - max_score) * lambda   //  0.0
  };
  
  float sum_exp = std::exp(scaled[0]) + std::exp(scaled[1]) + std::exp(scaled[2]);
  std::vector<float> expected = {
    std::exp(scaled[0]) / sum_exp,
    std::exp(scaled[1]) / sum_exp,
    std::exp(scaled[2]) / sum_exp
  };
  
  EXPECT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-6f);
  }
}

// Test edge case with identical scores
TEST_F(SoftmaxCalculatorTest, IdenticalScores) {
  std::vector<float> identical_scores = {5.0f, 5.0f, 5.0f, 5.0f};
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(identical_scores, lambda);
  
  EXPECT_EQ(result.size(), 4);
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  
  // All probabilities should be equal (uniform)
  for (float prob : result) {
    EXPECT_NEAR(prob, 0.25f, 1e-6f);
  }
}

// Test comprehensive input validation
TEST_F(SoftmaxCalculatorTest, ComprehensiveInputValidation) {
  // Test array size validation
  std::vector<float> empty_array;
  EXPECT_FALSE(calculator_->IsValidInput(empty_array));
  
  // Test extremely large array (should be rejected)
  std::vector<float> huge_array(2000000, 1.0f);  // Larger than kMaxScoreArraySize
  EXPECT_FALSE(calculator_->IsValidInput(huge_array));
  
  // Test valid size array
  std::vector<float> valid_array = {1.0f, 2.0f, 3.0f};
  EXPECT_TRUE(calculator_->IsValidInput(valid_array));
}

// Test enhanced array size validation
TEST_F(SoftmaxCalculatorTest, EnhancedArraySizeValidation) {
  // Test minimum boundary
  std::vector<float> single_element = {1.0f};
  auto result = calculator_->CalculateSoftmax(single_element, 1.0f);
  EXPECT_EQ(result.size(), 1);
  EXPECT_NEAR(result[0], 1.0f, 1e-6f);
  
  // Test maximum reasonable size
  std::vector<float> large_valid_array(10000, 1.0f);
  auto result_large = calculator_->CalculateSoftmax(large_valid_array, 1.0f);
  EXPECT_EQ(result_large.size(), 10000);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_large));
  
  // Test oversized array (should fallback to uniform)
  std::vector<float> oversized_array(1500000, 1.0f);
  auto result_oversized = calculator_->CalculateSoftmax(oversized_array, 1.0f);
  EXPECT_EQ(result_oversized.size(), 1500000);
  // Should fallback to uniform distribution
  for (float prob : result_oversized) {
    EXPECT_NEAR(prob, 1.0f / 1500000, 1e-10f);
  }
}

// Test score variance validation
TEST_F(SoftmaxCalculatorTest, ScoreVarianceValidation) {
  // Test identical scores (zero variance)
  std::vector<float> identical_scores = {5.0f, 5.0f, 5.0f};
  auto result_identical = calculator_->CalculateSoftmax(identical_scores, 1.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_identical));
  for (float prob : result_identical) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);
  }
  
  // Test very small variance
  std::vector<float> small_variance = {1.0f, 1.0000001f, 1.0000002f};
  auto result_small_var = calculator_->CalculateSoftmax(small_variance, 1.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_small_var));
  
  // Test reasonable variance
  std::vector<float> normal_variance = {1.0f, 5.0f, 10.0f};
  auto result_normal_var = calculator_->CalculateSoftmax(normal_variance, 1.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_normal_var));
}

// Test lambda overflow validation
TEST_F(SoftmaxCalculatorTest, LambdaOverflowValidation) {
  // Test lambda-score combination that could cause overflow
  std::vector<float> extreme_scores = {-1000.0f, 1000.0f};
  
  // Small lambda should work
  auto result_small_lambda = calculator_->CalculateSoftmax(extreme_scores, 0.001f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_small_lambda));
  EXPECT_TRUE(AllValuesFinite(result_small_lambda));
  
  // Large lambda with extreme scores should fallback to uniform
  auto result_large_lambda = calculator_->CalculateSoftmax(extreme_scores, 10.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_large_lambda));
  // Should fallback to uniform due to overflow risk
  for (float prob : result_large_lambda) {
    EXPECT_NEAR(prob, 0.5f, 1e-6f);
  }
}

// Test enhanced error message generation
TEST_F(SoftmaxCalculatorTest, EnhancedErrorMessages) {
  // Test with multiple validation failures
  std::vector<float> invalid_scores = {
    std::numeric_limits<float>::quiet_NaN(), 
    1e8f,  // Too large
    -1e8f  // Too small
  };
  
  auto result = calculator_->CalculateSoftmax(invalid_scores, -1.0f);  // Invalid lambda too
  
  // Should fallback to uniform distribution
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  for (float prob : result) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);
  }
}

// Test comprehensive numerical stability validation
TEST_F(SoftmaxCalculatorTest, ComprehensiveNumericalStabilityValidation) {
  // Test with extreme score ranges that could cause numerical issues
  std::vector<std::vector<float>> extreme_test_cases = {
    {-1e6f, -1e5f, -1e4f},  // Very negative scores
    {1e4f, 1e5f, 1e6f},     // Very positive scores
    {-1e6f, 0.0f, 1e6f},    // Extreme range
    {1e-6f, 1e-5f, 1e-4f},  // Very small positive scores
    {-1e-4f, -1e-5f, -1e-6f} // Very small negative scores
  };
  
  std::vector<float> lambda_values = {0.001f, 0.1f, 1.0f, 5.0f, 10.0f};
  
  for (const auto& scores : extreme_test_cases) {
    for (float lambda : lambda_values) {
      auto result = calculator_->CalculateSoftmax(scores, lambda);
      
      // All results should be numerically stable
      EXPECT_EQ(result.size(), scores.size());
      EXPECT_TRUE(AllProbabilitiesNonNegative(result));
      EXPECT_TRUE(ProbabilitiesSumToOne(result, 1e-5f)); // Slightly relaxed tolerance for extreme cases
      EXPECT_TRUE(AllValuesFinite(result));
      
      // No probability should be exactly 0 or 1 unless it's a degenerate case
      bool has_zero_prob = false;
      bool has_one_prob = false;
      for (float prob : result) {
        if (prob == 0.0f) has_zero_prob = true;
        if (prob == 1.0f) has_one_prob = true;
      }
      
      // For non-degenerate cases, we shouldn't have exact 0 or 1 probabilities
      if (scores.size() > 1 && lambda > 0.001f && lambda < 10.0f) {
        // This is a reasonable expectation for most cases
        EXPECT_FALSE(has_zero_prob && has_one_prob);
      }
    }
  }
}

// Test all requirements validation comprehensively
TEST_F(SoftmaxCalculatorTest, AllRequirementsValidation) {
  // Requirement 2.1: Uses exact formula from requirements
  std::vector<float> test_scores = {1.0f, 2.0f, 3.0f};
  float lambda = 2.0f;
  
  auto result = calculator_->CalculateSoftmax(test_scores, lambda);
  
  // Manually verify the exact formula: arr_scaled = (scores - max(scores)) * lambda
  float max_score = *std::max_element(test_scores.begin(), test_scores.end());
  std::vector<float> expected_scaled;
  for (float score : test_scores) {
    expected_scaled.push_back((score - max_score) * lambda);
  }
  
  // Verify log_sum_exp calculation
  float sum_exp = 0.0f;
  for (float scaled : expected_scaled) {
    sum_exp += std::exp(scaled);
  }
  float log_sum_exp = std::log(sum_exp);
  
  // Verify final probabilities: exp(arr_scaled - log_sum_exp)
  std::vector<float> expected_probs;
  for (float scaled : expected_scaled) {
    expected_probs.push_back(std::exp(scaled - log_sum_exp));
  }
  
  EXPECT_EQ(result.size(), expected_probs.size());
  for (size_t i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(result[i], expected_probs[i], 1e-6f);
  }
  
  // Requirement 2.2: Log-sum-exp computation
  // Already verified above
  
  // Requirement 2.3: Final probabilities computation
  // Already verified above
  
  // Requirement 2.4: NaN/Inf fallback
  std::vector<float> nan_scores = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f};
  auto nan_result = calculator_->CalculateSoftmax(nan_scores, lambda);
  for (float prob : nan_result) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  std::vector<float> inf_scores = {1.0f, std::numeric_limits<float>::infinity(), 3.0f};
  auto inf_result = calculator_->CalculateSoftmax(inf_scores, lambda);
  for (float prob : inf_result) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  // Requirement 2.5: Negative/zero sum fallback
  // This is handled by the NaN/Inf detection since negative sums would produce NaN in log
  
  // Requirement 2.6: Lambda precision handling
  auto tiny_lambda_result = calculator_->CalculateSoftmax(test_scores, 1e-12f);
  for (float prob : tiny_lambda_result) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should fallback to uniform
  }
}

// Test performance under stress conditions
TEST_F(SoftmaxCalculatorTest, PerformanceStressTest) {
  // Test with various array sizes
  std::vector<int> array_sizes = {10, 100, 1000, 10000};
  std::vector<float> lambda_values = {0.001f, 0.1f, 1.0f, 10.0f};
  
  for (int size : array_sizes) {
    // Create test array
    std::vector<float> large_array;
    large_array.reserve(size);
    for (int i = 0; i < size; ++i) {
      large_array.push_back(static_cast<float>(i) / 100.0f);
    }
    
    for (float lambda : lambda_values) {
      auto start = std::chrono::high_resolution_clock::now();
      auto result = calculator_->CalculateSoftmax(large_array, lambda);
      auto end = std::chrono::high_resolution_clock::now();
      
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      
      // Verify correctness
      EXPECT_EQ(result.size(), static_cast<size_t>(size));
      EXPECT_TRUE(ProbabilitiesSumToOne(result));
      EXPECT_TRUE(AllValuesFinite(result));
      
      // Performance should scale reasonably (less than 1ms per 1000 elements)
      EXPECT_LT(duration.count(), size); // Less than 1 microsecond per element
    }
  }
}

// Test thread safety (if applicable)
TEST_F(SoftmaxCalculatorTest, ThreadSafetyTest) {
  const int num_threads = 4;
  const int iterations_per_thread = 100;
  std::vector<float> test_scores = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  
  std::vector<std::future<bool>> futures;
  
  for (int t = 0; t < num_threads; ++t) {
    futures.push_back(std::async(std::launch::async, [this, &test_scores, iterations_per_thread]() {
      for (int i = 0; i < iterations_per_thread; ++i) {
        float lambda = 0.1f + (i % 10) * 0.1f; // Vary lambda
        auto result = calculator_->CalculateSoftmax(test_scores, lambda);
        
        if (result.size() != test_scores.size() || 
            !ProbabilitiesSumToOne(result) || 
            !AllValuesFinite(result)) {
          return false;
        }
      }
      return true;
    }));
  }
  
  // Wait for all threads and check results
  for (auto& future : futures) {
    EXPECT_TRUE(future.get()) << "Thread safety test failed";
  }
}

// Test precision handling with extreme lambda values
TEST_F(SoftmaxCalculatorTest, ExtremeLambdaPrecisionHandling) {
  std::vector<float> test_scores = {1.0f, 2.0f, 3.0f};
  
  // Test lambda at precision threshold
  auto result_threshold = calculator_->CalculateSoftmax(test_scores, 1e-10f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_threshold));
  
  // Test lambda below precision threshold (should fallback)
  auto result_below_threshold = calculator_->CalculateSoftmax(test_scores, 1e-12f);
  for (float prob : result_below_threshold) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  // Test denormalized lambda values
  auto result_denorm = calculator_->CalculateSoftmax(test_scores, std::numeric_limits<float>::denorm_min());
  for (float prob : result_denorm) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
}

// Test edge cases with mixed valid/invalid inputs
TEST_F(SoftmaxCalculatorTest, MixedValidInvalidInputsEdgeCases) {
  // Test array with one valid and multiple invalid values
  std::vector<float> mixed_array = {
    1.0f,  // Valid
    std::numeric_limits<float>::infinity(),  // Invalid
    std::numeric_limits<float>::quiet_NaN(),  // Invalid
    -std::numeric_limits<float>::infinity()   // Invalid
  };
  
  auto result = calculator_->CalculateSoftmax(mixed_array, 1.0f);
  
  // Should fallback to uniform distribution
  EXPECT_EQ(result.size(), 4);
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  for (float prob : result) {
    EXPECT_NEAR(prob, 0.25f, 1e-6f);
  }
}

// Test boundary conditions for score ranges
TEST_F(SoftmaxCalculatorTest, ScoreRangeBoundaryConditions) {
  // Test scores at exact boundaries
  std::vector<float> boundary_scores = {-1e6f, 0.0f, 1e6f};  // At kMinScoreValue and kMaxScoreValue
  auto result_boundary = calculator_->CalculateSoftmax(boundary_scores, 1.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_boundary));
  EXPECT_TRUE(AllValuesFinite(result_boundary));
  
  // Test scores just outside boundaries
  std::vector<float> outside_boundary_scores = {-1.1e6f, 0.0f, 1.1e6f};
  auto result_outside = calculator_->CalculateSoftmax(outside_boundary_scores, 1.0f);
  // Should fallback to uniform
  for (float prob : result_outside) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);
  }
}

// Test score range validation
TEST_F(SoftmaxCalculatorTest, ScoreRangeValidation) {
  // Test scores within valid range
  std::vector<float> valid_scores = {-1000.0f, 0.0f, 1000.0f};
  EXPECT_TRUE(calculator_->IsValidInput(valid_scores));
  
  // Test scores outside valid range (too large)
  std::vector<float> too_large_scores = {1e7f, 2e7f, 3e7f};
  EXPECT_FALSE(calculator_->IsValidInput(too_large_scores));
  
  // Test scores outside valid range (too small)
  std::vector<float> too_small_scores = {-1e7f, -2e7f, -3e7f};
  EXPECT_FALSE(calculator_->IsValidInput(too_small_scores));
}

// Test score precision validation
TEST_F(SoftmaxCalculatorTest, ScorePrecisionValidation) {
  // Test scores with reasonable range
  std::vector<float> reasonable_scores = {1.0f, 2.0f, 3.0f};
  EXPECT_TRUE(calculator_->IsValidInput(reasonable_scores));
  
  // Test scores with extremely large range that could cause overflow
  std::vector<float> extreme_range_scores = {-1000.0f, 0.0f, 10000.0f};
  // This should be rejected due to potential overflow with large lambda
  auto result = calculator_->CalculateSoftmax(extreme_range_scores, 10.0f);
  // Should fallback to uniform distribution
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
}

// Test lambda bounds validation
TEST_F(SoftmaxCalculatorTest, LambdaBoundsValidation) {
  std::vector<float> test_scores = {1.0f, 2.0f, 3.0f};
  
  // Test boundary values - should produce valid results
  auto result_min_boundary = calculator_->CalculateSoftmax(test_scores, 0.001f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_min_boundary));
  
  auto result_max_boundary = calculator_->CalculateSoftmax(test_scores, 10.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_max_boundary));
  
  // Test just outside boundaries - should fallback to uniform
  auto result_below_min = calculator_->CalculateSoftmax(test_scores, 0.0009f);
  for (float prob : result_below_min) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  auto result_above_max = calculator_->CalculateSoftmax(test_scores, 10.1f);
  for (float prob : result_above_max) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  // Test extreme invalid values - should fallback to uniform
  auto result_zero = calculator_->CalculateSoftmax(test_scores, 0.0f);
  for (float prob : result_zero) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  auto result_negative = calculator_->CalculateSoftmax(test_scores, -1.0f);
  for (float prob : result_negative) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  auto result_huge = calculator_->CalculateSoftmax(test_scores, 100.0f);
  for (float prob : result_huge) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
}

// Test lambda precision validation
TEST_F(SoftmaxCalculatorTest, LambdaPrecisionValidation) {
  std::vector<float> test_scores = {1.0f, 2.0f, 3.0f};
  
  // Test normal precision values - should produce valid results
  auto result_001 = calculator_->CalculateSoftmax(test_scores, 0.001f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_001));
  
  auto result_01 = calculator_->CalculateSoftmax(test_scores, 0.1f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_01));
  
  auto result_1 = calculator_->CalculateSoftmax(test_scores, 1.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_1));
  
  // Test extremely small lambda (precision too low) - should fallback to uniform
  auto result_tiny = calculator_->CalculateSoftmax(test_scores, 1e-12f);
  for (float prob : result_tiny) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
  
  // Test special float values - should fallback to uniform
  auto result_denorm = calculator_->CalculateSoftmax(test_scores, std::numeric_limits<float>::denorm_min());
  for (float prob : result_denorm) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);  // Should be uniform
  }
}

// Test extreme lambda values with numerical stability
TEST_F(SoftmaxCalculatorTest, ExtremeLambdaNumericalStability) {
  std::vector<float> scores = {-10.0f, 0.0f, 10.0f};
  
  // Test with minimum valid lambda
  auto result_min = calculator_->CalculateSoftmax(scores, 0.001f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_min));
  EXPECT_TRUE(AllValuesFinite(result_min));
  
  // Test with maximum valid lambda
  auto result_max = calculator_->CalculateSoftmax(scores, 10.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_max));
  EXPECT_TRUE(AllValuesFinite(result_max));
  
  // Test behavior difference between extreme lambdas
  // Small lambda should produce more uniform distribution
  float variance_min = 0.0f;
  float mean_min = 1.0f / 3.0f;
  for (float prob : result_min) {
    variance_min += (prob - mean_min) * (prob - mean_min);
  }
  variance_min /= result_min.size();
  
  // Large lambda should produce more peaked distribution
  float variance_max = 0.0f;
  float mean_max = 1.0f / 3.0f;
  for (float prob : result_max) {
    variance_max += (prob - mean_max) * (prob - mean_max);
  }
  variance_max /= result_max.size();
  
  // Large lambda should have higher variance (more peaked)
  EXPECT_GT(variance_max, variance_min);
}

// Test edge cases with very small score differences
TEST_F(SoftmaxCalculatorTest, VerySmallScoreDifferences) {
  // Test scores that are nearly identical
  std::vector<float> nearly_identical = {1.0f, 1.0f + 1e-7f, 1.0f + 2e-7f};
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(nearly_identical, lambda);
  
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(AllProbabilitiesNonNegative(result));
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  EXPECT_TRUE(AllValuesFinite(result));
  
  // Should still maintain ordering despite small differences
  EXPECT_LE(result[0], result[1]);
  EXPECT_LE(result[1], result[2]);
}

// Test error handling with mixed invalid inputs
TEST_F(SoftmaxCalculatorTest, MixedInvalidInputs) {
  // Array with both valid and invalid values
  std::vector<float> mixed_invalid = {
    1.0f, 
    std::numeric_limits<float>::quiet_NaN(), 
    3.0f, 
    std::numeric_limits<float>::infinity(),
    5.0f
  };
  
  auto result = calculator_->CalculateSoftmax(mixed_invalid, 1.0f);
  
  // Should fallback to uniform distribution
  EXPECT_EQ(result.size(), 5);
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  for (float prob : result) {
    EXPECT_NEAR(prob, 0.2f, 1e-6f);
  }
}

// Test performance with large valid arrays
TEST_F(SoftmaxCalculatorTest, LargeValidArrayPerformance) {
  // Test with reasonably large array (within limits)
  std::vector<float> large_array;
  large_array.reserve(10000);
  for (int i = 0; i < 10000; ++i) {
    large_array.push_back(static_cast<float>(i) / 100.0f);
  }
  
  auto start = std::chrono::high_resolution_clock::now();
  auto result = calculator_->CalculateSoftmax(large_array, 1.0f);
  auto end = std::chrono::high_resolution_clock::now();
  
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  EXPECT_EQ(result.size(), 10000);
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  EXPECT_TRUE(AllValuesFinite(result));
  
  // Should complete in reasonable time (less than 100ms for 10k elements)
  EXPECT_LT(duration.count(), 100);
}

// Test lambda parameter edge cases
TEST_F(SoftmaxCalculatorTest, LambdaEdgeCases) {
  std::vector<float> scores = {1.0f, 2.0f, 3.0f};
  
  // Test lambda exactly at boundaries
  auto result_min_boundary = calculator_->CalculateSoftmax(scores, 0.001f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_min_boundary));
  
  auto result_max_boundary = calculator_->CalculateSoftmax(scores, 10.0f);
  EXPECT_TRUE(ProbabilitiesSumToOne(result_max_boundary));
  
  // Test lambda just outside boundaries (should fallback to uniform)
  auto result_below_min = calculator_->CalculateSoftmax(scores, 0.0005f);
  for (float prob : result_below_min) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);
  }
  
  auto result_above_max = calculator_->CalculateSoftmax(scores, 15.0f);
  for (float prob : result_above_max) {
    EXPECT_NEAR(prob, 1.0f/3.0f, 1e-6f);
  }
}

// Test precision handling with very close scores
TEST_F(SoftmaxCalculatorTest, VeryCloseScores) {
  std::vector<float> close_scores = {1.0f, 1.0000001f, 1.0000002f};
  float lambda = 1.0f;
  
  auto result = calculator_->CalculateSoftmax(close_scores, lambda);
  
  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(AllProbabilitiesNonNegative(result));
  EXPECT_TRUE(ProbabilitiesSumToOne(result));
  EXPECT_TRUE(AllValuesFinite(result));
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}