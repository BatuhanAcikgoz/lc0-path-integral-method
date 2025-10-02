#pragma once

#include <vector>
#include <cmath>
#include "search/path_integral/interfaces.h"

namespace lczero {

// Simple softmax calculator using log-sum-exp implementation
class SoftmaxCalculator : public SoftmaxCalculatorInterface {
public:
  SoftmaxCalculator();
  ~SoftmaxCalculator() override;

  // Main softmax calculation using log-sum-exp for numerical stability
  std::vector<float> CalculateSoftmax(const std::vector<float>& scores, 
                                      float lambda) override;

  // Fallback uniform distribution for invalid inputs
  std::vector<float> UniformProbabilities(size_t count) override;

  // Input validation for score arrays
  bool IsValidInput(const std::vector<float>& scores) override;

private:
  // Core log-sum-exp calculation for numerical stability
  float CalculateLogSumExp(const std::vector<float>& scaled_scores);

  // Numerical stability checks
  bool HasNanOrInf(const std::vector<float>& values) const;
  bool IsFiniteAndValid(float value) const;

  // Scaling and normalization helpers
  std::vector<float> ScaleScores(const std::vector<float>& scores, 
                                 float max_score, float lambda);
  std::vector<float> ComputeProbabilities(const std::vector<float>& scaled_scores,
                                          float log_sum_exp);

  // Validation helpers
  bool ValidateProbabilitySum(const std::vector<float>& probabilities) const;
  void LogValidationWarning(const std::string& reason) const;

  // Constants for numerical stability
  static constexpr float kMinLambda = 0.001f;
  static constexpr float kMaxLambda = 10.0f;
  static constexpr float kEpsilon = 1e-8f;
  static constexpr float kMaxExpArg = 700.0f;  // Prevent exp() overflow
  static constexpr float kMinExpArg = -700.0f; // Prevent exp() underflow
  static constexpr size_t kMaxScoreArraySize = 1000000;      // Maximum array size
};

}  // namespace lczero