#include "search/path_integral/softmax.h"
#include "utils/logging.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace lczero {

SoftmaxCalculator::SoftmaxCalculator() = default;
SoftmaxCalculator::~SoftmaxCalculator() = default;

std::vector<float> SoftmaxCalculator::CalculateSoftmax(const std::vector<float>& scores, float lambda) {
  try {
    // Step 1: Basic input validation
    if (!IsValidInput(scores) || lambda < kMinLambda || lambda > kMaxLambda) {
      return UniformProbabilities(scores.size());
    }

    // Step 2: Find maximum score for numerical stability
    float max_score = *std::max_element(scores.begin(), scores.end());
    if (!std::isfinite(max_score)) {
      return UniformProbabilities(scores.size());
    }

    // Step 3: Calculate scaled scores (scores - max_score) * lambda
    std::vector<float> scaled_scores;
    scaled_scores.reserve(scores.size());
    
    for (float score : scores) {
      float scaled = (score - max_score) * lambda;
      // Clamp to prevent overflow/underflow
      scaled = std::max(kMinExpArg, std::min(kMaxExpArg, scaled));
      scaled_scores.push_back(scaled);
    }

    // Step 4: Calculate log-sum-exp for numerical stability
    float log_sum_exp = CalculateLogSumExp(scaled_scores);
    if (!std::isfinite(log_sum_exp)) {
      return UniformProbabilities(scores.size());
    }

    // Step 5: Compute final probabilities
    std::vector<float> probabilities;
    probabilities.reserve(scores.size());
    
    for (float scaled : scaled_scores) {
      float prob = std::exp(scaled - log_sum_exp);
      probabilities.push_back(prob);
    }

    // Step 6: Basic validation of result
    if (HasNanOrInf(probabilities)) {
      return UniformProbabilities(scores.size());
    }

    return probabilities;

  } catch (const std::exception&) {
    return UniformProbabilities(scores.size());
  }
}

float SoftmaxCalculator::CalculateLogSumExp(const std::vector<float>& scaled_scores) {
  // Calculate sum of exponentials
  float sum_exp = 0.0f;
  for (float scaled : scaled_scores) {
    sum_exp += std::exp(scaled);
  }
  
  if (sum_exp <= 0.0f || !std::isfinite(sum_exp)) {
    // Return a safe default that will lead to uniform distribution
    return 0.0f;
  }
  
  return std::log(sum_exp);
}

bool SoftmaxCalculator::HasNanOrInf(const std::vector<float>& values) const {
  return std::any_of(values.begin(), values.end(), 
                     [](float val) { return !std::isfinite(val); });
}

bool SoftmaxCalculator::IsFiniteAndValid(float value) const {
  return std::isfinite(value);
}

std::vector<float> SoftmaxCalculator::ScaleScores(const std::vector<float>& scores, 
                                                  float max_score, float lambda) {
  std::vector<float> scaled_scores;
  scaled_scores.reserve(scores.size());
  
  for (float score : scores) {
    float scaled = (score - max_score) * lambda;
    scaled = std::max(kMinExpArg, std::min(kMaxExpArg, scaled));
    scaled_scores.push_back(scaled);
  }
  
  return scaled_scores;
}

std::vector<float> SoftmaxCalculator::ComputeProbabilities(const std::vector<float>& scaled_scores,
                                                           float log_sum_exp) {
  std::vector<float> probabilities;
  probabilities.reserve(scaled_scores.size());
  
  for (float scaled : scaled_scores) {
    float prob = std::exp(scaled - log_sum_exp);
    probabilities.push_back(prob);
  }
  
  return probabilities;
}

bool SoftmaxCalculator::ValidateProbabilitySum(const std::vector<float>& probabilities) const {
  float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
  return std::abs(sum - 1.0f) < kEpsilon;
}

void SoftmaxCalculator::LogValidationWarning(const std::string& reason) const {
  CERR << "PathIntegral Softmax Warning: " << reason;
}

std::vector<float> SoftmaxCalculator::UniformProbabilities(size_t count) {
  if (count == 0) return {};
  
  float uniform_prob = 1.0f / static_cast<float>(count);
  return std::vector<float>(count, uniform_prob);
}

bool SoftmaxCalculator::IsValidInput(const std::vector<float>& scores) {
  return !scores.empty() && 
         scores.size() <= kMaxScoreArraySize &&
         std::all_of(scores.begin(), scores.end(), 
                     [](float score) { return std::isfinite(score); });
}

} // namespace lczero