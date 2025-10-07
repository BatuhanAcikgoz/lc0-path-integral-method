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

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>

#include "search/path_integral/controller_simple.h"
#include "search/path_integral/config.h"
#include "search/path_integral/performance_monitor.h"
#include "neural/backend.h"
#include "chess/position.h"
#include "chess/types.h"
#include "utils/optionsdict.h"

namespace lczero {

// Verification result for a single test scenario
struct VerificationResult {
  bool samples_match_requested = false;
  bool neural_net_used = false;
  bool timing_reasonable = false;
  bool backend_available = false;
  bool sampling_completed = false;
  
  // Detailed metrics
  int requested_samples = 0;
  int actual_samples = 0;
  int neural_net_evaluations = 0;
  int cached_evaluations = 0;
  int heuristic_evaluations = 0;
  double total_time_ms = 0.0;
  double avg_time_per_sample_ms = 0.0;
  
  // Analysis results
  std::string detailed_report;
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
  
  // Test configuration
  std::string position_fen;
  PathIntegralConfig config_used;
  
  // Helper methods
  bool IsValid() const {
    return samples_match_requested && sampling_completed && errors.empty();
  }
  
  double GetSamplesPerSecond() const {
    return total_time_ms > 0.0 ? (actual_samples * 1000.0 / total_time_ms) : 0.0;
  }
};

// Test scenario configuration
struct TestScenario {
  std::string name;
  std::string position_fen;
  PathIntegralConfig config;
  SearchLimits limits;
  
  // Expected behavior
  bool expect_neural_net_usage = true;
  double min_expected_time_ms = 0.0;
  double max_expected_time_ms = 10000.0;
  
  TestScenario(const std::string& n, const std::string& fen, 
               const PathIntegralConfig& cfg)
    : name(n), position_fen(fen), config(cfg) {}
};

// Comprehensive verification report
struct ComprehensiveVerificationReport {
  std::vector<VerificationResult> individual_results;
  
  // Summary statistics
  int total_tests = 0;
  int passed_tests = 0;
  int failed_tests = 0;
  int warnings_count = 0;
  int errors_count = 0;
  
  // Performance analysis
  double avg_samples_per_second = 0.0;
  double min_samples_per_second = 0.0;
  double max_samples_per_second = 0.0;
  
  // Backend usage analysis
  int tests_with_neural_net = 0;
  int tests_with_heuristics_only = 0;
  
  std::string summary_report;
  std::chrono::system_clock::time_point generated_at;
  
  bool IsOverallSuccess() const {
    return failed_tests == 0 && errors_count == 0;
  }
};

// Main verification tool class
class PathIntegralSamplingVerifier {
public:
  explicit PathIntegralSamplingVerifier(const OptionsDict& options);
  ~PathIntegralSamplingVerifier();
  
  // Single test verification
  VerificationResult VerifySampling(const Position& position, 
                                   const PathIntegralConfig& config,
                                   const SearchLimits& limits = SearchLimits{});
  
  // Comprehensive test suite
  ComprehensiveVerificationReport RunComprehensiveTest(
    const std::vector<std::string>& fen_positions = {});
  
  // Predefined test scenarios
  ComprehensiveVerificationReport RunStandardTestSuite();
  ComprehensiveVerificationReport RunPerformanceTestSuite();
  ComprehensiveVerificationReport RunEdgeCaseTestSuite();
  
  // Configuration management
  void SetBackend(std::unique_ptr<Backend> backend);
  void UpdateOptions(const OptionsDict& options);
  
  // Report generation
  bool ExportReport(const ComprehensiveVerificationReport& report,
                   const std::string& filename,
                   const std::string& format = "json") const;
  
  // Utility methods
  static std::vector<std::string> GetDefaultTestPositions();
  static std::vector<TestScenario> CreateStandardTestScenarios();
  static std::vector<TestScenario> CreatePerformanceTestScenarios();
  static std::vector<TestScenario> CreateEdgeCaseTestScenarios();

private:
  // Internal verification methods
  VerificationResult VerifyIndividualScenario(const TestScenario& scenario);
  bool ValidateSampleCounts(const PathIntegralPerformanceMonitor::SamplingMetrics& metrics,
                           int requested_samples);
  bool ValidateNeuralNetworkUsage(const PathIntegralPerformanceMonitor::SamplingMetrics& metrics);
  bool ValidateTimingReasonableness(const PathIntegralPerformanceMonitor::SamplingMetrics& metrics,
                                   const TestScenario& scenario);
  
  // Analysis methods
  void AnalyzePerformanceMetrics(VerificationResult& result,
                                const PathIntegralPerformanceMonitor::SamplingMetrics& metrics);
  void GenerateDetailedReport(VerificationResult& result, const TestScenario& scenario);
  void AddWarningsAndErrors(VerificationResult& result, const TestScenario& scenario);
  
  // Report generation helpers
  std::string GenerateJsonReport(const ComprehensiveVerificationReport& report) const;
  std::string GenerateTextReport(const ComprehensiveVerificationReport& report) const;
  std::string GenerateCsvReport(const ComprehensiveVerificationReport& report) const;
  
  // Summary generation
  void GenerateSummaryStatistics(ComprehensiveVerificationReport& report) const;
  
  // Components
  std::unique_ptr<SimplePathIntegralController> controller_;
  std::unique_ptr<Backend> backend_;
  OptionsDict options_;
  
  // Configuration
  bool verbose_output_ = false;
  std::string output_directory_ = "./verification_reports/";
  
  // Constants for validation
  static constexpr double kMinReasonableTimePerSampleMs = 0.001;  // 1 microsecond
  static constexpr double kMaxReasonableTimePerSampleMs = 1000.0; // 1 second
  static constexpr double kSampleCountTolerancePercent = 5.0;     // 5% tolerance
};

}  // namespace lczero