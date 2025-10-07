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

#include "tools/verify_path_integral_sampling.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <cmath>

#include "utils/logging.h"
#include "chess/board.h"
#include "chess/position.h"
#include "search/path_integral/interfaces.h"
#include "search/path_integral/options.h"

namespace lczero {

PathIntegralSamplingVerifier::PathIntegralSamplingVerifier(const OptionsDict& options)
    : options_(options) {
  verbose_output_ = options.GetOrDefault<bool>("verbose", false);
  output_directory_ = options.GetOrDefault<std::string>("output-dir", "./verification_reports/");
  
  // Create output directory if it doesn't exist
  std::filesystem::create_directories(output_directory_);
  
  // Initialize controller with options
  controller_ = std::make_unique<SimplePathIntegralController>(options);
  
  LOGFILE << "PathIntegralSamplingVerifier initialized with output directory: " 
          << output_directory_;
}

PathIntegralSamplingVerifier::~PathIntegralSamplingVerifier() = default;

void PathIntegralSamplingVerifier::SetBackend(std::unique_ptr<Backend> backend) {
  backend_ = std::move(backend);
  
  // Recreate controller with the new backend
  controller_ = std::make_unique<SimplePathIntegralController>(options_, backend_.get());
  
  if (verbose_output_) {
    LOGFILE << "Backend set for PathIntegralSamplingVerifier";
  }
}

void PathIntegralSamplingVerifier::UpdateOptions(const OptionsDict& options) {
  options_ = options;
  controller_->UpdateOptions(options);
  
  verbose_output_ = options.GetOrDefault<bool>("verbose", false);
  output_directory_ = options.GetOrDefault<std::string>("output-dir", "./verification_reports/");
}

VerificationResult PathIntegralSamplingVerifier::VerifySampling(
    const Position& position, 
    const PathIntegralConfig& config,
    const SearchLimits& limits) {
  
  VerificationResult result;
  result.position_fen = PositionToFen(position);
  result.config_used = config;
  result.requested_samples = config.samples;
  
  if (verbose_output_) {
    LOGFILE << "Verifying sampling for position: " << PositionToFen(position)
            << " with " << config.samples << " samples, lambda=" << config.lambda;
  }
  
  try {
    // Update controller configuration directly using SetConfig
    controller_->SetConfig(config);

    // Check backend availability
    result.backend_available = (backend_ != nullptr);
    if (!result.backend_available) {
      result.warnings.push_back("No neural network backend available - will use heuristic evaluation");
    }
    
    // Perform the sampling
    auto start_time = std::chrono::high_resolution_clock::now();
    
    Move selected_move = controller_->SelectMove(position, limits);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Get performance metrics
    auto metrics = controller_->GetLastSamplingMetrics();
    
    // Analyze results
    result.sampling_completed = !selected_move.is_null();
    result.actual_samples = metrics.actual_samples;
    result.neural_net_evaluations = metrics.neural_net_evaluations;
    result.cached_evaluations = metrics.cached_evaluations;
    result.heuristic_evaluations = metrics.heuristic_evaluations;
    result.avg_time_per_sample_ms = metrics.avg_time_per_sample_ms;
    
    // Validate sample counts
    result.samples_match_requested = ValidateSampleCounts(metrics, config.samples);
    
    // Validate neural network usage
    result.neural_net_used = ValidateNeuralNetworkUsage(metrics);
    
    // Create a test scenario for timing validation
    TestScenario temp_scenario("temp", PositionToFen(position), config);
    result.timing_reasonable = ValidateTimingReasonableness(metrics, temp_scenario);
    
    // Generate detailed analysis
    AnalyzePerformanceMetrics(result, metrics);
    GenerateDetailedReport(result, temp_scenario);
    
  } catch (const std::exception& e) {
    result.errors.push_back(std::string("Exception during sampling: ") + e.what());
    result.sampling_completed = false;
  }
  
  if (verbose_output_) {
    LOGFILE << "Verification completed. Valid: " << result.IsValid() 
            << ", Samples: " << result.actual_samples << "/" << result.requested_samples
            << ", Time: " << result.total_time_ms << "ms";
  }
  
  return result;
}

bool PathIntegralSamplingVerifier::ValidateSampleCounts(
    const PathIntegralPerformanceMonitor::SamplingMetrics& metrics,
    int requested_samples) {
  
  if (metrics.actual_samples == requested_samples) {
    return true;
  }
  
  // Allow small tolerance for edge cases
  double tolerance_percent = kSampleCountTolerancePercent / 100.0;
  int tolerance = static_cast<int>(requested_samples * tolerance_percent);
  tolerance = std::max(tolerance, 1); // At least 1 sample tolerance
  
  return std::abs(metrics.actual_samples - requested_samples) <= tolerance;
}

bool PathIntegralSamplingVerifier::ValidateNeuralNetworkUsage(
    const PathIntegralPerformanceMonitor::SamplingMetrics& metrics) {
  
  // If we have a backend, we should see some neural network evaluations
  if (backend_ != nullptr) {
    return metrics.neural_net_evaluations > 0 || metrics.cached_evaluations > 0;
  }
  
  // Without backend, heuristic evaluation is expected
  return metrics.heuristic_evaluations > 0;
}

bool PathIntegralSamplingVerifier::ValidateTimingReasonableness(
    const PathIntegralPerformanceMonitor::SamplingMetrics& metrics,
    const TestScenario& scenario) {
  
  if (metrics.avg_time_per_sample_ms < kMinReasonableTimePerSampleMs) {
    return false; // Too fast, might indicate skipped computation
  }
  
  if (metrics.avg_time_per_sample_ms > kMaxReasonableTimePerSampleMs) {
    return false; // Too slow, might indicate performance issues
  }
  
  // Check against scenario expectations if provided
  if (scenario.min_expected_time_ms > 0.0 && 
      metrics.total_time_ms < scenario.min_expected_time_ms) {
    return false;
  }
  
  if (scenario.max_expected_time_ms > 0.0 && 
      metrics.total_time_ms > scenario.max_expected_time_ms) {
    return false;
  }
  
  return true;
}

void PathIntegralSamplingVerifier::AnalyzePerformanceMetrics(
    VerificationResult& result,
    const PathIntegralPerformanceMonitor::SamplingMetrics& metrics) {
  
  // Check for suspicious performance patterns
  if (result.backend_available && metrics.neural_net_evaluations == 0) {
    result.warnings.push_back("Backend available but no neural network evaluations performed");
  }
  
  if (metrics.actual_samples > 0 && metrics.avg_time_per_sample_ms < 0.01) {
    result.warnings.push_back("Extremely fast sampling detected - verify computation is actually performed");
  }
  
  if (metrics.actual_samples != result.requested_samples) {
    std::ostringstream oss;
    oss << "Sample count mismatch: requested " << result.requested_samples 
        << ", actual " << metrics.actual_samples;
    result.warnings.push_back(oss.str());
  }
  
  // Calculate evaluation distribution
  int total_evaluations = metrics.neural_net_evaluations + 
                         metrics.cached_evaluations + 
                         metrics.heuristic_evaluations;
  
  if (total_evaluations == 0) {
    result.errors.push_back("No evaluations performed during sampling");
  } else if (total_evaluations < metrics.actual_samples) {
    result.warnings.push_back("Fewer evaluations than samples - possible evaluation reuse");
  }
}

void PathIntegralSamplingVerifier::GenerateDetailedReport(
    VerificationResult& result, 
    const TestScenario& scenario) {
  
  std::ostringstream report;
  
  report << "=== Path Integral Sampling Verification Report ===\n";
  report << "Position: " << result.position_fen << "\n";
  report << "Configuration:\n";
  report << "  - Lambda: " << result.config_used.lambda << "\n";
  report << "  - Samples: " << result.config_used.samples << "\n";
  report << "  - Mode: " << PathIntegralConfig::SamplingModeToString(result.config_used.sampling_mode) << "\n";
  report << "  - Reward Mode: " << PathIntegralConfig::RewardModeToString(result.config_used.reward_mode) << "\n";
  report << "\n";
  
  report << "Results:\n";
  report << "  - Sampling Completed: " << (result.sampling_completed ? "YES" : "NO") << "\n";
  report << "  - Samples Match Requested: " << (result.samples_match_requested ? "YES" : "NO") << "\n";
  report << "  - Neural Network Used: " << (result.neural_net_used ? "YES" : "NO") << "\n";
  report << "  - Timing Reasonable: " << (result.timing_reasonable ? "YES" : "NO") << "\n";
  report << "  - Backend Available: " << (result.backend_available ? "YES" : "NO") << "\n";
  report << "\n";
  
  report << "Performance Metrics:\n";
  report << "  - Requested Samples: " << result.requested_samples << "\n";
  report << "  - Actual Samples: " << result.actual_samples << "\n";
  report << "  - Neural Net Evaluations: " << result.neural_net_evaluations << "\n";
  report << "  - Cached Evaluations: " << result.cached_evaluations << "\n";
  report << "  - Heuristic Evaluations: " << result.heuristic_evaluations << "\n";
  report << "  - Total Time: " << std::fixed << std::setprecision(3) << result.total_time_ms << " ms\n";
  report << "  - Avg Time per Sample: " << std::fixed << std::setprecision(3) << result.avg_time_per_sample_ms << " ms\n";
  report << "  - Samples per Second: " << std::fixed << std::setprecision(1) << result.GetSamplesPerSecond() << "\n";
  report << "\n";
  
  if (!result.warnings.empty()) {
    report << "Warnings:\n";
    for (const auto& warning : result.warnings) {
      report << "  - " << warning << "\n";
    }
    report << "\n";
  }
  
  if (!result.errors.empty()) {
    report << "Errors:\n";
    for (const auto& error : result.errors) {
      report << "  - " << error << "\n";
    }
    report << "\n";
  }
  
  report << "Overall Result: " << (result.IsValid() ? "PASS" : "FAIL") << "\n";
  
  result.detailed_report = report.str();
}

ComprehensiveVerificationReport PathIntegralSamplingVerifier::RunComprehensiveTest(
    const std::vector<std::string>& fen_positions) {
  
  ComprehensiveVerificationReport report;
  report.generated_at = std::chrono::system_clock::now();
  
  std::vector<std::string> positions = fen_positions.empty() ? 
    GetDefaultTestPositions() : fen_positions;
  
  std::vector<TestScenario> scenarios = CreateStandardTestScenarios();
  
  if (verbose_output_) {
    LOGFILE << "Running comprehensive test with " << positions.size() 
            << " positions and " << scenarios.size() << " scenarios";
  }
  
  for (const auto& fen : positions) {
    for (auto& scenario : scenarios) {
      scenario.position_fen = fen;
      VerificationResult result = VerifyIndividualScenario(scenario);
      report.individual_results.push_back(result);
    }
  }
  
  GenerateSummaryStatistics(report);
  
  return report;
}

ComprehensiveVerificationReport PathIntegralSamplingVerifier::RunStandardTestSuite() {
  return RunComprehensiveTest(GetDefaultTestPositions());
}

ComprehensiveVerificationReport PathIntegralSamplingVerifier::RunPerformanceTestSuite() {
  ComprehensiveVerificationReport report;
  report.generated_at = std::chrono::system_clock::now();
  
  std::vector<TestScenario> scenarios = CreatePerformanceTestScenarios();
  std::vector<std::string> positions = GetDefaultTestPositions();
  
  // Use subset of positions for performance testing
  positions.resize(std::min(positions.size(), size_t(5)));
  
  for (const auto& fen : positions) {
    for (auto& scenario : scenarios) {
      scenario.position_fen = fen;
      VerificationResult result = VerifyIndividualScenario(scenario);
      report.individual_results.push_back(result);
    }
  }
  
  GenerateSummaryStatistics(report);
  return report;
}

ComprehensiveVerificationReport PathIntegralSamplingVerifier::RunEdgeCaseTestSuite() {
  ComprehensiveVerificationReport report;
  report.generated_at = std::chrono::system_clock::now();
  
  std::vector<TestScenario> scenarios = CreateEdgeCaseTestScenarios();
  std::vector<std::string> positions = GetDefaultTestPositions();
  
  // Use subset of positions for edge case testing
  positions.resize(std::min(positions.size(), size_t(3)));
  
  for (const auto& fen : positions) {
    for (auto& scenario : scenarios) {
      scenario.position_fen = fen;
      VerificationResult result = VerifyIndividualScenario(scenario);
      report.individual_results.push_back(result);
    }
  }
  
  GenerateSummaryStatistics(report);
  return report;
}

VerificationResult PathIntegralSamplingVerifier::VerifyIndividualScenario(
    const TestScenario& scenario) {
  
  VerificationResult result;
  Position position;

  try {
    position = Position::FromFen(scenario.position_fen);
  } catch (const Exception& e) {
    result.errors.push_back("Invalid FEN position: " + scenario.position_fen + " - " + e.what());
    return result;
  }
  
  return VerifySampling(position, scenario.config, scenario.limits);
}

void PathIntegralSamplingVerifier::GenerateSummaryStatistics(
    ComprehensiveVerificationReport& report) const {
  
  report.total_tests = report.individual_results.size();
  report.passed_tests = 0;
  report.failed_tests = 0;
  report.warnings_count = 0;
  report.errors_count = 0;
  
  std::vector<double> samples_per_second_values;
  
  for (const auto& result : report.individual_results) {
    if (result.IsValid()) {
      report.passed_tests++;
    } else {
      report.failed_tests++;
    }
    
    report.warnings_count += result.warnings.size();
    report.errors_count += result.errors.size();
    
    if (result.neural_net_used) {
      report.tests_with_neural_net++;
    } else {
      report.tests_with_heuristics_only++;
    }
    
    double sps = result.GetSamplesPerSecond();
    if (sps > 0.0) {
      samples_per_second_values.push_back(sps);
    }
  }
  
  // Calculate performance statistics
  if (!samples_per_second_values.empty()) {
    std::sort(samples_per_second_values.begin(), samples_per_second_values.end());
    
    report.min_samples_per_second = samples_per_second_values.front();
    report.max_samples_per_second = samples_per_second_values.back();
    
    double sum = 0.0;
    for (double sps : samples_per_second_values) {
      sum += sps;
    }
    report.avg_samples_per_second = sum / samples_per_second_values.size();
  }
  
  // Generate summary report
  std::ostringstream summary;
  summary << "=== Comprehensive Verification Summary ===\n";
  summary << "Total Tests: " << report.total_tests << "\n";
  summary << "Passed: " << report.passed_tests << "\n";
  summary << "Failed: " << report.failed_tests << "\n";
  summary << "Warnings: " << report.warnings_count << "\n";
  summary << "Errors: " << report.errors_count << "\n";
  summary << "\n";
  summary << "Performance Analysis:\n";
  summary << "  - Average Samples/sec: " << std::fixed << std::setprecision(1) 
          << report.avg_samples_per_second << "\n";
  summary << "  - Min Samples/sec: " << std::fixed << std::setprecision(1) 
          << report.min_samples_per_second << "\n";
  summary << "  - Max Samples/sec: " << std::fixed << std::setprecision(1) 
          << report.max_samples_per_second << "\n";
  summary << "\n";
  summary << "Backend Usage:\n";
  summary << "  - Tests with Neural Net: " << report.tests_with_neural_net << "\n";
  summary << "  - Tests with Heuristics Only: " << report.tests_with_heuristics_only << "\n";
  summary << "\n";
  summary << "Overall Result: " << (report.IsOverallSuccess() ? "SUCCESS" : "FAILURE") << "\n";
  
  report.summary_report = summary.str();
}

// Static methods for test data
std::vector<std::string> PathIntegralSamplingVerifier::GetDefaultTestPositions() {
  return {
    // Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    
    // Middle game positions
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10",
    "4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19",
    
    // Endgame positions
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11",
    "4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22",
    
    // Tactical positions
    "r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14",
    "r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13"
  };
}

std::vector<TestScenario> PathIntegralSamplingVerifier::CreateStandardTestScenarios() {
  std::vector<TestScenario> scenarios;
  
  // Standard configurations
  PathIntegralConfig config1;
  config1.lambda = 0.1f;
  config1.samples = 50;
  config1.sampling_mode = PathIntegralSamplingMode::kCompetitive;
  config1.enabled = true;
  scenarios.emplace_back("Standard Competitive", "", config1);
  
  PathIntegralConfig config2;
  config2.lambda = 0.1f;
  config2.samples = 50;
  config2.sampling_mode = PathIntegralSamplingMode::kQuantumLimit;
  config2.reward_mode = PathIntegralRewardMode::kHybrid;
  config2.enabled = true;
  scenarios.emplace_back("Standard Quantum Limit", "", config2);
  
  // Different lambda values
  PathIntegralConfig config3;
  config3.lambda = 0.01f;
  config3.samples = 25;
  config3.sampling_mode = PathIntegralSamplingMode::kCompetitive;
  config3.enabled = true;
  scenarios.emplace_back("Low Lambda", "", config3);
  
  PathIntegralConfig config4;
  config4.lambda = 1.0f;
  config4.samples = 25;
  config4.sampling_mode = PathIntegralSamplingMode::kCompetitive;
  config4.enabled = true;
  scenarios.emplace_back("High Lambda", "", config4);
  
  return scenarios;
}

std::vector<TestScenario> PathIntegralSamplingVerifier::CreatePerformanceTestScenarios() {
  std::vector<TestScenario> scenarios;
  
  // High sample count tests
  PathIntegralConfig config1;
  config1.lambda = 0.1f;
  config1.samples = 500;
  config1.sampling_mode = PathIntegralSamplingMode::kCompetitive;
  config1.enabled = true;
  scenarios.emplace_back("High Sample Count", "", config1);
  
  PathIntegralConfig config2;
  config2.lambda = 0.1f;
  config2.samples = 1000;
  config2.sampling_mode = PathIntegralSamplingMode::kCompetitive;
  config2.enabled = true;
  scenarios.emplace_back("Very High Sample Count", "", config2);
  
  return scenarios;
}

std::vector<TestScenario> PathIntegralSamplingVerifier::CreateEdgeCaseTestScenarios() {
  std::vector<TestScenario> scenarios;
  
  // Minimum samples
  PathIntegralConfig config1;
  config1.lambda = 0.1f;
  config1.samples = 1;
  config1.sampling_mode = PathIntegralSamplingMode::kCompetitive;
  config1.enabled = true;
  scenarios.emplace_back("Minimum Samples", "", config1);
  
  // Extreme lambda values
  PathIntegralConfig config2;
  config2.lambda = 0.001f;
  config2.samples = 100;
  config2.sampling_mode = PathIntegralSamplingMode::kCompetitive;
  config2.enabled = true;
  scenarios.emplace_back("Extreme Low Lambda", "", config2);
  
  PathIntegralConfig config3;
  config3.lambda = 10.0f;
  config3.samples = 100;
  config3.sampling_mode = PathIntegralSamplingMode::kCompetitive;
  config3.enabled = true;
  scenarios.emplace_back("Extreme High Lambda", "", config3);
  
  return scenarios;
}

bool PathIntegralSamplingVerifier::ExportReport(
    const ComprehensiveVerificationReport& report,
    const std::string& filename,
    const std::string& format) const {
  
  try {
    std::string full_path = output_directory_ + "/" + filename;
    std::ofstream file(full_path);
    
    if (!file.is_open()) {
      LOGFILE << "Failed to open file for writing: " << full_path;
      return false;
    }
    
    if (format == "json") {
      file << GenerateJsonReport(report);
    } else if (format == "csv") {
      file << GenerateCsvReport(report);
    } else {
      file << GenerateTextReport(report);
    }
    
    file.close();
    
    if (verbose_output_) {
      LOGFILE << "Report exported to: " << full_path;
    }
    
    return true;
    
  } catch (const std::exception& e) {
    LOGFILE << "Error exporting report: " << e.what();
    return false;
  }
}

std::string PathIntegralSamplingVerifier::GenerateJsonReport(
    const ComprehensiveVerificationReport& report) const {
  
  std::ostringstream json;
  json << "{\n";
  json << "  \"summary\": {\n";
  json << "    \"total_tests\": " << report.total_tests << ",\n";
  json << "    \"passed_tests\": " << report.passed_tests << ",\n";
  json << "    \"failed_tests\": " << report.failed_tests << ",\n";
  json << "    \"warnings_count\": " << report.warnings_count << ",\n";
  json << "    \"errors_count\": " << report.errors_count << ",\n";
  json << "    \"avg_samples_per_second\": " << report.avg_samples_per_second << ",\n";
  json << "    \"overall_success\": " << (report.IsOverallSuccess() ? "true" : "false") << "\n";
  json << "  },\n";
  json << "  \"individual_results\": [\n";
  
  for (size_t i = 0; i < report.individual_results.size(); ++i) {
    const auto& result = report.individual_results[i];
    json << "    {\n";
    json << "      \"position_fen\": \"" << result.position_fen << "\",\n";
    json << "      \"requested_samples\": " << result.requested_samples << ",\n";
    json << "      \"actual_samples\": " << result.actual_samples << ",\n";
    json << "      \"total_time_ms\": " << result.total_time_ms << ",\n";
    json << "      \"samples_per_second\": " << result.GetSamplesPerSecond() << ",\n";
    json << "      \"is_valid\": " << (result.IsValid() ? "true" : "false") << ",\n";
    json << "      \"neural_net_evaluations\": " << result.neural_net_evaluations << ",\n";
    json << "      \"heuristic_evaluations\": " << result.heuristic_evaluations << "\n";
    json << "    }";
    if (i < report.individual_results.size() - 1) json << ",";
    json << "\n";
  }
  
  json << "  ]\n";
  json << "}\n";
  
  return json.str();
}

std::string PathIntegralSamplingVerifier::GenerateTextReport(
    const ComprehensiveVerificationReport& report) const {
  
  std::ostringstream text;
  text << report.summary_report << "\n\n";
  
  text << "=== Individual Test Results ===\n";
  for (const auto& result : report.individual_results) {
    text << result.detailed_report << "\n";
    text << "----------------------------------------\n";
  }
  
  return text.str();
}

std::string PathIntegralSamplingVerifier::GenerateCsvReport(
    const ComprehensiveVerificationReport& report) const {
  
  std::ostringstream csv;
  
  // Header
  csv << "Position,Requested_Samples,Actual_Samples,Total_Time_ms,Samples_Per_Second,"
      << "Neural_Net_Evaluations,Cached_Evaluations,Heuristic_Evaluations,"
      << "Is_Valid,Warnings_Count,Errors_Count\n";
  
  // Data rows
  for (const auto& result : report.individual_results) {
    csv << "\"" << result.position_fen << "\","
        << result.requested_samples << ","
        << result.actual_samples << ","
        << result.total_time_ms << ","
        << result.GetSamplesPerSecond() << ","
        << result.neural_net_evaluations << ","
        << result.cached_evaluations << ","
        << result.heuristic_evaluations << ","
        << (result.IsValid() ? "1" : "0") << ","
        << result.warnings.size() << ","
        << result.errors.size() << "\n";
  }
  
  return csv.str();
}

}  // namespace lczero

