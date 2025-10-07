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

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tools/verify_path_integral_sampling.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/wrapper.h"
#include "utils/optionsparser.h"
#include "utils/logging.h"
#include "utils/commandline.h"

namespace lczero {

class PathIntegralVerificationTool {
public:
  PathIntegralVerificationTool() = default;
  
  void Run(int argc, char** argv) {
    try {
      // Initialize command line
      CommandLine::Init(argc, const_cast<const char**>(argv));
      
      // Parse command line options
      OptionsParser options_parser;
      SetupOptions(options_parser);
      
      if (!options_parser.ProcessAllFlags()) {
        std::cerr << "Error processing command line options." << std::endl;
        return;
      }
      
      auto options = options_parser.GetOptionsDict();
      
      // Show help if requested
      if (options.GetOrDefault<bool>("help", false)) {
        ShowHelp();
        return;
      }
      
      // Initialize logging
      InitializeLogging(options);
      
      // Create verification tool
      PathIntegralSamplingVerifier verifier(options);
      
      // Setup neural network backend if specified
      SetupBackend(verifier, options);
      
      // Run the appropriate test suite
      std::string test_suite = options.GetOrDefault<std::string>("test-suite", "standard");
      RunTestSuite(verifier, test_suite, options);
      
    } catch (const std::exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
    }
  }

private:
  void SetupOptions(OptionsParser& parser) {
    // Test configuration options
    parser.Add<StringOption>(OptionId{{"test-suite", "test-suite", 
      "Test suite to run: standard, performance, edge-case, comprehensive"}});
    parser.Add<StringOption>(OptionId{{"positions", "positions",
      "Comma-separated list of FEN positions to test (optional)"}});
    parser.Add<StringOption>(OptionId{{"output-format", "output-format",
      "Output format: text, json, csv"}});
    parser.Add<StringOption>(OptionId{{"output-file", "output-file",
      "Output file name (optional, defaults to stdout)"}});
    parser.Add<StringOption>(OptionId{{"output-dir", "output-dir",
      "Output directory for reports"}});
    
    // Path Integral configuration options
    parser.Add<FloatOption>(OptionId{{"PathIntegralLambda", "PathIntegralLambda",
      "Softmax temperature parameter"}}, 0.001f, 10.0f);
    parser.Add<IntOption>(OptionId{{"PathIntegralSamples", "PathIntegralSamples",
      "Number of samples to perform"}}, 1, 100000);
    parser.Add<StringOption>(OptionId{{"PathIntegralMode", "PathIntegralMode",
      "Sampling mode: competitive, quantum_limit"}});
    parser.Add<StringOption>(OptionId{{"PathIntegralRewardMode", "PathIntegralRewardMode",
      "Reward mode: policy, cp_score, hybrid"}});
    
    // Neural network backend options
    parser.Add<StringOption>(OptionId{{"backend", "backend",
      "Neural network backend to use (optional)"}});
    parser.Add<StringOption>(OptionId{{"weights", "weights",
      "Path to neural network weights file"}});
    parser.Add<IntOption>(OptionId{{"backend-opts", "backend-opts",
      "Backend-specific options"}}, 0, 1000);
    
    // General options
    parser.Add<BoolOption>(OptionId{{"verbose", "verbose",
      "Enable verbose output"}});
    parser.Add<BoolOption>(OptionId{{"help", "help",
      "Show this help message"}});
    parser.Add<IntOption>(OptionId{{"threads", "threads",
      "Number of threads to use"}}, 1, 256);
  }
  
  void ShowHelp() {
    std::cout << "Path Integral Sampling Verification Tool\n";
    std::cout << "========================================\n\n";
    std::cout << "This tool verifies that the LC0 Path Integral implementation\n";
    std::cout << "performs the correct number of samples and uses neural network\n";
    std::cout << "evaluation properly.\n\n";
    
    std::cout << "Usage: verify_path_integral [options]\n\n";
    
    std::cout << "Test Suites:\n";
    std::cout << "  standard     - Basic verification with standard configurations\n";
    std::cout << "  performance  - High sample count performance testing\n";
    std::cout << "  edge-case    - Edge cases and extreme parameter values\n";
    std::cout << "  comprehensive- All test suites combined\n\n";
    
    std::cout << "Examples:\n";
    std::cout << "  # Run standard test suite\n";
    std::cout << "  ./verify_path_integral --test-suite=standard\n\n";
    
    std::cout << "  # Run with specific neural network\n";
    std::cout << "  ./verify_path_integral --weights=network.pb.gz --backend=cuda\n\n";
    
    std::cout << "  # Test specific position with custom parameters\n";
    std::cout << "  ./verify_path_integral --positions=\"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\" \\\n";
    std::cout << "                         --PathIntegralSamples=100 --PathIntegralLambda=0.5\n\n";
    
    std::cout << "  # Export results to JSON\n";
    std::cout << "  ./verify_path_integral --output-format=json --output-file=results.json\n\n";
  }
  
  void InitializeLogging(const OptionsDict& options) {
    bool verbose = options.GetOrDefault<bool>("verbose", false);
    
    if (verbose) {
      // Enable detailed logging - for now just set a flag
      // The actual logging will be handled by LOGFILE and CERR macros
      std::cout << "Verbose logging enabled\n";
    }
  }
  
  void SetupBackend(PathIntegralSamplingVerifier& verifier, const OptionsDict& options) {
    std::string weights_path = options.GetOrDefault<std::string>("weights", "");
    std::string backend_name = options.GetOrDefault<std::string>("backend", "");
    
    if (weights_path.empty()) {
      std::cout << "No neural network weights specified. Using heuristic evaluation only.\n";
      return;
    }
    
    try {
      // Load neural network weights
      auto weights = LoadWeightsFromFile(weights_path);
      if (!weights) {
        std::cerr << "Failed to load weights from: " << weights_path << std::endl;
        return;
      }
      
      // Create backend
      std::unique_ptr<Backend> backend;
      if (!backend_name.empty()) {
        backend = CreateBackend(backend_name, options, weights);
      } else {
        backend = CreateBestBackend(options, weights);
      }
      
      if (!backend) {
        std::cerr << "Failed to create neural network backend" << std::endl;
        return;
      }
      
      std::cout << "Neural network backend initialized successfully\n";
      verifier.SetBackend(std::move(backend));
      
    } catch (const std::exception& e) {
      std::cerr << "Error setting up backend: " << e.what() << std::endl;
    }
  }
  
  void RunTestSuite(PathIntegralSamplingVerifier& verifier, 
                   const std::string& test_suite,
                   const OptionsDict& options) {
    
    std::cout << "Running " << test_suite << " test suite...\n\n";
    
    ComprehensiveVerificationReport report;
    
    if (test_suite == "standard") {
      report = verifier.RunStandardTestSuite();
    } else if (test_suite == "performance") {
      report = verifier.RunPerformanceTestSuite();
    } else if (test_suite == "edge-case") {
      report = verifier.RunEdgeCaseTestSuite();
    } else if (test_suite == "comprehensive") {
      report = verifier.RunComprehensiveTest();
    } else {
      // Custom test with specified positions
      std::vector<std::string> positions = ParsePositions(options);
      report = verifier.RunComprehensiveTest(positions);
    }
    
    // Output results
    OutputResults(verifier, report, options);
  }
  
  std::vector<std::string> ParsePositions(const OptionsDict& options) {
    std::string positions_str = options.GetOrDefault<std::string>("positions", "");
    std::vector<std::string> positions;
    
    if (positions_str.empty()) {
      return PathIntegralSamplingVerifier::GetDefaultTestPositions();
    }
    
    // Split by comma
    std::istringstream iss(positions_str);
    std::string position;
    while (std::getline(iss, position, ',')) {
      // Trim whitespace
      position.erase(0, position.find_first_not_of(" \t"));
      position.erase(position.find_last_not_of(" \t") + 1);
      if (!position.empty()) {
        positions.push_back(position);
      }
    }
    
    return positions;
  }
  
  void OutputResults(PathIntegralSamplingVerifier& verifier,
                    const ComprehensiveVerificationReport& report,
                    const OptionsDict& options) {
    
    std::string output_format = options.GetOrDefault<std::string>("output-format", "text");
    std::string output_file = options.GetOrDefault<std::string>("output-file", "");
    
    if (output_file.empty()) {
      // Output to stdout
      if (output_format == "json") {
        // For stdout JSON output, we'll use the export functionality
        std::cout << "JSON output to stdout not implemented. Use --output-file option.\n";
        std::cout << report.summary_report;
      } else {
        std::cout << report.summary_report;
      }
    } else {
      // Export to file
      bool success = verifier.ExportReport(report, output_file, output_format);
      if (success) {
        std::cout << "Results exported to: " << output_file << std::endl;
      } else {
        std::cerr << "Failed to export results to: " << output_file << std::endl;
      }
    }
    
    // Always show summary to stdout
    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "Total Tests: " << report.total_tests << "\n";
    std::cout << "Passed: " << report.passed_tests << "\n";
    std::cout << "Failed: " << report.failed_tests << "\n";
    std::cout << "Overall Result: " << (report.IsOverallSuccess() ? "SUCCESS" : "FAILURE") << "\n";
    
    if (!report.IsOverallSuccess()) {
      std::cout << "\nSome tests failed. Check the detailed report for more information.\n";
    }
  }
  
  std::optional<WeightsFile> LoadWeightsFromFile(const std::string& path) {
    try {
      return lczero::LoadWeightsFromFile(path);
    } catch (const std::exception& e) {
      std::cerr << "Error loading weights: " << e.what() << std::endl;
      return std::nullopt;
    }
  }
  
  std::unique_ptr<Backend> CreateBackend(const std::string& backend_name,
                                        const OptionsDict& options,
                                        const std::optional<WeightsFile>& weights) {
    try {
      // For now, we'll skip backend creation and focus on the verification logic
      // The verification tool can work without a neural network backend
      // by using heuristic evaluation
      std::cout << "Backend creation not implemented yet. Using heuristic evaluation.\n";
      return nullptr;
    } catch (const std::exception& e) {
      std::cerr << "Error creating backend: " << e.what() << std::endl;
      return nullptr;
    }
  }
  
  std::unique_ptr<Backend> CreateBestBackend(const OptionsDict& options,
                                            const std::optional<WeightsFile>& weights) {
    try {
      // For now, we'll skip backend creation and focus on the verification logic
      std::cout << "Backend creation not implemented yet. Using heuristic evaluation.\n";
      return nullptr;
    } catch (const std::exception& e) {
      std::cerr << "Error creating best backend: " << e.what() << std::endl;
      return nullptr;
    }
  }
};

}  // namespace lczero

int main(int argc, char** argv) {
  lczero::PathIntegralVerificationTool tool;
  tool.Run(argc, argv);
  return 0;
}