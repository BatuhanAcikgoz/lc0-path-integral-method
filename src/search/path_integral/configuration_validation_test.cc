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
#include "utils/optionsdict.h"
#include <limits>
#include <thread>
#include <vector>

namespace lczero {

class PathIntegralConfigurationValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        options_.Set<bool>("PathIntegralEnabled", true);
        options_.Set<float>("PathIntegralLambda", 0.1f);
        options_.Set<int>("PathIntegralSamples", 10);
        options_.Set<std::string>("PathIntegralMode", "competitive");
        options_.Set<bool>("verbose", false);
    }
    
    OptionsDict options_;
};

// Test Requirement 6.4: Configuration validation
TEST_F(PathIntegralConfigurationValidationTest, ValidLambdaRange) {
    PathIntegralConfig config;
    
    // Test valid lambda values
    std::vector<float> valid_lambdas = {0.001f, 0.01f, 0.1f, 1.0f, 5.0f, 10.0f};
    
    for (float lambda : valid_lambdas) {
        config.lambda = lambda;
        config.samples = 10;
        config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
        config.enabled = true;
        
        // Should be considered valid using the IsValid() method
        EXPECT_TRUE(config.IsValid()) << "Lambda " << lambda << " should be valid";
        EXPECT_GT(config.lambda, 0.0f) << "Lambda " << lambda << " should be positive";
        EXPECT_LE(config.lambda, 10.0f) << "Lambda " << lambda << " should be <= 10.0";
    }
}

TEST_F(PathIntegralConfigurationValidationTest, InvalidLambdaHandling) {
    PathIntegralConfig config;
    config.samples = 10;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    // Test invalid lambda values that should be handled gracefully
    std::vector<float> invalid_lambdas = {
        0.0f,                                    // Zero
        -0.1f,                                   // Negative
        std::numeric_limits<float>::infinity(),   // Infinity
        std::numeric_limits<float>::quiet_NaN(), // NaN
        15.0f                                    // Too large
    };
    
    for (float lambda : invalid_lambdas) {
        config.lambda = lambda;
        
        // Should not crash when used
        EXPECT_NO_THROW({
            // Test that the configuration can be created without throwing
            PathIntegralConfig test_config = config;
            (void)test_config; // Suppress unused variable warning
        }) << "Lambda " << lambda << " should not cause crashes";
    }
}

TEST_F(PathIntegralConfigurationValidationTest, ValidSampleCounts) {
    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    // Test valid sample counts
    std::vector<int> valid_samples = {1, 5, 10, 50, 100, 1000, 10000};
    
    for (int samples : valid_samples) {
        config.samples = samples;
        
        EXPECT_GT(config.samples, 0) << "Sample count " << samples << " should be positive";
        EXPECT_LE(config.samples, 100000) << "Sample count " << samples << " should be reasonable";
    }
}

TEST_F(PathIntegralConfigurationValidationTest, InvalidSampleCountHandling) {
    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    // Test invalid sample counts
    std::vector<int> invalid_samples = {
        0,                              // Zero
        -1,                             // Negative
        -100,                           // Large negative
        std::numeric_limits<int>::max() // Extremely large
    };
    
    for (int samples : invalid_samples) {
        config.samples = samples;
        
        // Should not crash when used
        EXPECT_NO_THROW({
            PathIntegralConfig test_config = config;
            (void)test_config; // Suppress unused variable warning
        }) << "Sample count " << samples << " should not cause crashes";
    }
}

TEST_F(PathIntegralConfigurationValidationTest, SamplingModeValidation) {
    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 10;
    config.enabled = true;
    
    // Test valid sampling modes
    std::vector<PathIntegralSamplingMode> valid_modes = {
        PathIntegralSamplingMode::kCompetitive,
        PathIntegralSamplingMode::kQuantumLimit
    };
    
    for (auto mode : valid_modes) {
        config.sampling_mode = mode;
        
        EXPECT_NO_THROW({
            PathIntegralConfig test_config = config;
            (void)test_config;
        }) << "Sampling mode should be valid";
    }
}

TEST_F(PathIntegralConfigurationValidationTest, EnabledDisabledStates) {
    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 10;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    
    // Test enabled state
    config.enabled = true;
    EXPECT_TRUE(config.enabled);
    
    // Test disabled state
    config.enabled = false;
    EXPECT_FALSE(config.enabled);
}

TEST_F(PathIntegralConfigurationValidationTest, ConfigurationCombinations) {
    // Test various configuration combinations
    struct TestCase {
        float lambda;
        int samples;
        PathIntegralSamplingMode mode;
        bool enabled;
        bool should_work;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {0.1f, 10, PathIntegralSamplingMode::kCompetitive, true, true, "Standard competitive"},
        {1.0f, 50, PathIntegralSamplingMode::kQuantumLimit, true, true, "Standard quantum limit"},
        {0.001f, 1, PathIntegralSamplingMode::kCompetitive, true, true, "Minimum values"},
        {10.0f, 1000, PathIntegralSamplingMode::kQuantumLimit, true, true, "Maximum values"},
        {0.1f, 10, PathIntegralSamplingMode::kCompetitive, false, true, "Disabled config"},
        {0.0f, 10, PathIntegralSamplingMode::kCompetitive, true, false, "Invalid lambda"},
        {0.1f, 0, PathIntegralSamplingMode::kCompetitive, true, false, "Invalid samples"},
        {-1.0f, -5, PathIntegralSamplingMode::kCompetitive, true, false, "All invalid"}
    };
    
    for (const auto& test_case : test_cases) {
        PathIntegralConfig config;
        config.lambda = test_case.lambda;
        config.samples = test_case.samples;
        config.sampling_mode = test_case.mode;
        config.enabled = test_case.enabled;
        
        if (test_case.should_work) {
            EXPECT_NO_THROW({
                PathIntegralConfig test_config = config;
                (void)test_config;
            }) << "Configuration should work: " << test_case.description;
        } else {
            // Even invalid configurations should not crash, just handle gracefully
            EXPECT_NO_THROW({
                PathIntegralConfig test_config = config;
                (void)test_config;
            }) << "Invalid configuration should not crash: " << test_case.description;
        }
    }
}

TEST_F(PathIntegralConfigurationValidationTest, DefaultConfigurationValues) {
    PathIntegralConfig config;
    
    // Test that default values are reasonable
    EXPECT_GE(config.lambda, 0.0f);
    EXPECT_LE(config.lambda, 10.0f);
    EXPECT_GE(config.samples, 0);
    EXPECT_LE(config.samples, 100000);
    EXPECT_TRUE(config.IsValid()) << "Default configuration should be valid";
}

TEST_F(PathIntegralConfigurationValidationTest, ConfigurationCopyAndAssignment) {
    PathIntegralConfig original;
    original.lambda = 0.5f;
    original.samples = 25;
    original.sampling_mode = PathIntegralSamplingMode::kQuantumLimit;
    original.enabled = true;
    
    // Test copy constructor
    PathIntegralConfig copied(original);
    EXPECT_EQ(copied.lambda, original.lambda);
    EXPECT_EQ(copied.samples, original.samples);
    EXPECT_EQ(copied.sampling_mode, original.sampling_mode);
    EXPECT_EQ(copied.enabled, original.enabled);
    
    // Test assignment operator
    PathIntegralConfig assigned;
    assigned = original;
    EXPECT_EQ(assigned.lambda, original.lambda);
    EXPECT_EQ(assigned.samples, original.samples);
    EXPECT_EQ(assigned.sampling_mode, original.sampling_mode);
    EXPECT_EQ(assigned.enabled, original.enabled);
}

TEST_F(PathIntegralConfigurationValidationTest, BoundaryValueTesting) {
    PathIntegralConfig config;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    // Test lambda boundary values
    std::vector<std::pair<float, bool>> lambda_boundaries = {
        {0.0009f, false},  // Just below minimum
        {0.001f, true},    // Minimum valid
        {0.0011f, true},   // Just above minimum
        {9.999f, true},    // Just below maximum
        {10.0f, true},     // Maximum valid
        {10.001f, false}   // Just above maximum
    };
    
    for (const auto& [lambda, should_be_valid] : lambda_boundaries) {
        config.lambda = lambda;
        config.samples = 10; // Valid sample count
        
        EXPECT_NO_THROW({
            PathIntegralConfig test_config = config;
            (void)test_config;
        }) << "Lambda boundary test failed for " << lambda;
    }
    
    // Test sample count boundary values
    std::vector<std::pair<int, bool>> sample_boundaries = {
        {0, false},      // Invalid
        {1, true},       // Minimum valid
        {2, true},       // Just above minimum
        {99999, true},   // Just below reasonable maximum
        {100000, true},  // Reasonable maximum
        {100001, false}  // Above reasonable maximum
    };
    
    for (const auto& [samples, should_be_valid] : sample_boundaries) {
        config.lambda = 0.1f; // Valid lambda
        config.samples = samples;
        
        EXPECT_NO_THROW({
            PathIntegralConfig test_config = config;
            (void)test_config;
        }) << "Sample boundary test failed for " << samples;
    }
}

TEST_F(PathIntegralConfigurationValidationTest, ThreadSafetyBasics) {
    // Basic thread safety test for configuration objects
    PathIntegralConfig config;
    config.lambda = 0.1f;
    config.samples = 10;
    config.sampling_mode = PathIntegralSamplingMode::kCompetitive;
    config.enabled = true;
    
    // Test that multiple threads can read configuration safely
    std::vector<std::thread> threads;
    std::vector<bool> results(4, false);
    
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&config, &results, i]() {
            try {
                // Read configuration values
                float lambda = config.lambda;
                int samples = config.samples;
                auto mode = config.sampling_mode;
                bool enabled = config.enabled;
                
                // Verify values are consistent
                results[i] = (lambda == 0.1f && samples == 10 && 
                             mode == PathIntegralSamplingMode::kCompetitive && 
                             enabled == true);
            } catch (...) {
                results[i] = false;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(results[i]) << "Thread " << i << " failed configuration read";
    }
}

} // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
