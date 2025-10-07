# Path Integral Sampling Verification Tool

This tool verifies that the LC0 C++ Path Integral implementation performs the correct number of samples and uses neural network evaluation properly.

## Purpose

The verification tool addresses the critical need to ensure that the Path Integral sampling implementation is working correctly. It was created to investigate suspicious performance metrics that suggested the C++ implementation might not be performing the same amount of computational work as expected.

## Features

- **Sample Count Verification**: Ensures the actual number of samples matches the requested count
- **Neural Network Usage Verification**: Confirms that neural network evaluations are being performed when a backend is available
- **Performance Analysis**: Measures timing and throughput to detect unreasonable performance patterns
- **Comprehensive Test Suites**: Multiple test scenarios covering standard, performance, and edge cases
- **Detailed Reporting**: Generates comprehensive reports in text, JSON, or CSV format

## Building

The verification tool is built automatically when Path Integral support is enabled:

```bash
meson setup builddir -Dpath_integral=true
meson compile -C builddir
```

This creates the `verify_path_integral` executable in the build directory.

## Usage

### Basic Usage

Run the standard test suite:
```bash
./verify_path_integral --test-suite=standard
```

### With Neural Network Backend

Test with a specific neural network:
```bash
./verify_path_integral --weights=network.pb.gz --backend=cuda --test-suite=standard
```

### Custom Configuration

Test with specific Path Integral parameters:
```bash
./verify_path_integral \
  --PathIntegralSamples=100 \
  --PathIntegralLambda=0.5 \
  --PathIntegralMode=competitive \
  --test-suite=performance
```

### Custom Positions

Test specific chess positions:
```bash
./verify_path_integral \
  --positions="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10" \
  --PathIntegralSamples=50
```

### Export Results

Export results to a file:
```bash
./verify_path_integral \
  --test-suite=comprehensive \
  --output-format=json \
  --output-file=verification_results.json \
  --output-dir=./reports/
```

## Command Line Options

### Test Configuration
- `--test-suite`: Test suite to run (standard, performance, edge-case, comprehensive)
- `--positions`: Comma-separated list of FEN positions to test
- `--output-format`: Output format (text, json, csv)
- `--output-file`: Output file name
- `--output-dir`: Output directory for reports

### Path Integral Configuration
- `--PathIntegralLambda`: Softmax temperature parameter (0.001-10.0, default 0.1)
- `--PathIntegralSamples`: Number of samples (1-100000, default 50)
- `--PathIntegralMode`: Sampling mode (competitive, quantum_limit)
- `--PathIntegralRewardMode`: Reward mode (policy, cp_score, hybrid)

### Neural Network Backend
- `--backend`: Neural network backend to use
- `--weights`: Path to neural network weights file
- `--backend-opts`: Backend-specific options

### General Options
- `--verbose`: Enable verbose output
- `--help`: Show help message
- `--threads`: Number of threads to use

## Test Suites

### Standard Test Suite
Basic verification with standard configurations:
- Different lambda values (0.01, 0.1, 1.0)
- Both competitive and quantum limit modes
- Standard sample counts (25-50)

### Performance Test Suite
High sample count performance testing:
- Sample counts from 200 to 500
- Focus on timing analysis
- Throughput measurement

### Edge Case Test Suite
Extreme parameter values and edge conditions:
- Minimum samples (1)
- Extreme lambda values (0.001, 10.0)
- Boundary condition testing

### Comprehensive Test Suite
Combines all test suites for complete verification.

## Output Formats

### Text Format (Default)
Human-readable summary with detailed analysis:
```
=== Comprehensive Verification Summary ===
Total Tests: 28
Passed: 26
Failed: 2
Warnings: 3
Errors: 0

Performance Analysis:
  - Average Samples/sec: 1250.5
  - Min Samples/sec: 890.2
  - Max Samples/sec: 1680.3

Overall Result: SUCCESS
```

### JSON Format
Structured data for programmatic analysis:
```json
{
  "summary": {
    "total_tests": 28,
    "passed_tests": 26,
    "failed_tests": 2,
    "avg_samples_per_second": 1250.5,
    "overall_success": true
  },
  "individual_results": [...]
}
```

### CSV Format
Tabular data for spreadsheet analysis with columns for position, sample counts, timing, and validation results.

## Verification Criteria

The tool validates several aspects of Path Integral sampling:

1. **Sample Count Accuracy**: Actual samples should match requested samples within 5% tolerance
2. **Neural Network Usage**: When a backend is available, neural network evaluations should occur
3. **Timing Reasonableness**: Sample timing should be within reasonable bounds (0.001ms - 1000ms per sample)
4. **Backend Availability**: Proper detection and usage of neural network backends
5. **Sampling Completion**: All sampling operations should complete successfully

## Troubleshooting

### No Neural Network Backend
If no weights file is specified, the tool will use heuristic evaluation and warn about backend unavailability. This is expected behavior.

### Compilation Issues
Ensure Path Integral support is enabled during build:
```bash
meson setup builddir -Dpath_integral=true
```

### Performance Warnings
If you see warnings about extremely fast sampling, this may indicate:
- Cached evaluations being reused
- Heuristic evaluation instead of neural network
- Potential implementation shortcuts

Check the detailed report for specific guidance.

## Integration with CI/CD

The verification tool can be integrated into continuous integration pipelines:

```bash
# Run verification and check exit code
./verify_path_integral --test-suite=standard --output-format=json --output-file=ci_results.json
if [ $? -eq 0 ]; then
    echo "Path Integral verification passed"
else
    echo "Path Integral verification failed"
    exit 1
fi
```

## Related Files

- `src/tools/verify_path_integral_sampling.h/cc`: Main verification implementation
- `src/tools/verify_path_integral_main.cc`: Command-line interface
- `src/tools/verify_path_integral_sampling_test.cc`: Unit tests
- `src/search/path_integral/`: Path Integral implementation being verified