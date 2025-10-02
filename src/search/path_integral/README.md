# Path Integral Search Implementation

This directory contains the implementation of Path Integral sampling methodology for the LC0 chess engine.

## Overview

Path Integral sampling is an advanced chess analysis technique that uses quantum-inspired sampling methods to explore chess positions. The implementation provides two main modes:

1. **Competitive Mode**: Optimized for competitive play, combining LC0's search capabilities with Path Integral sampling
2. **Quantum Limit Mode**: Research-oriented mode providing detailed analysis with policy and value head access

## Documentation

For comprehensive information about Path Integral Sampling, please refer to:

- **[PATH_INTEGRAL_DOCUMENTATION.md](PATH_INTEGRAL_DOCUMENTATION.md)**: Complete user guide and technical documentation
- **[PERFORMANCE_GUIDELINES.md](PERFORMANCE_GUIDELINES.md)**: Performance optimization and hardware recommendations
- **[INTEGRATION_EXAMPLES.md](INTEGRATION_EXAMPLES.md)**: Practical examples and integration patterns

## Quick Start

### Basic Configuration
```bash
# Enable Path Integral with default settings
setoption name PathIntegralMode value competitive
setoption name PathIntegralLambda value 0.1
setoption name PathIntegralSamples value 50
```

### Build Configuration
```bash
# Enable Path Integral during build
meson setup builddir -Dpath_integral=true
meson compile -C builddir
```

## Key Features

- Numerically stable log-sum-exp softmax implementation
- GPU-optimized batch processing for performance
- Configurable sampling parameters via UCI options
- Comprehensive error handling and fallback mechanisms
- Export capabilities for research and analysis

## UCI Options

- `PathIntegralLambda`: Softmax temperature (0.001-10.0, default 0.1)
- `PathIntegralSamples`: Number of samples (1-100000, default 50)
- `PathIntegralRewardMode`: Reward calculation method (policy/cp_score/hybrid, default hybrid)
- `PathIntegralMode`: Sampling mode (competitive/quantum_limit, default competitive)

## Implementation Architecture

The implementation follows a modular architecture with clear separation of concerns:

- **Controller**: Central coordination and UCI integration
- **Softmax Calculator**: Numerically stable probability calculations
- **Mode Handlers**: Separate implementations for competitive and quantum limit modes
- **Sampling Engine**: GPU-optimized sample generation with CPU fallback
- **Result Exporter**: JSON/CSV export functionality
- **Error Handler**: Comprehensive error handling and recovery
- **Performance Logger**: Detailed performance monitoring and statistics

## Performance Highlights

- GPU acceleration provides 3-5x performance improvement
- Adaptive depth conversion based on position complexity
- Efficient memory management and resource cleanup
- Graceful degradation to standard LC0 behavior on errors
- Minimal overhead in competitive mode (~15-25%)

## Testing and Validation

Comprehensive test suite includes:
- Unit tests for all core components
- Integration tests for UCI protocol compliance
- Performance benchmarks and regression tests
- Memory leak detection and resource validation
- Comprehensive requirements validation tests

## Research Applications

Path Integral Sampling enables advanced chess research:
- Move diversity analysis across different lambda values
- Position complexity evaluation
- Alternative move exploration
- Policy vs. value head analysis
- Quantum-inspired chess analysis methodologies

## Support and Troubleshooting

For issues and questions:
1. Check the performance guidelines for optimization tips
2. Review integration examples for usage patterns
3. Enable debug logging for detailed diagnostics
4. Consult the comprehensive documentation for technical details

## Contributing

When contributing to Path Integral functionality:
1. Follow the existing modular architecture
2. Add comprehensive tests for new features
3. Update documentation for user-facing changes
4. Ensure backward compatibility with existing UCI protocol
5. Validate performance impact and optimization opportunities