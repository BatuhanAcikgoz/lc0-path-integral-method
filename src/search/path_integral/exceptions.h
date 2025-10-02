#pragma once

#include <stdexcept>
#include <string>
#include <chrono>
#include <memory>
#include "utils/exception.h"

namespace lczero {

// Forward declarations for error context
struct ErrorContext;
class PathIntegralErrorHandler;

// Base exception class for Path Integral operations with enhanced error context
class PathIntegralException : public Exception {
public:
  explicit PathIntegralException(const std::string& message)
      : Exception("PathIntegral: " + message), 
        timestamp_(std::chrono::steady_clock::now()),
        error_code_(0),
        recoverable_(true) {}
  
  PathIntegralException(const std::string& message, int error_code, bool recoverable = true)
      : Exception("PathIntegral: " + message),
        timestamp_(std::chrono::steady_clock::now()),
        error_code_(error_code),
        recoverable_(recoverable) {}
  
  // Enhanced error information
  std::chrono::steady_clock::time_point GetTimestamp() const { return timestamp_; }
  int GetErrorCode() const { return error_code_; }
  bool IsRecoverable() const { return recoverable_; }
  
  // Error context management
  void SetContext(std::shared_ptr<ErrorContext> context) { context_ = context; }
  std::shared_ptr<ErrorContext> GetContext() const { return context_; }

private:
  std::chrono::steady_clock::time_point timestamp_;
  int error_code_;
  bool recoverable_;
  std::shared_ptr<ErrorContext> context_;
};

// Configuration-related exceptions
class PathIntegralConfigException : public PathIntegralException {
public:
  explicit PathIntegralConfigException(const std::string& message)
      : PathIntegralException("Config: " + message, 1001) {}
  
  PathIntegralConfigException(const std::string& message, const std::string& option_name)
      : PathIntegralException("Config: " + message + " (option: " + option_name + ")", 1001),
        option_name_(option_name) {}
  
  const std::string& GetOptionName() const { return option_name_; }

private:
  std::string option_name_;
};

// Softmax calculation exceptions with numerical details
class PathIntegralSoftmaxException : public PathIntegralException {
public:
  explicit PathIntegralSoftmaxException(const std::string& message)
      : PathIntegralException("Softmax: " + message, 2001) {}
  
  PathIntegralSoftmaxException(const std::string& message, float lambda, size_t array_size)
      : PathIntegralException("Softmax: " + message + " (lambda=" + std::to_string(lambda) + 
                             ", size=" + std::to_string(array_size) + ")", 2001),
        lambda_(lambda), array_size_(array_size) {}
  
  float GetLambda() const { return lambda_; }
  size_t GetArraySize() const { return array_size_; }

private:
  float lambda_ = 0.0f;
  size_t array_size_ = 0;
};

// Sampling operation exceptions with sample context
class PathIntegralSamplingException : public PathIntegralException {
public:
  explicit PathIntegralSamplingException(const std::string& message)
      : PathIntegralException("Sampling: " + message, 3001) {}
  
  PathIntegralSamplingException(const std::string& message, int sample_index, int total_samples)
      : PathIntegralException("Sampling: " + message + " (sample " + std::to_string(sample_index) + 
                             "/" + std::to_string(total_samples) + ")", 3001),
        sample_index_(sample_index), total_samples_(total_samples) {}
  
  int GetSampleIndex() const { return sample_index_; }
  int GetTotalSamples() const { return total_samples_; }

private:
  int sample_index_ = -1;
  int total_samples_ = 0;
};

// Neural network access exceptions with network state
class PathIntegralNeuralException : public PathIntegralException {
public:
  explicit PathIntegralNeuralException(const std::string& message)
      : PathIntegralException("Neural: " + message, 4001, false) {} // Usually not recoverable
  
  PathIntegralNeuralException(const std::string& message, const std::string& network_type)
      : PathIntegralException("Neural: " + message + " (network: " + network_type + ")", 4001, false),
        network_type_(network_type) {}
  
  const std::string& GetNetworkType() const { return network_type_; }

private:
  std::string network_type_;
};

// GPU/resource allocation exceptions with resource details
class PathIntegralResourceException : public PathIntegralException {
public:
  explicit PathIntegralResourceException(const std::string& message)
      : PathIntegralException("Resource: " + message, 5001) {}
  
  PathIntegralResourceException(const std::string& message, size_t requested_memory, size_t available_memory)
      : PathIntegralException("Resource: " + message + " (requested=" + std::to_string(requested_memory) + 
                             "B, available=" + std::to_string(available_memory) + "B)", 5001),
        requested_memory_(requested_memory), available_memory_(available_memory) {}
  
  size_t GetRequestedMemory() const { return requested_memory_; }
  size_t GetAvailableMemory() const { return available_memory_; }

private:
  size_t requested_memory_ = 0;
  size_t available_memory_ = 0;
};

// Export/IO exceptions with file context
class PathIntegralExportException : public PathIntegralException {
public:
  explicit PathIntegralExportException(const std::string& message)
      : PathIntegralException("Export: " + message, 6001) {}
  
  PathIntegralExportException(const std::string& message, const std::string& filename)
      : PathIntegralException("Export: " + message + " (file: " + filename + ")", 6001),
        filename_(filename) {}
  
  const std::string& GetFilename() const { return filename_; }

private:
  std::string filename_;
};

// MCTS integration exceptions with tree state
class PathIntegralMCTSException : public PathIntegralException {
public:
  explicit PathIntegralMCTSException(const std::string& message)
      : PathIntegralException("MCTS: " + message, 7001, false) {} // Usually not recoverable
  
  PathIntegralMCTSException(const std::string& message, int node_count, int depth)
      : PathIntegralException("MCTS: " + message + " (nodes=" + std::to_string(node_count) + 
                             ", depth=" + std::to_string(depth) + ")", 7001, false),
        node_count_(node_count), depth_(depth) {}
  
  int GetNodeCount() const { return node_count_; }
  int GetDepth() const { return depth_; }

private:
  int node_count_ = 0;
  int depth_ = 0;
};

// Data validation exceptions for corrupted data
class PathIntegralDataException : public PathIntegralException {
public:
  explicit PathIntegralDataException(const std::string& message)
      : PathIntegralException("Data: " + message, 8001) {}
  
  PathIntegralDataException(const std::string& message, const std::string& data_type, size_t data_size)
      : PathIntegralException("Data: " + message + " (type=" + data_type + 
                             ", size=" + std::to_string(data_size) + ")", 8001),
        data_type_(data_type), data_size_(data_size) {}
  
  const std::string& GetDataType() const { return data_type_; }
  size_t GetDataSize() const { return data_size_; }

private:
  std::string data_type_;
  size_t data_size_ = 0;
};

// Memory allocation failure exceptions
class PathIntegralMemoryException : public PathIntegralException {
public:
  explicit PathIntegralMemoryException(const std::string& message)
      : PathIntegralException("Memory: " + message, 9001) {}
  
  PathIntegralMemoryException(const std::string& message, size_t allocation_size)
      : PathIntegralException("Memory: " + message + " (size=" + std::to_string(allocation_size) + "B)", 9001),
        allocation_size_(allocation_size) {}
  
  size_t GetAllocationSize() const { return allocation_size_; }

private:
  size_t allocation_size_ = 0;
};

// Error context structure for detailed error information
struct ErrorContext {
  std::string operation_name;
  std::string component_name;
  std::chrono::steady_clock::time_point start_time;
  std::string position_fen;
  int sample_count = 0;
  float lambda = 0.0f;
  std::string additional_info;
  
  ErrorContext(const std::string& op_name, const std::string& comp_name)
      : operation_name(op_name), component_name(comp_name),
        start_time(std::chrono::steady_clock::now()) {}
};

}  // namespace lczero