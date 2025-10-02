#pragma once

// Main header for Path Integral functionality
#include "search/path_integral/config.h"
#include "search/path_integral/interfaces.h"
#include "search/path_integral/softmax.h"

namespace lczero {

// Forward declarations
class PathIntegralController;

// Main Path Integral integration point
class PathIntegralSystem {
public:
  PathIntegralSystem();
  ~PathIntegralSystem();

  // Initialize with configuration
  bool Initialize(const PathIntegralConfig& config);
  
  // Check if Path Integral is enabled and ready
  bool IsEnabled() const;
  
  // Get the controller for move selection
  PathIntegralController* GetController();

private:
  PathIntegralConfig config_;
  std::unique_ptr<PathIntegralController> controller_;
  bool initialized_ = false;
};

} // namespace lczero