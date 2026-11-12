#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace vleo {

// Thrown when a configuration file cannot be read or fails validation.
// Collects every problem found (not only the first) so users can fix a file
// in one pass. Each issue is "<key path>: <message>".
class ConfigError : public std::runtime_error {
 public:
  explicit ConfigError(std::vector<std::string> issues, const std::string& context = "configuration");

  const std::vector<std::string>& issues() const noexcept { return issues_; }

 private:
  std::vector<std::string> issues_;
};

}  // namespace vleo
