# vleo_io: YAML configuration schema/parser, resolved-config and run-metadata writers.
add_library(vleo_io STATIC
  Config.cpp
  ConfigParser.cpp
  ConfigWriter.cpp
  JsonWriter.cpp
  PlatformConfig.cpp
  RunMetadata.cpp
  YamlHelpers.cpp)
target_link_libraries(vleo_io
  PUBLIC vleo_core vleo_gsi
  PRIVATE yaml-cpp::yaml-cpp vleo_warnings)
