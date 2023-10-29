## This module cannot be used with the Motors2World module
## As such, all necessary dependencies are in the Motors2World/CMakeLists
option(enable_Brain_Motors2 "Enable Module Motors2Brain" OFF)
if (enable_Brain_Motors2)
  register_module(Brain Motors2)
  target_sources(${EXE} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Motors2Brain.cpp)
  target_sources(${EXE} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Motors2Brain.h)
endif()
