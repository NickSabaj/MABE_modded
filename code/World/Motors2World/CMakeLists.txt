option(enable_World_Motors2 "Enable Module Motors2World" ON)
if (enable_World_Motors2)
  register_module(World Motors2)

  include_directories(${CMAKE_CURRENT_LIST_DIR}) # allow "#include <animat_utilities>"
  target_sources(${EXE} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Motors2World.cpp)
  target_sources(${EXE} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Motors2World.h)
  target_sources(${EXE} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/animat_utilities.cpp)
  target_sources(${EXE} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/animat_utilities)

  # LAPACK / MKL
  find_package(LAPACK REQUIRED)
  include_directories(${LAPACK_INCLUDE_DIRS})
  target_link_libraries(${EXE} ${LAPACK_LIBRARIES})

  # ARMADILLO
  include_directories(${CMAKE_CURRENT_LIST_DIR}/armadillo/include)
  if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNO_BOUNDS_CHECKING")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
    message(STATUS "building with armadillo math library in *Release* mode")
  else()
    message(STATUS "building with armadillo math library in *Debug* mode")
  endif()

  # PPRINT
  include_directories(${CMAKE_CURRENT_LIST_DIR}/pprint/include)

  # X-PLATFORM MULTITHREADING
  find_package(Threads)
  target_link_libraries(${EXE} ${CMAKE_THREAD_LIBS_INIT})
endif()
