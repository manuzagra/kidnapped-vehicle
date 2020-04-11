project(kidnapped_vehicle)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++11)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

include_directories(include)

set(sources
    src/main.cpp
    src/particle_filter.cpp)

add_executable(particle_filter ${sources})

target_link_libraries(particle_filter z ssl uv uWS)
