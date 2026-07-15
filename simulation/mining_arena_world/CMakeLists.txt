cmake_minimum_required(VERSION 3.8)
project(mining_arena_world)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install directories
install(DIRECTORY
  worlds
  heightmaps
  models
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install scripts
install(PROGRAMS
  scripts/generate_heightmap.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
