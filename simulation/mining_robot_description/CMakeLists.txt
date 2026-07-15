cmake_minimum_required(VERSION 3.8)
project(mining_robot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install directories
install(DIRECTORY
  urdf
  meshes
  config
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# Install scripts
install(PROGRAMS
  scripts/generate_robot_meshes.py
  scripts/validate_urdf.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
