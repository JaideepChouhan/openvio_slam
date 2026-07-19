cmake_minimum_required(VERSION 3.8)
project(openvio_slam)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  openvio_slam/ros/vio_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME openvio_slam_node
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
