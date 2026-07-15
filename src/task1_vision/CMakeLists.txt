cmake_minimum_required(VERSION 3.0.2)
project(task1_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  kuavo_msgs
  cv_bridge
  compressed_depth_image_transport
  tf2_ros
  vision_msgs
  visualization_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    kuavo_msgs
    cv_bridge
    compressed_depth_image_transport
    tf2_ros
    vision_msgs
    visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/task1_vision_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  models
  resource
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES
  plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
