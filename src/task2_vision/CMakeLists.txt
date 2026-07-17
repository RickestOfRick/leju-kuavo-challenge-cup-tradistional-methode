cmake_minimum_required(VERSION 3.0.2)
project(task2_vision)

find_package(catkin REQUIRED COMPONENTS
  kuavo_msgs
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2_ros
  vision_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    kuavo_msgs
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    tf2_ros
    vision_msgs
)

catkin_install_python(PROGRAMS
  scripts/task2_vision_node.py
  scripts/task2_depth_xy_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  resource
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES
  plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
