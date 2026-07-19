cmake_minimum_required(VERSION 3.0.2)
project(task2_vision)

find_package(catkin REQUIRED COMPONENTS
  kuavo_msgs
  rospy
  std_msgs
  sensor_msgs
  tf2_ros
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    kuavo_msgs
    rospy
    std_msgs
    sensor_msgs
    tf2_ros
)

catkin_install_python(PROGRAMS
  scripts/task2_depth_xy_node.py
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
