cmake_minimum_required(VERSION 3.0.2)
project(task2_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
)

catkin_install_python(PROGRAMS
  scripts/task2_vision_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  resource
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
