cmake_minimum_required(VERSION 3.0.2)
project(task1_pick)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  kuavo_msgs
  task1_vision
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    kuavo_msgs
    task1_vision
)

catkin_install_python(PROGRAMS
  scripts/task1_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  scripts/task1_config.py
  scripts/task1_perception.py
  scripts/task1_motion.py
  scripts/task1_scheduler.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
