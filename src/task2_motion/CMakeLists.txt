cmake_minimum_required(VERSION 3.0.2)
project(task2_motion)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  kuavo_msgs
  task2_vision
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    kuavo_msgs
    task2_vision
)

catkin_install_python(PROGRAMS
  scripts/task2_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  scripts/task2_config.py
  scripts/task2_perception.py
  scripts/task2_geometry.py
  scripts/task2_motion.py
  scripts/task2_scheduler.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
