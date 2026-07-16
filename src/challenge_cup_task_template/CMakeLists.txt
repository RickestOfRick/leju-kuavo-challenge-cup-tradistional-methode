cmake_minimum_required(VERSION 3.0.2)
project(challenge_cup_task_template)

find_package(catkin REQUIRED COMPONENTS
  task1_vision
  task2_vision
  task2_motion
  kuavo_msgs
  task1_pick
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/challenge_task.py
  scripts/data.py
  scripts/test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
