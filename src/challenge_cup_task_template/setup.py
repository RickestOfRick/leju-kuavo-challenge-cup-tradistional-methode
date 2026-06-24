#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""catkin Python 包安装配置。

通过 CMakeLists.txt 中的 catkin_python_setup() 调用，使 src/ 下的
challenge_task_lib 包可在 devel/install 空间被正常 import。
"""

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=["challenge_task_lib"],
    package_dir={"": "src"},
)

setup(**setup_args)
