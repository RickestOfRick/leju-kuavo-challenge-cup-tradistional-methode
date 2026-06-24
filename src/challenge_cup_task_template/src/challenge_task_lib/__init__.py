#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""challenge_task_lib —— 挑战杯选手任务逻辑包。

按职责分层，便于阅读与调试：
    config        所有需标定的常量(★TODO 集中地)
    perception    合法传感器订阅与缓存
    hardware      行走/手臂/夹爪/IK 底层 IO
    vision        颜色定位 + 深度反投影到 base_link
    manipulation  抓取/放置/搜索动作宏
    scenes        三场景流水线
    runner        顶层组装入口(execute_task)

入口脚本只需 `from challenge_task_lib.runner import execute_task`。
"""

from .runner import execute_task

__all__ = ["execute_task"]
