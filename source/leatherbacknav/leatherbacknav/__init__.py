# Copyright (c) 2025, Nepher Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module for Leatherback waypoint navigation task.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *

# Export evaluation compatibility utilities for wheeled-waypoint-eval
from .tasks.manager_based.waypoint_nav.eval_compat import EvalCompatEnv, wrap_for_eval

__all__ = ["EvalCompatEnv", "wrap_for_eval"]

