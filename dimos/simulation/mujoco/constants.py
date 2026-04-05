# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

# Video/Camera constants (MuJoCo MJCF fovy = vertical degrees)
# ZED X (2.2mm lens): H 110° / V 80° / D 120° — match vertical FOV to MJCF head_camera / depth cameras.
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240
VIDEO_CAMERA_FOV = 80
DEPTH_CAMERA_FOV = 80

# Depth range (ZED X 2.2mm wide, Stereolabs datasheet): hardware ~0.3–20 m; typical “ideal” use ~0.3–12 m.
MIN_RANGE = 0.3
MAX_RANGE = 20.0

# Camera-frame vertical band (m); sim-only clutter filter — not a ZED hardware spec.
MAX_HEIGHT = 1.2

# Lidar constants
LIDAR_RESOLUTION = 0.05

# Simulation timing constants
VIDEO_FPS = 20
LIDAR_FPS = 2

LAUNCHER_PATH = Path(__file__).parent / "mujoco_process.py"
