from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
from reactivex import operators as ops
from reactivex.disposable import Disposable
from reactivex.subject import Subject

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import Pose, Vector3
from dimos.msgs.nav_msgs import OccupancyGrid
from dimos.msgs.sensor_msgs import Image
from dimos.utils.logging_config import setup_logger
from dimos.utils.reactive import backpressure

logger = setup_logger()


@dataclass
class CameraCalibration:
    """Per-camera calibration parameters."""

    name: str = "front"
    intrinsics: list[float] = field(default_factory=lambda: [500, 0, 320, 0, 500, 240, 0, 0, 1])
    extrinsics: list[float] = field(
        default_factory=lambda: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    )
    image_width: int = 640
    image_height: int = 480


@dataclass
class Config(ModuleConfig):
    model_path: str = ""
    num_cameras: int = 4
    cameras: list[CameraCalibration] = field(default_factory=list)

    # BEV grid — 5m x 5m around the robot at 5cm resolution
    grid_width: int = 100
    grid_height: int = 100
    resolution: float = 0.05
    x_min: float = -2.5
    x_max: float = 2.5
    y_min: float = -2.5
    y_max: float = 2.5
    occupancy_threshold: float = 0.5

    # Model input resolution
    input_width: int = 224
    input_height: int = 224

    # LSS depth discretization — indoor range
    num_depth_bins: int = 32
    depth_min: float = 0.3
    depth_max: float = 8.0

    # Frame synchronization tolerance (seconds)
    sync_tolerance_sec: float = 0.1

    # Publishing: 0=every inference, >0=interval in seconds, -1=never
    publish_interval: float = 0

    frame_id: str = "world"


class BEVOccupancyMapper(Module):
    """BEV occupancy grid mapper using multi-camera LSS inference.

    Drop-in replacement for VoxelGridMapper + CostMapper.
    Takes 4 camera image streams and produces an OccupancyGrid.
    """

    default_config = Config
    config: Config

    # Inputs: 4 camera image streams
    camera_front: In[Image]
    camera_right: In[Image]
    camera_back: In[Image]
    camera_left: In[Image]

    # Output: matches CostMapper for autoconnect compatibility
    global_costmap: Out[OccupancyGrid]

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        # Lazy import — only fails at runtime on machines without the C++ module
        try:
            from _bev_occupancy_net import BEVOccupancyNet
        except ImportError as e:
            raise ImportError(
                "Cannot import _bev_occupancy_net. "
                "Build it with: colcon build --packages-select vision_core "
                "(requires pybind11, TensorRT, CUDA)"
            ) from e

        self._net = BEVOccupancyNet(self._build_native_config())
        if not self._net.is_initialized():
            raise RuntimeError(
                f"BEVOccupancyNet failed to initialize with model: {self.config.model_path}"
            )

        # Latest frame buffer for synchronization
        self._latest_frames: dict[str, Image] = {}
        self._camera_names = ["front", "right", "back", "left"]

        logger.info(
            f"BEVOccupancyMapper initialized: {self.config.grid_width}x{self.config.grid_height} "
            f"@ {self.config.resolution}m/cell"
        )

    def _build_native_config(self) -> dict:
        """Build the config dict expected by the pybind11 BEVOccupancyNet constructor."""
        cameras = []
        for i, name in enumerate(["front", "right", "back", "left"]):
            cal = (
                self.config.cameras[i]
                if i < len(self.config.cameras)
                else CameraCalibration(name=name)
            )
            cameras.append(
                {
                    "name": cal.name,
                    "frame_id": f"camera_{cal.name}",
                    "intrinsics": np.array(cal.intrinsics, dtype=np.float32).reshape(3, 3),
                    "extrinsics": np.array(cal.extrinsics, dtype=np.float32).reshape(4, 4),
                    "image_width": cal.image_width,
                    "image_height": cal.image_height,
                }
            )

        return {
            "model_path": self.config.model_path,
            "num_cameras": self.config.num_cameras,
            "input_width": self.config.input_width,
            "input_height": self.config.input_height,
            "occupancy_threshold": self.config.occupancy_threshold,
            "grid": {
                "width": self.config.grid_width,
                "height": self.config.grid_height,
                "resolution": self.config.resolution,
                "x_min": self.config.x_min,
                "x_max": self.config.x_max,
                "y_min": self.config.y_min,
                "y_max": self.config.y_max,
                "num_depth_bins": self.config.num_depth_bins,
                "depth_min": self.config.depth_min,
                "depth_max": self.config.depth_max,
            },
            "cameras": cameras,
        }

    @rpc
    def start(self) -> None:
        super().start()

        self._publish_trigger: Subject[None] = Subject()
        self._disposables.add(
            backpressure(self._publish_trigger)
            .pipe(ops.map(lambda _: self._run_inference()))
            .subscribe()
        )

        # Subscribe each camera input
        for attr_name, cam_name in [
            ("camera_front", "front"),
            ("camera_right", "right"),
            ("camera_back", "back"),
            ("camera_left", "left"),
        ]:
            stream: In[Image] = getattr(self, attr_name)
            unsub = stream.subscribe(lambda img, n=cam_name: self._on_frame(n, img))
            self._disposables.add(Disposable(unsub))

        # Optional interval-based publishing
        if self.config.publish_interval > 0:
            from reactivex import interval

            self._disposables.add(
                interval(self.config.publish_interval).subscribe(
                    lambda _: self._publish_trigger.on_next(None)
                )
            )

    @rpc
    def stop(self) -> None:
        super().stop()
        self._latest_frames.clear()

    def _on_frame(self, camera_name: str, image: Image) -> None:
        self._latest_frames[camera_name] = image

        # Check if all cameras have recent frames
        if len(self._latest_frames) < len(self._camera_names):
            return

        # Check timestamp synchronization
        timestamps = [self._latest_frames[n].ts for n in self._camera_names]
        if max(timestamps) - min(timestamps) > self.config.sync_tolerance_sec:
            return

        if self.config.publish_interval == 0:
            self._publish_trigger.on_next(None)

    def _run_inference(self) -> None:
        if len(self._latest_frames) < len(self._camera_names):
            return

        # Extract BGR images in camera order
        images = []
        for name in self._camera_names:
            img = self._latest_frames[name]
            images.append(img.to_opencv())

        start_t = time.perf_counter()
        grid_data = self._net.infer(images)
        elapsed_ms = (time.perf_counter() - start_t) * 1000

        logger.debug(f"BEV inference: {elapsed_ms:.1f}ms")

        origin = Pose()
        origin.position = Vector3(self.config.x_min, self.config.y_min, 0.0)

        grid = OccupancyGrid(
            grid=grid_data,
            resolution=self.config.resolution,
            origin=origin,
            frame_id=self.config.frame_id,
            ts=time.time(),
        )
        self.global_costmap.publish(grid)


bev_occupancy = BEVOccupancyMapper.blueprint
