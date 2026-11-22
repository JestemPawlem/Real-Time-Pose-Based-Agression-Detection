from dataclasses import dataclass
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mkv', '.mov'}


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f'fps must be positive, got {self.fps}')
        if self.frame_count <= 0:
            raise ValueError(f'frame_count must be positive. got {self.frame_count}')
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f'dimensions must be positive, got {self.width}x{self.height}')

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps

    @property
    def resolution(self) -> str:
        return f'{self.width}x{self.height}'

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


def get_video_metadata(video_path: Path) -> VideoMetadata | None:
    """Open a video, verify readability, and return its metadata or None."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None

        ret, frame = cap.read()
        if not ret or frame is None:
            return None

        return VideoMetadata(
            fps=cap.get(cv2.CAP_PROP_FPS),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    except ValueError:
        return None
    finally:
        cap.release()
