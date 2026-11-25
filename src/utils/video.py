import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {'.avi', '.mp4', '.mkv', '.mov'}


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.fps) or self.fps <= 0:
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


def extract_source_group(video_path: Path) -> str:
    """Extract source clip prefix from filename."""
    stem = video_path.stem
    return stem.rsplit('_', 1)[0] if '_' in stem else stem


def compute_video_fingerprint(video_path: Path) -> tuple[str, np.ndarray | None]:
    """Compute SHA-256 hash and 64-bit averaged-background dHash for a video."""
    with video_path.open('rb') as f:
        sha256 = hashlib.file_digest(f, 'sha256').hexdigest()

    cap = cv2.VideoCapture(str(video_path))
    samples: list[np.ndarray] = []
    try:
        if not cap.isOpened():
            return sha256, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frac in (0.2, 0.5, 0.8):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * frac))
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                samples.append(cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA))

    finally:
        cap.release()

    if not samples:
        return sha256, None

    mean_frame = np.mean(samples, axis=0)
    if mean_frame.std() < 5.0:
        return sha256, None

    diff = mean_frame[:, 1:] > mean_frame[:, :-1]
    return sha256, np.packbits(diff)


def dhash_distance_matrix(hashes_a: np.ndarray, hashes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise Hamming distances between two arrays of packed 64-bit dHashes."""
    xor = hashes_a[:, None, :] ^ hashes_b[None, :, :]
    return np.bitwise_count(xor).sum(axis=-1)
