"""
MediaPipe Tasks HandLandmarker wrapper providing a clean interface for downstream modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

import cv2
import mediapipe as mp

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"

_SWAPPED_LABELS = {"Left": "Right", "Right": "Left"}


@dataclass
class HandData:
    """Minimal structure describing a detected hand."""

    landmarks: List[tuple[float, float, float]]
    handedness: str


def result_to_hands(result: Any, mirrored_input: bool) -> List[HandData]:
    """
    Convert a HandLandmarkerResult into ``HandData`` objects.

    MediaPipe documents that a horizontally mirrored image inverts the
    handedness labels, so they are swapped back when ``mirrored_input`` is set.
    """
    hands: List[HandData] = []
    for landmark_list, categories in zip(result.hand_landmarks, result.handedness):
        label = categories[0].category_name if categories else "Unknown"
        if mirrored_input:
            label = _SWAPPED_LABELS.get(label, label)
        landmarks = [(lm.x, lm.y, lm.z) for lm in landmark_list]
        hands.append(HandData(landmarks=landmarks, handedness=label))
    return hands


class MonotonicMs:
    """Strictly increasing millisecond timestamps, as VIDEO mode requires."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last = -1

    def next(self) -> int:
        now = int(self._clock() * 1000)
        if now <= self._last:
            now = self._last + 1
        self._last = now
        return now


class HandTracker:
    """Encapsulates HandLandmarker configuration and processing."""

    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.2,
        presence_confidence: float = 0.2,
        tracking_confidence: float = 0.2,
        mirrored_input: bool = True,
        model_path: Path = MODEL_PATH,
    ) -> None:
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Hand model not found at {model_path}. See the README section "
                "'Hand model' for where to download hand_landmarker.task."
            )
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mirrored_input = mirrored_input
        self._timestamps = MonotonicMs()

    def process_frame(self, frame) -> List[HandData]:
        """
        Run hand detection on the provided BGR frame.

        Returns a list of ``HandData`` objects—one per detected hand.
        """
        if frame is None:
            return []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(image, self._timestamps.next())
        return result_to_hands(result, self._mirrored_input)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()


def correct_fingertip(
    base: tuple[float, float, float],
    tip: tuple[float, float, float],
    factor: float = 1.35,
) -> tuple[float, float, float]:
    """
    Extend a fingertip along the base->tip direction to compensate for jitter.
    """
    dx = tip[0] - base[0]
    dy = tip[1] - base[1]
    dz = tip[2] - base[2]
    return (
        base[0] + dx * factor,
        base[1] + dy * factor,
        base[2] + dz * factor,
    )
