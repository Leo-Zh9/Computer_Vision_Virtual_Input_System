"""Headless tests: no camera, no screen, no pyautogui."""

from types import SimpleNamespace

import numpy as np

from vision.hand_tracker import HandTracker, MonotonicMs, result_to_hands
from vision.visualization import HAND_CONNECTIONS, draw_skeleton


def _fake_result(labels):
    landmarks = [[SimpleNamespace(x=i / 21, y=0.5, z=0.0) for i in range(21)] for _ in labels]
    handedness = [[SimpleNamespace(category_name=label)] for label in labels]
    return SimpleNamespace(hand_landmarks=landmarks, handedness=handedness)


def test_result_to_hands_converts_landmarks():
    hands = result_to_hands(_fake_result(["Left"]), mirrored_input=False)
    assert len(hands) == 1
    assert hands[0].handedness == "Left"
    assert len(hands[0].landmarks) == 21
    assert hands[0].landmarks[3] == (3 / 21, 0.5, 0.0)


def test_mirrored_input_swaps_handedness():
    # Regression pin only: it encodes MediaPipe's documented behaviour
    # ("a horizontally mirrored image will invert these outputs"). Whether the
    # swap is right is decided by the live check in the README, not by this test.
    hands = result_to_hands(_fake_result(["Left", "Right"]), mirrored_input=True)
    assert [h.handedness for h in hands] == ["Right", "Left"]


def test_timestamps_strictly_increase_even_if_clock_stalls():
    ticks = iter([1.0, 1.0, 0.999, 1.0005, 2.0])
    stamps = MonotonicMs(clock=lambda: next(ticks))
    values = [stamps.next() for _ in range(5)]
    assert values == sorted(set(values))
    assert values[-1] == 2000


def test_real_model_finds_no_hands_in_black_frame():
    tracker = HandTracker()
    try:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert tracker.process_frame(frame) == []
        assert tracker.process_frame(frame) == []
    finally:
        tracker.close()


def test_draw_skeleton_uses_library_connections():
    assert len(HAND_CONNECTIONS) == 21
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    hand = result_to_hands(_fake_result(["Right"]), mirrored_input=False)[0]
    draw_skeleton(frame, hand)
    assert frame.any()
