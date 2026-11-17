# Computer Vision–Based Virtual Input System

## Overview

A real-time hand-tracking system that converts webcam input into mouse control. Uses MediaPipe for hand detection, tracks palm center for cursor movement, and recognizes pinch gestures for left/right clicks. Maps a defined interaction zone to full screen dimensions for edge-to-edge control.

## Technologies & Libraries

- **MediaPipe Tasks (HandLandmarker)** - Hand landmark detection and tracking
- **OpenCV** - Video capture, frame processing, and visualization
- **PyAutoGUI** - OS-level mouse control (cursor movement, clicks)
- **Python 3.12** - Runtime; runs on macOS and Windows

## Features

- **Palm-center cursor tracking**: Stable cursor movement using wrist and middle MCP joint midpoint
- **Pinch gesture recognition**: 
  - Left-click: Index finger + thumb pinch
  - Right-click: Middle finger + thumb pinch
- **Two-zone spatial mapping**:
  - Tracking zone (green): Larger detection area for hand presence
  - Mapping zone (red): Inner region mapped 1:1 to full screen dimensions
- **Adaptive smoothing**: Exponential moving average with velocity-based prediction to reduce hand jitter
- **Micro-jitter filtering**: Deadzone threshold (~3px) to ignore involuntary hand tremors
- **Dynamic sensitivity control**: Configurable tracking/mapping scale factors for user comfort

## Installation

Dependencies are pinned in `requirements.txt` (generated from `requirements.in`), so every machine and CI installs the same versions.

**macOS** (with [uv](https://docs.astral.sh/uv/)):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Windows** (PowerShell, Python 3.12 from python.org):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first `import mediapipe` takes about 30 seconds while Matplotlib builds its font cache, with no window on screen. It only happens once. Run `python -c "import mediapipe"` after installing so the first launch isn't mistaken for a hang.

To change dependencies, edit `requirements.in`, then run `uv pip compile --universal --python-version 3.12 requirements.in -o requirements.txt`.

### macOS permissions

macOS grants these to the app that runs Python (Terminal, iTerm, VS Code, ...), not to Python itself:

1. **Camera**: System Settings > Privacy & Security > Camera > turn on your terminal app.
2. **Accessibility**: System Settings > Privacy & Security > Accessibility > turn on your terminal app. Without it, macOS silently ignores cursor moves and clicks; `main.py` prints a warning at startup.

Quit and reopen the terminal app after granting either one.

Windows needs no extra permission. Windows will not let it move the cursor over apps running as administrator.

### Hand model

`models/hand_landmarker.task` is MediaPipe's hand landmarker (float16), committed so the app works offline.

- Source: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
- Size: 7,819,105 bytes
- SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`

## Usage

```bash
python main.py
```

**Controls:**
- Move palm to control cursor
- Pinch index+thumb for left-click
- Pinch middle+thumb for right-click
- Open palm to enable, fist to disable
- Press 'q' (with the preview window focused) to quit

## Tests

```bash
python -m pytest
```

The tests run without a camera or screen. CI runs them on macOS and Windows. They cannot check the camera, cursor movement or clicks; those need the live check below.

### Live check

1. Raise your right hand: the overlay must show `Cursor Hand: Right`. (MediaPipe flips left/right on a mirrored image, and `HandTracker(mirrored_input=True)` swaps them back. If this shows `Left`, the swap is wrong.)
2. Make a fist: `Interaction Enabled: False`. Open your palm: `True`.
3. Move the cursor to all four screen edges.
4. Pinch index+thumb to left-click and middle+thumb to right-click.

## Configuration

In `main.py`:
- `TRACKING_SCALE` (default: 0.9) - Size of hand detection zone
- `MAPPING_SCALE` (default: 0.75) - Size of screen-mapped zone

