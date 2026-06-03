# Air Hockey Puck Tracker

Real-time detection and trajectory prediction of an air hockey puck using a camera, OpenCV, and a Kalman filter.

Developed as a Bachelor's thesis project at a Hungarian university.

---

## Features

- **Perspective correction** – homography transform maps the camera view to a flat top-down table view
- **Puck detection** – HSV colour masking combined with MOG2 background subtraction isolates the moving red puck
- **Kalman filter tracking** – 4-state filter (x, y, vx, vy) handles occlusions and noisy detections
- **Physics-based trajectory prediction** – simulates friction, wall bounces, and paddle collisions
- **Prediction evaluation** – compares predicted positions with actual ones at 5 / 10 / 20 frame horizons and plots the results

---

## Demo

Set `RECORD = True` in `main.py` before running to save a `demo_output.mp4` recording.

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.8+ |
| opencv-python | ≥ 4.8 |
| numpy | ≥ 1.24 |
| matplotlib | ≥ 3.7 |

---

## Installation & Running

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/air-hockey-tracker.git
cd air-hockey-tracker

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
py main.py
```

Press **Q** to quit. After the video ends, a prediction error plot is displayed and saved as `prediction_error.png`.

---

## Download (pre-built executable)

A standalone Windows executable that requires no Python installation is available:

**[Download AirHockeyTracker.exe](https://github.com/<your-username>/air-hockey-tracker/releases/latest)**

Just download and run — `test_video.mp4` is bundled inside.

---

## Building the executable yourself

```bat
build.bat
```

The resulting `AirHockeyTracker.exe` will appear in the `dist\` folder (~150 MB).
Requires PyInstaller (`pip install pyinstaller`).

---

## Camera Calibration

Run `calibrate.py` to define the four corners of the table in the camera view.
The result is saved as `calibration.npy` and loaded automatically on startup.

```bash
py calibrate.py
```

If no calibration file is found, the program falls back to hardcoded default points.

---

## Controls

| Key | Action |
|---|---|
| Q | Quit |

---

## Project Structure

```
air-hockey-tracker/
├── main.py          # Main loop, visualisation, evaluation plots
├── tracker.py       # AirHockeyTracker class (detection, Kalman, prediction)
├── calibrate.py     # Interactive camera calibration utility
├── utils.py         # resource_path() helper (dev + PyInstaller compatible)
├── requirements.txt
├── build.bat        # PyInstaller build script (Windows)
├── calibration.npy  # Saved homography source points
└── test_video.mp4   # Sample video for testing
```

---

## Algorithm Overview

```
Camera frame
    │
    ▼
Perspective warp (homography) → 800×400 px top-down view
    │
    ├─ HSV red mask  ──────────────────────┐
    └─ MOG2 motion mask  ──────────────────┤
                                           ▼
                               Combined mask (moving red objects)
                                           │
                                     Contour filtering
                                  (area + circularity)
                                           │
                              ┌────────────┴────────────┐
                           Paddles                    Puck candidates
                              │                           │
                              └──────────┬────────────────┘
                                         ▼
                               Kalman filter (x, y, vx, vy)
                                         │
                                         ▼
                          Physics simulation (friction + bounces)
                                         │
                                         ▼
                              Trajectory waypoints drawn
```
