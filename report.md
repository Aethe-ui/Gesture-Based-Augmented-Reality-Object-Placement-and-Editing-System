# Project Report — Gesture-Based AR Block Builder

**Date:** 23 February 2026  
**Project:** MP_2026 — Augmented Reality Block Building System  
**Status:** 🟡 Active Development (Core Loop Functional)

---

## 1. Project Overview

The **Gesture-Based AR Block Builder** is a standalone augmented reality (AR) desktop application built entirely in Python. It uses a standard webcam to overlay interactive virtual 3D cubes onto a real-world surface (floor, table, or any flat plane). Interaction is entirely **touchless** — users manipulate the virtual environment exclusively through **hand gestures** detected via the camera.

The primary use-case target is **civil engineering and architectural pre-visualization**: instead of sketching layouts on paper or using heavyweight CAD tools, users can physically walk around and interact with a virtual spatial blueprint.

### Technology Stack

| Component            | Library / Tool                                     |
| -------------------- | -------------------------------------------------- |
| Language             | Python 3                                           |
| Computer Vision      | OpenCV (`opencv-python`)                           |
| Hand Tracking        | MediaPipe Tasks API (`hand_landmarker.task` model) |
| 3D Math / Projection | NumPy + OpenCV `solvePnP` / `projectPoints`        |
| AR Pose Estimation   | ArUco Marker Detection (OpenCV `aruco` module)     |
| Unit Testing         | Python `unittest`                                  |

### Dependencies (`requirements.txt`)

```
opencv-python
mediapipe
numpy
```

---

## 2. Project File Structure

```
MP_2026/
├── main.py               # Main application loop (camera, gesture dispatch, rendering)
├── hand_tracker.py       # MediaPipe hand landmark detection & pinch gesture logic
├── block_manager.py      # Block data model: add, remove, move, snap-to-grid
├── ar_math.py            # 3D↔2D projection, ray casting, cube rendering, grid drawing
├── config.py             # Global constants (camera, block size, colors, thresholds)
├── generate_marker.py    # Utility to generate and save the ArUco marker PNG
├── marker_id0.png        # Pre-generated ArUco marker (DICT_6X6_250, ID 0)
├── hand_landmarker.task  # MediaPipe hand landmarker model file (~7.8 MB)
├── test_logic.py         # Unit tests for BlockManager and AR math
├── requirements.txt      # Python package dependencies
└── README.md             # Project overview and feature documentation
```

---

## 3. Architecture Overview

The application runs as a **single main loop** (`main.py`) with the following pipeline each frame:

```
[Camera Frame]
      │
      ▼
[ArUco Marker Detection]  ──── Found ────► Update camera pose (rvec, tvec) via solvePnP
      │ Not Found
      ▼
[Use last known or virtual fallback pose]
      │
      ▼
[Hand Landmark Detection]   (MediaPipe HandLandmarker — VIDEO mode)
      │
      ▼
[Gesture Processing]
  ├── Pinch detected → Mode dispatch (PLACE / MOVE / DELETE)
  └── Ray-cast pinch point → Ground plane (Z=0)
      │
      ▼
[AR Rendering]
  ├── Draw grid on Z=0 plane
  └── Draw solid 3D cubes (painter's algorithm face sorting)
      │
      ▼
[UI Overlay]   FPS counter, active mode label, tracking mode indicator
```

### Key Subsystems

#### 3.1 Hand Tracking (`hand_tracker.py`)

- Uses **MediaPipe Tasks API** (`HandLandmarker`) in `VIDEO` mode for real-time per-frame inference.
- Detects up to 2 hands and extracts 21 landmarks per hand.
- **Pinch detection**: measures pixel distance between thumb tip (landmark 4) and index finger tip (landmark 8). A pinch is registered when `distance < PINCH_THRESHOLD` (currently 30 px).
- Custom landmark drawing (joints + skeleton lines) drawn directly onto the OpenCV frame.

#### 3.2 Pose Estimation (`main.py`)

- **Primary mode — ArUco Marker**: Detects a printed `DICT_6X6_250` ID-0 marker in the camera view. Uses `cv2.solvePnP` against known object-space corner coordinates to compute the real camera pose (`rvec`, `tvec`) relative to the marker plane.
- **Fallback mode — Virtual Camera**: If no marker is found, a fixed synthetic camera pose is used (45° downward tilt, positioned at `[0, 200, 1000]`). This allows the application to run without a printed marker, although spatial anchoring is approximate.
- The `using_marker` flag tracks which mode is active and is displayed in the UI.

#### 3.3 AR Math (`ar_math.py`)

- **`get_camera_matrix(w, h)`**: Generates a pinhole camera intrinsic matrix assuming ~60° horizontal FOV.
- **`project_point_3d_to_2d(...)`**: Uses `cv2.projectPoints` to map any 3D world point to a 2D pixel coordinate.
- **`ray_cast_to_ground(uv, K, rvec, tvec)`**: Casts a ray from the camera through a 2D screen point and finds its intersection with the Z=0 world plane (the "ground" or surface plane). This is the core mechanism for translating a hand gesture into a 3D world position.
- **`draw_cube(img, center_3d, size, ...)`**: Renders a solid 3D cube by:
  1. Computing 8 corner vertices in world space.
  2. Projecting each to 2D.
  3. Sorting the 6 faces by camera-space depth (Painter's Algorithm).
  4. Drawing each face as a filled polygon (slightly darkened) with a bright border outline.
- **`draw_grid(img, rvec, tvec, K)`**: Renders a 10×10 reference grid on the Z=0 plane for spatial context.

#### 3.4 Block Manager (`block_manager.py`)

- Maintains a Python list of blocks: `[{'pos': (x, y, z), 'color': (B, G, R)}, ...]`.
- **Snap-to-grid**: All coordinates are rounded to the nearest `GRID_SPACING` (60 units) on placement, move, or lookup.
- **Stacking**: When placing in PLACE mode, `main.py` scans existing blocks at the same (x, y) column and stacks new blocks on top (Z increments by `BLOCK_SIZE`).
- **Collision detection**: `move_block` checks for occupied positions before confirming a move.
- **`get_block_at(x, y, z, tolerance)`**: Supports both exact grid lookup and distance-based tolerance lookup (used for imprecise gesture-based picking).

#### 3.5 Interaction Modes (`main.py`)

Three modes are cycled with the `M` key:

| Mode           | Key Color | Behaviour                                                                                                                              |
| -------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **PLACE** (0)  | Green     | Pinch → place a block at the ray-cast ground point, stacked above any existing block at that column. 0.5 s cooldown prevents spamming. |
| **MOVE** (1)   | Blue      | First pinch selects the nearest block; subsequent pinch positions drag the block to a new position.                                    |
| **DELETE** (2) | Red       | Pinch over a block → removes it. 0.5 s cooldown prevents accidental chain deletions.                                                   |

---

## 4. Current Progress (What Is Working)

| Feature                                | Status  | Notes                                                   |
| -------------------------------------- | ------- | ------------------------------------------------------- |
| Live camera feed (macOS)               | ✅ Done | Tries `CAP_AVFOUNDATION`, falls back to default index   |
| Hand landmark detection                | ✅ Done | MediaPipe Tasks API, VIDEO mode, up to 2 hands          |
| Pinch gesture detection                | ✅ Done | Thumb-index distance threshold                          |
| Ray-cast to ground plane               | ✅ Done | Correct world-space intersection                        |
| ArUco marker pose estimation           | ✅ Done | `solvePnP` from detected corners                        |
| Virtual fallback camera pose           | ✅ Done | 45° tilt, fixed position                                |
| Grid rendering on Z=0 plane            | ✅ Done | 10×10 configurable grid                                 |
| Block placement (PLACE mode)           | ✅ Done | Snap to grid + vertical stacking                        |
| Block selection & movement (MOVE mode) | ✅ Done | Index-based selection, collision check                  |
| Block deletion (DELETE mode)           | ✅ Done | Tolerance-based proximity hit test                      |
| 3D cube rendering (solid)              | ✅ Done | Painter's algo face sorting, filled polys               |
| FPS counter + mode UI overlay          | ✅ Done | Shown in corner rectangle                               |
| Tracking mode indicator                | ✅ Done | Shows `MARKER` or `VIRTUAL`                             |
| ArUco marker generator utility         | ✅ Done | `generate_marker.py` saves `marker_id0.png`             |
| Unit tests (BlockManager + AR math)    | ✅ Done | `test_logic.py` covers snap, add/remove, move, ray cast |
| Configurable parameters                | ✅ Done | All tunable values in `config.py`                       |

---

## 5. Currently In Development / Known Issues

### 5.1 Image Mirroring vs. Marker Accuracy

The camera feed is deliberately **not mirrored** (flip commented out in `main.py`). This is correct for ArUco marker detection (mirroring would break pose estimation), but makes the experience feel less natural for front-facing webcam use. A proper solution — detect on original, flip frame geometrically, correct rendered coordinates — is acknowledged in code comments but not yet implemented.

### 5.2 Virtual Mode Spatial Accuracy

When the ArUco marker is **not in view**, blocks still appear on-screen via the virtual camera pose, but their positions are **not anchored to any real-world surface**. A user moving the camera causes blocks to appear to drift. This is a fundamental limitation of not having real SLAM or depth sensing.

### 5.3 No Block Persistence

All blocks exist only **in memory**. Closing and reopening the application resets the scene. There is no save/load system.

### 5.4 Single Marker, Fixed ID

Only **ArUco marker ID 0** (DICT_6X6_250) is currently used. The system assumes a single flat surface defined by this one marker. Multiple markers (e.g., defining room corners) are not yet supported.

### 5.5 No Distortion Correction

The camera matrix is a **simplified approximation** (focal length = frame width). No lens distortion coefficients (`dist_coeffs`) are applied (`None` → treated as zero distortion). For a standard webcam this is acceptable, but it degrades accuracy for wide-angle cameras.

### 5.6 Z-Fighting / Rendering Artifacts

On some poses the Painter's Algorithm for face sorting may produce incorrect face draw order (a known limitation of the algorithm for non-convex camera angles), leading to visual flickering on cube faces.

### 5.7 Hand-to-World Mapping Stability

The pinch midpoint is mapped to world space via a single ray cast. There is no temporal filtering or smoothing on the resulting world coordinate, making precise block placement sensitive to small hand tremors.

---

## 6. What Is To Be Done Next (Planned Features)

### Phase 1 — Stability & Polish (Immediate Priority)

- [ ] **Smooth hand-position filtering**: Apply a low-pass / exponential moving-average filter on the ray-cast world coordinate to reduce jitter during placement and movement.
- [ ] **Mirror-mode fix**: Detect ArUco on original frame, render on flipped frame, and correct 3D coordinate handedness so the experience feels like a natural mirror for webcam users.
- [ ] **Calibration helper**: Add a simple one-time calibration step that estimates more accurate camera intrinsics using a checkerboard, and persists the result to a JSON file.

### Phase 2 — Core Feature Expansion

- [ ] **Scene save / load**: Serialize `block_manager.blocks` to a JSON file and restore it on startup, allowing persistent scenes.
- [ ] **Block color picker**: Allow users to cycle block colors (gesture or keypress), enabling color-coded structures.
- [ ] **Undo / redo stack**: Maintain an action history so users can reverse accidental placements or deletions (Ctrl+Z / Ctrl+Y).
- [ ] **Multi-hand support**: Use the second detected hand for mode switching (e.g., one-hand gesture vocabulary) instead of the keyboard `M` key.
- [ ] **Height control (Z-axis gesture)**: Allow vertical block placement at arbitrary heights, not just stacked columns — e.g., use vertical hand position or pinch + raise gesture to lift a block.

### Phase 3 — Spatial Understanding Improvements

- [ ] **Multiple ArUco markers (room-scale tracking)**: Place several markers around the workspace; fuse multiple pose estimates for more robust, wider-area tracking.
- [ ] **Optical flow stabilization**: Incorporate frame-to-frame optical flow to smooth camera pose when the ArUco marker is temporarily occluded.
- [ ] **Depth approximation**: Explore using stereo webcam or structured light (if hardware available) to obtain a rough depth map, enabling true surface detection without a physical marker.

### Phase 4 — UI & Export

- [ ] **On-screen virtual toolbar**: Replace keyboard mode switching with an on-screen gesture-activated toolbar (hover hand over icon to select mode).
- [ ] **Block count / structure stats overlay**: Display number of placed blocks, bounding box dimensions (in real-world units if marker size is calibrated).
- [ ] **Export to 3D format**: Export the current block layout to a simple 3D format (OBJ, STL, or voxel JSON) for use in external tools or 3D printing.
- [ ] **Screenshot / recording**: Allow capturing the current AR view or recording a short video clip of the session.

### Phase 5 — Platform & Performance

- [ ] **Android / iOS port**: Migrate the Python prototype to a mobile platform (e.g., Unity + AR Foundation, or Flutter + ARKit/ARCore) for true mobile AR.
- [ ] **GPU acceleration**: Profile frame pipeline and identify opportunities to move image processing to GPU (e.g., via OpenCV CUDA or a Metal/Vulkan backend on macOS).

---

## 7. Testing Status

| Test                    | File            | Status     |
| ----------------------- | --------------- | ---------- |
| `test_snap_to_grid`     | `test_logic.py` | ✅ Passing |
| `test_add_remove_block` | `test_logic.py` | ✅ Passing |
| `test_move_block`       | `test_logic.py` | ✅ Passing |
| `test_ray_cast`         | `test_logic.py` | ✅ Passing |

**Run tests with:**

```bash
python -m unittest test_logic.py -v
```

---

## 8. How to Run

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Generate marker image
python generate_marker.py
# Print or display marker_id0.png in front of the camera

# 4. Launch application
python main.py
```

**Keyboard Controls:**
| Key | Action |
|---|---|
| `M` | Cycle through modes: PLACE → MOVE → DELETE |
| `Q` | Quit the application |

**Gesture Controls:**
| Gesture | Action |
|---|---|
| Thumb + Index pinch | Activate current mode at hand position |
| Release pinch | Deselect (in MOVE mode) |

---

## 9. Summary

The project has a **solid, functional core**: hand tracking, marker-based pose estimation, ray casting, 3D rendering, and the full block manipulation lifecycle are all implemented and working. The application runs as a real-time loop at a reasonable frame rate on a standard Mac with a built-in webcam.

The immediate development focus should be on **stability improvements** (jitter filtering, mirror fix) and **scene persistence** (save/load), as these are the most impactful changes for usability. Beyond that, expanding to multi-marker tracking and a mobile platform would significantly elevate the project's practical and academic value.

---

_Report auto-generated from codebase analysis — 23 February 2026_
