import time

import cv2
import numpy as np

import ar_math
import config
from block_manager import BlockManager
from export import export_to_obj
from hand_tracker import HandTracker
from scene_manager import load_scene, save_scene


WINDOW_NAME = "Gesture-Based AR Builder - Phase 10"

MODE_PLACE = 0
MODE_MOVE = 1
MODE_DELETE = 2


def create_virtual_pose() -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(45.0)
    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float32,
    )
    rvec, _ = cv2.Rodrigues(rotation_x)
    tvec = np.array([[0.0], [200.0], [1000.0]], dtype=np.float32)
    return rvec.astype(np.float32), tvec


def open_camera() -> cv2.VideoCapture | None:
    attempts = [
        (0, cv2.CAP_AVFOUNDATION),
        (1, cv2.CAP_AVFOUNDATION),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_ANY),
    ]

    for index, backend in attempts:
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            continue

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        ok, _ = capture.read()
        if ok:
            print(f"Opened camera index {index} with backend {backend}")
            return capture

        capture.release()

    return None


def create_marker_object_points(marker_size: float) -> np.ndarray:
    half = marker_size / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def create_optimised_detector() -> cv2.aruco.ArucoDetector:
    """Create an ArUco detector with tuned parameters for stability."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()

    # ── Adaptive thresholding tuning ──────────────────────────────
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7

    # ── Corner refinement (built-in ArUco sub-pixel) ──────────────
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01

    # ── Relax detection to accept more markers ────────────────────
    params.minMarkerPerimeterRate = 0.02     # detect smaller markers
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.05
    params.minCornerDistanceRate = 0.05
    params.minDistanceToBorder = 3

    # ── Error correction ──────────────────────────────────────────
    params.errorCorrectionRate = 0.6         # allow some bit errors

    return cv2.aruco.ArucoDetector(aruco_dict, params)


def detect_marker_pose(
    frame: np.ndarray,
    gray: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None:
        return None, None, 0

    # Additional sub-pixel refinement on the greyscale image
    corners = ar_math.refine_marker_corners(gray, corners)

    base_object_points = create_marker_object_points(config.MARKER_SIZE)
    pose_list: list[tuple[np.ndarray, np.ndarray, float]] = []

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        marker_id_int = int(marker_id)
        if marker_id_int not in config.MARKER_IDS:
            continue

        marker_offset = np.array(config.MARKER_POSITIONS[marker_id_int], dtype=np.float32).reshape(1, 3)
        object_points = base_object_points + marker_offset

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners.reshape(4, 2).astype(np.float32),
            camera_matrix,
            np.zeros((5, 1), dtype=np.float32),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if success:
            # solvePnP can return multiple solutions with IPPE_SQUARE;
            # refine with iterative LM for sub-pixel accuracy.
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points,
                marker_corners.reshape(4, 2).astype(np.float32),
                camera_matrix,
                np.zeros((5, 1), dtype=np.float32),
                rvec,
                tvec,
            )
            pose_list.append((rvec.astype(np.float32), tvec.astype(np.float32), 1.0))

    if not pose_list:
        return None, None, 0

    fused_rvec, fused_tvec = ar_math.fuse_poses(pose_list)
    return fused_rvec, fused_tvec, len(pose_list)


def draw_overlay(
    frame: np.ndarray,
    fps: float,
    tracking_mode: str,
    mode_name: str,
    mode_color: tuple[int, int, int],
    current_color: tuple[int, int, int],
    shape_name: str = "CUBE",
) -> None:
    cv2.rectangle(frame, (10, 10), (480, 155), (40, 40, 40), cv2.FILLED)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.WHITE, 2)
    cv2.putText(
        frame,
        f"Mode: {mode_name}  Shape: {shape_name} (T)",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        mode_color,
        2,
    )
    tracking_color = config.GREEN if tracking_mode.startswith("MARKER") else (
        (255, 255, 0) if tracking_mode.startswith("FLOW") else config.YELLOW
    )
    cv2.putText(
        frame,
        f"Tracking: {tracking_mode}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        tracking_color,
        2,
    )
    # Gesture hint row (Phase 9)
    cv2.putText(
        frame,
        "Palm=mode  Fist=undo  Peace=del",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )
    cv2.rectangle(frame, (350, 20), (400, 70), current_color, cv2.FILLED)
    cv2.rectangle(frame, (350, 20), (400, 70), config.WHITE, 2)


def main() -> None:
    capture = open_camera()
    if capture is None:
        raise RuntimeError("Unable to open a webcam on macOS using AVFoundation or fallback backends.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.FRAME_WIDTH
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.FRAME_HEIGHT
    camera_matrix = ar_math.get_camera_matrix(width, height)
    virtual_rvec, virtual_tvec = create_virtual_pose()

    # Use the optimised detector with tuned parameters
    detector = create_optimised_detector()

    # Pose stabiliser for temporal smoothing
    stabilizer = ar_math.PoseStabilizer()
    was_tracking_markers = False

    # ── Optical-flow state (Phase 7) ──────────────────────────────
    prev_gray: np.ndarray | None = None
    flow_frame_count = 0
    last_marker_rvec: np.ndarray | None = None
    last_marker_tvec: np.ndarray | None = None

    tracker = HandTracker()
    blocks = BlockManager()
    blocks.set_blocks(load_scene())
    print("Controls: M mode, C color, S save, E export, Q quit, Undo: Ctrl+Z/Z/U, Redo: Ctrl+Y/Y/R")

    mode = MODE_PLACE
    mode_names = ["PLACE", "MOVE", "DELETE"]
    mode_colors = [config.GREEN, config.BLUE, config.RED]
    place_colors = [config.BLUE, config.RED, config.GREEN, config.YELLOW]
    color_index = 0
    current_shape = 0  # Phase 10: shape cycling (0=CUBE, 1=SLAB, 2=WALL)

    selected_index = -1
    was_pinching = False
    last_action_time = 0.0
    cooldown_s = 0.5

    # ── Gesture state machine (Phase 9) ──────────────────────────────
    palm_hold_frames = 0
    fist_triggered = False
    peace_triggered = False

    def top_block_index_at(sx: float, sy: float) -> int:
        best_i = -1
        best_z = -float("inf")
        for i, b in enumerate(blocks.get_blocks()):
            bx, by, bz = b["pos"]
            if bx == sx and by == sy and float(bz) > best_z:
                best_z = float(bz)
                best_i = i
        return best_i

    previous_time = time.time()

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Camera read failed; exiting.")
            break

        source_frame = frame

        # Pre-compute greyscale once — used for corner refinement & flow
        gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)

        marker_rvec, marker_tvec, marker_count = detect_marker_pose(
            source_frame, gray, detector, camera_matrix
        )

        if marker_count > 0:
            # ── Markers detected: use fused marker pose ───────────
            if not was_tracking_markers:
                stabilizer.reset()
                was_tracking_markers = True

            rvec, tvec = stabilizer.update(marker_rvec, marker_tvec)
            tracking_mode = f"MARKER({marker_count})"
            cv2.drawFrameAxes(
                source_frame,
                camera_matrix,
                np.zeros((5, 1), dtype=np.float32),
                rvec,
                tvec,
                60,
            )

            # Save state for optical-flow fallback
            last_marker_rvec = rvec.copy()
            last_marker_tvec = tvec.copy()
            flow_frame_count = 0
            prev_gray = gray.copy()

        elif (
            flow_frame_count < config.FLOW_MAX_FRAMES
            and prev_gray is not None
            and last_marker_rvec is not None
            and last_marker_tvec is not None
        ):
            # ── Markers lost: attempt optical-flow interpolation ──
            flow_rvec, flow_tvec, flow_ok = ar_math.interpolate_pose_with_flow(
                last_marker_rvec, last_marker_tvec, prev_gray, gray, camera_matrix
            )
            if flow_ok:
                rvec, tvec = flow_rvec, flow_tvec
                flow_frame_count += 1
                tracking_mode = f"FLOW({flow_frame_count})"
                # Update last known pose so next flow step chains correctly
                last_marker_rvec = rvec.copy()
                last_marker_tvec = tvec.copy()
                prev_gray = gray.copy()
            else:
                # Flow failed — fall back to virtual
                rvec = virtual_rvec
                tvec = virtual_tvec
                tracking_mode = "VIRTUAL"
                was_tracking_markers = False
        else:
            # ── Fully lost or flow budget exhausted ───────────────
            rvec = virtual_rvec
            tvec = virtual_tvec
            tracking_mode = "VIRTUAL"
            was_tracking_markers = False

        tracker.find_hands(source_frame, draw=True)
        lm_list = tracker.find_position(source_frame, hand_no=0)
        pinching = False
        pinch_center = (0, 0)
        ground = None
        hand_ground = None  # ray-cast from hand even when not pinching
        if lm_list:
            pinching, pinch_center = tracker.is_pinching(lm_list)
            # Always ray-cast from index fingertip (landmark 8) for ghost cursor
            idx_tip = (lm_list[8][1], lm_list[8][2])
            hand_ground = ar_math.ray_cast_to_ground(idx_tip, camera_matrix, rvec, tvec)

        # ── Gesture recognition (Phase 9) ─────────────────────────
        if lm_list:
            # A. Open palm → cycle mode (must hold 10+ frames)
            if tracker.is_open_palm(lm_list):
                palm_hold_frames += 1
                if palm_hold_frames == 10:
                    mode = (mode + 1) % 3
                    selected_index = -1
                    was_pinching = False
                    print(f"Gesture: PALM → mode = {mode_names[mode]}")
            else:
                palm_hold_frames = 0

            # B. Fist → undo (on transition)
            if tracker.is_fist(lm_list):
                if not fist_triggered:
                    fist_triggered = True
                    if blocks.undo():
                        selected_index = -1
                        print("Gesture: FIST → Undo")
            else:
                fist_triggered = False

            # C. Peace sign → toggle DELETE mode (on transition)
            if tracker.is_peace_sign(lm_list):
                if not peace_triggered:
                    peace_triggered = True
                    mode = MODE_DELETE
                    selected_index = -1
                    print("Gesture: PEACE → DELETE mode")
            else:
                peace_triggered = False
        else:
            palm_hold_frames = 0
            fist_triggered = False
            peace_triggered = False

        if pinching:
            cv2.circle(source_frame, pinch_center, 14, mode_colors[mode], cv2.FILLED)
            ground = ar_math.ray_cast_to_ground(pinch_center, camera_matrix, rvec, tvec)
            if ground is not None:
                gx, gy, gz = ground
                ground_uv = ar_math.project_point_3d_to_2d((gx, gy, gz), rvec, tvec, camera_matrix)
                cv2.circle(source_frame, ground_uv, 6, config.YELLOW, cv2.FILLED)

        pinch_started = pinching and not was_pinching
        pinch_ended = (not pinching) and was_pinching

        now = time.time()

        if mode == MODE_PLACE:
            if pinch_started and ground is not None and (now - last_action_time) >= cooldown_s:
                gx, gy, _ = ground
                sx, sy, _ = blocks.snap_to_grid(gx, gy, 0.0)

                # Get current shape size multipliers
                cur_shape_def = config.BLOCK_SHAPES.get(current_shape, config.BLOCK_SHAPES[0])
                cur_sx, cur_sy, cur_sz = cur_shape_def["size"]
                bs = float(config.BLOCK_SIZE)

                # New block XY footprint
                new_hx = bs * cur_sx / 2.0
                new_hy = bs * cur_sy / 2.0

                # Face snapping: find the highest top face of any block
                # whose XY footprint overlaps with the new block's footprint.
                max_top = 0.0  # ground level
                for b in blocks.get_blocks():
                    bx, by, bz = b["pos"]
                    b_shape = config.BLOCK_SHAPES.get(b.get("shape", 0), config.BLOCK_SHAPES[0])
                    b_ssx, b_ssy, b_ssz = b_shape["size"]
                    b_hx = bs * b_ssx / 2.0
                    b_hy = bs * b_ssy / 2.0
                    # Check XY overlap
                    if (sx - new_hx < bx + b_hx - 0.1 and sx + new_hx > bx - b_hx + 0.1 and
                            sy - new_hy < by + b_hy - 0.1 and sy + new_hy > by - b_hy + 0.1):
                        b_top = bz + bs * b_ssz / 2.0
                        max_top = max(max_top, b_top)

                # New block center z = top of stack + half of new block height
                new_z = max_top + bs * cur_sz / 2.0
                blocks.add_block(sx, sy, new_z, color=place_colors[color_index], shape=current_shape)
                last_action_time = now

        elif mode == MODE_MOVE:
            if pinch_started and ground is not None and selected_index == -1:
                gx, gy, _ = ground
                sx, sy, _ = blocks.snap_to_grid(gx, gy, 0.0)
                selected_index = top_block_index_at(sx, sy)

            if pinching and ground is not None and selected_index != -1:
                gx, gy, _ = ground
                old_z = float(blocks.get_blocks()[selected_index]["pos"][2])
                blocks.move_block(selected_index, gx, gy, old_z)

            if pinch_ended:
                selected_index = -1

        elif mode == MODE_DELETE:
            if pinch_started and ground is not None and (now - last_action_time) >= cooldown_s:
                gx, gy, _ = ground
                sx, sy, _ = blocks.snap_to_grid(gx, gy, 0.0)
                idx = top_block_index_at(sx, sy)
                if idx != -1:
                    bx, by, bz = blocks.get_blocks()[idx]["pos"]
                    blocks.remove_block(bx, by, bz)
                    if selected_index == idx:
                        selected_index = -1
                    elif selected_index > idx:
                        selected_index -= 1
                last_action_time = now

        was_pinching = pinching

        ar_math.draw_grid(source_frame, rvec, tvec, camera_matrix)

        for i, b in enumerate(blocks.get_blocks()):
            pos = b["pos"]
            col = b["color"]
            if i == selected_index:
                col = config.YELLOW
            b_shape = config.BLOCK_SHAPES.get(b.get("shape", 0), config.BLOCK_SHAPES[0])
            b_sx, b_sy, b_sz = b_shape["size"]
            ar_math.draw_cube(
                source_frame,
                center_3d=pos,
                size=float(config.BLOCK_SIZE),
                rvec=rvec,
                tvec=tvec,
                K=camera_matrix,
                color=col,
                sx=b_sx, sy=b_sy, sz=b_sz,
            )

        # ── Ghost cursor wireframe (Phase 8+10) ───────────────────
        # In PLACE mode, when hand is visible but NOT pinching,
        # show a wireframe preview at the predicted landing position.
        if mode == MODE_PLACE and lm_list and not pinching and hand_ground is not None:
            hgx, hgy, _ = hand_ground
            ghost_sx, ghost_sy, _ = blocks.snap_to_grid(hgx, hgy, 0.0)

            g_shape_def = config.BLOCK_SHAPES.get(current_shape, config.BLOCK_SHAPES[0])
            g_ssx, g_ssy, g_ssz = g_shape_def["size"]
            bs = float(config.BLOCK_SIZE)
            g_hx = bs * g_ssx / 2.0
            g_hy = bs * g_ssy / 2.0

            # Compute ghost Z using footprint overlap (same logic as placement)
            ghost_max_top = 0.0
            for b in blocks.get_blocks():
                bx, by, bz = b["pos"]
                b_shape = config.BLOCK_SHAPES.get(b.get("shape", 0), config.BLOCK_SHAPES[0])
                b_ssx, b_ssy, b_ssz = b_shape["size"]
                b_hx = bs * b_ssx / 2.0
                b_hy = bs * b_ssy / 2.0
                if (ghost_sx - g_hx < bx + b_hx - 0.1 and ghost_sx + g_hx > bx - b_hx + 0.1 and
                        ghost_sy - g_hy < by + b_hy - 0.1 and ghost_sy + g_hy > by - b_hy + 0.1):
                    b_top = bz + bs * b_ssz / 2.0
                    ghost_max_top = max(ghost_max_top, b_top)
            ghost_z = ghost_max_top + bs * g_ssz / 2.0

            ar_math.draw_wireframe_cube(
                source_frame,
                center_3d=(ghost_sx, ghost_sy, ghost_z),
                size=bs,
                rvec=rvec,
                tvec=tvec,
                K=camera_matrix,
                color=place_colors[color_index],
                alpha=0.4,
                sx=g_ssx, sy=g_ssy, sz=g_ssz,
            )

        # Mirror ONLY the camera feed for display.
        display_frame = cv2.flip(source_frame, 1)

        current_time = time.time()
        fps = 1.0 / max(current_time - previous_time, 1e-6)
        previous_time = current_time
        # Draw overlay after mirroring so it stays top-left.
        shape_name = config.BLOCK_SHAPES.get(current_shape, config.BLOCK_SHAPES[0])["name"]
        draw_overlay(
            display_frame,
            fps,
            tracking_mode,
            mode_names[mode],
            mode_colors[mode],
            place_colors[color_index],
            shape_name=shape_name,
        )

        cv2.imshow(WINDOW_NAME, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("m"), ord("M")):
            mode = (mode + 1) % 3
            selected_index = -1
            was_pinching = False
        if key in (ord("c"), ord("C")):
            color_index = (color_index + 1) % len(place_colors)
        if key in (ord("t"), ord("T")):
            current_shape = (current_shape + 1) % len(config.BLOCK_SHAPES)
            print(f"Shape: {config.BLOCK_SHAPES[current_shape]['name']}")
        if key in (ord("s"), ord("S")):
            save_scene(blocks.get_blocks())
            print("Scene saved to scene.json")
        if key in (ord("e"), ord("E")):
            path = export_to_obj(blocks.get_blocks(), filepath="export.obj", block_size=float(config.BLOCK_SIZE))
            print(f"Exported OBJ to {path}")
        if key in (26, ord("z"), ord("Z"), ord("u"), ord("U")):  # Ctrl+Z plus fallbacks
            if blocks.undo():
                selected_index = -1
                print("Undo")
        if key in (25, ord("y"), ord("Y"), ord("r"), ord("R")):  # Ctrl+Y plus fallbacks
            if blocks.redo():
                selected_index = -1
                print("Redo")

    tracker.close()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
