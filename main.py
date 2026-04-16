import time

import cv2
import numpy as np

import ar_math
import config
from hand_tracker import HandTracker


WINDOW_NAME = "Gesture-Based AR Builder - Phase 2"


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


def detect_marker_pose(
    frame: np.ndarray,
    detector: cv2.aruco.ArucoDetector,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None:
        return None, None, False

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    object_points = create_marker_object_points(config.MARKER_SIZE)

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        if int(marker_id) != 0:
            continue

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners[0].astype(np.float32),
            camera_matrix,
            np.zeros((5, 1), dtype=np.float32),
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if success:
            return rvec.astype(np.float32), tvec.astype(np.float32), True

    return None, None, False


def draw_overlay(frame: np.ndarray, fps: float, tracking_mode: str, pinching: bool) -> None:
    cv2.rectangle(frame, (10, 10), (310, 125), (40, 40, 40), cv2.FILLED)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.WHITE, 2)
    cv2.putText(
        frame,
        f"Tracking: {tracking_mode}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        config.GREEN if tracking_mode == "MARKER" else config.YELLOW,
        2,
    )
    if pinching:
        cv2.putText(frame, "PINCHING", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.YELLOW, 2)


def main() -> None:
    capture = open_camera()
    if capture is None:
        raise RuntimeError("Unable to open a webcam on macOS using AVFoundation or fallback backends.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or config.FRAME_WIDTH
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config.FRAME_HEIGHT
    camera_matrix = ar_math.get_camera_matrix(width, height)
    virtual_rvec, virtual_tvec = create_virtual_pose()

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    tracker = HandTracker()

    previous_time = time.time()

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Camera read failed; exiting.")
            break

        marker_rvec, marker_tvec, marker_found = detect_marker_pose(frame, detector, camera_matrix)
        if marker_found:
            rvec = marker_rvec
            tvec = marker_tvec
            tracking_mode = "MARKER"
            cv2.drawFrameAxes(frame, camera_matrix, np.zeros((5, 1), dtype=np.float32), rvec, tvec, 60)
        else:
            rvec = virtual_rvec
            tvec = virtual_tvec
            tracking_mode = "VIRTUAL"

        tracker.find_hands(frame, draw=True)
        lm_list = tracker.find_position(frame, hand_no=0)
        pinching = False
        if lm_list:
            pinching, pinch_center = tracker.is_pinching(lm_list)
            if pinching:
                cv2.circle(frame, pinch_center, 14, config.YELLOW, cv2.FILLED)

        ar_math.draw_grid(frame, rvec, tvec, camera_matrix)

        current_time = time.time()
        fps = 1.0 / max(current_time - previous_time, 1e-6)
        previous_time = current_time
        draw_overlay(frame, fps, tracking_mode, pinching)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    tracker.close()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
