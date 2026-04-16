import cv2
import numpy as np

import config


def get_camera_matrix(w: int, h: int) -> np.ndarray:
    focal_length = float(w)
    cx = w / 2.0
    cy = h / 2.0
    return np.array(
        [
            [focal_length, 0.0, cx],
            [0.0, focal_length, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _project_points(points_3d: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray) -> np.ndarray:
    points_2d, _ = cv2.projectPoints(
        points_3d.astype(np.float32),
        rvec.astype(np.float32),
        tvec.astype(np.float32),
        K.astype(np.float32),
        np.zeros((5, 1), dtype=np.float32),
    )
    return points_2d.reshape(-1, 2)


def draw_grid(img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray) -> None:
    half_extent = (config.GRID_SIZE * config.GRID_SPACING) / 2.0
    line_count = config.GRID_SIZE + 1

    for i in range(line_count):
        offset = -half_extent + (i * config.GRID_SPACING)

        x_line = np.array(
            [
                [-half_extent, offset, 0.0],
                [half_extent, offset, 0.0],
            ],
            dtype=np.float32,
        )
        y_line = np.array(
            [
                [offset, -half_extent, 0.0],
                [offset, half_extent, 0.0],
            ],
            dtype=np.float32,
        )

        x_line_2d = _project_points(x_line, rvec, tvec, K).astype(int)
        y_line_2d = _project_points(y_line, rvec, tvec, K).astype(int)

        cv2.line(img, tuple(x_line_2d[0]), tuple(x_line_2d[1]), config.GRID, 1, cv2.LINE_AA)
        cv2.line(img, tuple(y_line_2d[0]), tuple(y_line_2d[1]), config.GRID, 1, cv2.LINE_AA)
