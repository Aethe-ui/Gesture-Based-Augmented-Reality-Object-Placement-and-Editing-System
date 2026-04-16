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


def project_point_3d_to_2d(
    point_3d: tuple[float, float, float] | np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> tuple[int, int]:
    p = np.array([point_3d], dtype=np.float32).reshape(1, 3)
    uv = _project_points(p, rvec, tvec, K)[0]
    return int(round(float(uv[0]))), int(round(float(uv[1])))


def ray_cast_to_ground(
    uv: tuple[float, float] | tuple[int, int],
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> tuple[float, float, float] | None:
    u, v = float(uv[0]), float(uv[1])
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float32).reshape(3, 1)

    R, _ = cv2.Rodrigues(rvec.astype(np.float32))
    R_inv = R.T
    t = tvec.astype(np.float32).reshape(3, 1)

    origin_world = (-R_inv @ t).reshape(3)
    dir_world = (R_inv @ ray_cam).reshape(3)

    if abs(float(dir_world[2])) < 1e-6:
        return None

    lam = -float(origin_world[2]) / float(dir_world[2])
    if lam <= 0.0:
        return None

    p = origin_world + (dir_world * lam)
    return float(p[0]), float(p[1]), 0.0


def _darken(color: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    b, g, r = color
    return max(0, b - amount), max(0, g - amount), max(0, r - amount)


def draw_cube(
    img: np.ndarray,
    center_3d: tuple[float, float, float],
    size: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    cx, cy, cz = float(center_3d[0]), float(center_3d[1]), float(center_3d[2])
    hs = float(size) / 2.0

    corners = np.array(
        [
            [cx - hs, cy - hs, cz - hs],  # 0
            [cx + hs, cy - hs, cz - hs],  # 1
            [cx + hs, cy + hs, cz - hs],  # 2
            [cx - hs, cy + hs, cz - hs],  # 3
            [cx - hs, cy - hs, cz + hs],  # 4
            [cx + hs, cy - hs, cz + hs],  # 5
            [cx + hs, cy + hs, cz + hs],  # 6
            [cx - hs, cy + hs, cz + hs],  # 7
        ],
        dtype=np.float32,
    )

    corners_2d = _project_points(corners, rvec, tvec, K).astype(np.int32)

    R, _ = cv2.Rodrigues(rvec.astype(np.float32))
    t = tvec.astype(np.float32).reshape(3, 1)
    corners_cam = (R @ corners.T + t).T  # (8,3)

    faces: list[tuple[list[int], int]] = [
        ([0, 1, 2, 3], 70),   # -Z
        ([4, 5, 6, 7], 20),   # +Z
        ([0, 1, 5, 4], 90),   # -Y
        ([2, 3, 7, 6], 40),   # +Y
        ([0, 3, 7, 4], 110),  # -X
        ([1, 2, 6, 5], 55),   # +X
    ]

    def face_depth(face_indices: list[int]) -> float:
        return float(np.mean(corners_cam[face_indices, 2]))

    faces_sorted = sorted(faces, key=lambda f: face_depth(f[0]), reverse=True)

    for indices, shade in faces_sorted:
        pts = corners_2d[np.array(indices, dtype=np.int32)]
        cv2.fillPoly(img, [pts], _darken(color, shade), lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color, 2, lineType=cv2.LINE_AA)


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
