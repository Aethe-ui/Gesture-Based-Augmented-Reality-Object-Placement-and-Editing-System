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
    sx: float = 1.0,
    sy: float = 1.0,
    sz: float = 1.0,
) -> None:
    cx, cy, cz = float(center_3d[0]), float(center_3d[1]), float(center_3d[2])
    hs = float(size) / 2.0
    hx, hy, hz = hs * sx, hs * sy, hs * sz

    corners = np.array(
        [
            [cx - hx, cy - hy, cz - hz],  # 0
            [cx + hx, cy - hy, cz - hz],  # 1
            [cx + hx, cy + hy, cz - hz],  # 2
            [cx - hx, cy + hy, cz - hz],  # 3
            [cx - hx, cy - hy, cz + hz],  # 4
            [cx + hx, cy - hy, cz + hz],  # 5
            [cx + hx, cy + hy, cz + hz],  # 6
            [cx - hx, cy + hy, cz + hz],  # 7
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


def draw_wireframe_cube(
    img: np.ndarray,
    center_3d: tuple[float, float, float],
    size: float,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.4,
    sx: float = 1.0,
    sy: float = 1.0,
    sz: float = 1.0,
) -> None:
    """Draw a semi-transparent wireframe cube (12 edges only).

    The wireframe is rendered onto a copy of *img* and alpha-blended
    back, producing a ghost-like preview effect.
    """
    cx, cy, cz = float(center_3d[0]), float(center_3d[1]), float(center_3d[2])
    hs = float(size) / 2.0
    hx, hy, hz = hs * sx, hs * sy, hs * sz

    corners = np.array(
        [
            [cx - hx, cy - hy, cz - hz],  # 0
            [cx + hx, cy - hy, cz - hz],  # 1
            [cx + hx, cy + hy, cz - hz],  # 2
            [cx - hx, cy + hy, cz - hz],  # 3
            [cx - hx, cy - hy, cz + hz],  # 4
            [cx + hx, cy - hy, cz + hz],  # 5
            [cx + hx, cy + hy, cz + hz],  # 6
            [cx - hx, cy + hy, cz + hz],  # 7
        ],
        dtype=np.float32,
    )

    corners_2d = _project_points(corners, rvec, tvec, K).astype(np.int32)

    # The 12 edges of a cube, defined by pairs of corner indices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]

    overlay = img.copy()
    for a, b in edges:
        pt_a = tuple(corners_2d[a])
        pt_b = tuple(corners_2d[b])
        cv2.line(overlay, pt_a, pt_b, color, 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, dst=img)


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


def fuse_poses(
    pose_list: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    if not pose_list:
        raise ValueError("pose_list must contain at least one pose")

    total_weight = sum(max(float(weight), 0.0) for _, _, weight in pose_list)
    if total_weight <= 0.0:
        total_weight = float(len(pose_list))
        weighted = [(rvec, tvec, 1.0) for rvec, tvec, _ in pose_list]
    else:
        weighted = [(rvec, tvec, max(float(weight), 0.0)) for rvec, tvec, weight in pose_list]

    tvec_sum = np.zeros((3, 1), dtype=np.float32)
    rvec_sum = np.zeros((3, 1), dtype=np.float32)

    for rvec, tvec, weight in weighted:
        tvec_sum += tvec.astype(np.float32).reshape(3, 1) * weight
        rvec_sum += rvec.astype(np.float32).reshape(3, 1) * weight

    fused_tvec = (tvec_sum / total_weight).astype(np.float32)
    fused_rvec = (rvec_sum / total_weight).astype(np.float32)
    return fused_rvec, fused_tvec


# ═══════════════════════════════════════════════════════════════════════
# Pose Stabiliser — EMA smoothing + outlier rejection
# ═══════════════════════════════════════════════════════════════════════

class PoseStabilizer:
    """Temporal filter for marker poses.

    Applies exponential-moving-average (EMA) smoothing on both rvec and tvec
    and rejects sudden large jumps (outliers) that are typically caused by
    noisy / mis-detected corners.
    """

    def __init__(
        self,
        alpha: float = config.POSE_SMOOTHING_ALPHA,
        outlier_threshold: float = config.POSE_OUTLIER_THRESHOLD,
        outlier_patience: int = config.OUTLIER_PATIENCE,
    ) -> None:
        self._alpha = alpha
        self._outlier_threshold = outlier_threshold
        self._outlier_patience = outlier_patience

        self._prev_rvec: np.ndarray | None = None
        self._prev_tvec: np.ndarray | None = None
        self._consecutive_outliers = 0

    def reset(self) -> None:
        """Clear history (e.g. when switching from VIRTUAL back to MARKER)."""
        self._prev_rvec = None
        self._prev_tvec = None
        self._consecutive_outliers = 0

    def update(
        self, rvec: np.ndarray, tvec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Feed a raw rvec/tvec and return the stabilised version."""
        rvec = rvec.astype(np.float32).reshape(3, 1)
        tvec = tvec.astype(np.float32).reshape(3, 1)

        if self._prev_tvec is None:
            # First frame — accept as-is.
            self._prev_rvec = rvec.copy()
            self._prev_tvec = tvec.copy()
            self._consecutive_outliers = 0
            return rvec.copy(), tvec.copy()

        # ── Outlier rejection ──────────────────────────────────────
        jump = float(np.linalg.norm(tvec - self._prev_tvec))
        if jump > self._outlier_threshold:
            self._consecutive_outliers += 1
            if self._consecutive_outliers < self._outlier_patience:
                # Reject this frame, return the previous smoothed pose.
                return self._prev_rvec.copy(), self._prev_tvec.copy()
            else:
                # Too many consecutive outliers → force-accept (marker
                # probably really did move). Reset the filter.
                self._prev_rvec = rvec.copy()
                self._prev_tvec = tvec.copy()
                self._consecutive_outliers = 0
                return rvec.copy(), tvec.copy()

        self._consecutive_outliers = 0

        # ── EMA smoothing ──────────────────────────────────────────
        a = self._alpha
        smoothed_tvec = (a * tvec + (1.0 - a) * self._prev_tvec).astype(np.float32)

        # For rvec, ensure we handle the sign-flip ambiguity of Rodrigues:
        # If the angle between the old and new rvec is > π (dot < 0), flip
        # the new one before blending.  This prevents the EMA from averaging
        # across the ±π discontinuity.
        if float(np.dot(rvec.flatten(), self._prev_rvec.flatten())) < 0:
            rvec = -rvec
        smoothed_rvec = (a * rvec + (1.0 - a) * self._prev_rvec).astype(np.float32)

        self._prev_rvec = smoothed_rvec.copy()
        self._prev_tvec = smoothed_tvec.copy()
        return smoothed_rvec.copy(), smoothed_tvec.copy()


def refine_marker_corners(
    gray: np.ndarray,
    corners: tuple,
) -> tuple:
    """Sub-pixel refine ArUco corner detections for smoother solvePnP.

    Uses cv2.cornerSubPix on each detected marker's 4 corners, which
    significantly reduces the noise in the corner locations and leads
    to much more stable pose estimates.
    """
    win = config.CORNER_REFINE_WIN
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = []
    for marker_corners in corners:
        mc = marker_corners.copy().astype(np.float32)
        # cornerSubPix expects (N,1,2) shape
        if mc.ndim == 2:
            mc = mc.reshape(-1, 1, 2)
        elif mc.ndim == 3 and mc.shape[1] != 1:
            mc = mc.reshape(-1, 1, 2)
        mc = cv2.cornerSubPix(gray, mc, (win, win), (-1, -1), criteria)
        refined.append(mc.reshape(1, 4, 2))
    return tuple(refined)


# ═══════════════════════════════════════════════════════════════════════
# Optical-flow pose interpolation (Phase 7)
# ═══════════════════════════════════════════════════════════════════════

def interpolate_pose_with_flow(
    prev_rvec: np.ndarray,
    prev_tvec: np.ndarray,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extrapolate camera pose from optical flow when markers are lost.

    Detects good features in *prev_gray*, tracks them into *curr_gray*
    using Lucas-Kanade pyramidal optical flow, computes a homography
    from the surviving point correspondences, decomposes it into a
    rotation + translation delta, and applies the delta to the
    previous pose.

    Returns
    -------
    (rvec, tvec, success)
        success is False when fewer than MIN_FLOW_POINTS survive
        the tracking/homography step.
    """
    prev_rvec = prev_rvec.astype(np.float32).reshape(3, 1)
    prev_tvec = prev_tvec.astype(np.float32).reshape(3, 1)

    # ── Detect features in the previous frame ─────────────────────
    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7,
    )
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    if prev_pts is None or len(prev_pts) < config.MIN_FLOW_POINTS:
        return prev_rvec, prev_tvec, False

    # ── Track into current frame ──────────────────────────────────
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_params
    )
    if curr_pts is None or status is None:
        return prev_rvec, prev_tvec, False

    # Keep only successfully tracked points
    mask = status.flatten() == 1
    good_prev = prev_pts[mask].reshape(-1, 2)
    good_curr = curr_pts[mask].reshape(-1, 2)

    if len(good_prev) < config.MIN_FLOW_POINTS:
        return prev_rvec, prev_tvec, False

    # ── Compute homography ────────────────────────────────────────
    H, inlier_mask = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)
    if H is None:
        return prev_rvec, prev_tvec, False

    inlier_count = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if inlier_count < config.MIN_FLOW_POINTS:
        return prev_rvec, prev_tvec, False

    # ── Decompose homography into rotation + translation ──────────
    # decomposeHomographyMat returns up to 4 candidate solutions.
    num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

    if num_solutions == 0:
        return prev_rvec, prev_tvec, False

    # Pick the solution whose normal has a positive z component
    # (camera looking forward).  Fall back to the first solution.
    best_idx = 0
    for i in range(num_solutions):
        if normals[i][2, 0] > 0:
            best_idx = i
            break

    delta_R = rotations[best_idx].astype(np.float32)
    delta_t = translations[best_idx].astype(np.float32).reshape(3, 1)

    # ── Apply delta to previous pose ──────────────────────────────
    R_prev, _ = cv2.Rodrigues(prev_rvec)
    R_new = (delta_R @ R_prev).astype(np.float32)
    new_rvec, _ = cv2.Rodrigues(R_new)
    new_rvec = new_rvec.astype(np.float32)

    # Scale the translation delta conservatively — the decomposition's
    # translation is up-to-scale, so we use a small fraction of the
    # previous tvec norm to keep it proportional.
    scale = float(np.linalg.norm(prev_tvec)) * 0.01
    new_tvec = (prev_tvec + delta_t * scale).astype(np.float32)

    return new_rvec, new_tvec, True
