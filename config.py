CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

BLOCK_SIZE = 50
GRID_SIZE = 10
GRID_SPACING = 60

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
GRID = (200, 200, 200)

PINCH_THRESHOLD = 30
SMOOTHING_FACTOR = 0.5

MODEL_PATH = "hand_landmarker.task"

MARKER_SIZE = 100
MARKER_IDS = [0, 1, 2, 3]
MARKER_POSITIONS = {
    0: (0.0, 0.0, 0.0),
    1: (300.0, 0.0, 0.0),
    2: (300.0, 300.0, 0.0),
    3: (0.0, 300.0, 0.0),
}

# ── Pose stabilisation ──────────────────────────────────────────────
# EMA smoothing alpha for marker pose (0 = full smoothing, 1 = no smoothing).
# Lower values = smoother but more lag; 0.3 is a good compromise.
POSE_SMOOTHING_ALPHA = 0.3

# Maximum allowed jump (in world units) for tvec between frames before
# the new reading is treated as an outlier and rejected.
POSE_OUTLIER_THRESHOLD = 150.0

# Sub-pixel corner refinement window half-size (pixels).
CORNER_REFINE_WIN = 5

# Number of consecutive outlier frames before we force-accept the new pose
# (prevents getting stuck if the marker physically moves).
OUTLIER_PATIENCE = 15
