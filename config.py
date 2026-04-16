# Camera Settings
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# AR Settings
BLOCK_SIZE = 50  # Size of the cube in pixels (approximate for 2D, or units for 3D)
GRID_SIZE = 10   # 10x10 grid
GRID_SPACING = 60 # Distance between grid lines

# Colors (B, G, R)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_GRID = (200, 200, 200)

# Gesture Thresholds
PINCH_THRESHOLD = 30  # Distance between thumb and index tip to consider a pinch
SMOOTHING_FACTOR = 0.5 # For hand tracking smoothing

# Model Path
MODEL_PATH = "hand_landmarker.task"

# Marker Settings
MARKER_SIZE = 100 # Millimeters (physical size) - heavily affects Z estimation
