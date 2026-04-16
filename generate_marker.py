import cv2
import numpy as np


OUTPUT_PATH = "marker_id0.png"
MARKER_ID = 0
MARKER_SIZE_PX = 400


def generate_marker() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker = np.zeros((MARKER_SIZE_PX, MARKER_SIZE_PX), dtype=np.uint8)
    cv2.aruco.generateImageMarker(dictionary, MARKER_ID, MARKER_SIZE_PX, marker, 1)

    if not cv2.imwrite(OUTPUT_PATH, marker):
        raise RuntimeError(f"Failed to write marker image to {OUTPUT_PATH}")

    print(f"Saved ArUco marker to {OUTPUT_PATH}")
    print("Display it full-screen on another device or print it flat on paper.")
    print("Keep the marker unmirrored and fully visible to the webcam.")


if __name__ == "__main__":
    generate_marker()
