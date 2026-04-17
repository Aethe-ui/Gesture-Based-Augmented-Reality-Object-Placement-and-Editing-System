import cv2
import numpy as np


MARKER_IDS = [0, 1, 2, 3]
MARKER_SIZE_PX = 400


def generate_marker() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    for marker_id in MARKER_IDS:
        output_path = f"marker_id{marker_id}.png"
        marker = np.zeros((MARKER_SIZE_PX, MARKER_SIZE_PX), dtype=np.uint8)
        cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_SIZE_PX, marker, 1)

        if not cv2.imwrite(output_path, marker):
            raise RuntimeError(f"Failed to write marker image to {output_path}")

        print(f"Saved ArUco marker to {output_path}")

    print("Place markers at room/table corners.")
    print("Display it full-screen on another device or print it flat on paper.")
    print("Keep the marker unmirrored and fully visible to the webcam.")


if __name__ == "__main__":
    generate_marker()
