import cv2
import numpy as np

def generate_marker():
    # Load the dictionary that was used to generate the markers.
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    # Generate the marker
    marker_image = np.zeros((200, 200), dtype=np.uint8)
    marker_image = cv2.aruco.generateImageMarker(dictionary, 0, 200, marker_image, 1)

    cv2.imwrite("marker_id0.png", marker_image)
    print("Generated marker_id0.png. Open this on your phone or print it!")

if __name__ == "__main__":
    generate_marker()
