import numpy as np
import cv2
import config

def get_camera_matrix(w, h):
    """
    Returns a simple camera matrix assuming a field of view of approx 60 degrees.
    """
    # Focal length approx w (approx 60 deg hfov)
    f = w 
    cx, cy = w / 2, h / 2
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]], dtype=np.float32)

def project_point_3d_to_2d(point_3d, rvec, tvec, K, dist_coeffs=None):
    """
    Project a 3D point (x, y, z) to 2D image coordinates.
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    
    img_points, _ = cv2.projectPoints(np.array([point_3d], dtype=np.float32), rvec, tvec, K, dist_coeffs)
    return tuple(map(int, img_points[0][0]))

def draw_cube(img, center_3d, size, rvec, tvec, K, color):
    """
    Draws a 3D solid cube at center_3d.
    """
    half_size = size / 2
    x, y, z = center_3d

    # Define 8 corners of the cube
    corners_3d = np.array([
        [x - half_size, y - half_size, z - half_size], # 0: Front Bottom Left
        [x + half_size, y - half_size, z - half_size], # 1: Front Bottom Right
        [x + half_size, y + half_size, z - half_size], # 2: Front Top Right
        [x - half_size, y + half_size, z - half_size], # 3: Front Top Left
        [x - half_size, y - half_size, z + half_size], # 4: Back Bottom Left
        [x + half_size, y - half_size, z + half_size], # 5: Back Bottom Right
        [x + half_size, y + half_size, z + half_size], # 6: Back Top Right
        [x - half_size, y + half_size, z + half_size]  # 7: Back Top Left
    ], dtype=np.float32)

    # Project to 2D
    img_pts = []
    for pt in corners_3d:
        img_pts.append(project_point_3d_to_2d(pt, rvec, tvec, K))
    
    # Define faces (indices)
    # Z coordinate is usually UP/Forward. 
    # If we assume standard view, we can cheat painter's algo by drawing back faces first?
    # Or just draw all faces? With transparency?
    
    # Faces: Bottom, Top, Front, Back, Left, Right
    faces = [
        ([0, 1, 2, 3], color), # Front
        ([4, 5, 6, 7], color), # Back
        ([0, 1, 5, 4], color), # Bottom
        ([2, 3, 7, 6], color), # Top
        ([0, 3, 7, 4], color), # Left
        ([1, 2, 6, 5], color)  # Right
    ]

    # Simple painter's algo: Sort faces by depth
    # Calculate centroid depth for each face
    # Transform centroids to camera space
    R, _ = cv2.Rodrigues(rvec)
    
    def get_face_depth(indices):
        centroid = np.mean(corners_3d[indices], axis=0)
        # P_cam = R * P_world + t
        # We only care about Z_cam
        p_cam = R @ centroid + tvec
        return p_cam[2] # Z depth
        
    # Sort faces from furthest to closest (large Z to small Z)
    faces.sort(key=lambda f: get_face_depth(f[0]), reverse=True)

    for indices, col in faces:
        pts = np.array([img_pts[i] for i in indices])
        # Draw filled polygon
        # Darker shade for fill
        fill_color = tuple(max(0, c - 50) for c in col)
        cv2.fillPoly(img, [pts], fill_color)
        # Draw border
        cv2.polylines(img, [pts], True, col, 2)


def ray_cast_to_ground(uv, K, rvec, tvec):
    """
    Casts a ray from the camera through screen point uv (u, v)
    and finds the intersection with the ground plane (Z=0).
    Returns (x, y, 0).
    """
    # 1. Convert to normalized device coordinates
    u, v = uv
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Normalized ray direction in camera frame
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = 1.0
    ray_cam = np.array([x_cam, y_cam, z_cam])
    
    # 2. Transform ray to world coordinates
    # P_cam = R * P_world + t
    # P_world = R^T * (P_cam - t)
    # Ray origin in world: -R^T * t
    # Ray dir in world: R^T * ray_cam
    
    R, _ = cv2.Rodrigues(rvec)
    R_inv = R.T
    
    ray_origin_world = -R_inv @ tvec
    ray_dir_world = R_inv @ ray_cam
    
    # 3. Intersect with Plane Z = 0
    # Line: P = O + d * lambda
    # P.z = 0 => O.z + d.z * lambda = 0 => lambda = -O.z / d.z
    
    if abs(ray_dir_world[2]) < 1e-6:
        return None # Parallel to plane
        
    lmbda = -ray_origin_world[2] / ray_dir_world[2]
    
    if lmbda < 0:
        return None # Behind camera
        
    intersection = ray_origin_world + ray_dir_world * lmbda
    return intersection

def draw_grid(img, rvec, tvec, K):
    """
    Draws a reference grid on the Z=0 plane.
    """
    grid_sz = config.GRID_SIZE
    step = config.GRID_SPACING
    
    # Calculate offset to center grid roughly
    offset = (grid_sz * step) / 2
    
    for i in range(grid_sz + 1):
        # Lines parallel to X-axis
        p1 = np.array([-offset, -offset + i * step, 0])
        p2 = np.array([-offset + grid_sz * step, -offset + i * step, 0])
        
        pt1 = project_point_3d_to_2d(p1, rvec, tvec, K)
        pt2 = project_point_3d_to_2d(p2, rvec, tvec, K)
        cv2.line(img, pt1, pt2, config.COLOR_GRID, 1)

        # Lines parallel to Y-axis
        p3 = np.array([-offset + i * step, -offset, 0])
        p4 = np.array([-offset + i * step, -offset + grid_sz * step, 0])
        
        pt3 = project_point_3d_to_2d(p3, rvec, tvec, K)
        pt4 = project_point_3d_to_2d(p4, rvec, tvec, K)
        cv2.line(img, pt3, pt4, config.COLOR_GRID, 1)
