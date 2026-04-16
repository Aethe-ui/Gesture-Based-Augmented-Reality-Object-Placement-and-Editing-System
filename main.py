import cv2
import config
from hand_tracker import HandTracker
from block_manager import BlockManager
import ar_math
import numpy as np
import time

def main():
    # Try different camera indices and backends
    cap = None
    for index in [0, 1]:
        print(f"Trying camera index {index}...")
        # Try AVFoundation explicitly on Mac
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
             # Fallback to default
             cap = cv2.VideoCapture(index)
             cap = cv2.VideoCapture(index) # Fallback
        
        if cap.isOpened():
             print(f"Camera opened on index {index}")
             break
             
    if cap is None or not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(3, config.FRAME_WIDTH)
    cap.set(4, config.FRAME_HEIGHT)

    try:
        tracker = HandTracker()
    except Exception as e:
        print(f"Failed to load HandTracker/Model: {e}")
        return
    block_mgr = BlockManager()
    
    # Camera Matrix
    K = ar_math.get_camera_matrix(config.FRAME_WIDTH, config.FRAME_HEIGHT)
    
    # Virtual Camera Pose (Fixed overhead view for now, simulating look-down at table)
    # We want the camera to be "above" and "back" looking at the origin (0,0,0)
    # This is a simplification. Ideally we'd use Aruco markers to get real pose.
    # For now, we assume the camera is at a fixed position relative to the "table".
    
    # Rotated 45 degrees around X axis to look down
    # Rotation matrix for X axis
    # Virtual Camera Pose (Fixed overhead view)
    theta = np.deg2rad(45)
    r_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
    
    # Default Virtual Pose (Fallback)
    rvec_virtual, _ = cv2.Rodrigues(r_x)
    tvec_virtual = np.array([0, 200, 1000], dtype=np.float32) 
    
    # Current Pose (starts virtual)
    rvec, tvec = rvec_virtual, tvec_virtual
    using_marker = False
    
    # Aruco Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    # Create detector (OpenCV 4.7+)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    pTime = 0
    last_pinch_time = 0
    pinch_cooldown = 0.5 
    
    mode = 0 
    mode_names = ["PLACE", "MOVE", "DELETE"]
    mode_colors = [config.COLOR_GREEN, config.COLOR_BLUE, config.COLOR_RED]
    selected_block_idx = -1

    while True:
        success, img = cap.read()
        if not success:
            break
            
        # img = cv2.flip(img, 1) # Mirror view - CAUTION: Mirroring breaks marker pose estimation!
        # If we mirror, we must mirror marker detection too or flip back.
        # Standard AR practice: Don't mirror for back camera interactions, but for selfie/webcam often do.
        # If we flip, the marker ID text becomes mirrored and detection fails or pose is inverted.
        # Solution: Detect on original, then flip image and render? No, geometry breaks.
        # Best: Do NOT flip image if using markers. It feels less like a mirror but is correct for AR.
        # Or: Flip, detect, flip pose. Complex.
        # Let's removing flipping for Marker Mode to ensure stability.
        # User might find it weird if it was mirrored before.
        # Let's keep it un-flipped for now to ensure Aruco works.
        
        # 0. Aruco Detection
        # Detect markers
        corners, ids, rejected = aruco_detector.detectMarkers(img)
        
        marker_found = False
        if ids is not None:
            for i, marker_id in enumerate(ids):
                if marker_id[0] == 0: # We look for ID 0
                    # Estimate Pose
                    # standard estimatePoseSingleMarkers (might need to check version)
                    # For OpenCV 4.7+, try to use the objPoints logic or just the function if available
                    marker_size = 100 # Arbitrary units (e.g. mm) matching config
                    
                    # Create object points for the marker (assuming flat on Z=0)
                    # centered at 0,0,0
                    half = marker_size / 2
                    obj_points = np.array([
                        [-half, half, 0],
                        [half, half, 0],
                        [half, -half, 0],
                        [-half, -half, 0]
                    ], dtype=np.float32)
                    
                    # SolvePnP
                    # detectMarkers returns corners: [top-left, top-right, bottom-right, bottom-left]
                    # But Aruco order is TL, TR, BR, BL
                    image_points = corners[i][0]
                    
                    ret, rvec_est, tvec_est = cv2.solvePnP(obj_points, image_points, K, None)
                    
                    if ret:
                        rvec, tvec = rvec_est, tvec_est
                        marker_found = True
                        using_marker = True
                        
                        # Draw axis for debugging
                        cv2.drawFrameAxes(img, K, None, rvec, tvec, 50)
                        break
        
        if not marker_found:
             # Smoothly transition back or stick to last known?
             # For now, snap back to virtual if lost for too long?
             # Or just stay at last known.
             # If we haven't seen marker for X frames, revert? 
             # Let's stick to last known if we ever found it, else virtual.
             if not using_marker:
                 rvec, tvec = rvec_virtual, tvec_virtual

        # 1. Detect Hands
        try:
            tracker.find_hands(img) 
            lm_list = tracker.find_position(img)
        except Exception as e:
             print(f"Tracking error: {e}")
             continue
        
        # 2. Process Gestures
        if lm_list:
            is_pinching, center = tracker.is_pinching(lm_list)
            
            # Map hand center to the ground plane (Z=0 relative to marker or virtual floor)
            ground_point = ar_math.ray_cast_to_ground(center, K, rvec, tvec)
            
            if ground_point is not None:
                gx, gy, _ = ground_point
                
                # Visual cursor
                ground_center_2d = ar_math.project_point_3d_to_2d((gx, gy, 0), rvec, tvec, K)
                cv2.circle(img, ground_center_2d, 5, config.COLOR_YELLOW, cv2.FILLED)
                
                if is_pinching:
                    cv2.circle(img, center, 15, mode_colors[mode], cv2.FILLED)
                    
                    if mode == 0: # PLACE
                        if time.time() - last_pinch_time > pinch_cooldown:
                            sx, sy, _ = block_mgr.snap_to_grid(gx, gy, 0)
                            
                            max_z = -config.BLOCK_SIZE
                            for b in block_mgr.blocks:
                                bx, by, bz = b['pos']
                                if bx == sx and by == sy:
                                    if bz > max_z:
                                        max_z = bz
                            new_z = max_z + config.BLOCK_SIZE
                            
                            if block_mgr.add_block(sx, sy, new_z):
                                print(f"Added block at {sx:.1f}, {sy:.1f}, {new_z:.1f}")
                            last_pinch_time = time.time()
                            
                    elif mode == 1: # MOVE
                        if selected_block_idx == -1:
                            idx, _ = block_mgr.get_block_at(gx, gy, 0, tolerance=config.BLOCK_SIZE)
                            if idx != -1:
                                selected_block_idx = idx
                        else:
                            old_z = block_mgr.blocks[selected_block_idx]['pos'][2]
                            block_mgr.move_block(selected_block_idx, gx, gy, old_z)
                            
                    elif mode == 2: # DELETE
                        if time.time() - last_pinch_time > pinch_cooldown:
                            idx, _ = block_mgr.get_block_at(gx, gy, 0, tolerance=config.BLOCK_SIZE * 0.8)
                            if idx != -1:
                                block_mgr.blocks.pop(idx)
                                print("Block deleted")
                            last_pinch_time = time.time()
                else:
                    selected_block_idx = -1

        # 3. Render AR Scene
        try:
            ar_math.draw_grid(img, rvec, tvec, K)
        except Exception:
            pass 

        # Draw Blocks
        for i, block in enumerate(block_mgr.get_blocks()):
            x, y, z = block['pos']
            color = block['color']
            if i == selected_block_idx:
                color = config.COLOR_YELLOW
            ar_math.draw_cube(img, (x, y, z), config.BLOCK_SIZE, rvec, tvec, K, color)

        # 4. UI Overlay
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        
        cv2.rectangle(img, (0, 0), (400, 100), (50, 50, 50), cv2.FILLED)
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, config.COLOR_WHITE, 2)
        cv2.putText(img, f'Mode: {mode_names[mode]} (Press M)', (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, mode_colors[mode], 2)
        cv2.putText(img, f'Tracking: {"MARKER" if using_marker else "VIRTUAL"}', (10, 90), cv2.FONT_HERSHEY_PLAIN, 1.5, config.COLOR_WHITE, 1)
        
        cv2.imshow("AR Block Builder", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = (mode + 1) % 3

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
