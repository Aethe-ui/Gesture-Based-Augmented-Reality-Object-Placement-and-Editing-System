import unittest
import numpy as np
import config
from block_manager import BlockManager
import ar_math

class TestARLogic(unittest.TestCase):
    def setUp(self):
        self.mgr = BlockManager()
        self.mgr.grid_spacing = 10 # Simpler testing
        
    def test_snap_to_grid(self):
        self.assertEqual(self.mgr.snap_to_grid(12, 12, 0), (10, 10, 0))
        self.assertEqual(self.mgr.snap_to_grid(8, 8, 0), (10, 10, 0))
        self.assertEqual(self.mgr.snap_to_grid(4, 4, 0), (0, 0, 0))

    def test_add_remove_block(self):
        self.assertTrue(self.mgr.add_block(10, 10, 0))
        self.assertEqual(len(self.mgr.blocks), 1)
        
        # Duplicate check
        self.assertFalse(self.mgr.add_block(12, 12, 0)) # Snaps to 10,10
        self.assertEqual(len(self.mgr.blocks), 1)
        
        # Remove
        self.assertTrue(self.mgr.remove_block(10, 10, 0))
        self.assertEqual(len(self.mgr.blocks), 0)
        
    def test_move_block(self):
        self.mgr.add_block(0, 0, 0)
        idx, _ = self.mgr.get_block_at(0, 0, 0)
        self.assertNotEqual(idx, -1)
        
        # Move successfully
        self.assertTrue(self.mgr.move_block(idx, 20, 20, 0))
        self.assertEqual(self.mgr.blocks[0]['pos'], (20, 20, 0))
        
        # Move to occupied
        self.mgr.add_block(50, 50, 0)
        self.assertFalse(self.mgr.move_block(0, 50, 50, 0)) # Should fail
        
    def test_ray_cast(self):
        # Mock Camera Matrix
        w, h = 640, 480
        K = ar_math.get_camera_matrix(w, h)
        
        # Camera looking down from (0, 0, 100) exactly
        # No rotation for simplicity first (looking along Z)
        # Actually our ray_cast expects camera looking at Z-plane
        
        # Let's use the actual config setup approximately
        theta = np.deg2rad(45)
        r_x = np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])
        rvec, _ = cv2.Rodrigues(r_x)
        tvec = np.array([0, -200, 800], dtype=np.float32)
        
        # Center of screen should hit somewhere on ground
        uv = (w/2, h/2)
        pt = ar_math.ray_cast_to_ground(uv, K, rvec, tvec)
        self.assertIsNotNone(pt)
        # Should be roughly (0, some_y, 0)
        self.assertAlmostEqual(pt[0], 0, delta=1.0)
        self.assertAlmostEqual(pt[2], 0, delta=0.001)

import cv2
if __name__ == '__main__':
    unittest.main()
