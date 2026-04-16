import os
import tempfile
import unittest

import cv2
import numpy as np

import ar_math
import config
from block_manager import BlockManager
from export import export_to_obj
from scene_manager import load_scene, save_scene


class TestARLogic(unittest.TestCase):
    def setUp(self) -> None:
        self.mgr = BlockManager()

    def test_snap_to_grid(self) -> None:
        x, y, z = self.mgr.snap_to_grid(31, -31, 74)
        self.assertEqual(x, round(31 / config.GRID_SPACING) * config.GRID_SPACING)
        self.assertEqual(y, round(-31 / config.GRID_SPACING) * config.GRID_SPACING)
        self.assertEqual(z, round(74 / config.BLOCK_SIZE) * config.BLOCK_SIZE)

    def test_add_remove_block(self) -> None:
        self.assertTrue(self.mgr.add_block(0, 0, 0, config.BLUE))
        self.assertFalse(self.mgr.add_block(1, 1, 0, config.RED))  # snaps to same cell
        self.assertEqual(len(self.mgr.get_blocks()), 1)

        self.assertTrue(self.mgr.remove_block(0, 0, 0))
        self.assertFalse(self.mgr.remove_block(0, 0, 0))
        self.assertEqual(len(self.mgr.get_blocks()), 0)

    def test_move_block(self) -> None:
        self.mgr.add_block(0, 0, 0, config.BLUE)
        self.mgr.add_block(config.GRID_SPACING, 0, 0, config.RED)

        idx, _ = self.mgr.get_block_at(0, 0, 0)
        self.assertNotEqual(idx, -1)
        self.assertTrue(self.mgr.move_block(idx, 0, config.GRID_SPACING, 0))
        self.assertFalse(self.mgr.move_block(idx, config.GRID_SPACING, 0, 0))  # collision

    def test_undo_redo(self) -> None:
        self.assertTrue(self.mgr.add_block(0, 0, 0, config.BLUE))
        self.assertTrue(self.mgr.add_block(config.GRID_SPACING, 0, 0, config.RED))
        self.assertEqual(len(self.mgr.get_blocks()), 2)

        self.assertTrue(self.mgr.undo())
        self.assertEqual(len(self.mgr.get_blocks()), 1)
        self.assertTrue(self.mgr.redo())
        self.assertEqual(len(self.mgr.get_blocks()), 2)

    def test_ray_cast(self) -> None:
        w, h = 1280, 720
        K = ar_math.get_camera_matrix(w, h)

        theta = np.deg2rad(45.0)
        r_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(theta), -np.sin(theta)],
                [0.0, np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float32,
        )
        rvec, _ = cv2.Rodrigues(r_x)
        tvec = np.array([[0.0], [200.0], [1000.0]], dtype=np.float32)

        pt = ar_math.ray_cast_to_ground((w / 2.0, h / 2.0), K, rvec, tvec)
        self.assertIsNotNone(pt)
        assert pt is not None
        self.assertAlmostEqual(pt[2], 0.0, places=5)

    def test_save_load_scene(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            scene_path = os.path.join(td, "scene.json")
            blocks = [
                {"pos": (0.0, 0.0, 0.0), "color": config.BLUE},
                {"pos": (60.0, 0.0, 50.0), "color": config.RED},
            ]
            save_scene(blocks, scene_path)
            loaded = load_scene(scene_path)
            self.assertEqual(loaded, blocks)

    def test_obj_export(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "export.obj")
            blocks = [{"pos": (0.0, 0.0, 0.0), "color": config.BLUE}]
            path = export_to_obj(blocks, filepath=out_path, block_size=float(config.BLOCK_SIZE))
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
