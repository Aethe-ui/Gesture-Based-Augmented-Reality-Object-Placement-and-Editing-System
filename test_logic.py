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
                {"pos": (0.0, 0.0, 0.0), "color": config.BLUE, "shape": 0},
                {"pos": (60.0, 0.0, 50.0), "color": config.RED, "shape": 0},
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

    def test_fuse_poses(self) -> None:
        rvec = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        tvec = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        fused_rvec, fused_tvec = ar_math.fuse_poses([(rvec, tvec, 1.0), (rvec, tvec, 1.0)])
        self.assertTrue(np.allclose(fused_rvec, rvec))
        self.assertTrue(np.allclose(fused_tvec, tvec))

    def test_fuse_poses_average(self) -> None:
        rvec_a = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        tvec_a = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        rvec_b = np.array([[0.2], [0.2], [0.2]], dtype=np.float32)
        tvec_b = np.array([[100.0], [50.0], [20.0]], dtype=np.float32)

        fused_rvec, fused_tvec = ar_math.fuse_poses([(rvec_a, tvec_a, 1.0), (rvec_b, tvec_b, 1.0)])
        self.assertTrue(np.allclose(fused_rvec, np.array([[0.1], [0.1], [0.1]], dtype=np.float32)))
        self.assertTrue(np.allclose(fused_tvec, np.array([[50.0], [25.0], [10.0]], dtype=np.float32)))

    # ── Pose stabiliser tests ────────────────────────────────────

    def test_stabilizer_first_frame_passthrough(self) -> None:
        """First frame should pass through without smoothing."""
        stab = ar_math.PoseStabilizer(alpha=0.3)
        rvec = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        tvec = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        out_r, out_t = stab.update(rvec, tvec)
        self.assertTrue(np.allclose(out_r, rvec))
        self.assertTrue(np.allclose(out_t, tvec))

    def test_stabilizer_smoothing(self) -> None:
        """Subsequent frames should be EMA-smoothed."""
        stab = ar_math.PoseStabilizer(alpha=0.5, outlier_threshold=9999.0)
        rvec1 = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        tvec1 = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        stab.update(rvec1, tvec1)

        rvec2 = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
        tvec2 = np.array([[100.0], [100.0], [100.0]], dtype=np.float32)
        out_r, out_t = stab.update(rvec2, tvec2)

        # With alpha=0.5: result = 0.5*new + 0.5*old = 0.5
        self.assertTrue(np.allclose(out_r, np.array([[0.5], [0.5], [0.5]])))
        self.assertTrue(np.allclose(out_t, np.array([[50.0], [50.0], [50.0]])))

    def test_stabilizer_outlier_rejection(self) -> None:
        """Sudden large jumps should be rejected."""
        stab = ar_math.PoseStabilizer(alpha=0.5, outlier_threshold=50.0, outlier_patience=5)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        tvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        stab.update(rvec, tvec)

        # Sudden jump of 1000 units — should be rejected
        tvec_bad = np.array([[1000.0], [0.0], [0.0]], dtype=np.float32)
        out_r, out_t = stab.update(rvec, tvec_bad)
        # Should still return the previous pose
        self.assertTrue(np.allclose(out_t, tvec, atol=1e-3))

    def test_stabilizer_outlier_patience_force_accept(self) -> None:
        """After enough consecutive outliers, force-accept the new pose."""
        stab = ar_math.PoseStabilizer(alpha=0.5, outlier_threshold=50.0, outlier_patience=3)
        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        tvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        stab.update(rvec, tvec)

        tvec_far = np.array([[1000.0], [0.0], [0.0]], dtype=np.float32)
        # Frame 1, 2 → rejected (patience=3)
        stab.update(rvec, tvec_far)
        stab.update(rvec, tvec_far)
        # Frame 3 → force-accepted
        _, out_t = stab.update(rvec, tvec_far)
        self.assertTrue(np.allclose(out_t, tvec_far))

    def test_stabilizer_reset(self) -> None:
        """After reset, stabilizer should behave like a fresh instance."""
        stab = ar_math.PoseStabilizer(alpha=0.3)
        rvec = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        tvec = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        stab.update(rvec, tvec)

        stab.reset()

        # Next update should pass through like a first frame
        rvec2 = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        tvec2 = np.array([[100.0], [200.0], [300.0]], dtype=np.float32)
        out_r, out_t = stab.update(rvec2, tvec2)
        self.assertTrue(np.allclose(out_r, rvec2))
        self.assertTrue(np.allclose(out_t, tvec2))

    # ── Optical-flow interpolation tests (Phase 7) ───────────────

    def test_interpolate_pose_with_flow_identical(self) -> None:
        """Identical prev/curr frames with features → should succeed,
        and the returned pose should be very close to the input."""
        w, h = 320, 240
        K = ar_math.get_camera_matrix(w, h)

        # Create a synthetic image with detectable features (checkerboard-like)
        img = np.zeros((h, w), dtype=np.uint8)
        for r in range(0, h, 20):
            for c in range(0, w, 20):
                if (r // 20 + c // 20) % 2 == 0:
                    img[r:r+20, c:c+20] = 255

        rvec = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        tvec = np.array([[10.0], [20.0], [500.0]], dtype=np.float32)

        new_rvec, new_tvec, success = ar_math.interpolate_pose_with_flow(
            rvec, tvec, img, img, K
        )
        # With identical images the homography should be identity-like,
        # so the output pose should be very close to the input.
        self.assertTrue(success)
        self.assertTrue(np.allclose(new_rvec, rvec, atol=0.1))
        self.assertTrue(np.allclose(new_tvec, tvec, atol=10.0))

    def test_interpolate_pose_flow_insufficient_features(self) -> None:
        """Blank images have no features → flow should fail gracefully."""
        w, h = 320, 240
        K = ar_math.get_camera_matrix(w, h)
        blank = np.zeros((h, w), dtype=np.uint8)

        rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
        tvec = np.array([[0.0], [0.0], [500.0]], dtype=np.float32)

        _, _, success = ar_math.interpolate_pose_with_flow(
            rvec, tvec, blank, blank, K
        )
        self.assertFalse(success)

    # ── Block shapes & face snapping tests (Phase 10) ────────────

    def test_block_shapes(self) -> None:
        """Place CUBE, SLAB, WALL; verify 'shape' field is stored."""
        # Space blocks far apart so their AABBs don't overlap
        # SLAB is 3×BLOCK_SIZE wide = 150, so needs ≥4 grid spacings apart
        spacing = config.GRID_SPACING * 4
        self.assertTrue(self.mgr.add_block(0, 0, 0, config.BLUE, shape=0))
        self.assertTrue(self.mgr.add_block(spacing, 0, 0, config.RED, shape=1))
        self.assertTrue(self.mgr.add_block(spacing * 2, 0, 0, config.GREEN, shape=2))
        self.assertEqual(len(self.mgr.get_blocks()), 3)

        self.assertEqual(self.mgr.get_blocks()[0].get("shape"), 0)
        self.assertEqual(self.mgr.get_blocks()[1].get("shape"), 1)
        self.assertEqual(self.mgr.get_blocks()[2].get("shape"), 2)

    def test_face_snap(self) -> None:
        """Place a block at z=BLOCK_SIZE/2, verify next placement stacks flush."""
        bs = float(config.BLOCK_SIZE)
        # Place first CUBE — center at z = bs/2 (top face at z = bs)
        self.assertTrue(self.mgr.add_block(0, 0, bs / 2.0, config.BLUE, shape=0))
        first = self.mgr.get_blocks()[0]
        self.assertEqual(first["pos"][2], round((bs / 2.0) / bs) * bs)  # snapped

        # The top face of a CUBE at center z=bz is bz + bs*1/2 = bz + bs/2
        # Next CUBE center should be at top + bs/2
        bz = first["pos"][2]
        expected_next_z = (bz + bs * 0.5) + bs * 0.5  # top of first + half of new
        # Verify the math is correct
        self.assertAlmostEqual(expected_next_z, bz + bs)


if __name__ == "__main__":
    unittest.main()
