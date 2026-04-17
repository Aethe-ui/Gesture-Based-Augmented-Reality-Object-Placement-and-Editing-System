import math
from copy import deepcopy
from typing import Optional

import config


class BlockManager:
    def __init__(self) -> None:
        self.blocks: list[dict] = []
        self.history: list[list[dict]] = []
        self.redo_stack: list[list[dict]] = []

    def _snapshot(self) -> list[dict]:
        return deepcopy(self.blocks)

    def _push_undo_state(self) -> None:
        self.history.append(self._snapshot())
        self.redo_stack.clear()

    def snap_to_grid(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        s_xy = float(config.GRID_SPACING)
        s_z = float(config.BLOCK_SIZE)
        sx = round(float(x) / s_xy) * s_xy
        sy = round(float(y) / s_xy) * s_xy
        sz = round(float(z) / s_z) * s_z
        return sx, sy, sz

    @staticmethod
    def _get_aabb(
        pos: tuple[float, float, float], shape_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return (min_corner, max_corner) AABB for a block."""
        bs = float(config.BLOCK_SIZE)
        shape_def = config.BLOCK_SHAPES.get(shape_id, config.BLOCK_SHAPES[0])
        ssx, ssy, ssz = shape_def["size"]
        hx, hy, hz = bs * ssx / 2.0, bs * ssy / 2.0, bs * ssz / 2.0
        cx, cy, cz = pos
        return (cx - hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz + hz)

    @staticmethod
    def _aabbs_overlap(
        a_min: tuple[float, float, float],
        a_max: tuple[float, float, float],
        b_min: tuple[float, float, float],
        b_max: tuple[float, float, float],
        eps: float = 0.1,
    ) -> bool:
        """Check if two AABBs overlap on all 3 axes (with small tolerance)."""
        for i in range(3):
            if a_max[i] - eps <= b_min[i] or b_max[i] - eps <= a_min[i]:
                return False
        return True

    def add_block(
        self, x: float, y: float, z: float,
        color: tuple[int, int, int],
        shape: int = 0,
    ) -> bool:
        sx, sy, sz = self.snap_to_grid(x, y, z)
        new_min, new_max = self._get_aabb((sx, sy, sz), shape)

        for b in self.blocks:
            b_min, b_max = self._get_aabb(b["pos"], b.get("shape", 0))
            if self._aabbs_overlap(new_min, new_max, b_min, b_max):
                return False

        self._push_undo_state()
        self.blocks.append({"pos": (sx, sy, sz), "color": color, "shape": shape})
        return True

    def get_block_at(
        self,
        x: float,
        y: float,
        z: float,
        tolerance: Optional[float] = None,
    ) -> tuple[int, Optional[dict]]:
        if tolerance is None:
            sx, sy, sz = self.snap_to_grid(x, y, z)
            for i, b in enumerate(self.blocks):
                if b["pos"] == (sx, sy, sz):
                    return i, b
            return -1, None

        tol = float(tolerance)
        best_i = -1
        best_block: Optional[dict] = None
        best_dist = float("inf")

        for i, b in enumerate(self.blocks):
            bx, by, bz = b["pos"]
            d = math.sqrt((bx - x) ** 2 + (by - y) ** 2 + (bz - z) ** 2)
            if d <= tol and d < best_dist:
                best_dist = d
                best_i = i
                best_block = b

        return best_i, best_block

    def remove_block(self, x: float, y: float, z: float) -> bool:
        idx, _ = self.get_block_at(x, y, z)
        if idx == -1:
            return False
        self._push_undo_state()
        del self.blocks[idx]
        return True

    def move_block(
        self, index: int, new_x: float, new_y: float, new_z: float,
        shape: int | None = None,
    ) -> bool:
        if index < 0 or index >= len(self.blocks):
            return False

        sx, sy, sz = self.snap_to_grid(new_x, new_y, new_z)
        move_shape = shape if shape is not None else self.blocks[index].get("shape", 0)
        new_min, new_max = self._get_aabb((sx, sy, sz), move_shape)

        for i, b in enumerate(self.blocks):
            if i == index:
                continue
            b_min, b_max = self._get_aabb(b["pos"], b.get("shape", 0))
            if self._aabbs_overlap(new_min, new_max, b_min, b_max):
                return False

        self._push_undo_state()
        self.blocks[index]["pos"] = (sx, sy, sz)
        if shape is not None:
            self.blocks[index]["shape"] = shape
        return True

    def get_blocks(self) -> list[dict]:
        return self.blocks

    def set_blocks(self, blocks: list[dict]) -> None:
        self.blocks = deepcopy(blocks)
        self.history.clear()
        self.redo_stack.clear()

    def undo(self) -> bool:
        if not self.history:
            return False
        self.redo_stack.append(self._snapshot())
        self.blocks = self.history.pop()
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.history.append(self._snapshot())
        self.blocks = self.redo_stack.pop()
        return True
