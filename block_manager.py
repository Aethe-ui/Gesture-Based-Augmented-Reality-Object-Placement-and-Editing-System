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

    def add_block(self, x: float, y: float, z: float, color: tuple[int, int, int]) -> bool:
        sx, sy, sz = self.snap_to_grid(x, y, z)
        for b in self.blocks:
            if b["pos"] == (sx, sy, sz):
                return False

        self._push_undo_state()
        self.blocks.append({"pos": (sx, sy, sz), "color": color})
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

    def move_block(self, index: int, new_x: float, new_y: float, new_z: float) -> bool:
        if index < 0 or index >= len(self.blocks):
            return False

        sx, sy, sz = self.snap_to_grid(new_x, new_y, new_z)

        for i, b in enumerate(self.blocks):
            if i == index:
                continue
            if b["pos"] == (sx, sy, sz):
                return False

        self._push_undo_state()
        self.blocks[index]["pos"] = (sx, sy, sz)
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
