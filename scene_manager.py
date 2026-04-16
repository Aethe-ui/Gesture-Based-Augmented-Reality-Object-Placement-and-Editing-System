import json
from pathlib import Path


def save_scene(blocks: list[dict], filepath: str = "scene.json") -> None:
    serializable = []
    for b in blocks:
        x, y, z = b["pos"]
        cb, cg, cr = b["color"]
        serializable.append(
            {
                "pos": [x, y, z],
                "color": [cb, cg, cr],
            }
        )

    path = Path(filepath)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_scene(filepath: str = "scene.json") -> list[dict]:
    path = Path(filepath)
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []

    blocks: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pos = item.get("pos")
        color = item.get("color")
        if not isinstance(pos, list) or len(pos) != 3:
            continue
        if not isinstance(color, list) or len(color) != 3:
            continue
        blocks.append(
            {
                "pos": (float(pos[0]), float(pos[1]), float(pos[2])),
                "color": (int(color[0]), int(color[1]), int(color[2])),
            }
        )

    return blocks
