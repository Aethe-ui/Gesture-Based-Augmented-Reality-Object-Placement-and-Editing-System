from pathlib import Path


def export_to_obj(blocks: list[dict], filepath: str = "export.obj", block_size: float = 50.0) -> str:
    hs = float(block_size) / 2.0
    lines: list[str] = ["# Gesture-Based AR blocks export"]
    vertex_offset = 1

    for i, b in enumerate(blocks):
        cx, cy, cz = b["pos"]
        corners = [
            (cx - hs, cy - hs, cz - hs),
            (cx + hs, cy - hs, cz - hs),
            (cx + hs, cy + hs, cz - hs),
            (cx - hs, cy + hs, cz - hs),
            (cx - hs, cy - hs, cz + hs),
            (cx + hs, cy - hs, cz + hs),
            (cx + hs, cy + hs, cz + hs),
            (cx - hs, cy + hs, cz + hs),
        ]

        lines.append(f"o block_{i}")
        for vx, vy, vz in corners:
            lines.append(f"v {vx:.6f} {vy:.6f} {vz:.6f}")

        faces = [
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 4, 8, 7),
            (4, 1, 5, 8),
        ]
        for f in faces:
            a, b_, c, d = f
            lines.append(
                f"f {vertex_offset + a - 1} {vertex_offset + b_ - 1} "
                f"{vertex_offset + c - 1} {vertex_offset + d - 1}"
            )
        vertex_offset += 8

    path = Path(filepath)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
