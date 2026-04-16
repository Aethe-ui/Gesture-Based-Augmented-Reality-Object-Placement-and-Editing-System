import config
import math

class BlockManager:
    def __init__(self):
        self.blocks = [] # List of {'pos': (x,y,z), 'color': ...}
        self.grid_spacing = config.GRID_SPACING
        
    def snap_to_grid(self, x, y, z=0):
        """
        Snaps a raw 3D coordinate to the nearest grid point.
        """
        snapped_x = round(x / self.grid_spacing) * self.grid_spacing
        snapped_y = round(y / self.grid_spacing) * self.grid_spacing
        # Z is usually 0 (floor) or stacked
        snapped_z = round(z / self.grid_spacing) * self.grid_spacing 
        return snapped_x, snapped_y, snapped_z

    def add_block(self, x, y, z=0, color=config.COLOR_BLUE):
        sx, sy, sz = self.snap_to_grid(x, y, z)
        # Check if block exists
        for b in self.blocks:
            if b['pos'] == (sx, sy, sz):
                return False # Block already exists
        
        # In a real 3D sense for AR, Z might be UP. 
        # Here we assume Z=0 is floor, and positive Z is UP (away from floor).
        # Adjust based on coordinate system later.
        
        # Simple stack check: If trying to place at Z=0 but there's a block there,
        # we could automatically stack on top.
        # For now, let's keep it simple: strict placement.
        
        self.blocks.append({'pos': (sx, sy, sz), 'color': color})
        return True

    def get_block_at(self, x, y, z, tolerance=None):
        """
        Returns index and block if found at matching grid coordinates.
        tolerance: if set, checks distance (useful for raycast picking).
        """
        sx, sy, sz = self.snap_to_grid(x, y, z)
        
        # If tolerance is used, we check distance to center
        if tolerance:
            for i, b in enumerate(self.blocks):
                bx, by, bz = b['pos']
                dist = math.sqrt((bx - x)**2 + (by - y)**2 + (bz - z)**2)
                if dist < tolerance:
                    return i, b
        else:
            # Strict grid match
            for i, b in enumerate(self.blocks):
                if b['pos'] == (sx, sy, sz):
                    return i, b
        
        return -1, None

    def remove_block(self, x, y, z):
        # Allow removing by raw coord (snap inside)
        idx, _ = self.get_block_at(x, y, z)
        if idx != -1:
            del self.blocks[idx]
            return True
        return False
        
    def move_block(self, index, new_x, new_y, new_z):
        if 0 <= index < len(self.blocks):
            sx, sy, sz = self.snap_to_grid(new_x, new_y, new_z)
            # Check collision with other blocks (excluding itself)
            for i, b in enumerate(self.blocks):
                if i != index and b['pos'] == (sx, sy, sz):
                    return False # Occupied
            
            self.blocks[index]['pos'] = (sx, sy, sz)
            return True
        return False

    def get_blocks(self):
        return self.blocks
