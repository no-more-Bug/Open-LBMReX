"""Write exact native cell-plane arrays for the tests-owned pvpython consumers."""
from pathlib import Path
import numpy as np

def dump_planes(directory, name, fields, mask, spacing, dimensions, **metadata):
    """Save exact native cell slices for pvpython, without resampling or smoothing."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    depth = mask.shape[0]*spacing[2]
    if dimensions == 2 or depth < .5:
        position = depth/2
        planes = [('xy', int(position/spacing[2]), position)]
    else:
        planes = [
        ('xy', min(int(0.5001/spacing[2]), mask.shape[0]-1), 0.5001),
            ('xz', min(int(0.5001/spacing[1]), mask.shape[1]-1), 0.5001)]
    for plane, index, position in planes:
        axis = 0 if plane == 'xy' else 1
        selected = {key: np.take(value, index, axis=axis) for key, value in fields.items()}
        np.savez_compressed(directory/f'{name}-{plane}.npz', **selected,
            valid=np.take(mask, index, axis=axis), spacing=np.array(spacing), plane=plane,
            slice_coordinate=position, slice_index=index, **metadata)
