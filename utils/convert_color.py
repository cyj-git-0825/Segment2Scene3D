import sys
import numpy as np
from plyfile import PlyData, PlyElement

input_file = sys.argv[1] if len(sys.argv) > 1 else 'splat.ply'
output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.ply', '_rgb.ply')

ply = PlyData.read(input_file)
v = ply['vertex']

C0 = 0.28209479177387814
r = np.clip((0.5 + C0 * v['f_dc_0']) * 255, 0, 255).astype(np.uint8)
g = np.clip((0.5 + C0 * v['f_dc_1']) * 255, 0, 255).astype(np.uint8)
b = np.clip((0.5 + C0 * v['f_dc_2']) * 255, 0, 255).astype(np.uint8)

dtype_new = v.data.dtype.descr + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
new_data = np.empty(len(v.data), dtype=dtype_new)
for name in v.data.dtype.names:
    new_data[name] = v.data[name]
new_data['red'] = r
new_data['green'] = g
new_data['blue'] = b

el = PlyElement.describe(new_data, 'vertex')
PlyData([el], byte_order='<').write(output_file)
print(f"Done! Saved as {output_file}")
