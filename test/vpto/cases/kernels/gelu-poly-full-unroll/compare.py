import numpy as np
import sys

out = np.fromfile('v2.bin', dtype=np.float32)
gold = np.fromfile('golden_v2.bin', dtype=np.float32)
if out.shape != gold.shape:
    print(f'[ERROR] shape mismatch: {out.shape} vs {gold.shape}')
    sys.exit(1)
if not np.allclose(out, gold, rtol=1e-3, atol=1e-3):
    diff = np.abs(out - gold)
    idx = int(diff.argmax())
    print(f'[ERROR] mismatch max diff={diff[idx]} at idx={idx} golden={gold[idx]} out={out[idx]}')
    sys.exit(1)
print('[INFO] compare passed')
