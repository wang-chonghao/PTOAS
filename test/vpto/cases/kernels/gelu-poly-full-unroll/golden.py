import numpy as np

rng = np.random.default_rng(20260630)
x = rng.uniform(-8.0, 8.0, size=(16, 64)).astype(np.float32)
v1 = x * np.float32(0.5)
v2 = x * np.float32(0.7071)
v2 = np.minimum(v2, np.float32(3.92))
v2 = np.maximum(v2, np.float32(-3.92))
v3 = v2 * v2
v4 = v3 * np.float32(0.5344)
v4 = v4 + np.float32(7.5517)
v4 = v4 * v3
v4 = v4 + np.float32(101.62809)
v4 = v4 * v3
v4 = v4 + np.float32(1393.8015)
v4 = v4 * v3
v4 = v4 + np.float32(5063.7915)
v4 = v4 * v3
v4 = v4 + np.float32(29639.3848)
v4 = v4 * v2
v5 = v3 + np.float32(31.2128582)
v5 = v5 * v3
v5 = v5 + np.float32(308.569641)
v5 = v5 * v3
v5 = v5 + np.float32(3023.12476)
v5 = v5 * v3
v5 = v5 + np.float32(14243.3662)
v5 = v5 * v3
v5 = v5 + np.float32(26267.2246)
v5 = v4 / v5
v5 = v5 + np.float32(1.0)
y = v5 * v1
x.astype(np.float32).tofile('v1.bin')
y.astype(np.float32).tofile('golden_v2.bin')
