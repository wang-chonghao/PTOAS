# 11. Flash Attention Complete Walkthrough

This chapter walks through `examplesflash_attention_sketch.py` layer by layer, tracing a complete flash attention implementation from the user-facing Python wrapper down to hardware-bound sub-kernels. Every API discussed in Chapters 1–10 appears in context here.

The sketch computes **online-softmax flash attention** for one `(batch, head)` slice per launch instance. It partitions Q into blocks along the sequence dimension, iterates over KV blocks for each Q block, and maintains rolling softmax state across KV iterations.

## 11.1 Architecture overview

```
flash_attention(...)           L0  user-facing wrapper
  └─ @pto.jit flash_attention_kernel
       ├─ Tile Ops                 tile.load / tile.store at the GM↔UB boundary
       └─ @pto.ukernel  kv_block_process
            ├─ @pto.simt   materialize_tile_bounds
            ├─ @pto.cube   qk_matmul
            ├─ @pto.simd   online_softmax_rows
            ├─ @pto.cube   pv_matmul
            └─ @pto.simt   blend_output_rows
```

The dataflow for one KV block:

```
ukernel loads K/V block and sequences the pipeline
       │
       ├─ cube:  Q + K  ───────────────► S
       ├─ simd:  S + (m_prev, l_prev) ─► P, (m_next, l_next), alpha, beta
       ├─ cube:  P + V  ───────────────► PV
       └─ simt:  (o_prev, PV, alpha, beta) ─► o_next

After each KV block:
  (m_prev, l_prev, o_prev) := (m_next, l_next, o_next)
```

## 11.2 L0 — Python wrapper

```python
def flash_attention(Q, K, V, *, O=None, causal=False,
                    block_q=128, block_kv=128, stream=None):
    if O is None:
        O = pto.empty_like(Q)

    batch, seq_q, heads, dim = Q.shape
    _, seq_k, _, _ = K.shape

    compiled = flash_attention_kernel.compile(
        BLOCK_Q=block_q, BLOCK_KV=block_kv, CAUSAL=causal,
    )
    compiled[batch * heads, stream](Q, K, V, O)
    return O
```

This is plain Python — no PTO types, no IR. It handles ergonomic runtime concerns:

- **Output allocation**: `pto.empty_like(Q)` when the caller doesn't provide one.
- **Shape extraction**: reads `batch`, `seq_q`, `heads`, `dim` from the framework tensors.
- **Compile + launch**: `flash_attention_kernel.compile(...)` JIT-compiles the kernel with the given constexpr parameters, then launches it with a `[batch * heads]` grid — one block per `(batch, head)` slice.

L0 knows nothing about tiles, UB, or pipelines. It is the boundary between the user's tensor world and the PTO device world.

## 11.3 L1 — `@pto.jit` kernel entry

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"flash_attention_kernel","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"CAUSAL":false,"NUM_STAGES":2}} -->
```python
@pto.jit(target="a5")
def flash_attention_kernel(
    Q: pto.tensor_spec(rank=4, dtype=pto.f32),
    K: pto.tensor_spec(rank=4, dtype=pto.f32),
    V: pto.tensor_spec(rank=4, dtype=pto.f32),
    O: pto.tensor_spec(rank=4, dtype=pto.f32),
    *,
    BLOCK_Q: pto.constexpr = 128,
    BLOCK_KV: pto.constexpr = 128,
    CAUSAL: pto.constexpr = False,
    NUM_STAGES: pto.constexpr = 2,
):
    # Walkthrough body omitted in this signature overview.
    return
```

The `@pto.jit` decorator marks the compile + launch boundary. Inputs are Python-native tensors; outputs are written in-place to `O`. Keyword-only `constexpr` parameters (`BLOCK_Q`, `BLOCK_KV`, `CAUSAL`) are baked at compile time.

### 11.3.1 TensorView construction

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_tensor_views","symbol":"flash_attention_l1_tensor_views_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"CAUSAL":false,"NUM_STAGES":2}} -->
```python
q_view = pto.make_tensor_view(Q, shape=[batch, seq_q, heads, dim],
                              strides=Q.strides)
k_view = pto.make_tensor_view(K, shape=[batch, seq_k, heads, dim],
                              strides=K.strides)
v_view = pto.make_tensor_view(V, shape=[batch, seq_k, heads, dim],
                              strides=V.strides)
o_view = pto.make_tensor_view(O, shape=[batch, seq_q, heads, dim],
                              strides=O.strides)
```

`make_tensor_view` wraps each framework tensor with a PTO TensorView descriptor — a GM pointer paired with shape and stride metadata. These descriptors are what the rest of the kernel uses to address global memory. No data moves yet.

### 11.3.2 SPMD launch contract

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_tensor_views","symbol":"flash_attention_l1_tensor_views_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"CAUSAL":false,"NUM_STAGES":2}} -->
```python
block_idx = pto.get_block_idx()
block_num = pto.get_block_num()
subblock_idx = pto.get_subblock_idx()
subblock_num = pto.get_subblock_num()

batch_idx = block_idx // heads
head_idx = block_idx % heads
```

The launch grid is `[batch * heads]`. Each block computes one `(batch, head)` slice. `get_block_idx()` returns the current block's linear index; dividing by `heads` recovers the batch and head indices.

### 11.3.3 Per-head view partitioning

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_partitions","symbol":"flash_attention_l1_partitions_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"CAUSAL":false,"NUM_STAGES":2}} -->
```python
q_head = pto.partition_view(
    q_view,
    offsets=[batch_idx, 0, head_idx, 0],
    sizes=[1, seq_q, 1, dim],
)
k_head = pto.partition_view(
    k_view,
    offsets=[batch_idx, 0, head_idx, 0],
    sizes=[1, seq_k, 1, dim],
)
v_head = pto.partition_view(
    v_view,
    offsets=[batch_idx, 0, head_idx, 0],
    sizes=[1, seq_k, 1, dim],
)
o_head = pto.partition_view(
    o_view,
    offsets=[batch_idx, 0, head_idx, 0],
    sizes=[1, seq_q, 1, dim],
)
```

There is no dedicated `select_head_view` public helper anymore. Each `(batch, head)` working set is sliced from the 4D TensorView with the standard `partition_view(...)` surface, and further logical slicing composes on top of the same primitive.

### 11.3.4 Tile allocation

Three categories of tiles are allocated:

**MAT-backed bridge tiles** — the logical Q/K/V/P blocks that feed the cube path:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_tiles","symbol":"flash_attention_l1_tiles_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"HEAD_DIM":128}} -->
```python
q_mat = pto.alloc_tile(
    shape=[Br, D],
    dtype=pto.f32,
    memory_space=pto.MemorySpace.MAT,
    valid_shape=[full_br, dim],
    blayout="ColMajor",
    slayout="RowMajor",
)
k_mat = pto.alloc_tile(
    shape=[Bc, D],
    dtype=pto.f32,
    memory_space=pto.MemorySpace.MAT,
    valid_shape=[full_bc, dim],
    blayout="ColMajor",
    slayout="RowMajor",
)
v_mat = pto.alloc_tile(
    shape=[Bc, D],
    dtype=pto.f32,
    memory_space=pto.MemorySpace.MAT,
    valid_shape=[full_bc, dim],
    blayout="ColMajor",
    slayout="RowMajor",
)
p_mat = pto.alloc_tile(
    shape=[Br, Bc],
    dtype=pto.f32,
    memory_space=pto.MemorySpace.MAT,
    valid_shape=[full_br, full_bc],
    blayout="ColMajor",
    slayout="RowMajor",
)

o_prev_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
o_next_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
m_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
m_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
l_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
l_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")

s_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[full_br, full_bc])
p_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[full_br, full_bc])
pv_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
alpha_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
beta_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
```

The walkthrough keeps Q/K/V/P on the MAT path so the cube sub-kernels consume the same tile objects that the L1 schedule owns. Runtime tails still live in `valid_shape`; the physical tile shapes stay static.

**UB-resident state and scratch tiles** — the online-softmax state plus intermediate outputs:

```python
o_prev_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
o_next_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
m_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
m_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
l_prev_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
l_next_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")

s_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[full_br, full_bc])
p_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, valid_shape=[full_br, full_bc])
pv_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32, valid_shape=[full_br, dim])
alpha_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
beta_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, valid_shape=[full_br, one], blayout="ColMajor")
```

The online-softmax algorithm requires **ping-pong state tiles**: `m_prev`/`m_next`, `l_prev`/`l_next`, `o_prev`/`o_next`. After each KV block, `next` becomes `prev` for the following iteration.

**Cube-local scratch tiles** — allocated in specific memory spaces:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_tiles","symbol":"flash_attention_l1_tiles_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"HEAD_DIM":128}} -->
```python
q_l0a = pto.alloc_tile(shape=[Br, D], dtype=pto.f32,
                       memory_space=pto.MemorySpace.LEFT, valid_shape=[full_br, dim])
p_l0a = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32,
                       memory_space=pto.MemorySpace.LEFT, valid_shape=[full_br, full_bc])
rhs_l0b = pto.alloc_tile(shape=[Bc, D], dtype=pto.f32,
                         memory_space=pto.MemorySpace.RIGHT, valid_shape=[full_bc, dim])
qk_acc_tile = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32,
                             memory_space=pto.MemorySpace.ACC, valid_shape=[full_br, full_bc])
pv_acc_tile = pto.alloc_tile(shape=[Br, D], dtype=pto.f32,
                             memory_space=pto.MemorySpace.ACC, valid_shape=[full_br, dim])
```

Cube scratch tiles are NOT UB buffers. `LEFT`, `RIGHT`, and `ACC` are distinct hardware memory spaces inside the Cube unit. They serve as staging for matrix operands and accumulators.

### 11.3.5 SIMT metadata buffer

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_tiles","symbol":"flash_attention_l1_tiles_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128}} -->
```python
meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
meta_ptr = meta_tile.as_ptr()
```

A small UB tile stores three scalar loop bounds (`row_start`, `row_stop`, `valid_cols`). `meta_tile.as_ptr()` materializes a typed UB pointer into it, which is passed to the ukernel as scalar control metadata.

Notice that the row-wise softmax state tiles (`m_*`, `l_*`, `alpha_tile`,
`beta_tile`) are authored as `blayout="ColMajor"`. This is the intended public
surface for logical column vectors; it avoids forcing users to manufacture a
row-major padded physical width just to satisfy row-byte alignment.

### 11.3.6 Outer Q loop + inner KV loop

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.l1_loop_body","symbol":"flash_attention_l1_loop_body_probe","compile":{"BLOCK_Q":128,"BLOCK_KV":128,"HEAD_DIM":128,"CAUSAL":false,"NUM_STAGES":2}} -->
```python
with pto.for_(0, q_blocks, step=1) as qi:
    q_rows = _block_valid_extent(seq_q, qi, Br)
    q_part = pto.partition_view(q_head, offsets=[0, qi * Br, 0, 0],
                                sizes=[1, q_rows, 1, dim])
    o_part = pto.partition_view(o_head, offsets=[0, qi * Br, 0, 0],
                                sizes=[1, q_rows, 1, dim])

    q_mat.valid_shape = [q_rows, dim]
    o_prev_tile.valid_shape = [q_rows, dim]
    o_next_tile.valid_shape = [q_rows, dim]
    m_prev_tile.valid_shape = [q_rows, one]
    m_next_tile.valid_shape = [q_rows, one]
    l_prev_tile.valid_shape = [q_rows, one]
    l_next_tile.valid_shape = [q_rows, one]
    alpha_tile.valid_shape = [q_rows, one]
    beta_tile.valid_shape = [q_rows, one]
    p_mat.valid_shape = [q_rows, full_bc]
    pv_tile.valid_shape = [q_rows, dim]
    q_l0a.valid_shape = [q_rows, dim]

    pto.tile.load(q_part, q_mat)

    m_prev_tile.fill(float("-inf"))
    l_prev_tile.fill(0.0)
    o_prev_tile.fill(0.0)

    kv_loop = pto.for_(0, kv_blocks, step=1).carry(
        m=m_prev_tile, l=l_prev_tile, o=o_prev_tile,
    )
    with kv_loop:
        kj = kv_loop.iv
        m_cur = kv_loop.m
        l_cur = kv_loop.l
        o_cur = kv_loop.o
        kv_rows = _block_valid_extent(seq_k, kj, Bc)
        k_part = pto.partition_view(k_head,
                    offsets=[0, kj * Bc, 0, 0], sizes=[1, kv_rows, 1, dim])
        v_part = pto.partition_view(v_head,
                    offsets=[0, kj * Bc, 0, 0], sizes=[1, kv_rows, 1, dim])

        k_mat.valid_shape = [kv_rows, dim]
        v_mat.valid_shape = [kv_rows, dim]
        s_tile.valid_shape = [q_rows, kv_rows]
        p_tile.valid_shape = [q_rows, kv_rows]
        p_mat.valid_shape = [q_rows, kv_rows]
        pv_tile.valid_shape = [q_rows, dim]
        p_l0a.valid_shape = [q_rows, kv_rows]
        rhs_l0b.valid_shape = [kv_rows, dim]
        qk_acc_tile.valid_shape = [q_rows, kv_rows]
        pv_acc_tile.valid_shape = [q_rows, dim]

        kv_block_process(
            q_mat, k_part, v_part, k_mat, v_mat,
            o_cur, o_next_tile,
            m_cur, l_cur, m_next_tile, l_next_tile,
            s_tile, p_tile, p_mat, pv_tile,
            alpha_tile, beta_tile,
            q_l0a, p_l0a, rhs_l0b,
            qk_acc_tile, pv_acc_tile,
            meta_ptr,
        )

        kv_loop.update(m=m_next_tile, l=l_next_tile, o=o_next_tile)

    o_final_tile = kv_loop.final("o")
    pto.tile.store(o_final_tile, o_part)
```

Key points:

- **Static physical shape, dynamic valid extent**: `alloc_tile(shape=...)` stays constexpr. Tail handling is expressed by updating `valid_shape` before each block load and sub-kernel call.
- **`tile.load` at the L1 boundary**: Q is loaded once per Q block using a tile op into the MAT-backed bridge tile `q_mat`. The compiler auto-inserts the necessary `set_flag`/`wait_flag` pairs.
- **State initialization**: `fill(float("-inf"))` and `fill(0.0)` initialize the online-softmax accumulators before the first KV block.
- **Carry state**: the inner `kv_loop` carries three ping-pong tiles (`m`, `l`, `o`) across iterations using `.carry(...)` / `.update(...)` / `.final(...)`. After each KV block, the loop updates the carried values to the `_next` tiles. After the loop, `.final("o")` extracts the final output accumulator.
- **`tile.store` at the L1 boundary**: writes the final result for this Q block back to GM.

## 11.4 L2 — `@pto.ukernel`

```python
@pto.ukernel
def kv_block_process(
    q_mat, k_part, v_part, k_mat, v_mat,
    o_prev_tile, o_next_tile,
    m_prev_tile, l_prev_tile, m_next_tile, l_next_tile,
    s_tile, p_tile, p_mat, pv_tile,
    alpha_tile, beta_tile,
    q_l0a, p_l0a, rhs_l0b,
    qk_acc_tile, pv_acc_tile,
    meta_ptr,
):
```

The ukernel processes one KV block against an already-loaded Q tile. It owns the execution sandwich:

### Phase 0 — Stage K/V data

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.ukernel_phase","symbol":"flash_attention_ukernel_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
rows = k_mat.valid_shape[0]
cols = k_mat.valid_shape[1]
row_bytes = cols * pto.bytewidth(pto.f32)
gm_row_stride = k_part.strides[0] * pto.bytewidth(pto.f32)
mat_row_stride = k_mat.shape[1] * pto.bytewidth(pto.f32)
pto.mte_load(k_part.as_ptr(), k_mat.as_ptr(), 0, row_bytes,
             nburst=(rows, gm_row_stride, mat_row_stride))
pto.mte_load(v_part.as_ptr(), v_mat.as_ptr(), 0, row_bytes,
             nburst=(rows, gm_row_stride, mat_row_stride))
pto.pipe_barrier(pto.Pipe.ALL)
```

`mte_load` is the ptr-based GM→MAT DMA wrapper used by this walkthrough. The ukernel passes explicit GM/MAT pointers plus the DMA grouping parameters, and `pipe_barrier(Pipe.ALL)` makes the phase boundary explicit before the cube unit reads `k_mat`/`v_mat`.

### Phase 0b — Materialize loop bounds

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.ukernel_phase","symbol":"flash_attention_ukernel_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
materialize_tile_bounds(meta_ptr,
    q_mat.valid_shape[0],
    k_mat.valid_shape[0])
row_start = scalar.load(meta_ptr + 0)
row_stop  = scalar.load(meta_ptr + 1)
valid_cols = scalar.load(meta_ptr + 2)
```

The SIMT sub-kernel `materialize_tile_bounds` writes `{0, valid_rows, valid_cols}` into the metadata buffer. The ukernel then loads these scalars. They control the row iteration range in subsequent sub-kernels, handling partial tail blocks.

### Phase 1 — `S = Q @ K^T`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.ukernel_phase","symbol":"flash_attention_ukernel_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
qk_matmul(q_mat, k_mat, q_l0a, rhs_l0b, qk_acc_tile, s_tile)
pto.pipe_barrier(pto.Pipe.ALL)
```

Dispatches the cube sub-kernel. `pipe_barrier(Pipe.ALL)` separates the matrix multiply from the subsequent softmax.

### Phase 2 — Online softmax

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.ukernel_phase","symbol":"flash_attention_ukernel_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
online_softmax_rows(
    s_tile, p_tile,
    m_prev_tile, l_prev_tile,
    m_next_tile, l_next_tile,
    alpha_tile, beta_tile,
    row_start, row_stop, valid_cols,
)
pto.pipe_barrier(pto.Pipe.ALL)
```

The simd sub-kernel computes per-row softmax on `S`, updates the running `m`/`l` state, and writes `P`, `alpha`, and `beta`.

### Phase 3 — `PV = P @ V`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.ukernel_phase","symbol":"flash_attention_ukernel_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
pto.tile.mov(p_tile, p_mat)
pto.pipe_barrier(pto.Pipe.ALL)

pv_matmul(p_mat, v_mat, p_l0a, rhs_l0b, pv_acc_tile, pv_tile)
pto.pipe_barrier(pto.Pipe.ALL)
```

The probability tile is first staged onto the MAT path with `pto.tile.mov(p_tile, p_mat)`. Then the second cube dispatch reuses `rhs_l0b` for `V` and `pv_acc_tile` for the accumulator.

### Phase 4 — Blend output

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.ukernel_phase","symbol":"flash_attention_ukernel_phase_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
blend_output_rows(
    o_prev_tile, pv_tile, alpha_tile, beta_tile,
    o_next_tile, row_start, row_stop,
    v_mat.valid_shape[1],
)
pto.pipe_barrier(pto.Pipe.ALL)
```

The simt sub-kernel blends the old output accumulator with the new PV contribution, weighted by `alpha` and `beta`.

### Why the ukernel owns sync

Each `pipe_barrier(Pipe.ALL)` between phases is explicit in the ukernel body. This is intentional: at the L2 micro-instruction level, the user controls pipeline ordering. There is no auto-sync insertion — the ukernel is the single place where the hardware execution sequence is spelled out.

## 11.5 L3a — `@pto.cube` sub-kernels

### `qk_matmul` — `S = Q @ K^T`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.qk_cube_helper","symbol":"flash_attention_qk_cube_helper_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
@pto.cube
def qk_matmul(q_mat, k_mat, q_l0a, k_l0b, s_acc, s_tile):
    m = q_mat.valid_shape[0]
    k = q_mat.valid_shape[1]
    n = k_mat.valid_shape[0]

    pto.mte_l1_l0a(q_mat.as_ptr(), q_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(k_mat.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)
    pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)
    pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)
```

Four cube ops:

1. **`mte_l1_l0a`**: load Q tile from UB into LEFT scratch (`q_l0a`).
2. **`mte_l1_l0b`**: load K tile from UB into RIGHT scratch (`k_l0b`), with `transpose=True` for K^T.
3. **`mad`**: matrix multiply-accumulate — `s_acc = q_l0a @ k_l0b`.
4. **`mte_l0c_ub`**: write the accumulator result to the UB output tile `s_tile`.

The cube kernel does not allocate scratch — the caller (L1) owns scratch lifetime. The cube kernel only expresses dataflow.

### `pv_matmul` — `PV = P @ V`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.pv_cube_helper","symbol":"flash_attention_pv_cube_helper_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
@pto.cube
def pv_matmul(p_mat, v_mat, p_l0a, v_l0b, pv_acc, pv_tile):
    m = p_mat.valid_shape[0]
    k = p_mat.valid_shape[1]
    n = v_mat.valid_shape[1]

    pto.mte_l1_l0a(p_mat.as_ptr(), p_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(v_mat.as_ptr(), v_l0b.as_ptr(), k, n)
    pto.mad(p_l0a.as_ptr(), v_l0b.as_ptr(), pv_acc.as_ptr(), m, n, k)
    pto.mte_l0c_ub(pv_acc.as_ptr(), pv_tile.as_ptr(), m, n, n, n, 0)
```

Structurally identical to `qk_matmul`, but without transposition and with different input/output tiles. The scratch tiles `p_l0a`, `v_l0b`, and `pv_acc` are reused across KV blocks — the caller (L1) allocates them once.

## 11.6 L3b — `@pto.simd` online softmax

```python
@pto.simd
def online_softmax_rows(
    s_tile, p_tile,
    m_prev_tile, l_prev_tile,
    m_next_tile, l_next_tile,
    alpha_tile, beta_tile,
    row_start, row_stop, valid_cols,
):
```

The simd kernel iterates over rows with `pto.for_`, processing one row per iteration:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.online_softmax_loop","symbol":"flash_attention_online_softmax_loop_probe","compile":{"BLOCK":16}} -->
```python
with pto.for_(row_start, row_stop, step=1) as row:
    col_mask = pto.make_mask(pto.f32, valid_cols)

    s_row   = pto.vlds(s_tile[row, 0:])
    m_prev  = scalar.load(m_prev_tile[row, 0])
    l_prev  = scalar.load(l_prev_tile[row, 0])
```

- **Mask creation**: `make_mask(pto.f32, valid_cols)` generates a tail mask for the column dimension. On the last KV block, `valid_cols` may be less than the full block width.
- **Vector load**: `vlds(s_tile[row, 0:])` loads one entire row of `S` from UB into a vector register. The slice syntax `[row, 0:]` selects the full row.
- **Scalar load**: `lds` reads per-row scalars (`m_prev`, `l_prev`) from the state tiles.

### Softmax computation

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.online_softmax_compute","symbol":"flash_attention_online_softmax_compute_probe","compile":{"BLOCK":16}} -->
```python
    row_max   = pto.vcgmax(s_row, col_mask)
    m_next    = scalar.max(m_prev, row_max)

    s_shifted = pto.vsubs(s_row, m_next, col_mask)
    p_row     = pto.vexp(s_shifted, col_mask)

    row_sum   = pto.vcgadd(p_row, col_mask)
    l_scaled  = l_prev * scalar.exp(m_prev - m_next)
    l_next    = l_scaled + row_sum

    alpha = l_scaled / l_next
    beta  = 1.0 / l_next
```

This implements the online-softmax update from the Flash Attention paper:

- `vcgmax` (cross-lane max reduction) finds the row maximum.
- `max(m_prev, m_next)` combines with the running maximum.
- `vsubs` subtracts the scalar `m_next` from every lane (stabilized softmax).
- `vexp` computes `exp(s_shifted)` element-wise.
- `vcgadd` (cross-lane sum reduction) computes the row sum.
- `l_scaled` rescales the previous sum with the running-max correction factor.
- `alpha` and `beta` are the blending coefficients for the output update.

### Store results

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.online_softmax_store","symbol":"flash_attention_online_softmax_store_probe","compile":{"BLOCK":16}} -->
```python
    pto.vsts(p_row, p_tile[row, 0:], col_mask)
    scalar.store(m_next, m_next_tile[row, 0])
    scalar.store(l_next, l_next_tile[row, 0])
    scalar.store(alpha, alpha_tile[row, 0])
    scalar.store(beta, beta_tile[row, 0])
```

- `vsts` stores the vector `p_row` back to UB under the column mask.
- `sts` stores each scalar to its respective UB tile.

**Boundary contract**: vreg values (`s_row`, `p_row`, `row_max`, `row_sum`) never escape the simd kernel. All persistent state is written to UB tiles.

## 11.7 L3c — `@pto.simt` sub-kernels

### `materialize_tile_bounds` — scalar metadata

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.simt_materialize","symbol":"flash_attention_simt_materialize_probe","compile":{}} -->
```python
@pto.simt
def materialize_tile_bounds(meta_ptr, valid_rows, valid_cols):
    scalar.store(0, meta_ptr + 0)
    scalar.store(valid_rows, meta_ptr + 1)
    scalar.store(valid_cols, meta_ptr + 2)
```

Three scalar stores write the loop bounds into the metadata buffer. `meta_ptr` is a typed UB pointer; `+ 0`, `+ 1`, `+ 2` are element offsets into `i32` storage, not byte offsets. This is the simplest sub-kernel in the sketch — it handles scalar control metadata, not vector math.

### `blend_output_rows` — output accumulation

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.simt_blend","symbol":"flash_attention_simt_blend_probe","compile":{"BLOCK":8}} -->
```python
@pto.simt
def blend_output_rows(o_prev_tile, pv_tile, alpha_tile, beta_tile,
                      o_next_tile, row_start, row_stop, valid_dim):
    with pto.for_(row_start, row_stop, step=1) as row:
        alpha = scalar.load(alpha_tile[row, 0])
        beta  = scalar.load(beta_tile[row, 0])

        with pto.for_(0, valid_dim, step=1) as col:
            o_prev = scalar.load(o_prev_tile[row, col])
            pv_val = scalar.load(pv_tile[row, col])
            o_next = alpha * o_prev + beta * pv_val
            scalar.store(o_next, o_next_tile[row, col])
```

This is a scalar element-wise blend over the tile domain:

```
O_next[row, col] = alpha[row] * O_prev[row, col] + beta[row] * PV[row, col]
```

The SIMT kernel walks the tile element by element with nested `pto.for_` loops. Each iteration loads two scalars (`o_prev` and `pv_val`), computes the weighted sum, and stores the result. The `alpha`/`beta` coefficients are per-row (loaded once per row), while the blend is per-element.

**Why SIMT instead of SIMD?** The intent is to contrast with `online_softmax_rows`: softmax is dominated by row-wise vector reductions and exponentials — natural SIMD work. The final blend is a simple linear combination with per-row coefficients — expressing it as explicit scalar work-items makes the per-element access pattern explicit and leaves the compiler free to vectorize or fuse as it sees fit.

### Context manager alternative

For trivial sub-kernels like `materialize_tile_bounds`, a named function is overkill — the context manager form keeps the logic inline where it's used. The inline SIMT scope itself looks like this:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.inline_simt_scope","symbol":"flash_attention_inline_simt_scope_probe","compile":{"BLOCK_Q":16,"BLOCK_KV":16}} -->
```python
with pto.simt():
    scalar.store(0, meta_ptr + 0)
    scalar.store(q_mat.valid_shape[0], meta_ptr + 1)
    scalar.store(k_mat.valid_shape[0], meta_ptr + 2)
```

The `with pto.simt():` block acts as an anonymous inline sub-kernel scope. For 3-line helpers that have no reuse, the context manager avoids the indirection of a separate function. For complex, reusable logic like `online_softmax_rows` or `qk_matmul`, the named decorator form remains the better fit.

## 11.8 Putting it all together: one KV block execution

For one KV block, the full execution sequence is:

| Step | Layer | Operation | Hardware |
|------|-------|-----------|----------|
| 1 | L1 | `tile.load(q_part, q_mat)` | GM → MAT |
| 2 | L2 | `mte_load(k_part.as_ptr(), k_mat.as_ptr(), ...)` | GM → MAT |
| 3 | L2 | `mte_load(v_part.as_ptr(), v_mat.as_ptr(), ...)` | GM → MAT |
| 4 | L2 | `pipe_barrier(Pipe.ALL)` | — |
| 5 | L3c | `materialize_tile_bounds` | SIMT |
| 6 | L3a | `qk_matmul` (mte_l1_l0a, mte_l1_l0b, mad, mte_l0c_ub) | Cube |
| 7 | L2 | `pipe_barrier(Pipe.ALL)` | — |
| 8 | L3b | `online_softmax_rows` (vlds, vcgmax, vexp, vcgadd, vsts, ...) | SIMD |
| 9 | L2 | `pipe_barrier(Pipe.ALL)` | — |
| 10 | L2 | `tile.mov(p_tile, p_mat)` | Tile copy |
| 11 | L2 | `pipe_barrier(Pipe.ALL)` | — |
| 12 | L3a | `pv_matmul` | Cube |
| 13 | L2 | `pipe_barrier(Pipe.ALL)` | — |
| 14 | L3c | `blend_output_rows` | SIMT |
| 15 | L2 | `pipe_barrier(Pipe.ALL)` | — |

After all KV blocks: L1 issues `tile.store(o_final_tile, o_part)` to write the result back to GM.

## 11.9 Design patterns in this sketch

**Ping-pong state for online accumulators**: `m_prev`/`m_next`, `l_prev`/`l_next`, `o_prev`/`o_next` make the state transition explicit. After each KV block, the caller swaps the ping-pong pair (via `kv_loop.update(...)`) rather than aliasing in place.

**Scratch reuse**: `rhs_l0b` serves both `K` (in `qk_matmul`) and `V` (in `pv_matmul`). `pv_acc_tile` reuses the accumulator from QK^T. The caller (L1) allocates once; the ukernel passes them to both cube sub-kernels.

**Tile-level boundary vs micro-instruction boundary**: `tile.load`/`tile.store` appear only in `@pto.jit`. `mte_load` appears only in `@pto.ukernel`, and it is authored in the explicit ptr-based DMA form. This is the key abstraction split: L1 operates on tiles, L2 operates on micro-instructions.

**No vreg across sub-kernel boundaries**: vector registers are local to each `@pto.simd` kernel. Data crosses sub-kernel boundaries through UB tiles — the boundary contract is enforced by the type system.

**L3 invocation flexibility**: This sketch uses the explicit `@pto.ukernel` → L3 path for full control over MTE and sync. For simpler kernels that don't need that control, L3 sub-kernels can be called directly from `@pto.jit` (the compiler handles MTE + sync) or written inline as context managers (`with pto.simd():`, etc.). See Chapter 3 for details.
