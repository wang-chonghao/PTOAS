### TensorView Types

TensorView types represent multi‑dimensional (up to 5D) views into tensors residing in Global Memory (GM). They are used as kernel parameters for describing GM data and support slicing operations to create logical partitions for DMA load/store operations.

### TensorView Type Definition

TensorView types are parameterized by shape (a tuple of up to 5 dimensions) and element type:

```python
# Kernel parameter using TensorView
@pto.vkernel(target="a5", op="custom", dtypes=[(pto.AnyFloat, pto.AnyFloat, pto.AnyFloat)], priority=10)
def tiled_kernel(
    input_tensor: pto.TensorView,   # GM tensor view
    output_tensor: pto.TensorView,  # GM tensor view
    tile_buf: pto.Tile              # UB tile
):
    # Access tensor view properties
    shape = input_tensor.shape           # tuple of dimensions (dynamic or static, up to 5D)
    dtype = input_tensor.element_type    # e.g., pto.f32
    strides = input_tensor.strides       # stride in elements
```

**Important Notes:**
- TensorView is a **read-only descriptor** for GM data (though DMA store operations can write to it)
- Shape can be **static** (compile-time constants) or **dynamic** (determined at runtime)
- Strides are expressed in elements, not bytes
- Memory space is always GM (Global Memory)
- Maximum rank is 5 (PTO ISA right‑aligns lower‑rank shapes to 5D)
- When higher dimensions are 1, a 5D TensorView can be abbreviated to lower‑rank forms. For example, shape `(1,1,64,32,16)` can be written as `(64,32,16)` (3D), and shape `(1,1,1,32,16)` can be written as `(32,16)` (2D).

### TensorView Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Tensor dimensions (supports up to 5 dimensions, right-aligned to 5D in PTO ISA) |
| `element_type` | `Type` | Element data type (e.g., `pto.f32`, `pto.f16`) |
| `strides` | `tuple[int, ...]` | Stride in elements for each dimension |
| `offset` | `pto.i64` | Byte offset from base pointer (internal) |

### Padding Mode Enum

Padding mode controls how out-of-bounds accesses are handled during DMA load/store operations:

| Enum Value | Description |
|------------|-------------|
| `PadMode.PadNull` | No padding (out-of-bounds access is invalid) |
| `PadMode.PadFirstElem` | Pad using the first element of the source |
| `PadMode.PadValue` | Pad using a specified value (requires `pad_value` parameter) |

### Slicing Syntax

TensorView supports Python slicing syntax to create logical partitions:

```python
# Create a partition from a tensor view
partition = tensor_view[dim0_start:dim0_end, dim1_start:dim1_end]

# Example: extract a 16x16 tile from a larger tensor
tile_view = large_tensor[0:16, 0:16]

# Dynamic offsets and sizes
dim0_start = tensor_view.shape[0] // 2
dynamic_partition = tensor_view[dim0_start:tensor_view.shape[0], 4:20]

# Static positive step on dimension 0
stepped_partition = tensor_view[0:32:2, 0:16]

# Right-aligned shorthand on a 5D descriptor:
# if the leading 2 axes are logical singleton dimensions, a 3D-style slice
# maps to the trailing 3 physical axes.
partition_3d = tensor_view[d2_start:d2_end, d3_start:d3_end, d4_start:d4_end]

# Full 5D spelling remains available when needed
partition_5d = tensor_view[
    d0_start:d0_end,
    d1_start:d1_end,
    d2_start:d2_end,
    d3_start:d3_end,
    d4_start:d4_end,
]
```

**Constraints:**
- Slicing returns a new TensorView representing the logical partition
- The partition must be within the original tensor bounds
- When fewer than 5 slice axes are written, they are right-aligned to the
  trailing physical axes of the 5D descriptor
- `stop` must be explicit on all dimensions
- `start` may be static or dynamic
- `step` must be a static positive integer
- Dimension 0 may use `step > 1`
- Dimension 1 must keep `step == 1` (current implementation restriction for DMA operations)

### Alignment Type

The `pto.align` type is used for alignment carrier operations and maps to `!pto.align`.

