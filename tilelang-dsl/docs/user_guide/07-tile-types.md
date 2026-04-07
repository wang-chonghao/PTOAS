### Tile Types

Tile types represent data blocks in memory with layout and configuration information, corresponding to `!pto.tile_buf` in the VPTO IR. Tiles are commonly used as kernel parameters for tiled computations.

#### Tile Type Definition

```python
# Create a tile with shape, element type, and memory space
tile = pto.tile((256, 128), pto.f32, MemorySpace.UB)

# With explicit configuration
config = pto.tile_config(
    b_layout=pto.BLayout.ROW_MAJOR,
    s_layout=pto.SLayout.NONE_BOX,
    s_fractal_size=pto.i32(16),
    pad_value=pto.PadValue.ZERO
)
tile = pto.tile((256, 128), pto.f32, MemorySpace.UB, config=config)

# With valid shape (actual data dimensions within tile)
tile = pto.tile((256, 128), pto.f32, MemorySpace.UB, valid_shape=(240, 120))
```

**Important Notes on Shape and Valid Shape:**
- **Static Shape Requirement**: The `shape` parameter must be a compile-time constant. Tile dimensions are fixed at compilation time and cannot change at runtime.
- **Valid Shape Constraints**: The `valid_shape` parameter can be either static (compile-time constant) or dynamic (determined at runtime). It must be less than or equal to the physical `shape` in each dimension. This allows for variable-sized data within a fixed tile allocation.
- **Default Behavior**: When `valid_shape` is not specified, it defaults to the full `shape`.

#### Tile Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | **Static** full tile dimensions (compile-time constant) |
| `element_type` | `Type` | Element data type (e.g., `pto.f32`) |
| `memory_space` | `MemorySpace` | Memory space (GM, UB, etc.) |
| `valid_shape` | `tuple[int, ...]` | Actual data dimensions within tile (can be static/compile-time or dynamic/runtime). Must be ≤ shape in each dimension. |
| `config` | `TileConfig` | Layout and padding configuration |

#### Tile Configuration

The tile configuration includes layout and padding information:

```python
# Layout enums
pto.BLayout.ROW_MAJOR     # 0: row-major base layout
pto.BLayout.COL_MAJOR     # 1: column-major base layout

pto.SLayout.NONE_BOX      # 0: no secondary layout
pto.SLayout.ROW_MAJOR     # 1: row-major secondary layout  
pto.SLayout.COL_MAJOR     # 2: column-major secondary layout

pto.PadValue.NULL         # 0: no padding
pto.PadValue.ZERO         # 1: zero padding
pto.PadValue.MAX          # 2: maximum value padding
pto.PadValue.MIN          # 3: minimum value padding
```

#### Tile Shape Concepts

- **Static Physical Shape**: The `shape` parameter represents the **static physical dimensions** of the tile allocated in memory. This must be a **compile-time constant** because tile memory allocation is fixed during compilation. The shape determines the total memory footprint and cannot change at runtime.

- **Valid Shape**: The `valid_shape` parameter represents the logical dimensions of actual data within the tile. It can be either **static** (compile-time constant) or **dynamic** (determined at runtime). It must be less than or equal to the physical `shape` in each dimension. When `valid_shape` is not specified, it defaults to the full `shape`.

- **Key Distinction**:
  - `shape`: **Static, compile-time** - Fixed tile allocation
  - `valid_shape`: **Static or Dynamic** - Actual data region (must be ≤ shape)

- **Constraints**:
  - `valid_shape[i] ≤ shape[i]` for each dimension i
  - `shape` must be compile-time constants
  - `valid_shape` can be compile-time constants or runtime values

- **Use Cases**:
  - Fixed-size tile buffers with variable data (e.g., batch processing with different input sizes)
  - Padding scenarios where physical allocation is larger than actual data
  - Partial tile utilization in tiled algorithms

- **Fractal Layout**: The `s_fractal_size` in tile configuration specifies the size of fractal blocks for secondary layout. This is used for optimized memory access patterns in matrix operations.

- **Padding Behavior**: The `pad_value` determines how out-of-bounds accesses are handled when reading beyond `valid_shape` but within `shape`. Padding values are used for accesses in the padded region (between valid_shape and shape).

> **⚠️ Important: Shape Constraints**
> 
> The tile `shape` must be **compile-time constants**. `valid_shape` can be compile-time constants or determined at runtime, but must satisfy `valid_shape[i] ≤ shape[i]` for all dimensions i.

### Tile Operations

#### Basic Access Operations

```python
# Get tile properties
shape = tile.shape                    # (256, 128)
elem_type = tile.element_type         # pto.f32
mem_space = tile.memory_space         # MemorySpace.UB
valid_shape = tile.valid_shape        # (240, 120) or same as shape

# Get configuration properties
config = tile.config
b_layout = config.b_layout            # pto.BLayout.ROW_MAJOR
s_layout = config.s_layout            # pto.SLayout.NONE_BOX
s_fractal = config.s_fractal_size     # pto.i32(16)
pad = config.pad_value                # pto.PadValue.ZERO

# Dynamic properties
rank = tile.rank                      # 2
num_elements = tile.num_elements      # 32768 (256 * 128)
valid_elements = tile.valid_elements  # 28800 (240 * 120)
```

#### Layout and Stride Queries

```python
# Get layout descriptors
layout_desc = tile.layout_descriptor  # Returns layout description object

# Get strides (in elements)
strides = tile.strides                # (128, 1) for row-major 256x128

# Get byte strides
byte_strides = tile.byte_strides      # (512, 4) for f32 row-major

# Get base offset (in bytes)
offset = tile.offset                  # pto.i64(0) or specified offset
```

#### Conversion Operations

**Basic Mode Syntax**: Use tile element-indexing directly in vector operations:
```python
# 2D tile indexing
vec = pto.vlds(tile[row, col:])    
pto.vsts(vec, tile[row, col:], mask)

# 1D tile indexing  
vec = pto.vlds(tile[start:])
pto.vsts(vec, tile[start:], mask)
```

**Advanced Mode Syntax**: Convert tiles to typed pointers for byte-offset operations:
```python
# Convert tile to pointer
ptr = tile.as_ptr()                # Returns pto.ptr(pto.f32, MemorySpace.UB)

# Use pointer with byte offset
vec = pto.vlds(ptr, offset)
pto.vsts(vec, ptr, offset, mask)
```

**Tile Manipulation Operations**:
```python
# Extract slice of tile
slice_tile = tile.slice((0, 0), (64, 128))  # 64x128 slice from top-left corner

# Reshape tile (logical reshape, no data movement)
reshaped = tile.reshape((32768,))     # 1D reshape of 256x128 tile
```

#### Kernel Parameter Usage

```python
@pto.vkernel(target="a5", op="scale", dtypes=[(pto.AnyFloat, pto.AnyFloat)], priority=10)
def tiled_kernel(
    input_tile: pto.Tile,              # Tile parameter
    output_tile: pto.Tile,             # Another tile parameter
    scale: pto.f32
):
    # Tiles can be used directly in vector operations (no explicit conversion needed)
    all_mask = pto.make_mask(pto.f32, PAT.ALL)
    for i in range(0, 256, 64):
        # tile element-indexing syntax for basic mode vector operations
        vec = pto.vlds(input_tile[i, 0:])        # Load from row i, columns 0 to vector_lanes-1
        scaled = pto.vmuls(vec, scale, all_mask)
        pto.vsts(scaled, output_tile[i, 0:], all_mask)  # Store to same position
```

#### Tile Creation from Existing Buffers

```python
# Create tile from existing pointer with shape
ptr = pto.castptr(0, pto.ptr(pto.f32, MemorySpace.UB))
tile = pto.tile_from_ptr(ptr, (256, 128), pto.f32)

# Create tile with explicit stride
tile = pto.tile_with_strides((256, 128), pto.f32, MemorySpace.UB, 
                             strides=(256, 1))  # Column-major strides
```

