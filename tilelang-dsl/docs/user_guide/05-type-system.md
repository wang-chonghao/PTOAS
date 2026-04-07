## Type System

### Scalar Types

| DSL Type | Description | Bit Width |
|----------|-------------|-----------|
| `pto.i1` | Boolean | 1 |
| `pto.i8` | 8-bit integer | 8 |
| `pto.i16` | 16-bit integer | 16 |
| `pto.i32` | 32-bit integer | 32 |
| `pto.i64` | 64-bit integer | 64 |
| `pto.f16` | Half precision float | 16 |
| `pto.bf16` | Brain float 16 | 16 |
| `pto.f32` | Single precision float | 32 |

Python literals are automatically typed:
- `bool` → `pto.i1`
- `int` → Context-dependent (typically `pto.i32` or `pto.i64`)
- `float` → `pto.f32`

For explicit typing, use type constructors:
```python
x = pto.i32(1024)      # Explicit i32 constant
y: pto.i32 = 1024      # Type annotation
```

### Floating-Point Literal Forms

`pto.f16(...)`, `pto.bf16(...)`, and `pto.f32(...)` accept multiple literal forms.

```python
# Signed numeric literals
a = pto.f16(-1.5)
b = pto.bf16(+2.5)
c = pto.f32(-3.5)

# Special floating-point values
pos_inf = pto.f32("inf")
neg_inf = pto.f32("-inf")
qnan = pto.f32("nan")

# Bit-pattern form (hex string, interpreted by target dtype)
f16_neg_inf = pto.f16("0xFC00")
bf16_neg_inf = pto.bf16("0xFF80")
f32_neg_inf = pto.f32("0xFF800000")
```

Notes:
- Prefer dtype constructors for reduction seeds and boundary values (for example rowmax initialization).
- For float bit-pattern constants, pass a **string** hex literal to the matching dtype constructor.
- Avoid passing raw integer bit-patterns directly into vector broadcast/dup APIs when a floating vector is expected.
- `float(...)` function calls are not part of the TileLang DSL public call surface; use constructor forms above.

### Vector Register Type

Vector registers have fixed 256-byte width:

```python
v_f32 = pto.vreg(pto.f32)  # !pto.vreg<64xf32>
v_f16 = pto.vreg(pto.f16)  # !pto.vreg<128xf16>
v_i8 = pto.vreg(pto.i8)    # !pto.vreg<256xi8>
```

`pto.vreg(dtype)` only takes the element type. The frontend infers the element count automatically from the fixed 256-byte register width:

- `pto.f32` → `!pto.vreg<64xf32>`
- `pto.f16` → `!pto.vreg<128xf16>`
- `pto.bf16` → `!pto.vreg<128xbf16>`
- `pto.i32` → `!pto.vreg<64xi32>`
- `pto.i16` → `!pto.vreg<128xi16>`
- `pto.i8` → `!pto.vreg<256xi8>`

Constraint: `element_count × bitwidth(element_type) = 2048`

Use `pto.elements_per_vreg(dtype)` when you need the inferred element count explicitly:

```python
v_dtype = pto.vreg(pto.f32)
lanes0 = v_dtype.elements_per_vreg       # 64
lanes1 = pto.elements_per_vreg(pto.f32)  # 64
```

Current TileLang DSL v1 vector lowering supports `i8`, `i16`, `i32`, `f16`, `bf16`, and `f32` element types.

### Typed Masks

Masks are typed by their bit granularity:

| DSL Type | VPTO Type | Description |
|----------|-----------|-------------|
| `pto.mask_b8` | `!pto.mask<b8>` | 8-bit granularity mask |
| `pto.mask_b16` | `!pto.mask<b16>` | 16-bit granularity mask |
| `pto.mask_b32` | `!pto.mask<b32>` | 32-bit granularity mask |

```python
mask_ty = pto.mask_b32
mask: pto.mask_b32 = pto.make_mask(pto.f32, PAT.ALL)
```

Mask operations must match the vector element family:
- `f32` vectors use `mask_b32`
- `f16` vectors use `mask_b16`
- `i8` vectors use `mask_b8`

```python
# Correct: f32 vector with b32 mask
mask32 = pto.make_mask(pto.f32, PAT.ALL)
vec_f32 = pto.vlds(ptr, offset)
out = pto.vabs(vec_f32, mask32)

# Error: mismatched mask granularity
mask16 = pto.make_mask(pto.f16, PAT.ALL)
out = pto.vabs(vec_f32, mask16)  # Type error!
```

### Pointer Types [Advanced Tier]

Pointers combine element type and memory space:

```python
from pto import MemorySpace

ptr_gm = pto.ptr(pto.f32, MemorySpace.GM)    # GM pointer to f32
ptr_ub = pto.ptr(pto.f16, MemorySpace.UB)    # UB pointer to f16
```

The `MemorySpace` enum provides type-safe memory space specification:

| Enum Value | Description |
|------------|-------------|
| `MemorySpace.GM` | Global Memory (off-chip HBM/DDR) |
| `MemorySpace.UB` | Unified Buffer (on-chip SRAM, 256KB) |

This replaces string literals (`MemorySpace.GM`/`MemorySpace.UB`) with compile-time checked enums.

### Pointer Type Aliases [Advanced Tier]

For clarity in API documentation, the following type alias is used:

| Alias | Equivalent Type | Description |
|-------|----------------|-------------|
| `Tile` | `pto.tile(...)` | Tile buffer with layout and configuration |
