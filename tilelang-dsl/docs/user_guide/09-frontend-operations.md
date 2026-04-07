
### Frontend-only Authoring Operations

Operations in this family affect descriptor construction and code generation
shape. They are consumed by the frontend and do not correspond to runtime VPTO
instructions by themselves.

#### `pto.constexpr(value: bool) -> bool`

**Description**: Compile-time conditional construct for kernel specialization. Marks a boolean expression for evaluation during descriptor materialization, enabling branch elimination based on static compile-time information.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `bool` | Boolean expression that must be evaluable at compile time. |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `bool` | A frontend-only compile-time boolean used to guard `if` statements. |

**Behavior**:
- Evaluated during kernel descriptor materialization before semantic analysis and lowering.
- When used in `if pto.constexpr(...):` statements, only the selected branch is retained; the other branch is discarded entirely.
- If the condition cannot be proven static, descriptor materialization fails with a frontend diagnostic.
- Does not generate runtime control flow or value merging logic.

**Examples**:
```python
# Specialize based on element size
dtype = dst.element_type
elem_bytes = pto.bytewidth(dtype)

if pto.constexpr(elem_bytes == 2):
    # Specialized path for 16-bit types (f16/bf16)
    ...
else:
    # Fallback path for other types
    ...
```

```python
# Specialize based on tile shape
rows, cols = dst.shape

if pto.constexpr(rows == 1 and cols == 16):
    # Fast path for specific tile configuration
    ...
```

**Constraints**:
- `pto.constexpr` is a frontend-only authoring construct with no runtime representation.
- The condition must be statically evaluable from descriptor-time information (data types, tile shapes, literals, etc.).
- For kernel-level specialization, prefer `constraints=[...]` and `pto.select_kernel(...)`.
- See [Compile-time Specialization with `pto.constexpr`](04-template-kernels.md#compile-time-specialization-with-ptoconstexpr) for detailed usage guidelines.

### Type Query Operations

Operations for querying type properties.

#### `pto.bytewidth(dtype: Type) -> pto.i32`

**Description**: Returns the size in bytes of a single element of the given data type.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `Type` | Data type (e.g., `pto.f32`, `pto.f16`, `pto.i8`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `size` | `pto.i32` | Element size in bytes |

**Example**:
```python
f32_size = pto.bytewidth(pto.f32)  # Returns 4
f16_size = pto.bytewidth(pto.f16)  # Returns 2
i8_size = pto.bytewidth(pto.i8)    # Returns 1
```

**Common Use Case**: Calculate byte offsets for memory access:
```python
element_type = pto.f32
byte_offset = index * pto.bytewidth(element_type)
```

#### `pto.elements_per_vreg(dtype: Type) -> pto.i32`

**Description**: Returns the number of elements per vector register for a given element type, based on the hardware vector register size (256 bytes). This function computes `256 // bytewidth(dtype)`, which represents the maximum number of elements of the given type that can fit in a single vector register. Useful for determining vector width and loop stride calculations.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `Type` | Data type (e.g., `pto.f32`, `pto.f16`, `pto.i8`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `elems` | `pto.i32` | Number of elements per vector register for the given element type |

**Example**:
```python
f32_elems_per_vreg = pto.elements_per_vreg(pto.f32)  # Returns 64 (256 / 4)
f16_elems_per_vreg = pto.elements_per_vreg(pto.f16)  # Returns 128 (256 / 2)
i8_elems_per_vreg = pto.elements_per_vreg(pto.i8)    # Returns 256 (256 / 1)
```

**Common Use Case**: Loop stride calculation for vector operations:
```python
dtype = pto.f32
elems_per_vreg = pto.elements_per_vreg(dtype)  # Returns 64 for f32
for col in range(0, cols, elems_per_vreg):
    # Load/store vectors of 'elems_per_vreg' elements
    pass
```

**Relationship with `pto.bytewidth`**:
```python
# The relationship between bytewidth and elements per vector register:
elems = 256 // pto.bytewidth(dtype)
# This is equivalent to:
elems = pto.elements_per_vreg(dtype)
```

### Pointer Construction [Advanced Tier]

Operations for creating and manipulating typed pointers.

#### `pto.castptr(offset: pto.i64, ptr_type: Type) -> PtrType`

**Description**: Creates a typed pointer from an integer address, a memref-backed address value, or another typed pointer in the same memory space.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `offset` | `pto.i64` / address-like value | Integer address, memref-backed address value, or existing pointer |
| `ptr_type` | `Type` | Target pointer type (e.g., `pto.ptr(pto.f32, MemorySpace.GM)`) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `ptr` | `PtrType` | Typed pointer value |

**Example**:
```python
ub_ptr = pto.castptr(0, pto.ptr(pto.f32, MemorySpace.UB))
```

`TensorView.as_ptr()` and `Tile.as_ptr()` remain the preferred high-level APIs. They lower directly to address-extraction intrinsics (`pto.tensor_view_addr` / `pto.tile_buf_addr`) with pointer result types, while tile slice / buffer-view authoring paths continue to materialize memref results from the same intrinsics.

#### `pto.addptr(ptr: PtrType, offset: pto.i64) -> PtrType`

**Description**: Adds an element offset to an existing pointer. The offset is counted in elements, not bytes.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Source pointer |
| `offset` | `pto.i64` | Element offset to add (counted in elements, not bytes) |

**Returns**:
| Return Value | Type | Description |
|--------------|------|-------------|
| `new_ptr` | `PtrType` | Pointer with element offset applied |

**Example**:
```python
# Advance pointer by 1024 f32 elements (not bytes)
next_ptr = pto.addptr(ub_ptr, 1024)
```

