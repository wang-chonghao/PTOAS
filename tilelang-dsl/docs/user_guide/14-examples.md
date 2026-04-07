## Examples

### Template-based Kernel Examples

#### Unified Arithmetic Operations

A single kernel implementing multiple arithmetic operations using templates:

```python
T = pto.TypeVar('T')

@pto.vkernel(
    target="a5",
    ops=["tadd", "tsub", "tmul", "tdiv"],
    dtypes=[(T, T, T)],
    advanced=True,
    templates={
        "core": {
            "tadd": "vadd",
            "tsub": "vsub", 
            "tmul": "vmul",
            "tdiv": "vdiv",
        }
    }
)
def elementwise_arithmetic(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    """Single implementation for four arithmetic operations."""
    dtype = dst.element_type
    rows, cols = dst.valid_shape
    
    for row in range(0, rows, 1):
        remained = cols
        for col in range(0, cols, pto.elements_per_vreg(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            out = pto.tpl("core", lhs, rhs, mask)
            pto.vsts(out, dst[row, col:], mask)
```

#### Multiple Templates with Postprocess

Kernel using separate templates for arithmetic and postprocess operations:

```python
@pto.vkernel(
    target="a5",
    ops=["add_relu", "sub_relu", "add_abs", "sub_abs"],
    dtypes=[(T, T, T)],
    templates={
        "arithmetic": {
            "add_relu": "vadd",
            "sub_relu": "vsub",
            "add_abs": "vadd",
            "sub_abs": "vsub",
        },
        "postprocess": {
            "add_relu": "vrelu",
            "sub_relu": "vrelu",
            "add_abs": "vabs",
            "sub_abs": "vabs",
        }
    }
)
def elementwise_with_postprocess(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    dtype = dst.element_type
    rows, cols = dst.valid_shape
    
    for row in range(0, rows, 1):
        remained = cols
        for col in range(0, cols, pto.elements_per_vreg(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            
            # Use arithmetic template
            arith_result = pto.tpl("arithmetic", lhs, rhs, mask)
            
            # Apply postprocess template
            activated = pto.tpl("postprocess", arith_result, mask)
            
            pto.vsts(activated, dst[row, col:], mask)
```

#### Compile-time Substitution

Template substitution happens before semantic analysis and lowering:

```python
selected = pto.select_kernel("a5", "tadd", (ptype, ptype, ptype))
# frontend resolves:
# pto.tpl("core", lhs, rhs, mask)
# into:
# pto.vadd(lhs, rhs, mask)
```

#### Benefits of Template-based Authoring

1. **Code Reuse**: Single implementation serves multiple operations
2. **Maintenance**: Bug fixes and optimizations apply to all related operations
3. **Consistency**: Ensures uniform behavior across operation families
4. **Reduced Boilerplate**: Eliminates duplicate control flow and data movement code
5. **Type Safety**: Type variables ensure consistent operand types

### Simple Vector Copy

```python
@pto.vkernel(...)
def vector_copy(src: pto.Tile, dst: pto.Tile):
    all_mask: pto.mask_b32 = pto.make_mask(pto.f32, PAT.ALL)
    for offset in range(0, 256, 64):
        vec = pto.vlds(src, offset)
        pto.vsts(vec, dst, offset, all_mask)
```

### Conditional Computation

```python
@pto.vkernel(...)
def conditional_scale(src: pto.ptr(pto.f32, MemorySpace.GM),
                      dst: pto.ptr(pto.f32, MemorySpace.GM),
                      threshold: pto.f32):
    # ... setup ...

    with pto.strict_vecscope(ub_in, ub_out, threshold) as (vin, vout, thresh):
        for i in range(0, 1024, 64):
            vec = pto.vlds(vin, i)

            # Compare with threshold
            mask = pto.pge_b32(vec, thresh)

            # Scale values above threshold
            scaled = pto.vmuls(vec, pto.f32(2.0), mask)

            # Keep original values below threshold
            result = pto.vsel(scaled, vec, mask)

            pto.vsts(result, vout, i, all_mask)
```

### Loop with Carry

```python
@pto.vkernel(...)
def prefix_sum(src: pto.ptr(pto.i32, MemorySpace.UB),
               dst: pto.ptr(pto.i32, MemorySpace.UB)):
    all_mask = pto.make_mask(pto.i32, PAT.ALL)
    carry = all_mask

    for i in range(0, 256, 64):
        vec = pto.vlds(src, i)
        result, carry = pto.vaddcs(vec, vec, carry, all_mask)
        pto.vsts(result, dst, i, all_mask)
```
