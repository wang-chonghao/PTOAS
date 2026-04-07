### Synchronization & Buffer Control

Operations for pipeline synchronization and buffer management.

#### Enum Types for Synchronization

The following enum types provide type-safe parameter specification for synchronization operations:

- **`BarrierType`**: Memory barrier types for `pto.mem_bar`
  - `VV_ALL`: All prior vector ops complete before subsequent
  - `VST_VLD`: All prior vector stores visible before subsequent loads  
  - `VLD_VST`: All prior vector loads complete before subsequent stores

- **`Pipe`**: Hardware pipeline identifiers
  - `MTE2`: Memory Transfer Engine 2 pipeline
  - `V`: Vector pipeline
  - `MTE3`: Memory Transfer Engine 3 pipeline
  - `ALL`: All pipelines (for barrier operations)

- **`Event`**: Event identifiers for synchronization
  - `ID0`, `ID1`, `ID2`, `ID3`, ..., `ID31`: Event IDs 0-31 (A5 supports 32 event IDs, 0-15 for subblock 0, 16-31 for subblock 1)

#### `pto.set_flag(pipe_from: PIPE, pipe_to: PIPE, event: EVENT) -> None`

**Description**: Sets a synchronization flag between hardware pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `PIPE` | Source pipeline (e.g., `PIPE.MTE2`) |
| `pipe_to` | `PIPE` | Destination pipeline (e.g., `PIPE.V`) |
| `event` | `EVENT` | Event identifier (e.g., `EVENT.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE, EVENT

pto.set_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### `pto.wait_flag(pipe_from: PIPE, pipe_to: PIPE, event: EVENT) -> None`

**Description**: Waits for a synchronization flag between hardware pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe_from` | `PIPE` | Source pipeline (e.g., `PIPE.MTE2`) |
| `pipe_to` | `PIPE` | Destination pipeline (e.g., `PIPE.V`) |
| `event` | `EVENT` | Event identifier (e.g., `EVENT.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE, EVENT

pto.wait_flag(PIPE.MTE2, PIPE.V, EVENT.ID0)
```

#### `pto.pipe_barrier(pipes: PIPE) -> None`

**Description**: Executes a barrier across specified pipelines.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipes` | `PIPE` | Pipeline specification (e.g., `PIPE.ALL`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import PIPE

pto.pipe_barrier(PIPE.ALL)
```

#### `pto.get_buf(pipe: Pipe, buf_id: pto.i64, mode: pto.i64) -> None`

**Description**: Acquire buffer slot for inter-pipeline double-buffering coordination.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Pipeline identifier (e.g., `Pipe.MTE2`, `Pipe.V`, `Pipe.MTE3`) |
| `buf_id` | `pto.i64` | Buffer identifier |
| `mode` | `pto.i64` | Acquisition mode |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Pipe

# Acquire buffer for MTE2 pipeline
pto.get_buf(Pipe.MTE2, 0, 0)
```

#### `pto.rls_buf(pipe: Pipe, buf_id: pto.i64, mode: pto.i64) -> None`

**Description**: Release buffer slot to allow other pipeline to proceed.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `pipe` | `Pipe` | Pipeline identifier (e.g., `Pipe.MTE2`, `Pipe.V`, `Pipe.MTE3`) |
| `buf_id` | `pto.i64` | Buffer identifier |
| `mode` | `pto.i64` | Release mode |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Pipe

# Release buffer for MTE2 pipeline
pto.rls_buf(Pipe.MTE2, 0, 0)
```

#### `pto.mem_bar(barrier_type: BarrierType) -> None`

**Description**: Memory barrier for pipeline synchronization within vector scope. Required when UB addresses alias between vector load/store operations.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `barrier_type` | `BarrierType` | Barrier type: `BarrierType.VV_ALL` (all prior vector ops complete before subsequent), `BarrierType.VST_VLD` (all prior vector stores visible before subsequent loads), `BarrierType.VLD_VST` (all prior vector loads complete before subsequent stores) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import BarrierType

# Ensure stores are visible before loads to same UB region
pto.mem_bar(BarrierType.VST_VLD)
```

#### `pto.set_cross_core(core_id: pto.i64, event_id: Event) -> None`

**Description**: Signal event to another core (cross-core synchronization).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `core_id` | `pto.i64` | Target/source core identifier (platform-specific mapping) |
| `event_id` | `Event` | Cross-core event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Signal event ID0 to core 0
pto.set_cross_core(0, Event.ID0)
```

#### `pto.set_intra_block(block_id: pto.i64, event_id: Event) -> None`

**Description**: Signal event within a block (A5). Specifies trigger pipe. 1:1 per subblock.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `block_id` | `pto.i64` | Block/pipeline identifier specifying trigger pipe |
| `event_id` | `Event` | Event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Signal event ID0 on block/pipeline 0
pto.set_intra_block(0, Event.ID0)
```

#### `pto.set_intra_core(config: pto.i32) -> None`

**Description**: Configures intra-core synchronization settings.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `pto.i32` | Configuration value for intra-core synchronization |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_intra_core(3)
```

#### `pto.wait_flag_dev(core_id: pto.i64, event_id: Event) -> None`

**Description**: Wait for event from another core. SU-level blocking — entire core stalls.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `core_id` | `pto.i64` | Core identifier |
| `event_id` | `Event` | Event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Wait for event ID0 from core 0
pto.wait_flag_dev(0, Event.ID0)
```

#### `pto.wait_intra_core(block_id: pto.i64, event_id: Event) -> None`

**Description**: Wait for event within block (A5). Specifies which pipeline should wait — only that pipe stalls, SU and other pipes continue.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `block_id` | `pto.i64` | Block/pipeline identifier specifying which pipeline should wait |
| `event_id` | `Event` | Event identifier (e.g., `Event.ID0`) |

**Returns**: None (side-effect operation)

**Example**:
```python
from pto import Event

# Wait for event ID0 on block/pipeline 0
pto.wait_intra_core(0, Event.ID0)
```

### DMA Programming [Advanced Tier]

This section contains both DMA configuration operations (setting loop strides and sizes) and DMA execution operations (copying data).

#### Manual Configuration Example

```python
# DMA configuration example (requires careful parameter tuning)
pto.set_loop2_stride_outtoub(src_stride=32, dst_stride=128)  # Outer loop strides
pto.set_loop1_stride_outtoub(src_stride=1, dst_stride=32)    # Inner loop strides
pto.set_loop_size_outtoub(loop1=16, loop2=16)                # Transfer size
pto.copy_gm_to_ubuf(src=gm_ptr, dst=ub_ptr, n_burst=16, len_burst=128, gm_stride=128, ub_stride=128)

```

#### `pto.set_loop2_stride_outtoub(src_stride: pto.i64, dst_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA stride parameters for GM → UB transfers (loop2).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src_stride` | `pto.i64` | Source-side stride |
| `dst_stride` | `pto.i64` | Destination-side stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop1_stride_outtoub(src_stride: pto.i64, dst_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA stride parameters for GM → UB transfers (loop1).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src_stride` | `pto.i64` | Source-side stride |
| `dst_stride` | `pto.i64` | Destination-side stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop_size_outtoub(loop1: pto.i64, loop2: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA transfer size for GM → UB transfers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `loop1` | `pto.i64` | Inner loop trip count |
| `loop2` | `pto.i64` | Outer loop trip count |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_loop_size_outtoub(loop1=1, loop2=1)
```

#### `pto.set_loop2_stride_ubtoout(src_stride: pto.i64, dst_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA stride parameters for UB → GM transfers (loop2).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src_stride` | `pto.i64` | Source-side stride |
| `dst_stride` | `pto.i64` | Destination-side stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop1_stride_ubtoout(src_stride: pto.i64, dst_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA stride parameters for UB → GM transfers (loop1).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src_stride` | `pto.i64` | Source-side stride |
| `dst_stride` | `pto.i64` | Destination-side stride |

**Returns**: None (side-effect operation)

#### `pto.set_loop_size_ubtoout(loop1: pto.i64, loop2: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA transfer size for UB → GM transfers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `loop1` | `pto.i64` | Inner loop trip count |
| `loop2` | `pto.i64` | Outer loop trip count |

**Returns**: None (side-effect operation)

#### `pto.set_loop(loop_id: pto.i32, src_stride: pto.i64, dst_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA stride parameters for a generic loop.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `loop_id` | `pto.i32` | Loop identifier (e.g., 1 for inner loop, 2 for outer loop) |
| `src_stride` | `pto.i64` | Source-side stride |
| `dst_stride` | `pto.i64` | Destination-side stride |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_loop(1, src_stride=32, dst_stride=64)
```

#### `pto.set_loop_size(loop_id: pto.i32, size: pto.i64) -> None`  [Advanced Tier]

**Description**: Configures DMA transfer size for a generic loop.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `loop_id` | `pto.i32` | Loop identifier (e.g., 1 for inner loop, 2 for outer loop) |
| `size` | `pto.i64` | Loop trip count |

**Returns**: None (side-effect operation)

**Example**:
```python
pto.set_loop_size(1, 16)
```

#### DMA Execution Operations

**Note**: These operations execute DMA transfers but require manual configuration of DMA parameters (loop strides, loop sizes) using the `set_loop*_stride_*` and `set_loop_size_*` operations described above.

The following operations provide direct control over DMA transfers but require manual stride and size configuration.

#### `pto.copy_gm_to_ubuf(src: GMPtr, dst: UBPtr, sid: pto.i64 = 0, n_burst: pto.i64, len_burst: pto.i64, left_padding_count: pto.i64 = 0, right_padding_count: pto.i64 = 0, enable_ub_pad: pto.i1 = False, l2_cache_ctl: pto.i64 = 0, gm_stride: pto.i64, ub_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Copies data from Global Memory (GM) to Unified Buffer (UB).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `GMPtr` | Source GM pointer |
| `dst` | `UBPtr` | Destination UB pointer |
| `sid` | `pto.i64` | DMA stream/control operand, defaults to `0` |
| `n_burst` | `pto.i64` | Number of bursts |
| `len_burst` | `pto.i64` | Bytes copied by each burst |
| `left_padding_count` | `pto.i64` | Left padding count, defaults to `0` |
| `right_padding_count` | `pto.i64` | Right padding count, defaults to `0` |
| `enable_ub_pad` | `pto.i1` | Convenience alias for `data_select_bit`, defaults to `False` |
| `l2_cache_ctl` | `pto.i64` | L2 cache control operand, defaults to `0` |
| `gm_stride` | `pto.i64` | GM-side stride in bytes |
| `ub_stride` | `pto.i64` | UB-side stride in bytes |

**Returns**: None (side-effect operation)

**Notes**:
- In TileLang DSL, the keyword form above is the recommended public surface.
- The lowering still maps to the underlying low-level PTO operand ABI in positional order.

**Example**:
```python
pto.copy_gm_to_ubuf(
    src=gm_ptr,
    dst=ub_ptr,
    n_burst=32,
    len_burst=128,
    gm_stride=128,
    ub_stride=128,
    enable_ub_pad=False,
)
```

#### `pto.copy_ubuf_to_ubuf(src: UBPtr, dst: UBPtr, src_offset: pto.i64, src_stride0: pto.i64, src_stride1: pto.i64, dst_offset: pto.i64, dst_stride0: pto.i64, dst_stride1: pto.i64) -> None`  [Advanced Tier]

**Description**: Copies data within Unified Buffer (UB → UB).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `UBPtr` | Source UB pointer |
| `dst` | `UBPtr` | Destination UB pointer |
| `src_offset` | `pto.i64` | Source offset |
| `src_stride0` | `pto.i64` | Source stride dimension 0 |
| `src_stride1` | `pto.i64` | Source stride dimension 1 |
| `dst_offset` | `pto.i64` | Destination offset |
| `dst_stride0` | `pto.i64` | Destination stride dimension 0 |
| `dst_stride1` | `pto.i64` | Destination stride dimension 1 |

**Returns**: None (side-effect operation)

#### `pto.copy_ubuf_to_gm(src: UBPtr, dst: GMPtr, sid: pto.i64 = 0, n_burst: pto.i64, len_burst: pto.i64, reserved: pto.i64 = 0, gm_stride: pto.i64, ub_stride: pto.i64) -> None`  [Advanced Tier]

**Description**: Copies data from Unified Buffer (UB) to Global Memory (GM).

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `UBPtr` | Source UB pointer |
| `dst` | `GMPtr` | Destination GM pointer |
| `sid` | `pto.i64` | DMA stream/control operand, defaults to `0` |
| `n_burst` | `pto.i64` | Number of bursts |
| `len_burst` | `pto.i64` | Bytes copied by each burst |
| `reserved` | `pto.i64` | Reserved operand, defaults to `0` |
| `gm_stride` | `pto.i64` | GM-side stride in bytes |
| `ub_stride` | `pto.i64` | UB-side stride in bytes |

**Returns**: None (side-effect operation)

**Notes**:
- In TileLang DSL, the keyword form above is the recommended public surface.
- `gm_stride`/`ub_stride` are ergonomic aliases for the low-level `burst_dst_stride`/`burst_src_stride` operands.
- The lowering still maps to the underlying low-level PTO operand ABI in positional order.

**Example**:
```python
pto.copy_ubuf_to_gm(
    src=ub_ptr,
    dst=gm_ptr,
    n_burst=32,
    len_burst=128,
    gm_stride=128,
    ub_stride=128,
)
```
