# FlashAttention Softmax VPTO Sync E2E Result

Case: `flash_attention_softmax_block`, shape `32x32`, dtype `f32`, single VPTO softmax kernel.

Both paths enable explicit sync insertion and disable ccec vector instruction rescheduling:

- `--enable-insert-sync`
- `-mllvm -cce-aicore-vec-misched=0` in the VPTO object emission command

## Commands

The validation uses `PTOAS/test/vpto/scripts/run_host_vpto_validation.sh` with generated case wrappers under `_cases/`.

Default flags:

```text
--pto-arch a5 --pto-backend=vpto --pto-level=level2 --enable-op-fusion --enable-insert-sync
```

Costmodel flags:

```text
--pto-arch a5 --pto-backend=vpto --pto-level=level2 --enable-op-fusion --enable-insert-sync --use-vfsim-fusion-planner --dump-vfsim-unroll-test
```

## Results

| Path | Compare | Total tick | VF execute time | Selected unroll |
|---|---:|---:|---:|---:|
| default | passed | 2707 | 377 | n/a |
| costmodel | passed | 2834 | 512 | 8 |

Costmodel candidate estimates:

| Unroll | Valid | Estimated cycles |
|---:|---:|---:|
| 1 | yes | 364 |
| 2 | yes | 362 |
| 3 | no | n/a |
| 4 | yes | 366 |
| 5 | no | n/a |
| 6 | no | n/a |
| 7 | no | n/a |
| 8 | yes | 343 |

## Saved Files

- `default/vpto_final.mlir`
- `costmodel/vpto_final.mlir`
- `default/dumps/core0.veccore0.instr_log.dump`
- `default/dumps/core0.veccore0.instr_popped_log.dump`
- `default/dumps/core0.veccore0.rvec.IDU.dump`
- `costmodel/dumps/core0.veccore0.instr_log.dump`
- `costmodel/dumps/core0.veccore0.instr_popped_log.dump`
- `costmodel/dumps/core0.veccore0.rvec.IDU.dump`

The final VPTO IR contains explicit `PIPE_V -> PIPE_MTE3` sync before `pto.copy_ubuf_to_gm` in both paths.
