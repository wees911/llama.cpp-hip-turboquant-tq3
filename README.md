# llama.cpp-tq3 — HIP/ROCm patch (Targeting RDNA2/3/4, Validated on RDNA2)

Patch to build [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3)
with the HIP/ROCm backend on AMD GPUs, enabling TQ3_1S and TQ3_4S weight inference
with full flash attention support across all decode turns.

Based on [flamme-demon/llama.cpp-hip-turboquant-tq3](https://github.com/flamme-demon/llama.cpp-hip-turboquant-tq3),
which provided the initial HIP shims validated on gfx1100 (RX 7900 XTX). This patch
extends that work with flash attention kernel instances for TQ3_1S/TQ3_4S as key
types, fixing multi-turn inference on AMD GPUs. Validated on gfx1030 (RDNA2).

> **Note:** This patch was developed with the assistance of [Claude](https://claude.ai) (Anthropic)
> through iterative debugging and code generation.
>
> If [flamme-demon/llama.cpp-hip-turboquant-tq3](https://github.com/flamme-demon/llama.cpp-hip-turboquant-tq3)
> has been updated to include flash attention support for TQ3_1S/TQ3_4S, please check that repository first.

## Problem solved

On AMD GPUs, the upstream fork produces correct output on the first inference turn,
but subsequent turns generate completely unrelated content (e.g. outputting "assistant"
or starting a math problem). The root cause is missing `fattn-vec` kernel instances
for TQ3_1S/TQ3_4S as K types — decode-phase attention silently falls back to a
broken path, corrupting the KV cache from turn 2 onward.

## Validated on

| Hardware | ROCm | OS |
|---|---|---|
| AMD Radeon RX 6000 series (gfx1030) | 7.2.1 | Ubuntu 24.04 |

`-mwavefrontsize32` is enforced automatically for gfx10xx/gfx11xx/gfx12xx targets.
Other RDNA2/3/4 architectures are expected to work but have not been independently tested.

## Measured performance (gfx1030)

| Model | VRAM | Config | Generation |
|---|---|---|---|
| Qwopus3.5-27B-v3-TQ3_4S | 16 GB | `-ctk q4_0 -ctv tq3_0 -fa on -c 16384` | **~22 t/s** |
| Qwen3.6-35B-A3B-TQ3_4S | 16 GB | `-ctk q4_0 -ctv tq3_0 -fa on -c 16384` | **~60 t/s** |

## What the patch changes

7 files changed, 6 new files added:

| File | Change |
|---|---|
| `ggml/src/ggml-cuda/fattn-common.cuh` | Add `vec_dot_fattn_vec_KQ_tq3_1s` and `vec_dot_fattn_vec_KQ_tq3_4s`; register in `get_vec_dot_KQ` |
| `ggml/src/ggml-cuda/fattn-vec.cuh` | Extend `nthreads_KQ` and `Q_q8_1` conditions to cover TQ3_1S/TQ3_4S as K types |
| `ggml/src/ggml-cuda/mmq.cuh` | Add `TQ3_4S` tile size to `mmq_get_dp4a_tile_x_sizes` |
| `ggml/src/ggml-cuda/vendors/hip.h` | Rewrite `__shfl_*_sync` macros for 3/4-arg dispatch; add `__ballot_sync` |
| `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-tq3_{0,1s,4s}-{f16,q8_0}.cu` | 6 new fattn-vec instance files (K=TQ3_x, V=f16 or q8_0) |
| `ggml/src/ggml-hip/CMakeLists.txt` | Register new instances; enforce `-mwavefrontsize32` for all HIP TUs on RDNA targets |

### Why these changes

`fattn-vec` dispatches via `get_vec_dot_KQ<type_K>`. The upstream fork only
registered `TQ3_0` as a valid K type. `TQ3_1S` and `TQ3_4S` hit the
`static_assert(type_K == -1, "bad type")` fallback, causing decode attention
to produce garbage values that corrupted the KV cache every turn after the first.

The fix adds proper KQ dot product implementations for both types — matching the
centroid/scale arithmetic already present in `vecdotq.cuh` — and registers the
six necessary `(K type, V type)` flash attention instance combinations.

## Apply

```bash
git clone https://github.com/turbo-tan/llama.cpp-tq3
cd llama.cpp-tq3
git apply turboquant-hip-fix.patch

mkdir build && cd build
cmake .. \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1030 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

Replace `gfx1030` with your GPU architecture (e.g. `gfx1100` for RX 7000 series).

## Supported types

| Type | Role | Flash attention K |
|---|---|---|
| TQ3_0 | Weights / KV cache | ✅ |
| TQ3_1S | Weights | ✅ (this patch) |
| TQ3_4S | Weights | ✅ (this patch) |

V side of flash attention supports `f16` (default KV cache) and `q8_0`.
`tq3_0` is also valid as V type via the existing `dequantize_V_tq3_0` path.

## Recommended server config

```bash
./build/bin/llama-server \
  -m /path/to/model-TQ3_4S.gguf \
  -ngl 99 -c 16384 -np 1 \
  -ctk q4_0 -ctv tq3_0 -fa on \
  --no-warmup --jinja \
  --cache-reuse 256
```

## Credits

- [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) — TQ3_1S/TQ3_4S weight quantization and CUDA kernels
- [flamme-demon/llama.cpp-hip-turboquant-tq3](https://github.com/flamme-demon/llama.cpp-hip-turboquant-tq3) — base HIP port (gfx1100) that this patch extends
- [Claude](https://claude.ai) (Anthropic) — assisted in debugging and developing this patch

## License

MIT — same as [llama.cpp](https://github.com/ggml-org/llama.cpp)
