## llama.cpp-hip-turboquant                                                                                                                                                                              
                                                                                                                                                                                                           
  Arch Linux `PKGBUILD` and minimal patch to build the                                                                                                                                                     
  [`turbo-tan/llama.cpp-tq3`](https://github.com/turbo-tan/llama.cpp-tq3) fork                                                                                                                             
  with the HIP/ROCm backend, enabling TurboQuant-quantized models                                                                                                                                          
  (TQ3_1S, **TQ3_4S**) and TQ3_0 KV cache on AMD GPUs.                                                                                                                                                     
                                              
  The upstream fork is CUDA-only — this patch only adds the HIP shims required                                                                                                                             
  for the GPU kernels. No change to the TurboQuant logic itself.
                                                                                                                                                                                                           
  ### Validated on                                          
                                                                                                                                                                                                           
  - **GPU**: AMD Radeon RX 7900 XTX (gfx1100)                                                                                                                                                              
  - **ROCm**: 7.2.0 / HIP 7.2.26043                                                                                                                                                                        
  - **OS**: Manjaro Linux (kernel 6.19)                                                                                                                                                                    
  - **Model**: `YTan2000/Qwen3.6-35B-A3B-TQ3_4S` + BF16 mmproj
                                              
  ### Measured performance (llama-server, 65k context)                                                                                                                                                     
                                              
  | Config                                     | Prompt   | Generation |                                                                                                                                   
  |--------------------------------------------|----------|------------|
  | `-ctk q4_0 -ctv tq3_0 -fa on`              | 238 t/s  | **78.2 t/s** |                                                                                                                                 
  | Default f16 KV cache                       | 159 t/s  | 89.6 t/s   |
                                                                                                                                                                                                           
  VRAM usage: ~22.7 / 24 GB (full 35B model on GPU).        
                                                                                                                                                                                                           
  ### What the patch changes              
                                                                                                                                                                                                           
  112 lines, 4 files in `ggml/src/ggml-cuda/`:                                                                                                                                                             
   
  - `vendors/hip.h` — variadic `__shfl_*_sync` macros (3/4-arg), `__ballot_sync`,                                                                                                                          
    `cudaEventCreate` / `cudaEventElapsedTime` shims.       
  - `tq3-native.cuh` — conditional HIP vs CUDA include for `fp16.h` and                                                                                                                                    
    `GGML_COMMON_DECL_*`.                                   
  - `tq3-native.cu` — include order (`common.cuh` before `tq3-native.cuh`).                                                                                                                                
  - `vecdotq.cuh` — replaces 4× `__dp4a` with `ggml_cuda_dp4a` (portable wrapper
    that maps to `v_dot4c` on RDNA3).                                                                                                                                                                      
                                                            
  ### Install                                                                                                                                                                                              
                                                                                                                                                                                                           
  ```bash                                                                                                                                                                                                  
  git clone <this-repo> llama.cpp-hip-turboquant                                                                                                                                                           
  cd llama.cpp-hip-turboquant                                                                                                                                                                              
  makepkg -si                                               
                                              
  The package conflicts with llama.cpp and llama.cpp-hip (both ship
  /usr/bin/llama-server). pacman will handle the transition.
                                                                                                                                                                                                           
  Limitations                             
                                                                                                                                                                                                           
  - GGML_RPC=OFF — the RPC backend is disabled because of a static_assert on                                                                                                                               
  GGML_OP_COUNT that changed upstream. Patch separately if you need it.
  - llama-cli: add --no-warmup to avoid a rare kernel that hangs during                                                                                                                                    
  warmup. llama-server is not affected.                                                                                                                                                                    
                                          
  Supported types in this build                                                                                                                                                                            
                                                                                                                                                                                                           
  ┌────────┬─────┬─────────────────┐          
  │  Type  │ ID  │      Role       │                                                                                                                                                                       
  ├────────┼─────┼─────────────────┤                        
  │ TQ3_1S │ 44  │ Weights         │                                                                                                                                                                       
  ├────────┼─────┼─────────────────┤
  │ TQ3_4S │ 46  │ Weights (4 bpw) │                                                                                                                                                                       
  ├────────┼─────┼─────────────────┤                        
  │ TQ3_0  │ 200 │ KV cache        │      
  └────────┴─────┴─────────────────┘
                                                                                                                                                                                                           
  The TURBO2_0 / TURBO3_0 / TURBO4_0 / TQ4_1S variants from the
  https://github.com/domvox/llama.cpp-turboquant-hip fork are not                                                                                                                                          
  supported — the two KV cache designs have diverged.       
                                                                                                                                                                                                           
  Credits
                                                                                                                                                                                                           
  - Upstream TurboQuant (CUDA): https://github.com/turbo-tan/llama.cpp-tq3
  - Original HIP port (different KV cache design):                                                                                                                                                         
  https://github.com/domvox/llama.cpp-turboquant-hip        
  - Base PKGBUILD: llama.cpp-hip by Orion-zhen / txtsd on AUR                                                                                                                                              
  - Paper: https://arxiv.org/abs/2504.19874   
                                                                                                                                                                                                           
  License                                                                                                                                                                                                  
                                                                                                                                                                                                           
  MIT (same as upstream llama.cpp).
