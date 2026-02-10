# NPU Backend Compatibility Status

## Overview
This document tracks the compatibility status of all 23 operators for the NPU (Ascend) backend.

## Compatibility Matrix

| Category | Operator | Status | Notes |
|----------|----------|--------|-------|
| **Pointwise** | add | ✅ Ready | Simple element-wise, BLOCK_N=1024 |
| **Pointwise** | fill | ✅ Ready | Store-only operation |
| **Pointwise** | fill_ | ✅ Ready | In-place fill |
| **Pointwise** | zeros | ✅ Ready | Store zeros |
| **Pointwise** | exponential_ | ✅ Ready | Uses tl.log, tl.maximum |
| **Pointwise** | contiguous | ✅ Ready | 1D and 2D strided copy |
| **Matmul** | mm | ✅ Ready | Verified with examples |
| **Matmul** | bmm | ✅ Ready | Batched matmul |
| **Matmul** | addmm | ✅ Ready | Add + matmul |
| **Index** | embedding | ✅ Ready | Index lookup |
| **Index** | cat | ✅ Ready | Concatenation |
| **Index** | nonzero | ✅ Ready | Non-zero indices |
| **Index** | reshape_and_cache_flash | ✅ Ready | KV cache reshape |
| **Reduce** | sum | 🔄 Updated | FlagGems Ascend two-pass pattern |
| **Reduce** | max | 🔄 Updated | FlagGems Ascend two-pass pattern |
| **Reduce** | argmax | 🔄 Updated | FlagGems Ascend two-pass pattern |
| **Reduce** | topk | ⚠️ Original | May need two-stage + bitonic sort |
| **Normalization** | softmax | 🔄 Updated | FlagGems Ascend iterative pattern |
| **Normalization** | rms_norm | 🔄 Updated | Iterative variance computation |
| **Normalization** | fused_add_rms_norm | 🔄 Updated | FlagGems Ascend pattern |
| **Fusion** | apply_rotary_pos_emb | 🔄 Updated | FlagGems Ascend pattern |
| **Fusion** | rwkv_ka_fusion | 🔄 Updated | Element-wise, NPU-compatible |
| **Fusion** | rwkv_mm_sparsity | 🔄 Updated | Uses tl.dot, NPU-compatible |

## Legend
- ✅ Ready: Works on NPU without modifications
- 🔄 Updated: Modified for NPU compatibility using FlagGems Ascend patterns
- ⚠️ Original: Original implementation, may need further testing/modification

## Key Changes Made

### CMakeLists.txt
- Removed NPU conditional skip for reduce/normalization/fusion operators
- All 23 operators now compile for NPU backend

### Reduce Operators (sum, max, argmax)
- Implemented two-pass reduction strategy from FlagGems Ascend
- First pass: block-wise reduction
- Second pass: reduce block results to final value
- Grid size capped at 4096 for NPU compatibility

### Normalization Operators (softmax, rms_norm, fused_add_rms_norm)
- Implemented iterative pattern from FlagGems Ascend
- Uses online algorithm for softmax (avoids full-tensor reduction)
- Block sizes capped at 8192 for NPU

### Fusion Operators
- apply_rotary_pos_emb: Based on FlagGems Ascend rotary_embedding.py
- rwkv_*: Simple operations (element-wise, tl.dot) that are NPU-compatible

## Known Limitations

1. **Small tensor sizes**: Kernels may have issues when tensor size < BLOCK_N (1024)
   - Tests skip shape {1} for this reason

2. **topk**: May need bitonic sort implementation for full NPU compatibility
   - Current implementation falls back to PyTorch for correctness

3. **Block size constraints**: NPU has different memory constraints
   - Max BLOCK_SIZE capped at 8192 (vs 4096 for some operations)
   - 40 AIV cores on Ascend 910B3/B4

## FlagGems Reference Files
- `/data/baai_user_home/chwork/FlagGems/src/flag_gems/runtime/backend/_ascend/ops/`
- `/data/baai_user_home/chwork/FlagGems/src/flag_gems/runtime/backend/_ascend/fused/`

## Verification Commands

```bash
# Build all operators
cmake --build build/ --parallel

# Run specific operator test
TORCH_CPP_LOG_LEVEL=INFO build/operators/<category>/<op>/test_<op>

# Run all tests
python tests/run_all_tests.py --backend NPU --build-dir build
```
