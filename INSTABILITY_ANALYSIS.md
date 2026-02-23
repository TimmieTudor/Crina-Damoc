# Training Instability Analysis

## Problem Summary
After approximately 290 iterations (EPOCH 9, ITER 3), the model produces NaN values in the forward pass.

## Root Cause
The ALIF neuron's `adapt_strength` parameter has **zero gradients** throughout training:
- All 24 ALIF neurons (8 layers × 3 neurons per layer) show `adapt_strength` gradient = 0.0
- The `base_threshold` parameter receives gradients correctly
- This indicates the adaptive threshold mechanism is not being trained

## Evidence from train_log.txt
```
EPOCH 9, ITER 3: NaNs detected in Forward Output!
EPOCH 9, ITER 3: Gradient of blocks.0.attn.lif_neuron.adapt_strength is vanishingly small!
EPOCH 9, ITER 3: Gradient of blocks.0.ff.lif1.adapt_strength is vanishingly small!
...
(All adapt_strength gradients are exactly 0.0)
```

## Likely Causes

### 1. **CUDA Kernel Issue** (Most Likely)
The `alif_backward` CUDA kernel may not be computing `grad_adapt_strength` correctly.

**Check**: `cuda_lib/alif.cu` backward pass
- Line where `grad_adapt_strength` is accumulated
- Ensure the gradient w.r.t. `adapt_strength` in the threshold dynamics is computed

### 2. **Detached Computation**
The `adapt` state might be detached from the computation graph.

**Check**: `crina_tinyshakespeare.py` line 582-583
```python
self.v[:B].copy_(v_f.detach().unsqueeze(1))
self.adapt[:B].copy_(a_f.detach().unsqueeze(1))
```
The `.detach()` is correct for preventing gradient accumulation across time, but we need to ensure the gradient flows through the current timestep.

### 3. **Zero Initialization**
`adapt_strength` is initialized to 0.1, but if the gradient computation has a bug, it stays at initialization.

### 4. **Surrogate Gradient Issue**
The surrogate gradient might not be propagating through the adaptive threshold term.

## Recommended Fixes

### Fix 1: Verify CUDA Kernel Gradient Computation
Check `cuda_lib/alif.cu` backward kernel:
```cpp
// Ensure this line exists and is correct:
atomicAdd(&grad_adapt_strength[d], grad_i * spike_hard[idx] * ...);
```

### Fix 2: Add Gradient Clipping
Even if gradients are computed, they might be vanishing. Add to training loop:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Fix 3: Initialize adapt_strength Higher
Change initialization from 0.1 to 0.5 or 1.0 to make the effect more pronounced:
```python
self.adapt_strength = nn.Parameter(torch.ones(size) * 0.5)
```

### Fix 4: Add Regularization
Add L2 regularization to prevent parameters from collapsing:
```python
# In training loop
l2_reg = sum(p.pow(2).sum() for p in model.parameters())
loss = ce_loss + 1e-5 * l2_reg
```

### Fix 5: Reduce Surrogate Slope
The slope of 25.0 is very steep, which can cause vanishing gradients:
```python
self.surrogate_slope = 10.0  # Instead of 25.0
```

## Immediate Action
1. Run `diagnose_nan.py` with hooks to catch the exact layer where NaN first appears
2. Check if `grad_adapt_strength` is being computed in the CUDA kernel
3. Add gradient clipping to prevent explosion
4. Consider using PyTorch's native ALIF implementation temporarily to verify the architecture works

## Long-term Solutions
1. Implement gradient checkpointing for long sequences
2. Add batch normalization or layer normalization before ALIF neurons
3. Use a warmup schedule for learning rate
4. Implement adaptive gradient clipping (per-parameter)
