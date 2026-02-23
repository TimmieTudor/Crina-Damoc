# crina_tinyshakespeare.py
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import math
import time
import os
import requests
from tqdm import tqdm
import wandb
import crina_cuda
from typing import Optional, List, Tuple
# from torch._subclasses.fake_tensor import is_fake # No longer needed manually

# -----------------------------------------------------------------------------
# Custom Op Registration for torch.compile
# -----------------------------------------------------------------------------

@torch.library.custom_op("crina_cuda::linear_sigmoid_forward", mutates_args=())
def linear_sigmoid_forward_op(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return crina_cuda.linear_sigmoid_forward(input, weight, bias)

@linear_sigmoid_forward_op.register_fake
def _(input, weight, bias=None):
    output_shape = input.shape[:-1] + (weight.shape[0],)
    return input.new_empty(output_shape)

@torch.library.custom_op("crina_cuda::linear_sigmoid_backward", mutates_args=())
def linear_sigmoid_backward_op(grad_output: torch.Tensor, output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    res = crina_cuda.linear_sigmoid_backward(grad_output, output, input, weight)
    return tuple(res)

@linear_sigmoid_backward_op.register_fake
def _(grad_output, output, input, weight):
    grad_input = torch.empty_like(input)
    grad_weight = torch.empty_like(weight)
    grad_bias = torch.empty((weight.shape[0],), device=input.device, dtype=input.dtype)
    return grad_input, grad_weight, grad_bias

@torch.library.custom_op("crina_cuda::fused_gate_forward", mutates_args=())
def fused_gate_forward_op(state_gate: torch.Tensor, raw_sum: torch.Tensor, prev_s: torch.Tensor) -> torch.Tensor:
    return crina_cuda.fused_gate_forward(state_gate, raw_sum, prev_s)

@fused_gate_forward_op.register_fake
def _(state_gate, raw_sum, prev_s):
    return raw_sum.new_empty(raw_sum.shape)

@torch.library.custom_op("crina_cuda::fused_gate_backward", mutates_args=())
def fused_gate_backward_op(grad_output: torch.Tensor, state_gate: torch.Tensor, raw_sum: torch.Tensor, prev_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    res = crina_cuda.fused_gate_backward(grad_output, state_gate, raw_sum, prev_s)
    return tuple(res)

@fused_gate_backward_op.register_fake
def _(grad_output, state_gate, raw_sum, prev_s):
    grad_state_gate = torch.empty_like(state_gate)
    grad_raw_sum = torch.empty_like(raw_sum)
    grad_prev_s = torch.empty_like(prev_s)
    return grad_state_gate, grad_raw_sum, grad_prev_s

@torch.library.custom_op("crina_cuda::addadd_forward", mutates_args=())
def addadd_forward_op(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return crina_cuda.addadd_forward(a, b, c)

@addadd_forward_op.register_fake
def _(a, b, c):
    return a.new_empty(a.shape)

@torch.library.custom_op("crina_cuda::addadd_backward", mutates_args=())
def addadd_backward_op(grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    res = crina_cuda.addadd_backward(grad_output, a, b, c)
    return tuple(res)

@addadd_backward_op.register_fake
def _(grad_output, a, b, c):
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    grad_c = torch.empty_like(c)
    return grad_a, grad_b, grad_c

@torch.library.custom_op("crina_cuda::rotate_half_forward", mutates_args=())
def rotate_half_forward_op(x: torch.Tensor) -> torch.Tensor:
    return crina_cuda.rotate_half_forward(x)

@rotate_half_forward_op.register_fake
def _(x):
    return x.new_empty(x.shape)

@torch.library.custom_op("crina_cuda::rotate_half_backward", mutates_args=())
def rotate_half_backward_op(grad_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # returns list in C++, convert to tensor/tuple? 
    # The binding returns vector<Tensor> of size 1. wrapper returned scalar tensor.
    # Wait, existing wrapper returned `grad_input` (scalar tensor from vector[0]?)
    # C++ returns {grad_input}.
    # Python sees list [Tensor].
    # Existing wrapper: `return grad_input` (line 66)
    # Line 67: `return crina_cuda.rotate_half_backward(...)`. This returns [Tensor].
    # If original code worked, autograd handled list of 1 tensor as 1 tensor??
    # Or rotate_half_backward_wrapper implementation was returning [Tensor]?
    # Line 224: `grad_input = rotate_half_backward_wrapper(...)`.
    # Line 225: `return grad_input`. If list, returns list.
    # Autograd expects tuple. If 1 element, maybe list is ok?
    # Safer to unpack.
    res = crina_cuda.rotate_half_backward(grad_output, x)
    return res[0]

@rotate_half_backward_op.register_fake
def _(grad_output, x):
    return torch.empty_like(x)

@torch.library.custom_op("crina_cuda::add_product_forward", mutates_args=())
def add_product_forward_op(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return crina_cuda.add_product_forward(a, b, c, d)

@add_product_forward_op.register_fake
def _(a, b, c, d):
    return a.new_empty(a.shape)

@torch.library.custom_op("crina_cuda::add_product_backward", mutates_args=())
def add_product_backward_op(grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    res = crina_cuda.add_product_backward(grad_output, a, b, c, d)
    return tuple(res)

@add_product_backward_op.register_fake
def _(grad_output, a, b, c, d):
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    grad_c = torch.empty_like(c)
    grad_d = torch.empty_like(d)
    return grad_a, grad_b, grad_c, grad_d

@torch.library.custom_op("crina_cuda::swiglu_activation_forward", mutates_args=())
def swiglu_activation_forward_op(input: torch.Tensor) -> torch.Tensor:
    return crina_cuda.swiglu_activation_forward(input)

@swiglu_activation_forward_op.register_fake
def _(input):
    output_shape = input.shape[:-1] + (input.shape[-1] // 2,)
    return input.new_empty(output_shape)

@torch.library.custom_op("crina_cuda::swiglu_activation_backward", mutates_args=())
def swiglu_activation_backward_op(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    return crina_cuda.swiglu_activation_backward(input, grad_output)

@swiglu_activation_backward_op.register_fake
def _(grad_output, input):
    return torch.empty_like(input)

@torch.library.custom_op("crina_cuda::lateral_mixing_forward", mutates_args=())
def lateral_mixing_forward_op(input: torch.Tensor, weight: torch.Tensor, causal: bool) -> torch.Tensor:
    return crina_cuda.lateral_mixing_forward(input, weight, causal)

@lateral_mixing_forward_op.register_fake
def _(input, weight, causal):
    return input.new_empty(input.shape)

@torch.library.custom_op("crina_cuda::lateral_mixing_backward", mutates_args=())
def lateral_mixing_backward_op(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, causal: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    res = crina_cuda.lateral_mixing_backward(grad_output, input, weight, causal)
    return tuple(res)

@lateral_mixing_backward_op.register_fake
def _(grad_output, input, weight, causal):
    return torch.empty_like(input), torch.empty_like(weight)

@torch.library.custom_op("crina_cuda::rope_forward", mutates_args=())
def rope_forward_op(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return crina_cuda.rope_forward(x, cos, sin)

@rope_forward_op.register_fake
def _(x, cos, sin):
    return x.new_empty(x.shape)

@torch.library.custom_op("crina_cuda::rope_backward", mutates_args=())
def rope_backward_op(grad_output: torch.Tensor, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    res = crina_cuda.rope_backward(grad_output, x, cos, sin)
    return tuple(res)

@rope_backward_op.register_fake
def _(grad_output, x, cos, sin):
    return torch.empty_like(x), torch.empty_like(cos), torch.empty_like(sin)

@torch.library.custom_op("crina_cuda::base_proj_forward", mutates_args=())
def base_proj_forward_op(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
    return crina_cuda.base_proj_forward(x, weight, bias)

@base_proj_forward_op.register_fake
def _(x, weight, bias):
    output_shape = x.shape[:-1] + (1,)
    return x.new_empty(output_shape)

@torch.library.custom_op("crina_cuda::base_proj_backward", mutates_args=())
def base_proj_backward_op(grad_output: torch.Tensor, x: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    res = crina_cuda.base_proj_backward(grad_output, x, weight)
    return tuple(res)

@base_proj_backward_op.register_fake
def _(grad_output, x, weight):
    return torch.empty_like(x), torch.empty_like(weight), torch.empty((1,), device=x.device, dtype=x.dtype)

@torch.library.custom_op("crina_cuda::rmsnorm_forward", mutates_args=())
def rmsnorm_forward_op(input: torch.Tensor, weight: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    res = crina_cuda.rmsnorm_forward(input, weight, eps)
    return tuple(res)

@rmsnorm_forward_op.register_fake
def _(input, weight, eps):
    return torch.empty_like(input), torch.empty(input.shape[:-1], device=input.device, dtype=input.dtype)

@torch.library.custom_op("crina_cuda::rmsnorm_backward", mutates_args=())
def rmsnorm_backward_op(grad_output: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, inv_rms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    res = crina_cuda.rmsnorm_backward(grad_output, input, weight, inv_rms)
    return tuple(res)

@rmsnorm_backward_op.register_fake
def _(grad_output, input, weight, inv_rms):
    return torch.empty_like(input), torch.empty_like(weight)

@torch.library.custom_op("crina_cuda::tree_cell_forward", mutates_args=())
def tree_cell_forward_op(raw_sums: torch.Tensor, prev_s: Optional[torch.Tensor], residual: torch.Tensor, gate_weights: torch.Tensor, norm1_w: torch.Tensor, norm2_w: torch.Tensor, eps: float, level_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    res = crina_cuda.tree_cell_forward(raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, eps, level_idx)
    return tuple(res)

@tree_cell_forward_op.register_fake
def _(raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, eps, level_idx):
    return torch.empty_like(raw_sums), torch.empty(raw_sums.shape[:-1], device=raw_sums.device, dtype=raw_sums.dtype), torch.empty(raw_sums.shape[:-1], device=raw_sums.device, dtype=raw_sums.dtype)

@torch.library.custom_op("crina_cuda::tree_cell_backward", mutates_args=())
def tree_cell_backward_op(grad_output: torch.Tensor, raw_sums: torch.Tensor, prev_s: Optional[torch.Tensor], residual: torch.Tensor, gate_weights: torch.Tensor, norm1_w: torch.Tensor, norm2_w: torch.Tensor, inv_rms1: torch.Tensor, inv_rms2: torch.Tensor, level_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    res = crina_cuda.tree_cell_backward(grad_output, raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, inv_rms1, inv_rms2, level_idx)
    return tuple(res)

@tree_cell_backward_op.register_fake
def _(grad_output, raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, inv_rms1, inv_rms2, level_idx):
    return torch.empty_like(raw_sums), torch.empty_like(raw_sums), torch.empty_like(residual), torch.empty_like(gate_weights), torch.empty_like(norm1_w), torch.empty_like(norm2_w)

@torch.library.custom_op("crina_cuda::alif_forward", mutates_args=())
def alif_forward_op(i_inj: torch.Tensor, v_state: torch.Tensor, a_state: torch.Tensor, bt: torch.Tensor, as_w: torch.Tensor, tm: float, ta: float, slope: float, training: bool) -> List[torch.Tensor]:
    res = crina_cuda.alif_forward(i_inj, v_state, a_state, bt, as_w, tm, ta, slope, training)
    return res # Returns list, custom_op can handle list if annotated? 
    # Annotation: List[torch.Tensor].

@alif_forward_op.register_fake
def _(i_inj, v_state, a_state, bt, as_w, tm, ta, slope, training):
    B, D = i_inj.size(0), i_inj.size(-1)
    return [
        torch.empty_like(i_inj), 
        torch.empty((B, D), device=i_inj.device, dtype=i_inj.dtype), 
        torch.empty((B, D), device=i_inj.device, dtype=i_inj.dtype),
        torch.empty((0,), device=i_inj.device, dtype=i_inj.dtype),
        torch.empty((0,), device=i_inj.device, dtype=i_inj.dtype)
    ]

@torch.library.custom_op("crina_cuda::alif_backward", mutates_args=())
def alif_backward_op(grad_spike: torch.Tensor, grad_vf: torch.Tensor, grad_af: torch.Tensor, vt: torch.Tensor, td: torch.Tensor, sh: torch.Tensor, bt: torch.Tensor, as_w: torch.Tensor, tm: float, ta: float, slope: float) -> List[torch.Tensor]:
    res = crina_cuda.alif_backward(grad_spike, grad_vf, grad_af, vt, td, sh, bt, as_w, tm, ta, slope)
    return res

@alif_backward_op.register_fake
def _(grad_spike, grad_vf, grad_af, vt, td, sh, bt, as_w, tm, ta, slope):
    return [torch.empty_like(grad_spike), torch.empty((0,), device=grad_spike.device), torch.empty((0,), device=grad_spike.device), torch.empty_like(bt), torch.empty_like(as_w)]

# Aliases to maintain compatibility with existing calls in autograd.Function classes
linear_sigmoid_forward_wrapper = linear_sigmoid_forward_op
linear_sigmoid_backward_wrapper = linear_sigmoid_backward_op
fused_gate_forward_wrapper = fused_gate_forward_op
fused_gate_backward_wrapper = fused_gate_backward_op
addadd_forward_wrapper = addadd_forward_op
addadd_backward_wrapper = addadd_backward_op
rotate_half_forward_wrapper = rotate_half_forward_op
rotate_half_backward_wrapper = rotate_half_backward_op
add_product_forward_wrapper = add_product_forward_op
add_product_backward_wrapper = add_product_backward_op
swiglu_activation_forward_wrapper = swiglu_activation_forward_op
swiglu_activation_backward_wrapper = swiglu_activation_backward_op
lateral_mixing_forward_wrapper = lateral_mixing_forward_op
lateral_mixing_backward_wrapper = lateral_mixing_backward_op
rope_forward_wrapper = rope_forward_op
rope_backward_wrapper = rope_backward_op
base_proj_forward_wrapper = base_proj_forward_op
base_proj_backward_wrapper = base_proj_backward_op
rmsnorm_forward_wrapper = rmsnorm_forward_op
rmsnorm_backward_wrapper = rmsnorm_backward_op
tree_cell_forward_wrapper = tree_cell_forward_op
tree_cell_backward_wrapper = tree_cell_backward_op
alif_forward_wrapper = alif_forward_op
alif_backward_wrapper = alif_backward_op


class FusedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state_gate, raw_sum, prev_s):
        output = fused_gate_forward_wrapper(state_gate, raw_sum, prev_s)
        ctx.save_for_backward(state_gate, raw_sum, prev_s)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        state_gate, raw_sum, prev_s = ctx.saved_tensors
        grad_state_gate, grad_raw_sum, grad_prev_s = fused_gate_backward_wrapper(grad_output, state_gate, raw_sum, prev_s)
        return grad_state_gate, grad_raw_sum, grad_prev_s

class AddAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c):
        output = addadd_forward_wrapper(a, b, c)
        ctx.save_for_backward(a, b, c)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        a, b, c = ctx.saved_tensors
        grad_a, grad_b, grad_c = addadd_backward_wrapper(grad_output, a, b, c)
        return grad_a, grad_b, grad_c

class RotateHalf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = rotate_half_forward_wrapper(x)
        ctx.save_for_backward(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_input = rotate_half_backward_wrapper(grad_output, x)
        return grad_input

class AddProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, d):
        output = add_product_forward_wrapper(a, b, c, d)
        ctx.save_for_backward(a, b, c, d)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        a, b, c, d = ctx.saved_tensors
        grad_a, grad_b, grad_c, grad_d = add_product_backward_wrapper(grad_output, a, b, c, d)
        return grad_a, grad_b, grad_c, grad_d

class SwiGLUActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = swiglu_activation_forward_wrapper(input)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = swiglu_activation_backward_wrapper(grad_output, input)
        return grad_input

class LateralMixingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, causal):
        output = lateral_mixing_forward_wrapper(input, weight, causal)
        ctx.save_for_backward(input, weight)
        ctx.causal = causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        causal = ctx.causal
        grad_input, grad_weight = lateral_mixing_backward_wrapper(grad_output, input, weight, causal)
        return grad_input, grad_weight, None

class RoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        output = rope_forward_wrapper(x, cos, sin)
        ctx.save_for_backward(x, cos, sin)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, cos, sin = ctx.saved_tensors
        grad_x, grad_cos, grad_sin = rope_backward_wrapper(grad_output, x, cos, sin)
        return grad_x, grad_cos, grad_sin

class ALIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i_inj, v_state, a_state, bt, as_w, tm, ta, slope, training):
        # Explicit training flag ensures C++ saves buffers correctly
        # torch.is_grad_enabled() is usually False inside Function.forward
        spike, v_final, a_final, saved_vt, saved_td = alif_forward_wrapper(i_inj, v_state, a_state, bt, as_w, tm, ta, slope, training)
        
        ctx.save_for_backward(spike, saved_vt, saved_td, bt, as_w)
        ctx.params = (tm, ta, slope)

        return spike, v_final, a_final, saved_vt, saved_td

    @staticmethod
    def backward(ctx, grad_spike, grad_vf, grad_af, grad_vt, grad_td):
        spike, vt, td, bt, as_w = ctx.saved_tensors
        tm, ta, slope = ctx.params
        
        # grad_i_inj, grad_v_state, grad_a_state, grad_base_thresh, grad_adapt_strength
        res = alif_backward_wrapper(grad_spike, grad_vf, grad_af, vt, td, spike, bt, as_w, tm, ta, slope)
        
        return res[0], res[1], res[2], res[3], res[4], None, None, None, None

class FusedLinearSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # Call the wrapper
        output = linear_sigmoid_forward_wrapper(input, weight, bias)
        # Save tensors needed for backward. Bias is not needed for gradient calculation.
        ctx.save_for_backward(output, input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, input, weight = ctx.saved_tensors
        # Call the wrapper
        grad_input, grad_weight, grad_bias = linear_sigmoid_backward_wrapper(grad_output, output, input, weight)
        return grad_input, grad_weight, grad_bias

# Helper module
class FusedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features).cuda())
        self.bias = torch.nn.Parameter(torch.randn(out_features).cuda())
        
    def forward(self, x):
        return FusedLinearSigmoid.apply(x, self.weight, self.bias)

class BaseProjFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        output = base_proj_forward_wrapper(input, weight, bias)
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = base_proj_backward_wrapper(grad_output, input, weight)
        return grad_input, grad_weight, grad_bias

class BaseProj(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 2).cuda())
        self.bias = torch.nn.Parameter(torch.randn(1).cuda())
        
    def forward(self, x):
        return BaseProjFunction.apply(x, self.weight, self.bias)

class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eps):
        output, inv_rms = rmsnorm_forward_wrapper(input, weight, eps)
        ctx.save_for_backward(input, weight, inv_rms)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, inv_rms = ctx.saved_tensors
        grad_input, grad_weight = rmsnorm_backward_wrapper(grad_output, input, weight, inv_rms)
        return grad_input, grad_weight, None

class FusedRMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model).cuda())
        
    def forward(self, x):
        return RMSNormFunction.apply(x, self.weight, self.eps)

def fused_gate(state_gate, raw_sum, prev_s):
    return FusedGate.apply(state_gate, raw_sum, prev_s)

class TreeCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, eps, level_idx):
        output, inv_rms1, inv_rms2 = tree_cell_forward_wrapper(raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, eps, level_idx)
        ctx.save_for_backward(raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, inv_rms1, inv_rms2)
        ctx.level_idx = level_idx
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, inv_rms1, inv_rms2 = ctx.saved_tensors
        level_idx = ctx.level_idx
        
        grad_raw_sums, grad_prev_s, grad_residual, grad_gate_weights, grad_norm1_w, grad_norm2_w = \
            tree_cell_backward_wrapper(grad_output, raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, inv_rms1, inv_rms2, level_idx)
        
        # Return gradients for all inputs to forward
        # (raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, eps, level_idx)
        # Note: prev_s might be None but is passed as a Tensor of zeros if not present
        return grad_raw_sums, grad_prev_s, grad_residual, grad_gate_weights, grad_norm1_w, grad_norm2_w, None, None

def tree_cell_fused(raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, eps, level_idx):
    return TreeCellFunction.apply(raw_sums, prev_s, residual, gate_weights, norm1_w, norm2_w, eps, level_idx)


def addadd(a, b, c):
    return AddAdd.apply(a, b, c)

def add_product(a, b, c, d):
    return AddProduct.apply(a, b, c, d)

def swiglu_activation(input):
    return SwiGLUActivation.apply(input)

def lateral_mixing(input, weight, causal):
    return LateralMixingFunction.apply(input, weight, causal)

# ---------------------- TinyShakespeare Dataset ----------------------
class TinyShakespeareDataset(Dataset):
    def __init__(self, data, block_size=128):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

# ---------------------- LIF Neuron (from your project) ----------------------
class LIFNeuron(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.threshold = nn.Parameter(torch.ones(size) * 0.5)
        self.tau_mem = 0.99
        self.register_buffer("v", torch.zeros(1, 1, size))

    def reset(self, batch_idx=None):
        if batch_idx is None:
            self.v.zero_()
        else:
            if batch_idx >= self.v.shape[0]:
                # Expand buffer to accommodate the batch index
                old_v = self.v
                self.v = torch.zeros(batch_idx + 1, 1, self.size, device=old_v.device)
                # Keep old values if they existed
                min_b = min(old_v.shape[0], self.v.shape[0])
                self.v[:min_b].copy_(old_v[:min_b])
            self.v[batch_idx].zero_()


    def detach_state(self):
        self.v.detach_()



    def forward(self, i_inj):
        # i_inj can be [B, T, D] or [B, T/2, 2, D]
        B = i_inj.shape[0]
        D = i_inj.shape[-1]
        
        # Auto-expand state if batch size increased
        if self.v.shape[0] < B:
            old_v = self.v
            self.v = torch.zeros(B, 1, self.size, device=old_v.device)
            self.v[:old_v.shape[0]].copy_(old_v)
        elif self.v.shape[0] > B:
            # If batch size decreased (e.g. final batch), slice for this pass
            # Note: We don't overwrite self.v permanently here to avoid confusing compile
            target_v = self.v[:B]
        else:
            target_v = self.v

        # Explicitly broadcast state over all middle dimensions
        v_shape = [B] + [1] * (i_inj.dim() - 2) + [D]
        v = target_v.view(*v_shape).expand_as(i_inj)


        v = self.tau_mem * v + i_inj * 5.0   # ← input scaling


        spike = (v >= self.threshold).float()
        v = v - spike * self.threshold

        #v = v - spike * self.threshold

        # Save state: [B, 1, D]
        # Flatten all middle dimensions to ensure [B, 1, D] regardless of input rank
        v_flat = v.view(B, -1, self.size)
        
        # CLAMP FIX: Prevent state explosion
        v_flat = torch.clamp(v_flat, min=-100.0, max=100.0)
        self.v.copy_(v_flat.detach()[:, -1:, :])




        if self.training:
            surrogate = torch.sigmoid(25.0 * (v - self.threshold))
            spike = surrogate + spike.detach() - spike.detach()

        return spike

class ALIFNeuron(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire (ALIF) Neuron
    - Standard LIF dynamics
    - Adaptive threshold that increases with each spike (spike-frequency adaptation)
    - Fully vectorized, supports [B, N, D] shapes (e.g., for tree nodes)
    - Surrogate gradient for training
    """
    def __init__(self, size, threshold=0.5, tau_mem=0.99, tau_adapt=0.95, adapt_strength=0.1):
        super().__init__()
        self.size = size

        # Base parameters
        self.base_threshold = nn.Parameter(torch.ones(size) * threshold)
        self.adapt_strength = nn.Parameter(torch.ones(size) * adapt_strength)  # How much threshold rises per spike
        self.tau_mem = tau_mem
        self.tau_adapt = tau_adapt  # Decay rate of adaptation

        # Persistent states: [1, 1, size] for broadcasting
        self.register_buffer("v", torch.zeros(1, 1, size))        # Membrane potential
        self.register_buffer("adapt", torch.zeros(1, 1, size))    # Adaptation state

        # Surrogate gradient slope
        self.surrogate_slope = 25.0

    def reset(self, batch_idx=None):
        """Reset membrane and adaptation state"""
        if batch_idx is None:
            self.v.zero_()
            self.adapt.zero_()
        else:
            if batch_idx >= self.v.shape[0]:
                # Expand buffers
                old_v = self.v
                old_adapt = self.adapt
                new_size = batch_idx + 1
                self.v = torch.zeros(new_size, 1, self.size, device=old_v.device)
                self.adapt = torch.zeros(new_size, 1, self.size, device=old_adapt.device)
                # Copy old
                min_b = min(old_v.shape[0], new_size)
                self.v[:min_b].copy_(old_v[:min_b])
                self.adapt[:min_b].copy_(old_adapt[:min_b])
                
            self.v[batch_idx].zero_()
            self.adapt[batch_idx].zero_()


    def detach_state(self):
        self.v.detach_()
        self.adapt.detach_()



    def forward(self, i_inj):
        """
        i_inj: [B, N, size] or [B, size] — input current
        Returns: spike tensor same shape as i_inj
        """
        B = i_inj.shape[0]
        if i_inj.dim() == 2:
            i_inj = i_inj.unsqueeze(1) # [B, 1, D]

        # Auto-expand states
        if self.v.shape[0] < B:
            old_v, old_a = self.v, self.adapt
            self.v = torch.zeros(B, 1, self.size, device=old_v.device)
            self.adapt = torch.zeros(B, 1, self.size, device=old_a.device)
            self.v[:old_v.shape[0]].copy_(old_v)
            self.adapt[:old_a.shape[0]].copy_(old_a)
        
        target_v = self.v[:B].squeeze(1) # [B, D]
        target_a = self.adapt[:B].squeeze(1) # [B, D]

        # CALL FUSED ALIF KERNEL
        spike, v_f, a_f, saved_v_temp, saved_thresh_dyn = ALIFFunction.apply(
            i_inj, target_v, target_a, 
            self.base_threshold, self.adapt_strength,
            self.tau_mem, self.tau_adapt, self.surrogate_slope,
            torch.is_grad_enabled()
        )

        # In-place update of persistent state
        # CRITICAL: Detach here to prevent the autograd graph from growing across time-windows
        
        # CLAMP FIX: Prevent state explosion leading to NaNs
        v_f = torch.clamp(v_f, min=-100.0, max=100.0)
        a_f = torch.clamp(a_f, min=0.0, max=100.0) # Adaptation is additive, so min=0 is safe
        
        self.v[:B].copy_(v_f.detach().unsqueeze(1))
        self.adapt[:B].copy_(a_f.detach().unsqueeze(1))

        return spike.squeeze(1) if i_inj.shape[1] == 1 else spike

class FeedForwardSNN(nn.Module):
    def __init__(self, d_model, hidden_dim, sparsity_level=0.3):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.lif1 = ALIFNeuron(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.lif2 = ALIFNeuron(d_model)

    def reset(self, batch_idx=None):
        self.lif1.reset(batch_idx)
        self.lif2.reset(batch_idx)

    def detach_state(self):
        self.lif1.detach_state()
        self.lif2.detach_state()

    def forward(self, x):
        x = self.fc1(x)
        x = self.lif1(x)
        x = self.fc2(x)
        x = self.lif2(x)
        return x

# ---------------------- TreeSelfAttention (our novel mechanism) ----------------------
class TreeSelfAttention(nn.Module):
    def __init__(self, d_model, sparsity_level=0.7, tree_depth=4):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        self.num_nodes = (1 << tree_depth) - 1

        self.node_projs = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(self.num_nodes)
        ])
        self.lif_neurons = nn.ModuleList([
            ALIFNeuron(d_model) for _ in range(self.num_nodes)
        ])

        # Learnable weights for mixing node outputs per position
        self.node_weights = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        device = x.device

        # --- Tree setup ---
        padded_T = 1 << math.ceil(math.log2(T))
        effective_depth = min(self.tree_depth, int(math.log2(padded_T)))
        num_leaves = 1 << effective_depth
        sub_seq = T // num_leaves

        if padded_T > T:
            x = F.pad(x, (0, 0, 0, padded_T - T))

        leaves = x.view(B, num_leaves, sub_seq, D).mean(dim=2)  # [B, leaves, D]

        node_states = [None] * self.num_nodes
        leaf_start = (1 << (effective_depth - 1)) - 1

        # --- Bottom-up spiking propagation ---
        for leaf_idx in range(num_leaves):
            gid = leaf_start + leaf_idx
            if gid >= self.num_nodes:
                continue
            vec = leaves[:, leaf_idx]
            proj = self.node_projs[gid](torch.cat([vec, vec], dim=-1))
            spike = self.lif_neurons[gid](proj.unsqueeze(1)).squeeze(1)
            node_states[gid] = proj * spike

        # Process internal nodes (bottom-up)
        for level in range(effective_depth - 2, -1, -1):
            level_start = (1 << level) - 1
            level_size = 1 << level
            for i in range(level_size):
                nid = level_start + i
                if nid >= self.num_nodes:
                    continue
                left = 2 * nid + 1
                right = 2 * nid + 2
                l_out = node_states[left] if left < self.num_nodes and node_states[left] is not None else torch.zeros(B, D, device=device)
                r_out = node_states[right] if right < self.num_nodes and node_states[right] is not None else torch.zeros(B, D, device=device)
                fused = torch.cat([l_out, r_out], dim=-1)
                proj = self.node_projs[nid](fused)
                spike = self.lif_neurons[nid](proj.unsqueeze(1)).squeeze(1)
                node_states[nid] = proj * spike

        # --- Per-position mixture using learned node weights ---
        # Stack all active node outputs: [B, num_active_nodes, D]
        active_states = torch.stack([s for s in node_states if s is not None], dim=1)  # [B, N, D]
        active_weights = self.node_weights[:active_states.shape[1]]  # [N, D]

        # Weighted sum per position (broadcast weights)
        weights = F.softmax(active_weights, dim=0)  # [N, D]
        mixture = torch.einsum('nd,bnd->bd', weights, active_states)  # [B, D]

        # Expand to sequence length
        output = mixture.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        output = self.norm(output + x[:, :T])  # Residual + norm

        return output[:, :T]  # Trim padding

class TreeSelfAttentionHybrid(nn.Module):
    def __init__(self, d_model, sparsity_level=0.7, tree_depth=4, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        self.chunk_size = chunk_size
        self.num_nodes = (1 << tree_depth) - 1

        self.node_projs = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(self.num_nodes)
        ])
        self.lif_neurons = nn.ModuleList([
            ALIFNeuron(d_model) for _ in range(self.num_nodes)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Persistent state for recurrent/chunkwise
        self.register_buffer("node_states", torch.zeros(1, self.num_nodes, d_model))
        self.register_buffer("leaf_buffer", torch.zeros(1, 1 << (tree_depth-1), d_model))
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

    def reset(self):
        self.node_states.zero_()
        self.leaf_buffer.zero_()
        self.step_counter.zero_()

    def forward(self, x, mode="parallel"):
        """
        x: [B, T, d_model] or [B, 1, d_model] in recurrent mode
        mode: "parallel" | "recurrent" | "chunkwise"
        """
        if mode == "parallel":
            return self._forward_parallel(x)
        elif mode == "recurrent":
            return self._forward_recurrent(x)
        elif mode == "chunkwise":
            return self._forward_chunkwise(x)
        else:
            raise ValueError("mode must be parallel, recurrent, or chunkwise")

    def _forward_parallel(self, x):
        # Your existing parallel implementation (with mean-pool leaves)
        B, T, D = x.shape
        padded_T = 1 << math.ceil(math.log2(T))
        eff_depth = min(self.tree_depth, int(math.log2(padded_T)))
        num_leaves = 1 << eff_depth
        sub_seq = T // num_leaves

        if padded_T > T:
            x = F.pad(x, (0, 0, 0, padded_T - T))
        leaves = x.view(B, num_leaves, sub_seq, D).mean(dim=2)

        node_states = [None] * self.num_nodes
        leaf_start = (1 << (eff_depth - 1)) - 1

        # Leaves
        for i in range(num_leaves):
            gid = leaf_start + i
            if gid >= self.num_nodes: continue
            vec = leaves[:, i]
            proj = self.node_projs[gid](torch.cat([vec, vec], dim=-1))
            spike = self.lif_neurons[gid](proj.unsqueeze(1)).squeeze(1)
            node_states[gid] = proj * spike

        # Internal nodes
        for level in range(eff_depth - 2, -1, -1):
            start = (1 << level) - 1
            size = 1 << level
            for i in range(size):
                nid = start + i
                if nid >= self.num_nodes: continue
                l = 2 * nid + 1
                r = 2 * nid + 2
                l_out = node_states[l] if l < self.num_nodes and node_states[l] is not None else torch.zeros(B, D, device=x.device)
                r_out = node_states[r] if r < self.num_nodes and node_states[r] is not None else torch.zeros(B, D, device=x.device)
                fused = torch.cat([l_out, r_out], dim=-1)
                proj = self.node_projs[nid](fused)
                spike = self.lif_neurons[nid](proj.unsqueeze(1)).squeeze(1)
                node_states[nid] = proj * spike

        root = node_states[0] if node_states[0] is not None else torch.zeros(B, D, device=x.device)
        output = root.unsqueeze(1).expand(-1, T, -1)
        return self.norm(output[:, :T] + x[:, :T])

    def _forward_recurrent(self, x_t):
        # One token at a time
        B, _, D = x_t.shape
        leaf_idx = int(self.step_counter.item() % (1 << (self.tree_depth - 1)))
        self.leaf_buffer[:, leaf_idx] = x_t.squeeze(1)
        self.step_counter += 1

        # Rebuild tree from current leaf buffer
        return self._run_tree_from_leaves(self.leaf_buffer.clone())

    def _forward_chunkwise(self, x_chunk):
        # Process chunk in parallel, update persistent state
        B, T, D = x_chunk.shape
        assert T <= self.chunk_size, "Chunk size exceeded"

        # Pad chunk and run parallel
        padded = F.pad(x_chunk, (0, 0, 0, self.chunk_size - T))
        output = self._forward_parallel(padded)

        # Update persistent leaf buffer (sliding window)
        start_idx = int(self.step_counter.item() % (1 << (self.tree_depth - 1)))
        self.leaf_buffer[:, start_idx:start_idx + T] = x_chunk.mean(dim=1)  # mean-pool chunk into leaves
        self.step_counter += T

        return output[:, :T]

    def _run_tree_from_leaves(self, leaves):
        # Shared bottom-up logic used by recurrent/chunkwise
        B, num_leaves, D = leaves.shape
        node_states = self.node_states.clone()

        leaf_start = (1 << (self.tree_depth - 1)) - 1
        for i in range(num_leaves):
            gid = leaf_start + i
            if gid >= self.num_nodes: continue
            vec = leaves[:, i]
            proj = self.node_projs[gid](torch.cat([vec, vec], dim=-1))
            spike = self.lif_neurons[gid](proj.unsqueeze(1)).squeeze(1)
            node_states[:, gid] = proj * spike

        for level in range(self.tree_depth - 2, -1, -1):
            start = (1 << level) - 1
            size = 1 << level
            for i in range(size):
                nid = start + i
                if nid >= self.num_nodes: continue
                l = 2 * nid + 1
                r = 2 * nid + 2
                l_out = node_states[:, l] if l < self.num_nodes else torch.zeros(B, D, device=leaves.device)
                r_out = node_states[:, r] if r < self.num_nodes else torch.zeros(B, D, device=leaves.device)
                fused = torch.cat([l_out, r_out], dim=-1)
                proj = self.node_projs[nid](fused)
                spike = self.lif_neurons[nid](proj.unsqueeze(1)).squeeze(1)
                node_states[:, nid] = proj * spike

        self.node_states = node_states.detach()
        root = node_states[:, 0]
        return self.norm(root.unsqueeze(1))

class TreeSelfAttentionGPU(nn.Module):
    def __init__(self, d_model, tree_depth=4):
        super().__init__()
        self.d_model = d_model
        self.tree_depth = tree_depth
        self.num_nodes = (1 << tree_depth) - 1

        self.node_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.lif_levels = nn.ModuleList([ALIFNeuron(d_model) for _ in range(tree_depth)])
        self.node_weights = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.02)
        self.norm = nn.RMSNorm(d_model)

        # Proper topology
        parent, left, right = [], [], []
        for n in range(self.num_nodes):
            l = 2 * n + 1
            r = 2 * n + 2
            if l < self.num_nodes:
                parent.append(n)
                left.append(l)
                right.append(r if r < self.num_nodes else -1)
        self.register_buffer('parent_idx', torch.tensor(parent))
        self.register_buffer('left_idx', torch.tensor(left))
        self.register_buffer('right_idx', torch.tensor(right))

    def forward(self, x):
        B, T, D = x.shape
        device = x.device

        padded_T = 1 << math.ceil(math.log2(T))
        if padded_T > T:
            x = F.pad(x, (0, 0, 0, padded_T - T))

        eff_depth = min(self.tree_depth, int(math.log2(padded_T)))
        num_leaves = 1 << eff_depth
        sub_seq = padded_T // num_leaves

        leaves = x.view(B, num_leaves, sub_seq, D)
        leaves = leaves.mean(dim=2) if sub_seq > 1 else leaves.squeeze(2)

        node_states = torch.zeros(B, self.num_nodes, D, device=device)
        leaf_start = (1 << (eff_depth - 1)) - 1
        valid = min(num_leaves, self.num_nodes - leaf_start)
        node_states[:, leaf_start:leaf_start + valid] = leaves[:, :valid]

        # Vectorized bottom-up
        for level in range(eff_depth - 1, -1, -1):
            level_start = (1 << level) - 1
            level_end = min((1 << (level + 1)) - 1, self.num_nodes)
            nodes = torch.arange(level_start, level_end, device=device)
            if len(nodes) == 0: continue

            l_idx = 2 * nodes + 1
            r_idx = 2 * nodes + 2
            valid = l_idx < self.num_nodes

            l_states = node_states[:, l_idx[valid]]
            r_states = node_states[:, r_idx[valid]] if (r_idx[valid] < self.num_nodes).any() else torch.zeros_like(l_states)

            fused = torch.cat([l_states, r_states], dim=-1)
            proj = self.node_proj(fused)
            spike = self.lif_levels[level](proj)
            node_states[:, nodes[valid]] = proj * spike

        # Hierarchical mixture with depth bias
        weights = F.softmax(self.node_weights, dim=0)
        mixture = torch.einsum('nd,bnd->bd', weights, node_states)
        output = mixture.unsqueeze(1).expand(-1, T, -1)
        return self.norm(output + x[:, :T])[:, :T]

def rope(x, cos, sin):
    return RoPEFunction.apply(x, cos, sin)

def rotate_half_old(x):
    x_dim = x.shape[-1] // 2
    x1, x2 = x.split(x_dim, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def rotate_half(x):
    return RotateHalf.apply(x)

def apply_rotary_emb(x, cos, sin):
    # x: [B, N, D]
    # cos, sin: [D] (pre-expanded and broadcastable)
    return rope(x, cos, sin)

class LowRankLinear(nn.Module):
    """Two-stage linear projection: ``in_features → rank → out_features``.

    The effective weight matrix is ``W = B @ A`` where ``A`` maps the input to a
    low-dimensional latent space (``rank``) and ``B`` maps back to the full
    output dimension.  This reduces parameters and FLOPs when ``rank <<
    out_features`` while keeping the same functional form as a regular linear
    layer.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        rank (int, optional): Bottleneck dimension. If ``None`` a default of
            ``max(1, out_features // 4)`` is used.
        bias (bool): Whether to include a bias term on the second projection
            (mirrors ``nn.Linear`` behaviour).
    """

    def __init__(self, in_features: int, out_features: int,
                 rank: int | None = None, bias: bool = True, groups: int = 1):
        super().__init__()
        if rank is None:
            rank = max(1, out_features // 4)

        # Ensure rank is divisible by groups
        if rank % groups != 0:
            rank = ((rank + groups - 1) // groups) * groups

        self.rank = rank
        self.groups = groups
        self.in_features = in_features
        self.out_features = out_features

        if groups == 1:
            self.proj_in = nn.Linear(in_features, rank, bias=False)
            self.proj_out = nn.Linear(rank, out_features, bias=bias)
            # Initialise the factors with sensible defaults
            nn.init.kaiming_uniform_(self.proj_in.weight, a=math.sqrt(5))
            nn.init.xavier_uniform_(self.proj_out.weight)
            if bias:
                nn.init.zeros_(self.proj_out.bias)
        else:
            # Use custom parameters for grouped projection via einsum
            self.proj_in_weight = nn.Parameter(torch.empty(groups, rank // groups, in_features // groups))
            self.proj_out_weight = nn.Parameter(torch.empty(groups, out_features // groups, rank // groups))
            if bias:
                self.proj_out_bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter('proj_out_bias', None)
            self.reset_parameters_grouped()

    def reset_parameters_grouped(self):
        # Initialize weights per group to mimic nn.Linear initialization
        for g in range(self.groups):
            nn.init.kaiming_uniform_(self.proj_in_weight[g], a=math.sqrt(5))
            nn.init.xavier_uniform_(self.proj_out_weight[g])
        if self.proj_out_bias is not None:
            nn.init.zeros_(self.proj_out_bias)

    def forward(self, x: torch.Tensor, input_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the low-rank factorisation.

        Shape: ``(B, *, in_features) → (B, *, out_features)``.
        """
        if self.groups == 1:
            if input_bias is not None:
                # Fuse input bias addition: proj_in(x + bias) = proj_in(x) + proj_in(bias)
                #hidden = self.proj_in(x) + self.proj_in(input_bias)
                hidden = F.linear(x, self.proj_in.weight, input_bias)
            else:
                hidden = self.proj_in(x)
            return self.proj_out(hidden)
        else:
            # Optimized implementation using permute + bmm to avoid einsum overhead
            # Flatten batch dimensions to (N, D)
            x_flat = x.view(-1, self.in_features)
            N = x_flat.shape[0]
            
            # Reshape to (N, G, I_per_group) and permute to (G, N, I_per_group) for BMM
            x_grouped = x_flat.view(N, self.groups, -1).permute(1, 0, 2)
            
            # Projections using BMM: (G, N, I) @ (G, I, O) -> (G, N, O)
            hidden = torch.bmm(x_grouped, self.proj_in_weight.transpose(1, 2))
            
            if input_bias is not None:
                # Project bias separately and add to hidden to avoid large tensor addition
                b_flat = input_bias.view(-1, self.in_features)
                b_grouped = b_flat.view(b_flat.shape[0], self.groups, -1).permute(1, 0, 2)
                b_hidden = torch.bmm(b_grouped, self.proj_in_weight.transpose(1, 2))
                hidden = hidden + b_hidden

            out_grouped = torch.bmm(hidden, self.proj_out_weight.transpose(1, 2))
            
            # Permute back to (N, G, O) and reshape to original dimensions
            out = out_grouped.permute(1, 0, 2).reshape(*x.shape[:-1], self.out_features)
            
            if self.proj_out_bias is not None:
                out = out + self.proj_out_bias
            return out

class LowRankSwiGLU(nn.Module):
    """Low-rank factorised SwiGLU shared across tree levels.

    The original ``SharedSwiGLU`` used a single ``nn.Linear`` mapping
    ``in_features -> out_features * 2``.  To reduce parameters and FLOPs we
    factorise this projection into two smaller linear layers:

    ``in_features -> rank`` (no bias) followed by ``rank -> out_features * 2``
    (with bias).  The effective weight matrix is ``W = B @ A`` where ``A`` maps
    to the low-dimensional space and ``B`` maps back to the full output size.

    ``rank`` can be tuned; a sensible default is ``max(1, out_features // 4)``.
    """
    def __init__(self, in_features, out_features, rank=None, num_levels=16, groups=1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if rank is None:
            # Default rank is a quarter of the output dimension (at least 1)
            rank = max(1, out_features // 4)
        self.rank = rank
        # Low‑rank linear projection that outputs ``out_features * 2`` values
        # (value + gate for SwiGLU). Bias is kept on the second projection to
        # match the behaviour of a standard ``nn.Linear``.
        self.low_rank = LowRankLinear(in_features, out_features * 2,
                                      rank=self.rank, bias=True, groups=groups)

        # Per‑level bias (same as original implementation)
        self.level_bias = nn.Parameter(torch.zeros(num_levels, in_features))
        self.act = nn.SiLU()

    def forward(self, x, level_idx, extra_bias=None):
        # Add per‑level bias before the low‑rank projection
        bias = self.level_bias[level_idx]
        if extra_bias is not None:
            #bias = bias + extra_bias
            x = addadd(x, bias, extra_bias)
        else:
            x = x + bias
        # Low‑rank factorisation via the helper module
        projected = self.low_rank(x)
        # SAFETY CLAMP: Prevent Inf in forward (causing NaN in RMSNorm) and NaN in backward (SwiGLU grad)
        #projected = torch.clamp(projected, min=-100.0, max=100.0)
        return swiglu_activation(projected)

class SharedSwiGLU(nn.Module):
    """Parameter-efficient Gated Linear Unit shared across tree levels."""
    def __init__(self, in_features, out_features, num_levels=16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features * 2)
        # Revert to input-space bias: smaller dimension -> less memory movement
        self.level_bias = nn.Parameter(torch.zeros(num_levels, in_features))
        self.act = nn.SiLU()

    def forward(self, x, level_idx):
        # Add bias to input before projection
        # This is more efficient when in_features < out_features * 2 (which is true for SwiGLU)
        projected = self.linear(x + self.level_bias[level_idx].to(x.dtype))
        # SAFETY CLAMP: Prevent Inf in forward and NaN in backward
        #projected = torch.clamp(projected, min=-100.0, max=100.0)
        return swiglu_activation(projected)

class LateralMixing(nn.Module):
    """
    Applies a depthwise 1D convolution using efficient einsum and unfolding.
    This allows information to leak horizontally between neighbors (cousins)
    before they are merged vertically, reducing boundary artifacts in the tree.
    """
    def __init__(self, d_model, kernel_size=3, causal=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal = causal
        # Causal padding: (kernel_size - 1) on the left. Symmetric: kernel_size // 2 on both sides.
        self.padding = kernel_size - 1 if causal else kernel_size // 2
        
        # Depthwise weights: [D, K]. Initialized close to identity.
        self.weight = nn.Parameter(torch.zeros(d_model, kernel_size))
        with torch.no_grad():
            center_idx = kernel_size - 1 if causal else kernel_size // 2
            self.weight[:, center_idx] = 1.0
            self.weight += torch.randn_like(self.weight) * 0.02

    def forward(self, x):
        # x: [B, N, D]
        return lateral_mixing(x, self.weight, self.causal)

class TestTreeSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, tree_depth=4, is_causal=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.is_causal = is_causal
        self.tree_depth = tree_depth
        self.rank = max(1, d_model // 4)
        
        # Base processing
        self.node_proj = nn.Linear(d_model, d_model)
        self.lif_neuron = ALIFNeuron(d_model)
        self.base_proj = BaseProj()
        
        # Initialize base_proj to act as a mean (0.5, 0.5) initially for stability
        with torch.no_grad():
            self.base_proj.weight.fill_(0.5)
            self.base_proj.bias.zero_()
        
        # Hierarchical layers (Recursive Heterogeneity)
        # Replaced ModuleLists with SharedSwiGLU for parameter efficiency and expressiveness
        self.merge_layer = LowRankSwiGLU(d_model * 2, d_model, rank=self.rank, groups=n_heads)
        self.broadcast_layer = LowRankSwiGLU(d_model, d_model, rank=self.rank, groups=n_heads)
        self.context_merge_layer = LowRankSwiGLU(d_model, d_model, rank=self.rank, groups=n_heads)
        self.skip_layer = LowRankSwiGLU(d_model, d_model, rank=self.rank, groups=n_heads)
        
        # Lateral Mixing (Suggestion #2)
        self.lateral_mix = LateralMixing(d_model, kernel_size=3, causal=is_causal)
        
        self.gate_proj = FusedLinear(d_model, d_model)

        # Memory‑efficient state handling (Suggestion #5)
        # Store node states in reduced precision to cut memory usage roughly in half.
        # Users can disable compression via the flag if full precision is required.
        self.use_state_compression = False
        self.state_dtype = torch.float16

        # Per-Level Path encodings (Fractal Positional Awareness)
        # 16 levels supports up to 65k sequence length (2^16)
        # Shape: [16, 2, D] -> [Level, Side, D]
        self.level_paths = nn.Parameter(torch.randn(16, 2, d_model) * 0.02)

        # Stateful Tree Nodes (Temporal Memory)
        self.node_states = None # Will hold a single Arena Tensor [B, Total_Nodes, D]
        self.state_gate = nn.Parameter(torch.ones(16) * -2.0) # Initialized to bias towards new info
        
        # TREE RoPE: Level-specific frequencies
        # We need D/2 frequencies for D-dimensional rotation
        self.rope_theta = nn.Parameter(torch.randn(16, d_model // 2) * 0.02)

        # Normalization layers
        self.norm_merge = FusedRMSNorm(d_model)
        self.post_merge_norm = FusedRMSNorm(d_model)
        self.norm_broadcast = FusedRMSNorm(d_model)

        self.register_buffer("zero_ctx", torch.zeros(1, 1, d_model))
        self.register_buffer("rope_cos", torch.zeros(16, d_model), persistent=False)
        self.register_buffer("rope_sin", torch.zeros(16, d_model), persistent=False)
        self._rope_cache_init = False

    def reset_state(self, batch_idx=None):
        if batch_idx is None:
            if self.node_states is not None:
                self.node_states.zero_()
        elif self.node_states is not None and batch_idx < self.node_states.shape[0]:
            self.node_states[batch_idx].zero_()

        self.lif_neuron.reset(batch_idx)

    def detach_state(self):
        if self.node_states is not None:
            self.node_states.detach_()
        self.lif_neuron.detach_state()

    def collect_state(self):
        return {
            "node_states": self.node_states.clone() if self.node_states is not None else None,
            "lif_v": self.lif_neuron.v.clone(),
            "lif_a": self.lif_neuron.adapt.clone() if hasattr(self.lif_neuron, 'adapt') else None
        }

    def load_state(self, state):
        """Load a previously saved state.

        If ``use_state_compression`` is enabled we expect the stored tensors to be
        in ``self.state_dtype`` and convert them back to the model's default dtype.
        """
        if state["node_states"] is not None:
            # Convert back to model dtype if compressed
            if self.use_state_compression:
                self.node_states = state["node_states"].to(torch.get_default_dtype())
            else:
                self.node_states = state["node_states"].clone()
        else:
            self.node_states = None
        self.lif_neuron.v.copy_(state["lif_v"])
        if state["lif_a"] is not None:
            self.lif_neuron.adapt.copy_(state["lif_a"])


    def forward(self, x):
        # Force detach previous states to strictly truncate BPTT within the graph
        prev_node_states = None
        if self.node_states is not None:
            # Detach and, if compressed, cast back to the model's default dtype
            if self.use_state_compression:
                prev_node_states = self.node_states.to(torch.get_default_dtype()).detach()
            else:
                prev_node_states = self.node_states.clone().detach()
        
        # x shape: [B, T, D]

        B, T, D = x.shape

        # Pad to power of 2 (Safe for torch.compile)
        orig_T = T
        if int(math.log2(T)) != math.log2(T):
            padded_T = 1
            while padded_T < T:
                padded_T *= 2
            
            if padded_T > T:
                x = F.pad(x, (0, 0, 0, padded_T - T))
            T = padded_T
        
        # num_levels
        num_levels = 0
        temp_T = T
        while temp_T > 1:
            temp_T //= 2
            num_levels += 1
        
        # Optimization: Pre-calculate level offsets for Static Memory Arena
        # Level 0: [0, T/2)
        # Level 1: [T/2, T/2 + T/4)
        # ...
        level_offsets = [0]
        curr_len = T // 2
        for _ in range(num_levels): 
            level_offsets.append(level_offsets[-1] + curr_len)
            curr_len //= 2
        total_nodes = level_offsets[-1]
        
        # Invalidate previous state if the tree structure (total_nodes) has changed
        # This prevents shape mismatch errors when sequence length varies (e.g. 4 -> 8)
        if prev_node_states is not None and prev_node_states.shape[1] != total_nodes:
            prev_node_states = None
        
        # Ensure persistent state buffer exists and is correct size
        # We do this BEFORE any computation so we can use copy_() later
        if self.node_states is None or self.node_states.shape != (B, total_nodes, D):
            self.node_states = torch.zeros(B, total_nodes, D, device=x.device, dtype=self.state_dtype if self.use_state_compression else x.dtype)
        
        
        # Precompute RoPE cos/sin for all levels to avoid re-computation in loop
        # rope_theta: [16, D/2] -> [16, D]
        if not self._rope_cache_init or (self.training and self.rope_theta.requires_grad):
            cos = torch.repeat_interleave(self.rope_theta.cos(), 2, dim=-1)
            sin = torch.repeat_interleave(self.rope_theta.sin(), 2, dim=-1)
            self.rope_cos.copy_(cos.detach())
            self.rope_sin.copy_(sin.detach())
            self._rope_cache_init = True
        else:
            cos, sin = self.rope_cos, self.rope_sin

        # 1. BASE LEVEL PROCESSING (T -> T/2)
        # Apply Path Encodings for Level 0
        #x_pairs = x.reshape(B, T // 2, 2, D).clone()
        x_pairs = x.view(B, T // 2, 2, D)
        
        # Use F.linear for bias to avoid module hook issues with Parameter inputs
        effective_bias = F.linear(self.level_paths[0], self.node_proj.weight, self.node_proj.bias)
        base_out = self.lif_neuron(F.linear(x_pairs, self.node_proj.weight, effective_bias))
        
        # Allocate Static Memory Arena for Bottom-Up Summaries
        # [B, Total_Nodes, D]
        # Use zeros to prevent NaNs from uninitialized memory in compiled mode
        summary_arena = torch.zeros(B, total_nodes, D, device=x.device, dtype=x.dtype)
        
        # Level 0 Processing
        current_seq = base_out.reshape(B, T, D)
        #print(current_seq)
        
        # NEW: Learnable linear combination of left and right child states
        reshaped_base = base_out.reshape(B, T//2, D, 2)
        raw_base_sum = self.base_proj(reshaped_base).squeeze(-1)

        # Pre-calculate residual for fusion
        sibling_proj = self.base_proj(base_out.view(B, T//2, D, 2)).squeeze(-1)
        sibling_residual = self.skip_layer(sibling_proj, 0)
        
        #all_gates = torch.sigmoid(self.state_gate)
        
        if prev_node_states is not None:
            # Level 0 is at offset 0
            prev_s = prev_node_states[:, 0:T//2] 
            if prev_s.shape[0] == B:
                # FUSED TREE CELL (Level 0) - Handles [Norm1 -> Gate -> Norm2 -> AddResidual]
                current_sums = tree_cell_fused(
                    raw_base_sum, prev_s, sibling_residual,
                    self.state_gate, self.norm_merge.weight, self.post_merge_norm.weight,
                    1e-6, 0
                )
            else:
                # Apply both norms to match kernel topology (Norm2(Norm1(x)) + res)
                current_sums = self.post_merge_norm(self.norm_merge(raw_base_sum)) + sibling_residual
        else:
            current_sums = self.post_merge_norm(self.norm_merge(raw_base_sum)) + sibling_residual
        
        # Write Level 0 to Arena
        summary_arena[:, level_offsets[0]:level_offsets[1]] = current_sums
        
        # 2. BOTTOM-UP PASS (Iterative merging)
        # Each step: [B, N, D] -> [B, N/2, D]
        # Level 1 and above
        for l in range(num_levels - 1):
            # current_sums is already set from previous iteration
            
            level_idx = l + 1 # We are calculating Level 1, 2, ...
            N = current_sums.shape[1]
            if N < 2: break
            
            # Apply Lateral Mixing to allow info to flow between cousins before merging
            # Optimization: Skip for very short sequences where mixing adds overhead with little gain
            if N >= 4:
                current_sums = self.lateral_mix(current_sums)
            
            # Combine siblings and apply level-specific path encodings (Level l+1)
            siblings = current_sums.view(B, N // 2, 2, D)
            
            # Optimization: Fuse path encoding addition into the merge layer's bias
            path_bias = self.level_paths[level_idx].view(-1) # Flatten [2, D] -> [2*D]
            raw_sums = self.merge_layer(siblings.view(B, N // 2, 2 * D), level_idx, extra_bias=path_bias)
            
            # Suggestion #4: Add skip connection in bottom-up loop
            reshaped_siblings = siblings.view(B, N // 2, D, 2)
            sibling_proj = self.base_proj(reshaped_siblings).squeeze(-1)
            sibling_residual = self.skip_layer(sibling_proj, level_idx)

            # Mix with previous state if exists (now both normalized)
            #state_gate = all_gates[level_idx]
            if prev_node_states is not None:
                # Extract previous state for this level from arena
                prev_s = prev_node_states[:, level_offsets[l+1]:level_offsets[l+2]]
                if prev_s.shape[0] == B:
                    # FUSED TREE CELL (Level L)
                    current_sums = tree_cell_fused(
                        raw_sums, prev_s, sibling_residual,
                        self.state_gate, self.norm_merge.weight, self.post_merge_norm.weight,
                        1e-6, level_idx
                    )
                else:
                    # CRITICAL FIX: Apply normalization even without prev state
                    # Prevents signal explosion in deep trees
                    current_sums = self.post_merge_norm(self.norm_merge(raw_sums)) + sibling_residual
            else:
                current_sums = self.post_merge_norm(self.norm_merge(raw_sums)) + sibling_residual
            
            # Write to Arena
            summary_arena[:, level_offsets[l+1]:level_offsets[l+2]] = current_sums

        # 3. TOP-DOWN PASS (Iterative Descent - O(N))
        
        # Allocate Context Arena (same structure as summary arena)
        # We will write into this directly to avoid stack/cat overhead
        context_arena = torch.zeros(B, total_nodes, D, device=x.device, dtype=x.dtype)
        
        # Start with Root Summary from previous window (if exists) as context
        # Root is at the last populated offset
        root_start = level_offsets[num_levels-1]
        root_end = level_offsets[num_levels]
        
        if prev_node_states is not None:
            prev_root = prev_node_states[:, root_start:root_end]
            if prev_root.shape[0] == B:
                context_arena[:, root_start:root_end] = prev_root
            else:
                context_arena[:, root_start:root_end] = self.zero_ctx.expand(B, -1, -1)
        else:
            context_arena[:, root_start:root_end] = self.zero_ctx.expand(B, -1, -1)
        
        for l in reversed(range(num_levels)):
            # Read parent context from arena
            current_ctx = context_arena[:, level_offsets[l]:level_offsets[l+1]]
            N_parent = current_ctx.shape[1]
            
            # 1. Project and expand parent context for both children
            parent_ctx_processed = self.broadcast_layer(current_ctx, l)
            # Reshape to [B, 1, N_parent, D] and expand to [B, 2, N_parent, D]
            # This avoids torch.cat, which is a major source of overhead for torch.compile
            children_ctx = parent_ctx_processed.unsqueeze(1).expand(-1, 2, -1, -1)
            
            #print(children_ctx.shape)
            
            # 2. Apply path encodings and rotations to the expanded context
            # self.level_paths[l] is [2, D] -> unsqueeze to [2, 1, D] to broadcast
            children_ctx = children_ctx + self.level_paths[l].unsqueeze(1)
            
            # Now slicing dim 1 keeps the remaining dimensions packed linearly
            left_ctx_view = children_ctx[:, 0, :, :]
            right_ctx_view = children_ctx[:, 1, :, :]

            #print(children_ctx.is_contiguous())
            #print(right_ctx_view.is_contiguous())

            if self.is_causal:
                # Right child gets RoPE and sibling summary injection
                rotated_right = apply_rotary_emb(right_ctx_view, cos[l], sin[l])
                
                if l > 0:
                    # Read child summaries from summary_arena
                    child_level_summaries = summary_arena[:, level_offsets[l-1]:level_offsets[l]].view(B, -1, 2, D)
                    left_child_sums = child_level_summaries[:, :, 0, :]
                    final_right = rotated_right + self.context_merge_layer(left_child_sums, l)
                    
                    # Fused Write (Causal)
                    dest = context_arena[:, level_offsets[l-1]:level_offsets[l]].view(B, N_parent, 2, D)
                    dest[:, :, 0, :] = left_ctx_view
                    dest[:, :, 1, :] = final_right
                else:
                    left_tokens = current_seq.view(B, N_parent, 2, D)[:, :, 0, :]
                    final_right = rotated_right + self.context_merge_layer(left_tokens, l)
                    current_ctx = torch.stack([left_ctx_view, final_right], dim=2).view(B, N_parent * 2, D)
            else:
                # Non-causal: Use symmetric rotations for balance
                new_left = apply_rotary_emb(left_ctx_view, self.rope_cos[l], -self.rope_sin[l])
                new_right = apply_rotary_emb(right_ctx_view, self.rope_cos[l], self.rope_sin[l])
                new_left = apply_rotary_emb(left_ctx_view, cos[l], -sin[l])
                new_right = apply_rotary_emb(right_ctx_view, cos[l], sin[l])
                
                if l > 0:
                    # Fused Write (Non-Causal)
                    dest = context_arena[:, level_offsets[l-1]:level_offsets[l]].view(B, N_parent, 2, D)
                    dest[:, :, 0, :] = new_left
                    dest[:, :, 1, :] = new_right
                else:
                    current_ctx = torch.stack([new_left, new_right], dim=2).view(B, N_parent * 2, D)
        
        # 4. FINAL GATED UPDATE
        # current_ctx at the end of loop is Level 0 context (leaves)
        #gate = torch.sigmoid(self.gate_proj(current_seq))
        gate = self.gate_proj(current_seq)
        #current_seq = current_seq + gate * self.norm_broadcast(current_ctx)
        current_seq = torch.addcmul(current_seq, gate, self.norm_broadcast(current_ctx))

        current_seq = current_seq[:, :orig_T]
        #print(current_seq)

        # Final Update of persistent states for the NEXT window
        # Store states for the next forward pass. Apply optional compression to
        # reduce memory footprint.
        # CRITICAL FIX: Use copy_() to update persistent buffer in-place.
        # This prevents the tensor address from changing, which is required for CUDA Graphs.
        if self.use_state_compression:
            self.node_states.copy_(summary_arena.detach().to(self.state_dtype))
        else:
            self.node_states.copy_(summary_arena.detach())

        return current_seq

class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # 1. Initialize weights: standard practice is normal or uniform distribution
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # PyTorch defaults to normal distribution initialization
        init.normal_(self.weight)
        # If padding_idx is used, that row must be zeroed out
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input):
        # 2. Reimplementing the lookup via simple tensor indexing
        # This returns the rows of the weight matrix corresponding to the input indices
        return self.weight[input]

# ---------------------- Crina-Synapse Block ----------------------
class CrinaBlock(nn.Module):
    def __init__(self, d_model, tree_depth=4):
        super().__init__()
        self.pre_norm_attn = nn.RMSNorm(d_model)
        #self.attn = TreeSelfAttentionGPU(d_model, tree_depth=tree_depth)
        self.attn = TestTreeSelfAttention(d_model, tree_depth=tree_depth)
        self.post_norm_attn = nn.RMSNorm(d_model)
        
        self.pre_norm_ff = nn.RMSNorm(d_model)
        self.ff = FeedForwardSNN(d_model, hidden_dim=4 * d_model)
        self.post_norm_ff = nn.RMSNorm(d_model)

        # Learnable residual scaling factors (initialized <1.0 for stability)
        self.residual_scale_attn = nn.Parameter(torch.tensor(0.1))  # Start at 0.1
        self.residual_scale_ff = nn.Parameter(torch.tensor(0.1))    # Start at 0.1

    def reset_state(self, batch_idx=None):
        self.attn.reset_state(batch_idx)
        self.ff.reset(batch_idx)

    def detach_state(self):
        self.attn.detach_state()
        self.ff.detach_state()

    def collect_state(self):
        return {"attn": self.attn.collect_state(), "ff": [self.ff.lif1.v.clone(), self.ff.lif1.adapt.clone(), self.ff.lif2.v.clone(), self.ff.lif2.adapt.clone()]}

    def load_state(self, state):
        self.attn.load_state(state["attn"])
        self.ff.lif1.v.copy_(state["ff"][0])
        self.ff.lif1.adapt.copy_(state["ff"][1])
        self.ff.lif2.v.copy_(state["ff"][2])
        self.ff.lif2.adapt.copy_(state["ff"][3])




    def forward(self, x):

        # Sandwich attention with scaled residual
        attn_out = self.post_norm_attn(self.attn(self.pre_norm_attn(x)))
        x = x + self.residual_scale_attn * attn_out
        
        # Sandwich FFN with scaled residual
        ff_out = self.post_norm_ff(self.ff(self.pre_norm_ff(x)))
        x = x + self.residual_scale_ff * ff_out
        
        return x

# ---------------------- Full Model ----------------------
class CrinaSynapse(nn.Module):
    def __init__(self, vocab_size=65, d_model=256, n_layers=8, tree_depth=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([CrinaBlock(d_model, tree_depth=tree_depth) for _ in range(n_layers)])
        self.ln_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def reset_state(self, batch_idx=None):
        for block in self.blocks:
            block.reset_state(batch_idx)

    def detach_state(self):
        for block in self.blocks:
            block.detach_state()

    def collect_state(self):
        return [block.collect_state() for block in self.blocks]

    def load_state(self, states):
        for block, state in zip(self.blocks, states):
            block.load_state(state)

    def forward(self, idx):
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def reset_all_lif_neurons(model):
    for module in model.modules():
        if isinstance(module, LIFNeuron) or isinstance(module, ALIFNeuron):
            module.reset()

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-6, eta_min=1e-5):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

    def get_lr(self, epoch):
        default_lr = self.optimizer.defaults['lr']
        if epoch < self.warmup_epochs:
            # Linear warmup
            return (default_lr - self.warmup_start_lr) / self.warmup_epochs * epoch + self.warmup_start_lr
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return self.eta_min + 0.5 * (default_lr - self.eta_min) * (1 + math.cos(math.pi * progress))

    def step(self, epoch):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(epoch)

# ---------------------- Training Loop ----------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists('tiny_shakespeare.txt'):
        print('Downloading TinyShakespeare dataset...')
        url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        response = requests.get(url)
        with open('tiny_shakespeare.txt', 'w') as f:
            f.write(response.text)

    # Load TinyShakespeare
    with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_ds = TinyShakespeareDataset(train_data, block_size=128)
    val_ds = TinyShakespeareDataset(val_data, block_size=128)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    num_epochs = 10
    max_train_iters = num_epochs * len(train_loader)
    warmup_iters = 100
    model = CrinaSynapse(vocab_size=vocab_size, d_model=256, n_layers=8, tree_depth=4).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = CosineWarmupScheduler(optimizer, warmup_start_lr=1e-4, warmup_epochs=warmup_iters, max_epochs=max_train_iters)

    torch.set_float32_matmul_precision('high')
    model = torch.compile(model, fullgraph=True)
    print(f"Model parameters: {sum([p.numel() for p in model.parameters()]):_}")
    wandb.init(project="crina_tinyshakespeare", name="crina_tinyshakespeare", config={
        "model": "CrinaSynapse",
        "vocab_size": vocab_size,
        "d_model": 256,
        "n_layers": 8,
        "tree_depth": 4,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "num_epochs": 1,
        "warmup_iters": warmup_iters,
        "max_train_iters": max_train_iters
    })
    current_iter = 0
    for epoch in range(1):
        reset_all_lif_neurons(model)
        model.train()
        #print(list(enumerate(train_loader))[0])
        pbar = tqdm(train_loader, desc="Training...")
        for x, y in pbar:
            x, y = x.cuda(), y.cuda()
            model.reset_state() # Reset state because batches are shuffled and unrelated
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients to prevent NaN            
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            scheduler.step(current_iter)
            current_iter += 1
            wandb.log({"train_loss": loss.item()})
        print(f"Epoch {epoch} | Train loss: {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            pbar2 = tqdm(val_loader, desc="Validating...")
            for x, y in pbar2:
                x, y = x.cuda(), y.cuda()
                logits = model(x)
                loss_item = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()
                val_loss += loss_item
                pbar2.set_postfix({'val_loss': loss_item})
            val_loss /= len(val_loader)
            wandb.log({"val_loss": val_loss})
        print(f"Epoch {epoch} | Val loss: {val_loss:.4f}")
        
    
    # Save the original model's state dict (uncompiled) to avoid _orig_mod prefix
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(raw_model.state_dict(), "crina_tinyshakespeare.pth")
    wandb.save("crina_tinyshakespeare.pth")
    wandb.finish()

if __name__ == "__main__":
    train()