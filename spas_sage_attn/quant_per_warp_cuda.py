"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from typing import Any, List, Literal, Optional, Tuple, Union

from . import _fused
_fused = torch.ops.spas_sage_attn_fused

def per_block_int8(
    q: torch.Tensor, 
    k: torch.Tensor, 
    km: Optional[torch.Tensor] = None,
    BLKQ: int =128, 
    BLKK: int =64, 
    sm_scale: Optional[float] = None, 
    tensor_layout: str ="HND"
):
    """
    Quantize the query tensor `q` and the key tensor `k` with per block quantization.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    km : Optional[torch.Tensor]
        The mean tensor of `k` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]``.
        Should be of the same dtype as `k` if provided. Default is None.
    
    sm_scale : Optional[float]
        The scale factor for the softmax operation. Default is ``head_dim**-0.5``. 
        It will be multiplied by ``1.44269504`` to work together with the triton attention kernel.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - The quantized query tensor. Shape: Same as `q` but with `int8` dtype.
        - The scale tensor of the query tensor. Shape: ``[batch_size, num_qo_heads, (qo_len + BLKQ - 1) // BLKQ]`` with `float32` dtype.
        - The quantized key tensor. Shape: Same as `k` but with `int8` dtype.
        - The scale tensor of the key tensor. Shape: ``[batch_size, num_kv_heads, (kv_len + BLKK - 1) // BLKK]`` with `float32` dtype.
    
    Note
    ----
    - The tensors `q` and `k` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    """

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = torch.empty((b, h_qo, (qo_len + BLKQ - 1) // BLKQ), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + BLKK - 1) // BLKK), device=q.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5
    
    sm_scale *= 1.44269504

    _fused.quant_per_block_int8_cuda(q, q_int8, q_scale, sm_scale, BLKQ, _tensor_layout)
    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, BLKK, _tensor_layout)
    else:
        _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, BLKK, _tensor_layout)

    return q_int8, q_scale, k_int8, k_scale

def per_warp_int8(
    q: torch.Tensor, 
    k: torch.Tensor, 
    km: Optional[torch.Tensor] = None, 
    tensor_layout: str ="HND"
):
    """
    Quantize the query tensor `q` with per warp quantization and the key tensor `k` with per block quantization.
    Warp size of quantizing `q` is 32, with a block size of 128.
    Block size of quantizing `k` is 64.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    km : Optional[torch.Tensor]
        The mean tensor of `k` along the sequence length dimension. Shape: ``[batch_size, num_kv_heads, head_dim]``.
        Should be of the same dtype as `k` if provided. Default is None.
    
    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        A tuple containing:
        - The quantized query tensor. Shape: Same as `q` but with `int8` dtype.
        - The scale tensor of the query tensor. Shape: ``[batch_size, num_qo_heads, (qo_len + BLKQ - 1) // 128 * 4]`` with `float32` dtype.
        - The quantized key tensor. Shape: Same as `k` but with `int8` dtype.
        - The scale tensor of the key tensor. Shape: ``[batch_size, num_kv_heads, (kv_len + BLKK - 1) // 64]`` with `float32` dtype.
    
    Note
    ----
    - The tensors `q` and `k` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    """

    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    
    _tensor_layout = 0 if tensor_layout == "NHD" else 1

    q_scale = torch.empty((b, h_qo, ((qo_len + 127) // 128) * (128 // 32)), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, (kv_len + 63) // 64), device=q.device, dtype=torch.float32)

    _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, _tensor_layout)

    if km is not None:
        km = km.squeeze(1) if _tensor_layout == 0 else km.squeeze(2)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km, k_int8, k_scale, 64, _tensor_layout)
    else:
        _fused.quant_per_block_int8_cuda(k, k_int8, k_scale, 64, _tensor_layout)
    
    return q_int8, q_scale, k_int8, k_scale

    