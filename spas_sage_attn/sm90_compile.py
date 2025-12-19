from math import ceil

import torch

from . import _qattn_sm90
_qattn_sm90 = torch.ops.spas_sage_attn_qattn_sm90


@torch.library.register_fake("spas_sage_attn_qattn_sm90::qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90")
def qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90_fake_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lut: torch.Tensor,
    valid_block_num: torch.Tensor,
    pv_threshold: torch.Tensor,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    tensor_layout: int,
    is_causal: int,
    qk_quant_gran: int,
    sm_scale: float,
    return_pv_count: int,
) -> torch.Tensor:
    if not return_pv_count:
        return torch.empty((0,))

    batch_size = query.size(0)
    if tensor_layout == 0:
        num_qo_heads = query.size(2)
        qo_len = query.size(1)
    else:
        num_qo_heads = query.size(1)
        qo_len = query.size(2)

    CTA_Q = 64
    WARP_Q = 16
    size = ceil(qo_len / CTA_Q) * (CTA_Q // WARP_Q)
    return torch.empty((batch_size, num_qo_heads, size), dtype=torch.int32, device=query.device)
