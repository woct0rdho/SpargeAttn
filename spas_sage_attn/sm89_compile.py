from math import ceil

import torch

from . import _qattn_sm89
_qattn_sm89 = torch.ops.spas_sage_attn_qattn_sm89


def sm89_fake_impl(
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

    CTA_Q = 128
    WARP_Q = 32
    size = ceil(qo_len / CTA_Q) * (CTA_Q // WARP_Q)
    return torch.empty((batch_size, num_qo_heads, size), dtype=torch.int32, device=query.device)


torch.library.register_fake("spas_sage_attn_qattn_sm89::qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold")(sm89_fake_impl)
torch.library.register_fake("spas_sage_attn_qattn_sm89::qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold")(sm89_fake_impl)
