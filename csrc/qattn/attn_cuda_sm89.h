/*
 * Copyright (c) 2025 by SpargeAttn team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

void qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor lut,
    Tensor valid_block_num,
    Tensor query_scale,
    Tensor key_scale,
    Tensor value_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    int64_t qk_quant_gran,
    double sm_scale);

Tensor qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor lut,
    Tensor valid_block_num,
    Tensor pv_threshold,
    Tensor query_scale,
    Tensor key_scale,
    Tensor value_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    int64_t qk_quant_gran,
    double sm_scale,
    int64_t return_pv_count);

Tensor qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor output,
    Tensor lut,
    Tensor valid_block_num,
    Tensor pv_threshold,
    Tensor query_scale,
    Tensor key_scale,
    Tensor value_scale,
    int64_t tensor_layout,
    int64_t is_causal,
    int64_t qk_quant_gran,
    double sm_scale,
    int64_t return_pv_count);
