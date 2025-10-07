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

#include "../pytorch_extensions_utils.cuh"
#include "decl.cuh"

void qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor lut,
                    torch::Tensor valid_block_num,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    int64_t tensor_layout,
                    int64_t is_causal,
                    int64_t qk_quant_gran,
                    double sm_scale)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(lut);
  CHECK_CUDA(valid_block_num);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(lut);
  CHECK_CONTIGUOUS(valid_block_num);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, at::ScalarType::Char);
  CHECK_DTYPE(key, at::ScalarType::Char);
  CHECK_DTYPE(value, torch::kHalf);
  CHECK_DTYPE(lut, at::ScalarType::Int);
  CHECK_DTYPE(valid_block_num, at::ScalarType::Int);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(lut, 4);
  CHECK_DIMS(valid_block_num, 3);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int head_dim = query.size(3);
  const int batch_size = query.size(0);

  int stride_bz_q = query.stride(0);
  int stride_bz_k = key.stride(0);
  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);

    stride_seq_q = query.stride(1);
    stride_seq_k = key.stride(1);
    stride_seq_v = value.stride(1);
    stride_seq_o = output.stride(1);

    stride_h_q = query.stride(2);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_h_o = output.stride(2);
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);

    stride_seq_q = query.stride(2);
    stride_seq_k = key.stride(2);
    stride_seq_v = value.stride(2);
    stride_seq_o = output.stride(2);

    stride_h_q = query.stride(1);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_h_o = output.stride(1);
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }

  auto output_dtype = output.scalar_type();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
        DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
          constexpr int CTA_Q = 128;
          constexpr int CTA_K = 64;
          constexpr int WARP_Q = (HEAD_DIM == 64) ? 32 : 16;
          constexpr int WARP_K = 64;

          if constexpr (QK_QUANT_GRAN == 1)
          {
            CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q));
            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K));
          }
          else if constexpr (QK_QUANT_GRAN == 2)
          {
            CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
          }
          else if constexpr (QK_QUANT_GRAN == 3)
          {
            CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
            CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
          }
          else
          {
            static_assert(QK_QUANT_GRAN == 1 || QK_QUANT_GRAN == 2 || QK_QUANT_GRAN == 3, "Unsupported quantization granularity");
          }

          CHECK_SHAPE(lut, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q), div_ceil(kv_len, CTA_K));
          CHECK_SHAPE(valid_block_num, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q));
          
          SpargeAttentionSM80Dispatched<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, QK_QUANT_GRAN, float, true, 0, DTypeOut, IS_CAUSAL, false>(
            reinterpret_cast<int8_t*>(query.data_ptr()),
            reinterpret_cast<int8_t*>(key.data_ptr()),
            reinterpret_cast<half*>(value.data_ptr()),
            reinterpret_cast<DTypeOut*>(output.data_ptr()),
            nullptr,
            reinterpret_cast<int32_t*>(lut.data_ptr()),
            reinterpret_cast<int32_t*>(valid_block_num.data_ptr()),
            nullptr,
            reinterpret_cast<float*>(query_scale.data_ptr()),
            reinterpret_cast<float*>(key_scale.data_ptr()),
            batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
            stride_bz_q, stride_seq_q, stride_h_q,
            stride_bz_k, stride_seq_k, stride_h_k,
            stride_bz_v, stride_seq_v, stride_h_v,
            stride_bz_o, stride_seq_o, stride_h_o,
            sm_scale);
        });
      });
    });
  });
}

torch::Tensor qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(torch::Tensor query,
                    torch::Tensor key,
                    torch::Tensor value,
                    torch::Tensor output,
                    torch::Tensor lut,
                    torch::Tensor valid_block_num,
                    torch::Tensor pv_threshold,
                    torch::Tensor query_scale,
                    torch::Tensor key_scale,
                    int64_t tensor_layout,
                    int64_t is_causal,
                    int64_t qk_quant_gran,
                    double sm_scale,
                    int64_t return_pv_count)
{
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(output);
  CHECK_CUDA(lut);
  CHECK_CUDA(valid_block_num);
  CHECK_CUDA(pv_threshold);
  CHECK_CUDA(query_scale);
  CHECK_CUDA(key_scale);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_LASTDIM_CONTIGUOUS(value);
  CHECK_LASTDIM_CONTIGUOUS(output);
  CHECK_CONTIGUOUS(lut);
  CHECK_CONTIGUOUS(valid_block_num);
  CHECK_CONTIGUOUS(pv_threshold);
  CHECK_CONTIGUOUS(query_scale);
  CHECK_CONTIGUOUS(key_scale);

  CHECK_DTYPE(query, at::ScalarType::Char);
  CHECK_DTYPE(key, at::ScalarType::Char);
  CHECK_DTYPE(value, torch::kHalf);
  CHECK_DTYPE(lut, at::ScalarType::Int);
  CHECK_DTYPE(valid_block_num, at::ScalarType::Int);
  CHECK_DTYPE(pv_threshold, at::ScalarType::Float);
  CHECK_DTYPE(query_scale, at::ScalarType::Float);
  CHECK_DTYPE(key_scale, at::ScalarType::Float);

  CHECK_DIMS(query, 4);
  CHECK_DIMS(key, 4);
  CHECK_DIMS(value, 4);
  CHECK_DIMS(output, 4);
  CHECK_DIMS(lut, 4);
  CHECK_DIMS(valid_block_num, 3);
  CHECK_DIMS(pv_threshold, 1);
  CHECK_DIMS(query_scale, 3);
  CHECK_DIMS(key_scale, 3);

  const int head_dim = query.size(3);
  const int batch_size = query.size(0);

  int stride_bz_q = query.stride(0);
  int stride_bz_k = key.stride(0);
  int stride_bz_v = value.stride(0);
  int stride_bz_o = output.stride(0);

  int qo_len, kv_len, num_qo_heads, num_kv_heads;
  int stride_seq_q, stride_seq_k, stride_seq_v, stride_seq_o;
  int stride_h_q, stride_h_k, stride_h_v, stride_h_o;

  if (tensor_layout == 0)
  {
    qo_len = query.size(1);
    kv_len = key.size(1);
    num_qo_heads = query.size(2);
    num_kv_heads = key.size(2);
    CHECK_SHAPE(key, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(value, batch_size, kv_len, num_kv_heads, head_dim);
    CHECK_SHAPE(output, batch_size, qo_len, num_qo_heads, head_dim);

    stride_seq_q = query.stride(1);
    stride_seq_k = key.stride(1);
    stride_seq_v = value.stride(1);
    stride_seq_o = output.stride(1);

    stride_h_q = query.stride(2);
    stride_h_k = key.stride(2);
    stride_h_v = value.stride(2);
    stride_h_o = output.stride(2);
  }
  else if (tensor_layout == 1)
  {
    qo_len = query.size(2);
    kv_len = key.size(2);
    num_qo_heads = query.size(1);
    num_kv_heads = key.size(1);
    CHECK_SHAPE(key, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(value, batch_size, num_kv_heads, kv_len, head_dim);
    CHECK_SHAPE(output, batch_size, num_qo_heads, qo_len, head_dim);

    stride_seq_q = query.stride(2);
    stride_seq_k = key.stride(2);
    stride_seq_v = value.stride(2);
    stride_seq_o = output.stride(2);

    stride_h_q = query.stride(1);
    stride_h_k = key.stride(1);
    stride_h_v = value.stride(1);
    stride_h_o = output.stride(1);
  }
  else
  {
    throw std::invalid_argument("tensor_layout must be 0 or 1");
  }

  if (num_qo_heads % num_kv_heads != 0) {
    std::ostringstream err_msg;
    err_msg << "num_qo_heads (" << num_qo_heads << ") must be divisible by num_kv_heads (" << num_kv_heads << ")";
    throw std::invalid_argument(err_msg.str());  
  }
  
  torch::Tensor pv_count = torch::empty({0});

  auto output_dtype = output.scalar_type();

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_CAUSAL(is_causal, IS_CAUSAL, {
      DISPATCH_RETURN_PV_COUNT(return_pv_count, RETURN_PV_COUNT, {
        DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, {
          DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, DTypeOut, {
            constexpr int CTA_Q = 128;
            constexpr int CTA_K = 64;
            constexpr int WARP_Q = (HEAD_DIM == 64) ? 32 : 16;
            constexpr int WARP_K = 64;
            
            if constexpr (RETURN_PV_COUNT)
            {
              pv_count = torch::empty({batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q)}, query.options().dtype(at::ScalarType::Int));
            }

            if constexpr (QK_QUANT_GRAN == 1)
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K));
            }
            else if constexpr (QK_QUANT_GRAN == 2)
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q));
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K));
            }
            else if constexpr (QK_QUANT_GRAN == 3)
            {
              CHECK_SHAPE(query_scale, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q) * (CTA_Q / WARP_Q) * 8);
              CHECK_SHAPE(key_scale, batch_size, num_kv_heads, div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K) * 4);
            }
            else
            {
              static_assert(QK_QUANT_GRAN == 1 || QK_QUANT_GRAN == 2 || QK_QUANT_GRAN == 3, "Unsupported quantization granularity");
            }

            CHECK_SHAPE(lut, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q), div_ceil(kv_len, CTA_K));
            CHECK_SHAPE(valid_block_num, batch_size, num_qo_heads, div_ceil(qo_len, CTA_Q));
            CHECK_SHAPE(pv_threshold, num_qo_heads);

            SpargeAttentionSM80Dispatched<CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, QK_QUANT_GRAN, float, true, 1, DTypeOut, IS_CAUSAL, RETURN_PV_COUNT>(
              reinterpret_cast<int8_t*>(query.data_ptr()),
              reinterpret_cast<int8_t*>(key.data_ptr()),
              reinterpret_cast<half*>(value.data_ptr()),
              reinterpret_cast<DTypeOut*>(output.data_ptr()),
              (RETURN_PV_COUNT) ? reinterpret_cast<int32_t*>(pv_count.data_ptr()) : nullptr,
              reinterpret_cast<int32_t*>(lut.data_ptr()),
              reinterpret_cast<int32_t*>(valid_block_num.data_ptr()),
              reinterpret_cast<float*>(pv_threshold.data_ptr()),
              reinterpret_cast<float*>(query_scale.data_ptr()),
              reinterpret_cast<float*>(key_scale.data_ptr()),
              batch_size, qo_len, kv_len, num_qo_heads, num_kv_heads,
              stride_bz_q, stride_seq_q, stride_h_q,
              stride_bz_k, stride_seq_k, stride_h_k,
              stride_bz_v, stride_seq_v, stride_h_v,
              stride_bz_o, stride_seq_o, stride_h_o,
              sm_scale);
          });
        });
      });
    });
  });

  return pv_count;
}
