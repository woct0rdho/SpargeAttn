/*
 * Copyright (c) 2024 by SageAttention team.
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

#pragma once
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#define div_ceil(M, N) (((M) + (N)-1) / (N))

#define CHECK_CUDA(x) \
  STD_TORCH_CHECK(x.is_cuda(), "Tensor " #x " must be on CUDA")
#define CHECK_DTYPE(x, true_dtype)     \
  STD_TORCH_CHECK(x.scalar_type() == true_dtype, \
              "Tensor " #x " must have dtype (" #true_dtype ")")
#define CHECK_DIMS(x, true_dim)    \
  STD_TORCH_CHECK(x.dim() == true_dim, \
              "Tensor " #x " must have dimension number (" #true_dim ")")
#define CHECK_NUMEL(x, minimum)     \
  STD_TORCH_CHECK(x.numel() >= minimum, \
              "Tensor " #x " must have at last " #minimum " elements")
// https://github.com/Dao-AILab/flash-attention/blob/add175637c5d54b74bc25372e49ce282d6f236fc/hopper/flash_api_stable.cpp#L98
#define CHECK_SHAPE(x, ...)                                   \
  do { \
      auto expected_dims = std::vector<int64_t>{__VA_ARGS__}; \
      STD_TORCH_CHECK(x.dim() == static_cast<int64_t>(expected_dims.size()), #x " must have " + std::to_string(expected_dims.size()) + " dimensions, got " + std::to_string(x.dim())); \
      for (size_t i = 0; i < expected_dims.size(); ++i) { \
          STD_TORCH_CHECK(x.size(i) == expected_dims[i], #x " dimension " + std::to_string(i) + " must have size " + std::to_string(expected_dims[i]) + ", got " + std::to_string(x.size(i))); \
      } \
  } while (0)
#define CHECK_CONTIGUOUS(x) \
  STD_TORCH_CHECK(x.is_contiguous(), "Tensor " #x " must be contiguous")
#define CHECK_LASTDIM_CONTIGUOUS(x) \
  STD_TORCH_CHECK(x.stride(-1) == 1,    \
              "Tensor " #x " must be contiguous at the last dimension")

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)          \
if (head_dim == 64) {                                       \
  constexpr int HEAD_DIM = 64;                              \
__VA_ARGS__                                                 \
} else if (head_dim == 128) {                               \
  constexpr int HEAD_DIM = 128;                             \
__VA_ARGS__                                                 \
} else {                                                    \
  std::ostringstream err_msg;                               \
  err_msg << "Unsupported head dim: " << int(head_dim);     \
  throw std::invalid_argument(err_msg.str());               \
}

#define DISPATCH_CAUSAL(is_causal, IS_CAUSAL, ...)          \
if (is_causal == 1) {                                       \
  constexpr bool IS_CAUSAL = true;                          \
  __VA_ARGS__                                               \
} else if (is_causal == 0) {                                \
  constexpr bool IS_CAUSAL = false;                         \
  __VA_ARGS__                                               \
}  else {                                                   \
  std::ostringstream err_msg;                               \
  err_msg << "Unsupported causal mode: " << int(is_causal); \
  throw std::invalid_argument(err_msg.str());               \
}

#define DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, ...) \
if (qk_quant_gran == 1) {                                         \
  constexpr int QK_QUANT_GRAN = 1;                                \
  __VA_ARGS__                                                     \
} else if (qk_quant_gran == 2) {                                  \
  constexpr int QK_QUANT_GRAN = 2;                                \
  __VA_ARGS__                                                     \
} else if (qk_quant_gran == 3) {                                  \
  constexpr int QK_QUANT_GRAN = 3;                                \
  __VA_ARGS__                                                     \
}  else {                                                         \
  std::ostringstream err_msg;                                     \
  err_msg << "Unsupported qk_quant_gran: " << int(qk_quant_gran); \
  throw std::invalid_argument(err_msg.str());                     \
}

#define DISPATCH_RETURN_PV_COUNT(return_pv_count, RETURN_PV_COUNT, ...)  \
if (return_pv_count == 1) {                                              \
  constexpr bool RETURN_PV_COUNT = true;                                 \
  __VA_ARGS__                                                            \
} else if (return_pv_count == 0) {                                       \
  constexpr bool RETURN_PV_COUNT = false;                                \
  __VA_ARGS__                                                            \
}  else {                                                                \
  std::ostringstream err_msg;                                            \
  err_msg << "Unsupported return_pv_count: " << int(return_pv_count);    \
  throw std::invalid_argument(err_msg.str());                            \
}

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)              \
if (pytorch_dtype == torch::headeronly::ScalarType::Half) {                           \
  using c_type = half;                                                                \
  __VA_ARGS__                                                                         \
} else if (pytorch_dtype == torch::headeronly::ScalarType::BFloat16) {                \
  using c_type = nv_bfloat16;                                                         \
  __VA_ARGS__                                                                         \
} else {                                                                              \
  std::ostringstream oss;                                                             \
  oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << #pytorch_dtype;    \
  STD_TORCH_CHECK(false, oss.str());                                                  \
}

#define DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, ...)        \
if (block_size == 64) {                                         \
  constexpr int BLOCK_SIZE = 64;                                \
  __VA_ARGS__                                                   \
} else if (block_size == 128) {                                 \
  constexpr int BLOCK_SIZE = 128;                               \
  __VA_ARGS__                                                   \
} else {                                                        \
  std::ostringstream err_msg;                                   \
  err_msg << "Unsupported block_size " << int(block_size);      \
  throw std::invalid_argument(err_msg.str());                   \
}