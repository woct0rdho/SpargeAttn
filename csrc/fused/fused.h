#include <torch/csrc/stable/tensor.h>

using torch::stable::Tensor;

void transpose_pad_permute_cuda(
                Tensor input,
                Tensor output,
                int64_t tensor_layout);

void scale_fuse_quant_cuda(
                Tensor input,
                Tensor output,
                Tensor scale,
                int64_t num_tokens,
                double scale_max,
                int64_t tensor_layout);