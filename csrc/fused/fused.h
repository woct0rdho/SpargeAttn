#include <torch/all.h>

void transpose_pad_permute_cuda(
                torch::Tensor input,
                torch::Tensor output,
                int64_t tensor_layout);

void scale_fuse_quant_cuda(
                torch::Tensor input,
                torch::Tensor output,
                torch::Tensor scale,
                int64_t num_tokens,
                double scale_max,
                int64_t tensor_layout);
