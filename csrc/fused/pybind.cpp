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

#include <Python.h>
#include <torch/csrc/stable/library.h>

#include "fused.h"

extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the TORCH_LIBRARY static initializers
       below are run. */
    PyObject* PyInit__fused(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_fused",       /* name of module */
            NULL,           /* module documentation, may be NULL */
            -1,             /* size of per-interpreter state of the module,
                               or -1 if the module keeps state in global variables. */
            NULL,           /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

void boxed_transpose_pad_permute_cuda(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto input = to<Tensor>(stack[0]);
    auto output = to<Tensor>(stack[1]);
    auto tensor_layout = to<int64_t>(stack[2]);

    transpose_pad_permute_cuda(input, output, tensor_layout);
}

void boxed_scale_fuse_quant_cuda(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs
) {
    auto input = to<Tensor>(stack[0]);
    auto output = to<Tensor>(stack[1]);
    auto scale = to<Tensor>(stack[2]);
    auto num_tokens = to<int64_t>(stack[3]);
    auto scale_max = to<double>(stack[4]);
    auto tensor_layout = to<int64_t>(stack[5]);

    scale_fuse_quant_cuda(input, output, scale, num_tokens, scale_max, tensor_layout);
}

// Defines the operators
STABLE_TORCH_LIBRARY(spas_sage_attn_fused, m) {
    m.def("transpose_pad_permute_cuda("
            "Tensor input, "
            "Tensor(a!) output, "
            "int tensor_layout"
          ") -> ()");
    m.def("scale_fuse_quant_cuda("
            "Tensor input, "
            "Tensor(a!) output, "
            "Tensor scale, "
            "int num_tokens, "
            "float scale_max, "
            "int tensor_layout"
          ") -> ()");
}

// Registers CUDA implementations
STABLE_TORCH_LIBRARY_IMPL(spas_sage_attn_fused, CUDA, m) {
    m.impl("transpose_pad_permute_cuda", &boxed_transpose_pad_permute_cuda);
    m.impl("scale_fuse_quant_cuda", &boxed_scale_fuse_quant_cuda);
}