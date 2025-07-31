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

#include <Python.h>
#include <torch/library.h>

#include "attn_cuda_sm89.h"

extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
       The import from Python will load the .so consisting of this file
       in this extension, so that the TORCH_LIBRARY static initializers
       below are run. */
    PyObject* PyInit__qattn_sm89(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_qattn_sm89",  /* name of module */
            NULL,           /* module documentation, may be NULL */
            -1,             /* size of per-interpreter state of the module,
                               or -1 if the module keeps state in global variables. */
            NULL,           /* methods */
        };
        return PyModule_Create(&module_def);
    }
}

// Defines the operators
TORCH_LIBRARY(spas_sage_attn_qattn_sm89, m) {
    m.def("qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor lut, "
            "Tensor valid_block_num, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "Tensor value_scale, "
            "int tensor_layout, "
            "int is_causal, "
            "int qk_quant_gran, "
            "float sm_scale"
          ") -> ()");
    m.def("qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor lut, "
            "Tensor valid_block_num, "
            "Tensor pv_threshold, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "Tensor value_scale, "
            "int tensor_layout, "
            "int is_causal, "
            "int qk_quant_gran, "
            "float sm_scale, "
            "int return_pv_count"
          ") -> Tensor");
    m.def("qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold("
            "Tensor query, "
            "Tensor key, "
            "Tensor value, "
            "Tensor(a!) output, "
            "Tensor lut, "
            "Tensor valid_block_num, "
            "Tensor pv_threshold, "
            "Tensor query_scale, "
            "Tensor key_scale, "
            "Tensor value_scale, "
            "int tensor_layout, "
            "int is_causal, "
            "int qk_quant_gran, "
            "float sm_scale, "
            "int return_pv_count"
          ") -> Tensor");
}

// Registers CUDA implementations
TORCH_LIBRARY_IMPL(spas_sage_attn_qattn_sm89, CUDA, m) {
    m.impl("qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale", &qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale);
    m.impl("qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold", &qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold);
    m.impl("qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold", &qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold);
}
