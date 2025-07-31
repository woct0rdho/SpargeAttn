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

import os
from pathlib import Path
import subprocess
from packaging.version import parse, Version
from typing import List, Set
import warnings

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

def run_instantiations(src_dir: str):
    base_path = Path(src_dir)
    py_files = [
        path for path in base_path.rglob('*.py')
        if path.is_file()
    ]

    for py_file in py_files:
        print(f"Running: {py_file}")
        os.system(f"python {py_file}")

def get_instantiations(src_dir: str):
    # get all .cu files under src_dir
    base_path = Path(src_dir)
    return [
        os.path.join(src_dir, str(path.relative_to(base_path)))
        for path in base_path.rglob('*')
        if path.is_file() and path.suffix == ".cu"
    ]

# Compiler flags.
if os.name == "nt":
    # TODO: Detect MSVC rather than OS
    CXX_FLAGS = ["/O2", "/openmp", "/std:c++17", "/bigobj", "-DENABLE_BF16"]
    LINK_FLAGS = []
else:
    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
    LINK_FLAGS = []
CXX_FLAGS += ["-DPy_LIMITED_API=0x03090000"]
NVCC_FLAGS_COMMON = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    f"--threads={os.cpu_count()}",
    # "-Xptxas=-v",
    "-diag-suppress=174", # suppress the specific warning
    "-diag-suppress=177",
    "-diag-suppress=221",
]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS_COMMON += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version

compute_capabilities = set()
if os.getenv("TORCH_CUDA_ARCH_LIST"):
    # TORCH_CUDA_ARCH_LIST is separated by space or semicolon
    for x in os.getenv("TORCH_CUDA_ARCH_LIST").replace(";", " ").split():
        compute_capabilities.add(x)
else:
    # Iterate over all GPUs on the current machine.
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
            continue
        compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    raise RuntimeError("No GPUs found. Please specify TORCH_CUDA_ARCH_LIST or build on a machine with GPUs.")
else:
    print(f"Detected compute capabilities: {compute_capabilities}")

def has_capability(target):
    return any(cc.startswith(target) for cc in compute_capabilities)

# Validate the NVCC CUDA version.
if nvcc_cuda_version < Version("12.0"):
    raise RuntimeError("CUDA 12.0 or higher is required to build the package.")
if nvcc_cuda_version < Version("12.4") and has_capability("8.9"):
    raise RuntimeError(
        "CUDA 12.4 or higher is required for compute capability 8.9.")
if nvcc_cuda_version < Version("12.3") and has_capability("9.0"):
    raise RuntimeError(
        "CUDA 12.3 or higher is required for compute capability 9.0.")
if nvcc_cuda_version < Version("12.8") and has_capability("12.0"):
    raise RuntimeError(
        "CUDA 12.8 or higher is required for compute capability 12.0.")

# Add target compute capabilities to NVCC flags.
def get_nvcc_flags(allowed_capabilities):
    NVCC_FLAGS = []
    for capability in compute_capabilities:
        if capability not in allowed_capabilities:
            continue

        # capability: "8.0+PTX" -> num: "80"
        num = capability.split("+")[0].replace(".", "")
        if num in {"90", "120"}:
            # need to use sm90a instead of sm90 to use wgmma ptx instruction.
            # need to use sm120a to use mxfp8/mxfp4/nvfp4 instructions.
            num += "a"

        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
        if capability.endswith("+PTX"):
            NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

    NVCC_FLAGS += NVCC_FLAGS_COMMON
    return NVCC_FLAGS

ext_modules = []

if has_capability(("8.0", "8.6")):
    sources = [
        "csrc/qattn/pybind_sm80.cpp",
        "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
    ]
    run_instantiations("csrc/qattn/instantiations_sm80")
    sources += get_instantiations("csrc/qattn/instantiations_sm80")
    qattn_extension = CUDAExtension(
        name="spas_sage_attn._qattn_sm80",
        sources=sources,
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": get_nvcc_flags(["8.0", "8.6"]),
        },
        extra_link_args=LINK_FLAGS,
        py_limited_api=True,
    )
    ext_modules.append(qattn_extension)

if has_capability(("8.9", "12.0")):
    sources = [
        "csrc/qattn/pybind_sm89.cpp",
        "csrc/qattn/qk_int_sv_f8_cuda_sm89.cu",
    ]
    run_instantiations("csrc/qattn/instantiations_sm89")
    sources += get_instantiations("csrc/qattn/instantiations_sm89")
    qattn_extension = CUDAExtension(
        name="spas_sage_attn._qattn_sm89",
        sources=sources,
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": get_nvcc_flags(["8.9", "12.0"]),
        },
        extra_link_args=LINK_FLAGS,
        py_limited_api=True,
    )
    ext_modules.append(qattn_extension)

if has_capability(("9.0",)):
    sources = [
        "csrc/qattn/pybind_sm90.cpp",
        "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
    ]
    run_instantiations("csrc/qattn/instantiations_sm90")
    sources += get_instantiations("csrc/qattn/instantiations_sm90")
    qattn_extension = CUDAExtension(
        name="spas_sage_attn._qattn_sm90",
        sources=sources,
        libraries=["cuda"],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": get_nvcc_flags(["9.0"]),
        },
        extra_link_args=LINK_FLAGS,
        py_limited_api=True,
    )
    ext_modules.append(qattn_extension)

fused_extension = CUDAExtension(
    name="spas_sage_attn._fused",
    sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": get_nvcc_flags(["8.0", "8.6", "8.9", "9.0", "12.0"]),
    },
    extra_link_args=LINK_FLAGS,
    py_limited_api=True,
)
ext_modules.append(fused_extension)

setup(
    name='spas_sage_attn',
    version='0.1.0' + os.environ.get("SPAS_SAGE_ATTN_WHEEL_VERSION_SUFFIX", ""),
    author='Jintao Zhang, Chendong Xiang, Haofeng Huang',
    author_email='jt-zhang6@gmail.com',
    packages=find_packages(),
    description='Accurate and efficient Sparse SageAttention.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thu-ml/SpargeAttn',
    license='BSD 3-Clause License',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
