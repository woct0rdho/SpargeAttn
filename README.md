# [SpargeAttn](https://github.com/thu-ml/SpargeAttn) fork for Windows wheels and easy installation

This repo makes it easy to build SpargeAttention (also known as SparseSageAttention, and the package name is `spas_sage_attn`) for multiple Python, PyTorch, and CUDA versions, then distribute the wheels to other people.

The latest wheels support GTX 16xx, RTX 20xx/30xx/40xx/50xx, V100, A100, H100, AGX Orin (sm70/75/80/86/87/89/90/120). There are also reports that it works with B200 (sm100) and DGX Spark (sm121), but I did not bundle these kernels in the wheels, and you need to build from source.

## Installation

It's similar to SageAttention. Install a wheel in the release page: https://github.com/woct0rdho/SpargeAttn/releases

## Use notes

Before using SpargeAttention in larger projects like ComfyUI, please run [test_spargeattn.py](https://github.com/woct0rdho/SpargeAttn/blob/main/tests/test_spargeattn.py) to test if SpargeAttention itself works.

SpargeAttention is usually not used directly, and [RadialAttention](https://github.com/mit-han-lab/radial-attention) is built on top of it. To apply RadialAttention in ComfyUI, you can use my node [ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn), or `WanVideoSetRadialAttention` node in kijai's [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper).

[SLA](https://github.com/thu-ml/SLA) (Sparse-Linear Attention) is also built on top of it.

## Build from source

(This is for developers)

If you need to build and run SpargeAttention on your own machine:
1. Install Visual Studio (MSVC and Windows SDK), and CUDA toolkit
2. Clone this repo
   * Checkout `abi3_stable` branch if you want ABI3 and libtorch stable ABI, which supports PyTorch >= 2.9
   * Checkout `abi3` branch if you want ABI3, which supports PyTorch >= 2.4
   * The purpose of ABI3 and libtorch stable ABI is to avoid building many wheels. I've also added `torch.compile` support in these branches. There is no other functional difference from the main branch
3. Install the dependencies in [`pyproject.toml`](https://github.com/woct0rdho/SpargeAttn/blob/main/pyproject.toml), including the desired torch version such as `torch 2.7.1+cu128`
4. Run `python setup.py install --verbose` to install directly, or `python setup.py bdist_wheel --verbose` to build a wheel. This avoids the environment checks of pip

## Dev notes

* The wheels are built using the [workflow](https://github.com/woct0rdho/SpargeAttn/blob/main/.github/workflows/build-spargeattn.yml)
* [Block-Sparse-SageAttention-2.0](https://github.com/Radioheading/Block-Sparse-SageAttention-2.0) is included, which is used in RadialAttention
* Compared to the official repo, I've modified the API so we can use the same functions `spas_sage2_attn_meansim_cuda, spas_sage2_attn_meansim_topk_cuda, block_sparse_sage2_attn_cuda` for sm >= 80
* For RTX 20xx, you need to use the Triton kernel `spas_sage_attn_meansim` in `spas_sage_attn/triton_kernel_example.py`
