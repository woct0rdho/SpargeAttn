# [SpargeAttn](https://github.com/thu-ml/SpargeAttn) fork for build system integration

This repo makes it easy to build SpargeAttention (also known as SparseSageAttention, and the package name is `spas_sage_attn`) for multiple Python, PyTorch, and CUDA versions, then distribute the wheels to other people. See [releases](https://github.com/woct0rdho/SpargeAttn/releases) for the wheels, and the [workflow](https://github.com/woct0rdho/SpargeAttn/blob/main/.github/workflows/build-spargeattn.yml) to build them on Windows.

If you only need to build and run on your own machine, you can clone this repo, install the dependencies in [`pyproject.toml`](https://github.com/woct0rdho/SpargeAttn/blob/main/pyproject.toml) (include the correct torch version such as `torch 2.7.1+cu128`), then run `python setup.py install` (this avoids the environment checks of pip).

[Block-Sparse-SageAttention-2.0](https://github.com/Radioheading/Block-Sparse-SageAttention-2.0) is included, which is used in [RadialAttention](https://github.com/mit-han-lab/radial-attention).

To use RadialAttention in ComfyUI, you need to first install the pip package `spas_sage_attn` here, then install my node [ComfyUI-RadialAttn](https://github.com/woct0rdho/ComfyUI-RadialAttn), or `WanVideoSetRadialAttention` node in kijai's [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper).
