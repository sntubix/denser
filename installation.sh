#!/bin/bash

# Install PyTorch and torchvision with CUDA support
echo "Installing PyTorch and torchvision with CUDA 11.8 support..."
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install the CUDA toolkit
echo "Installing CUDA toolkit 11.8..."
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Install ninja and tiny-cuda-nn
echo "Installing ninja and tiny-cuda-nn..."
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install pytorch3d
echo "Installing PyTorch3D..."

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

pip install "numpy<2"

pip install gsplat==0.1.12

echo "Environment setup complete!"
