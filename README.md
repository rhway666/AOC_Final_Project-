# My Project: AoC Final

Welcome to the AoC Final project repository. This work addresses the challenges of deploying the DeiT image-classification model on resource-constrained devices by introducing a systolic-array-based hardware accelerator with an integrated non-linear module, enabling full-model computation on the edge.

DeiT relies heavily on linear operations within its attention layers, which become the primary bottleneck for real-time inference on low-power hardware. To overcome this, we designed and implemented a dedicated systolic-array accelerator tailored to the matrix multiplications and linear projections in DeiT. Our accelerator:

- **Maximizes performance** through parallelized linear operations, significantly reducing execution time.  
- **Improves energy efficiency** and is optimized for low-power, edge-device requirements.  
- **Preserves model accuracy**, enabling real-time inference without compromising classification quality.

Experimental results demonstrate that our accelerator successfully speeds up the key computation modules of the DeiT model, offering a practical hardware solution for Transformer architectures in edge-computing scenarios. This project paves the way for deploying deep learning models in environments with strict resource limitations.

## Overview

This repository contains code and resources for:

- **Performance Analysis**: scripts and notebooks for profiling and evaluating DeiT’s linear operations on the accelerator.  
- **Quantization Pipeline**: PyTorch-based tools to quantize DeiT weights and activations for low-precision inference.  
- **Hardware Implementation**: RTL sources, testbenches, and synthesis scripts for the systolic-array accelerator with non-linear module support.

> **Note:** Large model files and weights are excluded to keep the repo lightweight. See **Setup Instructions** below for download links and steps.

## Setup Instructions

There are three main folders—`analysis/`, `quantization/`, and `hardware/`.  
Each folder contains its own README with detailed usage and examples. To get started:


### Prerequisites
- Python 3.10
- PyTorch 2.7.0 

```
conda create -n aoc_final python=3.10
conda activate aoc_final
conda install numpy pandas matplotlib seaborn pytorch torchvision torchmetrics -c pytorch
conda install transformers datasets huggingface-hub -c conda-forge
```



