# **Deep Learning Training Examples with PyTorch**

This repository provides a comprehensive guide and practical examples for training deep learning models using PyTorch across various parallelism strategies. Whether you are working on single-GPU training or scaling to multi-GPU setups with Distributed Data Parallel (DDP) or Fully Sharded Data Parallel (FSDP), these examples will guide you through the process.

---

## **Contents**

### 01. **Introduction to Deep Learning**
- Foundational concepts of deep learning and PyTorch.
- Basics of tensors, datasets, and model building.

---

### 02. **Single-GPU Training**
- Efficiently training models on a single GPU.
- Profiling tools and techniques to optimize performance.

---

### 03. **Multi-GPU Training with Data Parallelism (DP)**
- Scaling models across multiple GPUs using `torch.nn.DataParallel`.
- Profiling and optimizing data parallel workloads.

---

### 04. **Distributed Data Parallel (DDP) Training**
- Leveraging `torch.nn.parallel.DistributedDataParallel` for efficient multi-GPU training.
- Setting up process groups, distributed samplers, and profiling DDP workloads.

---

### 05. **Fully Sharded Data Parallel (FSDP) Training**
- Training large models with memory efficiency using Fully Sharded Data Parallel (FSDP).
- Fine-tuning large-scale models like CodeLlama with gradient checkpointing and parameter sharding.

---

## **Resources**

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Basics_Pytorch](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html)
- [Data_Parallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
- [Distributed_Data_parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

---


---
## **NOTE**
- If you are already familiar with deep learning with PyTorch, you can skip [01. Introduction to Deep Learning](./01_introduction_to_deeplearning/) and go directly to [02. Single-GPU Training](./02_singlegpu_training/).
---

