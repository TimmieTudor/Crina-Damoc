# Dr. Timmie's Mad Science Project: Crina Damoc

> While brainstorming, Dr. Timmie studied some forbidden knowledge from Kronstadt and submerged some ideas, thus discovering Crina Damoc. Since much of our technology was developed in German-speaking regions, this is not at all impossible.

This repository contains various scripts, experiments, and prototypes for the **Crina Damoc** project — a full-stack rebellion against cloud-locked AI.

## Summary

The Crina Damoc system is a novel hardware and software solution built for the future of artificial intelligence. It is made of two tightly integrated subcomponents:

- **Crina** (Cognitive Reasoning for Integrated Neural Architectures) – a multimodal spiking neural network that fuses text, images, audio, and video using tree-structured hierarchical processing, spiking neurons, and extreme sparsity.
- **Damoc** (Deep Augmented Model of Computation, pronounced *duh-MOK*) – a hybrid hardware architecture designed to run Crina natively on-device with <5 W power, zero internet dependency, and millisecond latency.

Together, they aim to deliver private, efficient, multimodal AI directly on consumer devices — without the need for massive data centers or profit-driven APIs.

## Goals

- Prove that **architectural innovation** beats brute-force scale for the next leap toward AGI.
- Enable truly **personal** AI that runs entirely on your hardware.
- Demonstrate **spiking, tree-based** computation as a viable alternative to dense Transformers.
- Build toward **CMD** – an agentic, multimodal terminal powered by Crina Damoc (a privacy-first alternative to Command Prompt/PowerShell).

## Key Features

- **TreeSelfAttentionGPU** – O(log n) hierarchical attention with spiking gating (no softmax, no quadratic cost).
- **ALIF neurons** – Adaptive LIF with spike-frequency adaptation for richer temporal dynamics.
- **Sandwich RMSNorm + learnable residual scaling** – Stable training in deep spiking stacks.
- **Full spiking feedforward (FeedForwardSNN)** – Consistent event-driven compute throughout.
- **Byte-level modeling** – Perfect character reasoning (e.g., correctly counts Rs in "strawberry").
- **~50% fewer FLOPs** than equivalent Transformer (measured on same config).

## Repository Structure

- `crina_burn/` - A rewrite of the Crina model in Rust using the Burn crate.
- `damoc/` - Various experiments and prototypes for the Damoc hardware.
- `funi/` - A list of funny findings and insights while working on the project.
- `notes/` - A collection of notes and ideas for the project.
- `cmd/` - A collection of scripts for the CMD project.
- `benchmark_crina_vs_llama.py` - A script to benchmark Crina against Llama.
- `crina_model.py` - The old implementation of the Crina model in Python.
- `crina_tinyshakespeare.py` - The new implementation of the Crina model in Python, along with a script for training it on the Tiny Shakespeare dataset.
- `crina_openwebtext.py` - An experimental script for training the Crina model on the OpenWebText dataset.
- `test_attention.py` - The latest implementation of the Crina model in Python