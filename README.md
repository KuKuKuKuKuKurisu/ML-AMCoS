# Adaptive Covariance and Multi-Layer Alignment for Out-of-Distribution Detection (ML-AMCoS)

Official PyTorch implementation of **ML-AMCoS** (NeurIPS 2026).

## 📌 Overview

**ML-AMCoS** is a fully training-free framework designed for robust cross-domain Out-of-Distribution (OOD) detection across natural and medical imaging benchmarks without requiring real OOD validation data.

* **Adaptive Metric (AMCoS):** Dynamically bridges class-conditional Mahalanobis precision and angular Cosine similarity via covariance trace weighting, suppressing covariance noise in high-variance regimes while capturing fine-grained geometry in low-variance domains.
* **OOD-Free Multi-Layer Selection:** Employs a corner-cutmix pseudo-OOD synthesis technique to evaluate and retain the top-k most discriminative layers, preventing feature dilution.

## 🚀 Quick Start

Our method and experiments are built upon the OpenOOD (https://github.com/jingkang50/openood) and OpenMIBOOD (https://github.com/remic-othr/OpenMIBOOD) benchmarks. You can reproduce our method by following these steps:

1. Prepare the requirements for both benchmarks following their respective instructions.
2. Place our method's code (`mlamcos.py`) and configuration file (`mlamcos.yml`) into the paths `openood/postprocessors/` and `configs/postprocessors/` respectively.
3. Update `openood/evaluation_api/postprocessor.py` and `openood/postprocessors/__init__.py` to ensure our method is correctly imported.
4. You are ready to go!

## ⚖ License
This repository is licensed under the [Apache-2.0 License](LICENSE).
