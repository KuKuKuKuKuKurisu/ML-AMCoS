# Adaptive Covariance and Multi-Layer Alignment for Out-of-Distribution Detection (ML-AMCoS)

Official PyTorch implementation of **ML-AMCoS** (NeurIPS 2026).

## 💡 Run

Our method and experiments are built upon the OpenOOD (https://github.com/jingkang50/openood) and OpenMIBOOD (https://github.com/remic-othr/OpenMIBOOD) benchmarks. You can reproduce our method by following these steps:

1. Prepare the requirements for both benchmarks following their respective instructions.
2. Place our method's code (`mlamcos.py`) and configuration file (`mlamcos.yml`) into the paths `openood/postprocessors/` and `configs/postprocessors/` respectively.
3. Update `openood/evaluation_api/postprocessor.py` and `openood/postprocessors/__init__.py` to ensure our method is correctly imported.
4. You are ready to go!

## ⚖ License
This repository is licensed under the [Apache-2.0 License](LICENSE).
