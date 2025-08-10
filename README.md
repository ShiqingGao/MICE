<div align="center">

# Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration

[[Paper]](https://openreview.net/pdf?id=nq5bt0mRTC)  [[Github]](https://github.com/ShiqingGao/MICE) 


<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#🎉news" style="text-decoration: none; font-weight: bold;">🎉 News</a> •
    <a href="#✨getting-started" style="text-decoration: none; font-weight: bold;">✨ Getting Started</a> •
    <a href="#📖introduction" style="text-decoration: none; font-weight: bold;">📖 Introduction</a>
  </p>
  <p>
    <a href="#🎈citation" style="text-decoration: none; font-weight: bold;">🎈 Citation</a> •
    <a href="#🌻acknowledgement" style="text-decoration: none; font-weight: bold;">🌻 Acknowledgement</a> 
  </p>
</div>

</div>


# 🎉News

- **[2025/06/20]** 🎉 **Code Release**: Our complete implementation is now available on GitHub, including all experiments and evaluation scripts.
- **[2025/06/18]** 🎉 **Paper Publication**: Our paper is officially released and available at [OpenReview](https://openreview.net/pdf?id=nq5bt0mRTC).
- **[2025/06/9]** 🎉 **ICML 2025 Oral Presentation**: Our paper has been selected for **oral presentation** at ICML 2025, representing the **top 1%** of all submissions.
- **[2025/05/01]** 🎉 **ICML 2025 Spotlight**: Our paper has been accepted as **spotlight** presentation by ICML 2025. 

# ✨Getting started

This repository implements the **Memory-driven Intrinsic Cost Estimation (MICE)** algorithm, built upon the [OmniSafe](https://github.com/PKU-Alignment/omnisafe) framework for safe reinforcement learning.

## 🚀 Quick Start

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ShiqingGao/MICE.git
cd MICE
```

2. **Install dependencies:**
```bash
# Install OmniSafe base framework
pip install omnisafe

# Install project-specific dependencies
pip install -r requirements.txt
```

### 🎯 Running Experiments

Navigate to the experiments directory and run the training script:

```bash
cd experiments
python train_mice.py
```


# 📖Introduction

## 🎯 Problem Statement

Constrained Reinforcement Learning (CRL) faces a critical challenge: **value underestimation bias** that leads to excessive constraint violations during exploration. Traditional approaches often struggle to balance safety requirements with policy performance.

## 💡 Our Solution: MICE Algorithm

We propose the **Memory-driven Intrinsic Cost Estimation (MICE)** algorithm, which addresses the underestimation bias through three key innovations:

### 🔧 Core Components

1. **Memory Module**: Records and maintains high-risk regions encountered during training
2. **Intrinsic Cost Generation**: Dynamically generates intrinsic costs based on historical risk data
3. **Extrinsic-Intrinsic Cost Update Scheme**: Combines external and internal cost signals for robust constraint satisfaction

### 🎯 Key Features

- **Significantly Reduces Violations**: Our approach dramatically decreases constraint violations while preserving policy performance
- **Theoretical Guarantees**: Provides a theoretical upper bound on constraint violations
- **Proven Convergence**: Establishes value function convergence under mild assumptions
- **Memory-Efficient**: Lightweight memory module with minimal computational overhead

### 🔬 Technical Contributions

- **Novel Cost Estimation**: Introduces memory-driven intrinsic cost estimation mechanism
- **Theoretical Analysis**: Provides rigorous theoretical analysis of constraint violation bounds
- **Empirical Validation**: Comprehensive evaluation across multiple safety-critical environments


# 🎈Citation

If you find this paper or repository helpful in your research, please cite our work:

```bibtex
@inproceedings{gaocontrolling,
  title={Controlling Underestimation Bias in Constrained Reinforcement Learning for Safe Exploration},
  author={Gao, Shiqing and Ding, Jiaxin and Fu, Luoyi and Wang, Xinbing},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

# 🌻Acknowledgement
We implement our reinforcement learning algorithm extending from [omnisafe](https://github.com/volcengine/verl). We utilize [safety-gymnaisum](https://github.com/PKU-Alignment/safety-gymnasium) for our environments.


