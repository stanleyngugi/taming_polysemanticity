# Polysemanticity Toy Model Experiment

## Overview

This repository contains a PyTorch-based toy model for studying incidental polysemanticity in neural networks, inspired by foundational work in mechanistic interpretability (MI). Polysemanticity refers to the phenomenon where individual neurons or latents activate for multiple unrelated features, complicating model auditability and safety. While superposition theory attributes this to capacity constraints, incidental polysemanticity arises from training artifacts like regularization or noise, even in overcapacity regimes.

The experiment uses a simple MLP trained on synthetic correlated data, followed by a sparse autoencoder (SAE) to decompose representations and measure entanglement via metrics like Mean Absolute Cosine Similarity (MACS) and weighted Interference. Through ablations on initialization (random vs. orthogonal), regularization (L1 vs. L2), activation functions (ReLU vs. GELU), and noise types (bipolar vs. positive kurtosis), we test mitigations for incidental polysemanticity. Key findings: L2 regularization reduces polysemanticity by ~18% compared to L1, challenging the "sparse = interpretable" assumption in overcomplete settings.

This is an exploratory toy—designed for quick iteration and insight, not production. It demonstrates practical pitfalls (e.g., dying ReLUs) and mitigations (e.g., LeakyReLU revival). For full details, see the blog post on my research site: [Taming Incidental Polysemanticity in Toy Models](https://stanleyngugi.netlify.app/files/taming_polysemanticity).

## Features
- **Synthetic Data Generation**: Correlated feature groups with tunable co-occurrence, sparsity, and non-linear targets to simulate "temptation" for entanglement.
- **Modular Architecture**: MLP network with configurable init/reg/act/noise; overcomplete SAE for decomposition.
- **Ablation Framework**: CLI-driven runs for 8 strategic combinations, from baseline (high poly) to max mitigation (low poly).
- **Metrics Suite**: MACS (raw entanglement), calibrated MACS (null-adjusted), Interference (importance-weighted poly cost), sparsity measures (L0, W4), and reconstruction/performance MSE.
- **Reproducibility**: Seeded runs, pinned dependencies, logging, and overlap plots for visualization.

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/stanleyngugi/taming_polysemanticity.git
   cd taming_polysemanticity
   ```

2. **Set Up Virtual Environment** (Recommended to avoid conflicts):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   The `requirements.txt` pins versions for reproducibility (e.g., torch==2.4.1, numpy==2.1.1). Note: If you encounter conflicts (e.g., with torchvision or numba), use the virtual env or uninstall unused packages.

## Usage

Run ablations via the CLI in `main.py`. Each command executes a specific combination over `--seeds` (default 10) for variance estimation. Results are logged to `experiment.log` and printed, with cosine overlap plots saved as PNGs.

### Example Commands for Key Ablations
1. **Baseline (Random + Bipolar + L1 + ReLU)**:  
   `python main.py --init random --noise bipolar --reg l1 --act relu --seeds 10`

2. **Orthogonal Init Only**:  
   `python main.py --init orthogonal --noise bipolar --reg l1 --act relu --seeds 10`

3. **Positive Kurtosis Noise Only**:  
   `python main.py --init random --noise positive_kurtosis --reg l1 --act relu --seeds 10`

4. **L2 Reg + GELU Act Only**:  
   `python main.py --init random --noise bipolar --reg l2 --act gelu --seeds 10`

5. **Orthogonal + L2 + GELU**:  
   `python main.py --init orthogonal --noise bipolar --reg l2 --act gelu --seeds 10`

6. **Orthogonal + Positive Kurtosis + L1 + ReLU**:  
   `python main.py --init orthogonal --noise positive_kurtosis --reg l1 --act relu --seeds 10`

7. **Random + Positive Kurtosis + L2 + ReLU**:  
   `python main.py --init random --noise positive_kurtosis --reg l2 --act relu --seeds 10`

8. **Max Mitigation (Orthogonal + Positive Kurtosis + L2 + GELU)**:  
   `python main.py --init orthogonal --noise positive_kurtosis --reg l2 --act gelu --seeds 10`

To run all in batch, create a script like `run_ablations.sh`:
```
#!/bin/bash
python main.py --init random --noise bipolar --reg l1 --act relu --seeds 10
# Add other commands...
```
Then: `bash run_ablations.sh`.

### Interpreting Outputs
- **Logs**: Per-seed metrics in `experiment.log` (e.g., "Seed 0 Metrics: MACS 0.2954...").
- **Averages**: Printed at end (e.g., "Average Interference: 6.3689 +/- 1.2965").
- **Plots**: Cosine overlap matrices as `overlap_[config]_seed[N].png`—high off-diagonals indicate polysemanticity.
- **Key Insights from Runs**: L2 combos show lower poly (Inter ~6.4-7.1 vs. L1 ~8.1-9.1), with max mitigation achieving -26.8% Interference reduction.

## Project Structure
- `models.py`: Defines Config dataclass, Network (MLP with ablatable init/act/reg), and SAE (with LeakyReLU/bias fixes for revival).
- `utils.py`: Data generation (correlated groups, importances, targets), noise injection, metrics (MACS/calib, Interference, sparsity), and plotting.
- `main.py`: CLI orchestration—data gen, training loops (with hooks for hidden acts), SAE training, metric computation, logging/plots.
- `requirements.txt`: Pinned deps (e.g., torch==2.4.1) for repro.
- `experiment.log`: Output logs from runs.
- `overlap_*.png`: Generated visualization files.

## Limitations and Future Work
- **Toy-Only**: Synthetic data limits real-world applicability—extend to PEFT/LoRA fine-tuning on LLMs like Llama to probe poly in adapters without catastrophic forgetting.
- **Volatility**: SAE_MSE high variance (CV 50-492%) from pos kurt tails—add gradient clipping (norm=1.0) to stabilize.
- **Scope**: Overcapacity focus; test mild bottlenecks (hidden_dim=9) for phase boundaries.
- **Enhancements**: Incorporate ProLU/JumpReLU for SAEs (+15-20% fidelity); full PSI metrics; lambda/sparsity sweeps.

## Acknowledgments and References
Thanks to the MI community for inspiration. Key references:
- Lecomte, V., et al. (2023). What Causes Polysemanticity? An Alternative Origin Story... arXiv:2312.03096. [Link](https://arxiv.org/abs/2312.03096)
- Elhage, N., et al. (2022). Toy Models of Superposition. arXiv:2209.10652. [Link](https://arxiv.org/abs/2209.10652)
- Authors of PSI (2024). Disentangling Polysemantic Neurons... arXiv:2508.16950. [Link](https://arxiv.org/abs/2508.16950)
- Bricken, T., et al. (2023). Towards Monosemanticity... [Link](https://transformer-circuits.pub/2023/monosemantic-features)
- Templeton, A., et al. (2024). Scaling Monosemanticity... Anthropic. [Link](https://anthropic.com/research/scaling-monosemanticity)

For feedback or collaborations, reach me at sngugi.research@gmail.com. Happy experimenting!
