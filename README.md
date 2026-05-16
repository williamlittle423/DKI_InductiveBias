# Is Microbial Keystone Species Identification Invariant to Model Inductive Bias?

A comparative study of microbial keystone species identification across deep learning architectures to evaluate the robustness of the DKI framework to model inductive bias.

## Overview

This project investigates whether the keystone species identified by the Data-driven Keystone-species Identification (DKI) framework are robust across predictive architectures with different inductive biases.

The original DKI framework proposed by Wang et al. used the cNODE2 Neural ODE architecture to learn mappings from binary microbial species presence vectors to relative abundance compositions. Keystone species are then inferred by simulating species removal experiments within the learned model.

In this work, multiple architectures are trained on the same microbial abundance prediction task:

- Linear Model
- Residual MLP
- Attention-Based Model
- Neural ODE (NODE)
- Original cNODE2

The resulting keystone species rankings are then compared using:

- Spearman Rank Correlation
- Top-k Jaccard Similarity
- Pairwise Keystoneness Scatter Plots

The main finding is that keystone rankings remain highly consistent across architectures despite substantial differences in inductive bias and predictive formulation.

---

## Repository Structure

```text
.
├── data/
│   ├── Ptrain.csv
│   ├── Ptest.csv
│   ├── Sample_id.csv
│   └── Species_id.csv
│
├── results/
│   ├── qtst_*.csv
│   ├── qtrn_*.csv
│   ├── loss_train_*.csv
│   ├── loss_val_*.csv
│   ├── keystoneness_by_model.csv
│   └── figures/
│
├── train_models.py
├── compute_keystoneness.py
├── compare_models_rankings.py
├── plot_results.py
└── README.md
```

---

## Methods

### 1. Abundance Prediction

Each model learns a mapping:

```math
z \rightarrow p
```

where:

- `z` = binary microbial species presence vector
- `p` = relative abundance composition

Training minimizes Bray-Curtis dissimilarity.

---

### 2. Keystone Species Identification

After training:

1. A species is removed from the binary presence vector
2. The model predicts the perturbed abundance composition
3. Bray-Curtis dissimilarity between the original and perturbed compositions is computed

This perturbation score defines the species keystoneness.

---

### 3. Ranking Comparison

Global keystone rankings are compared across architectures using:

- Spearman correlation
- Top-10 / Top-20 Jaccard similarity
- Scatter plots of median keystoneness

---

## Models

| Model | Inductive Bias |
|---|---|
| Linear | Linear microbial interactions |
| Residual MLP | Nonlinear hierarchical feature learning |
| Attention | Contextual pairwise species interactions |
| NODE | Continuous dynamical systems |
| cNODE2 | Replicator-style ecological dynamics |

---

## Training

Implemented in PyTorch with CUDA acceleration.

Key implementation details:

- Adam optimizer
- Bray-Curtis loss
- Fixed-step Euler integration for cNODE2
- Equilibrium-based integration for NODE
- GPU training using Google Colab

---

## Running the Project

### Train models

```bash
python train_models.py
```

This saves:
- trained predictions
- validation losses
- perturbed abundance predictions

to:

```text
results/
```

---

### Compute keystoneness

```bash
python compute_keystoneness.py
```

Outputs:

```text
results/keystoneness_by_model.csv
```

---

### Compare rankings

```bash
python compare_models_rankings.py
```

Computes:
- Spearman correlations
- Jaccard similarities
- ranking comparison tables

---

### Generate plots

```bash
python plot_results.py
```

Generates:
- correlation matrices
- scatter plots
- keystone ranking visualizations

---

## Main Results

- Pairwise Spearman correlations exceeded `0.99`
- Top keystone taxa were highly consistent across architectures
- ODE-based models achieved lower Bray-Curtis loss
- Non-ODE models produced larger perturbation magnitudes

These findings suggest that DKI-style keystone identification may capture stable ecological signals rather than artifacts of a single predictive architecture.

---

## References

- Wang et al. — *Identifying keystone species in microbial communities using deep learning*
- Chen et al. — *Neural Ordinary Differential Equations*

---

## Acknowledgements

This project builds upon the original DKI framework proposed by Wang et al.
