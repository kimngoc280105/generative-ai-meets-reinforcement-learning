# Diffusion-DICE 2D Bandit Toy Case Demo

This directory contains a self-contained demonstration of the **2D Bandit Toy Case** from the NeurIPS 2024 paper **"Diffusion-DICE: In-Sample Diffusion Guidance for Offline Reinforcement Learning"** (Mao et al., 2024), presented in the ICML 2025 tutorial *"Generative AI Meets Reinforcement Learning"*.

The purpose of this toy case is to demonstrate the phenomenon of **value-function error exploitation** in offline reinforcement learning and how Diffusion-DICE successfully alleviates it using **In-Sample Guidance Learning (IGL)** compared to QGPO and IDQL.

---

## Mathematical Background & Problem Setup

### 1. Annular Dataset (Data Support)
The offline dataset represents limited data coverage. The actions $a \in \mathbb{R}^2$ follow a bivariate standard normal distribution restricted to an **annular (ring-shaped)** region:
$$1.5 \leq \|a\|_2 \leq 2.5$$
The area inside the ring ($\|a\|_2 < 1.5$) represents the **out-of-distribution (OOD)** region, where no training data exists.

### 2. Reward & Critic Overestimation
*   **Ground Truth Reward $R(a)$**: Peaks on the outer circle ($\|a\|_2 = 2.5$) at $\theta = 0, \pi$. It is 0 at the origin $(0, 0)$.
*   **Learned Critic $\hat{R}(a)$**: Trained on offline dataset $\mathcal{D}$. Due to the lack of training data in the center OOD region, the MLP extrapolates incorrectly, producing an erroneously high reward at the center — simulating the **overestimation error** that plagues offline RL.

### 3. Comparison of Algorithms
*   **IDQL (Select-only)**: Generates 500 candidate actions unguided from the behavior diffusion model, then selects the 64 with the highest $\hat{R}(a)$. Because of overestimation at the center, IDQL selects OOD actions.
*   **QGPO (Guide-only)**: Guides the reverse diffusion process using $\nabla_{a_t} \hat{R}(a_t)$. Since $\hat{R}$ peaks at the center, gradient guidance pulls all generated actions into the OOD center.
*   **Diffusion-DICE (Guide-then-select)**: Learns in-sample weights $w^*(a)$ via DICE dual optimization. Trains a guidance network $g_\theta(a_t, t)$ with IGL loss purely on in-sample data. Sampling guided by $\nabla_{a_t} g_\theta$ pushes actions toward high-value *in-distribution* regions. Final select step avoids OOD overestimation.

---

## File Structure

```
demo/
├── toycase_bandit.py          # Self-contained training and evaluation script
├── requirements.txt           # Python package dependencies
├── toycase_results.png        # Output: Annular dataset experiment results
├── toycase_cluster_results.png # Output: 4-Cluster dataset extension results
└── toycase_tuning.png         # Output: Hyperparameter tuning sweep curves
```

---

## Hardware & Software Environment

This demo runs entirely on **CPU** — no GPU or cloud resources required.

| Component | Specification |
|---|---|
| **OS** | Windows 10/11, macOS, or Linux |
| **Python** | >= 3.8 (tested on Python 3.12.0) |
| **RAM** | Minimum 2 GB |
| **Storage** | ~100 MB |
| **Runtime** | ~3–5 minutes on a dual-core CPU |
| **GPU** | Not required |

---

## Installation and Running

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Demo
```bash
# Windows
py toycase_bandit.py

# Linux/macOS
python toycase_bandit.py
```

The script will automatically:
1. Train all models (Reward MLP, Behavior Diffusion, DICE weights, Guidance Network)
2. Sample actions from IDQL, QGPO, and Diffusion-DICE
3. Generate `toycase_results.png` in the `demo/` folder

---

## Expected Results

After running, `toycase_results.png` is generated in `demo/`. It contains a 2×3 grid:

| | |
|---|---|
| **Dataset Actions** $\pi^{\mathcal{D}}$ | **Ground Truth Reward** $R$ |
| **Learned Reward** $\hat{R}$ (OOD overestimation at center) | |
| **QGPO** (all actions collapse to OOD center) | **IDQL** (many actions leak into OOD center) |
| **Diffusion-DICE** (actions correctly on outer peaks) | |

---

## Extensions (Pre-generated Results)

Two additional experiments were performed and their outputs are already saved:

### Extension 1: 4-Cluster Dataset (`toycase_cluster_results.png`)
Dataset of 4 Gaussian clusters at $(\pm 2.2, \pm 2.2)$ with $\sigma = 0.25$ and an OOD center.  
**Result**: QGPO and IDQL collapse to center; Diffusion-DICE correctly covers all 4 clusters.

### Extension 2: Hyperparameter Tuning (`toycase_tuning.png`)
Sweep of guidance scale $\eta \in [0.0, 2.0]$, measuring:
- **ID Ratio** (safety): fraction of actions within the data support
- **True Reward** (performance): average ground truth reward

**Result**: QGPO's ID Ratio collapses to 0 at $\eta \geq 0.5$; Diffusion-DICE maintains ~95–100% ID Ratio at all scales and peaks in True Reward at $\eta = 1.0$.
