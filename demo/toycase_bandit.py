import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create demo directory if it doesn't exist
os.makedirs("demo", exist_ok=True)

# Helper function to copy output images to the report folder
def copy_to_report(filename):
    src = os.path.join("demo", filename)
    dst = os.path.join("report", "images", filename)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        print(f"[Sync] Copied {src} to {dst}")
    else:
        print(f"[Sync Warning] Source file {src} not found for copying.")

# ==============================================================================
# PART 1: Standard Annular Dataset Setup & Model Definitions
# ==============================================================================

# 1. Dataset Generation (Annular region: 1.5 <= ||a||_2 <= 2.5)
def generate_annular_data(num_samples=1500):
    samples = []
    while len(samples) < num_samples:
        a = np.random.randn(2)
        norm = np.linalg.norm(a)
        if 1.5 <= norm <= 2.5:
            samples.append(a)
    return np.array(samples)

dataset_actions = generate_annular_data(1500)
dataset_actions_t = torch.FloatTensor(dataset_actions)

# Ground truth reward: peaks at outer circle (r=2.5) at theta = 0, pi
def ground_truth_reward(a):
    x, y = a[..., 0], a[..., 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    r_term = np.exp(-((r - 2.5)**2) / 0.5)
    theta_term = (np.cos(2 * theta) + 1.0) / 2.0
    return r_term * theta_term

dataset_rewards = ground_truth_reward(dataset_actions)
dataset_rewards_t = torch.FloatTensor(dataset_rewards).unsqueeze(1)

# 2. Reward MLP Model
class RewardMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# 3. Behavior Diffusion Model
class SimpleScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.net = nn.Sequential(
            nn.Linear(2 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        xt = torch.cat([x, t_embed], dim=-1)
        return self.net(xt)

# 4. Guidance Network (IGL)
class GuidanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        self.net = nn.Sequential(
            nn.Linear(2 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x, t):
        t_embed = self.time_embed(t)
        xt = torch.cat([x, t_embed], dim=-1)
        return self.net(xt).squeeze(-1)

# DDPM Parameters
num_steps = 16
betas = torch.linspace(1e-4, 0.02, num_steps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_t = alphas_cumprod[t].unsqueeze(1)
    return torch.sqrt(alpha_t) * x0 + torch.sqrt(1.0 - alpha_t) * noise

# Generic sampling function supporting customizable models
def sample_diffusion(guidance_type="none", guidance_scale=1.0, num_samples=200, 
                     model=None, reward_fn=None, g_net=None):
    xt = torch.randn(num_samples, 2)
    for t_idx in reversed(range(num_steps)):
        t_tensor = torch.full((num_samples,), t_idx, dtype=torch.long)
        t_float = t_tensor.float().unsqueeze(1) / num_steps
        
        with torch.no_grad():
            pred_noise = model(xt, t_float)
        
        guided_noise = pred_noise
        if guidance_type == "qgpo" and reward_fn is not None:
            xt_grad = xt.clone().detach().requires_grad_(True)
            rewards = reward_fn(xt_grad)
            grad = torch.autograd.grad(torch.sum(rewards), xt_grad)[0]
            std_t = torch.sqrt(1.0 - alphas_cumprod[t_idx])
            guided_noise = pred_noise - guidance_scale * std_t * grad
        elif guidance_type == "dice" and g_net is not None:
            xt_grad = xt.clone().detach().requires_grad_(True)
            g_vals = g_net(xt_grad, t_float)
            grad = torch.autograd.grad(torch.sum(g_vals), xt_grad)[0]
            std_t = torch.sqrt(1.0 - alphas_cumprod[t_idx])
            guided_noise = pred_noise - guidance_scale * std_t * grad
            
        alpha_t = alphas[t_idx]
        alpha_t_cumprod = alphas_cumprod[t_idx]
        beta_t = betas[t_idx]
        
        if t_idx > 0:
            noise = torch.randn_like(xt)
        else:
            noise = torch.zeros_like(xt)
            
        mean = (1.0 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1.0 - alpha_t_cumprod)) * guided_noise)
        sigma = torch.sqrt(beta_t)
        xt = mean + sigma * noise
        
    return xt.detach().numpy()

# ==============================================================================
# TRAINING & EVALUATION: ANNULAR DATASET
# ==============================================================================
print("--- Training models on Annular Dataset ---")

reward_mlp = RewardMLP()
reward_optimizer = optim.Adam(reward_mlp.parameters(), lr=1e-3)
for epoch in range(1000):
    reward_optimizer.zero_grad()
    pred = reward_mlp(dataset_actions_t)
    loss = nn.MSELoss()(pred, dataset_rewards_t)
    loss.backward()
    reward_optimizer.step()

def learned_reward_fn(a_tensor):
    with torch.no_grad():
        pred = reward_mlp(a_tensor)
    r = torch.norm(a_tensor, p=2, dim=-1, keepdim=True)
    overestimate = 1.5 * torch.clamp(1.0 - r, min=0.0)
    return pred + overestimate

diff_model = SimpleScoreNet()
diff_optimizer = optim.Adam(diff_model.parameters(), lr=2e-3)
for epoch in range(2000):
    diff_optimizer.zero_grad()
    t = torch.randint(0, num_steps, (1500,))
    noise = torch.randn_like(dataset_actions_t)
    xt = q_sample(dataset_actions_t, t, noise)
    t_float = t.float().unsqueeze(1) / num_steps
    pred_noise = diff_model(xt, t_float)
    loss = nn.MSELoss()(pred_noise, noise)
    loss.backward()
    diff_optimizer.step()

# DICE Dual Optimization
V = torch.zeros(1, requires_grad=True)
alpha = 0.5
dice_optimizer = optim.Adam([V], lr=1e-2)
with torch.no_grad():
    rewards_learned = learned_reward_fn(dataset_actions_t).squeeze()

for epoch in range(500):
    dice_optimizer.zero_grad()
    sp_term = (rewards_learned - V) / alpha
    f_star = torch.where(sp_term >= 0, sp_term**2 / 4 + sp_term, torch.exp(torch.clamp(sp_term, max=1.0)) - 1)
    loss = V + alpha * torch.mean(f_star)
    loss.backward()
    dice_optimizer.step()

with torch.no_grad():
    sp_term = (rewards_learned - V) / alpha
    w_star = torch.where(sp_term >= 0, sp_term / 2 + 1, torch.exp(sp_term))
    w_star = w_star / w_star.mean()

# Guidance Network (IGL)
guidance_net = GuidanceNet()
guidance_optimizer = optim.Adam(guidance_net.parameters(), lr=1e-3)
for epoch in range(2000):
    guidance_optimizer.zero_grad()
    t = torch.randint(0, num_steps, (1500,))
    t_float = t.float().unsqueeze(1) / num_steps
    noise = torch.randn_like(dataset_actions_t)
    xt = q_sample(dataset_actions_t, t, noise)
    g_val = guidance_net(xt, t_float)
    loss = torch.mean(w_star * torch.exp(-g_val) + g_val)
    loss.backward()
    guidance_optimizer.step()

# Sampling for Annular Dataset
print("Sampling IDQL (unguided) on Annular...")
idql_candidates = sample_diffusion(guidance_type="none", num_samples=500, model=diff_model, reward_fn=learned_reward_fn, g_net=guidance_net)
idql_candidates_t = torch.FloatTensor(idql_candidates)
with torch.no_grad():
    idql_rewards = learned_reward_fn(idql_candidates_t).squeeze().numpy()
idql_best_indices = np.argsort(idql_rewards)[-64:]
idql_actions = idql_candidates[idql_best_indices]

print("Sampling QGPO (guide-based) on Annular...")
qgpo_actions = sample_diffusion(guidance_type="qgpo", guidance_scale=0.5, num_samples=64, model=diff_model, reward_fn=learned_reward_fn, g_net=guidance_net)

print("Sampling Diffusion-DICE on Annular...")
dice_candidates = sample_diffusion(guidance_type="dice", guidance_scale=1.0, num_samples=500, model=diff_model, reward_fn=learned_reward_fn, g_net=guidance_net)
dice_candidates_t = torch.FloatTensor(dice_candidates)
with torch.no_grad():
    dice_rewards = learned_reward_fn(dice_candidates_t).squeeze().numpy()
dice_best_indices = np.argsort(dice_rewards)[-64:]
dice_actions = dice_candidates[dice_best_indices]

# Plotting Annular Dataset Results
print("Generating Annular visualization plot...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Dataset actions
axes[0, 0].scatter(dataset_actions[:, 0], dataset_actions[:, 1], c='blue', alpha=0.3, s=5)
axes[0, 0].set_title(r"Dataset Actions $\pi^{\mathcal{D}}$")
axes[0, 0].set_xlim(-4, 4)
axes[0, 0].set_ylim(-4, 4)
axes[0, 0].set_aspect('equal')

# Plot 2: Ground truth reward
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
grid_actions = np.stack([X, Y], axis=-1)
Z_gt = ground_truth_reward(grid_actions)

im1 = axes[0, 1].contourf(X, Y, Z_gt, levels=50, cmap='viridis')
axes[0, 1].set_title("Ground Truth Reward R")
axes[0, 1].set_xlim(-4, 4)
axes[0, 1].set_ylim(-4, 4)
axes[0, 1].set_aspect('equal')
fig.colorbar(im1, ax=axes[0, 1])

# Plot 3: Learned reward with overestimation peak in center
grid_actions_t = torch.FloatTensor(grid_actions.reshape(-1, 2))
with torch.no_grad():
    Z_learned = learned_reward_fn(grid_actions_t).numpy().reshape(100, 100)

im2 = axes[0, 2].contourf(X, Y, Z_learned, levels=50, cmap='viridis')
axes[0, 2].set_title(r"Learned Reward $\hat{R}$ (with OOD Overestimation)")
axes[0, 2].set_xlim(-4, 4)
axes[0, 2].set_ylim(-4, 4)
axes[0, 2].set_aspect('equal')
fig.colorbar(im2, ax=axes[0, 2])

# Plot 4: QGPO
axes[1, 0].scatter(qgpo_actions[:, 0], qgpo_actions[:, 1], c='red', alpha=0.7, s=15)
axes[1, 0].set_title("QGPO (Guide-only)")
axes[1, 0].set_xlim(-4, 4)
axes[1, 0].set_ylim(-4, 4)
axes[1, 0].set_aspect('equal')

# Plot 5: IDQL
axes[1, 1].scatter(idql_actions[:, 0], idql_actions[:, 1], c='purple', alpha=0.7, s=15)
axes[1, 1].set_title("IDQL (Select-only)")
axes[1, 1].set_xlim(-4, 4)
axes[1, 1].set_ylim(-4, 4)
axes[1, 1].set_aspect('equal')

# Plot 6: Diffusion-DICE
axes[1, 2].scatter(dice_actions[:, 0], dice_actions[:, 1], c='green', alpha=0.7, s=15)
axes[1, 2].set_title("Diffusion-DICE (Ours)")
axes[1, 2].set_xlim(-4, 4)
axes[1, 2].set_ylim(-4, 4)
axes[1, 2].set_aspect('equal')

plt.tight_layout()
output_fig_path = "demo/toycase_results.png"
plt.savefig(output_fig_path, dpi=300)
plt.close()
print(f"Annular results saved to {output_fig_path}")

# ==============================================================================
# EXTENSION 1: 4-CLUSTER DATASET EXPERIMENT
# ==============================================================================
print("\n--- Running Extension 1: 4-Cluster Dataset ---")

def generate_cluster_data(num_samples=1500):
    centers = np.array([
        [2.2, 2.2],
        [2.2, -2.2],
        [-2.2, 2.2],
        [-2.2, -2.2]
    ])
    samples_per_center = num_samples // len(centers)
    samples = []
    for c in centers:
        samples.append(c + 0.25 * np.random.randn(samples_per_center, 2))
    rem = num_samples - samples_per_center * len(centers)
    if rem > 0:
        samples.append(centers[0] + 0.25 * np.random.randn(rem, 2))
    return np.vstack(samples)

dataset_actions_c = generate_cluster_data(1500)
dataset_actions_c_t = torch.FloatTensor(dataset_actions_c)

def ground_truth_reward_cluster(a):
    centers = np.array([
        [2.2, 2.2],
        [2.2, -2.2],
        [-2.2, 2.2],
        [-2.2, -2.2]
    ])
    a_expanded = np.expand_dims(a, axis=-2)  # shape (..., 1, 2)
    dist_sq = np.sum((a_expanded - centers)**2, axis=-1)  # shape (..., 4)
    r_terms = np.exp(-dist_sq / 0.5)
    return np.max(r_terms, axis=-1)

dataset_rewards_c = ground_truth_reward_cluster(dataset_actions_c)
dataset_rewards_c_t = torch.FloatTensor(dataset_rewards_c).unsqueeze(1)

# Train Reward MLP on Cluster Dataset
reward_mlp_c = RewardMLP()
reward_optimizer_c = optim.Adam(reward_mlp_c.parameters(), lr=1e-3)
for epoch in range(1000):
    reward_optimizer_c.zero_grad()
    pred = reward_mlp_c(dataset_actions_c_t)
    loss = nn.MSELoss()(pred, dataset_rewards_c_t)
    loss.backward()
    reward_optimizer_c.step()

def learned_reward_fn_c(a_tensor):
    with torch.no_grad():
        pred = reward_mlp_c(a_tensor)
    r = torch.norm(a_tensor, p=2, dim=-1, keepdim=True)
    overestimate = 1.5 * torch.clamp(1.0 - r, min=0.0)
    return pred + overestimate

# Train Behavior Diffusion on Cluster Dataset
diff_model_c = SimpleScoreNet()
diff_optimizer_c = optim.Adam(diff_model_c.parameters(), lr=2e-3)
for epoch in range(2000):
    diff_optimizer_c.zero_grad()
    t = torch.randint(0, num_steps, (1500,))
    noise = torch.randn_like(dataset_actions_c_t)
    xt = q_sample(dataset_actions_c_t, t, noise)
    t_float = t.float().unsqueeze(1) / num_steps
    pred_noise = diff_model_c(xt, t_float)
    loss = nn.MSELoss()(pred_noise, noise)
    loss.backward()
    diff_optimizer_c.step()

# DICE Dual Optimization on Cluster Dataset
V_c = torch.zeros(1, requires_grad=True)
dice_optimizer_c = optim.Adam([V_c], lr=1e-2)
with torch.no_grad():
    rewards_learned_c = learned_reward_fn_c(dataset_actions_c_t).squeeze()

for epoch in range(500):
    dice_optimizer_c.zero_grad()
    sp_term = (rewards_learned_c - V_c) / alpha
    f_star = torch.where(sp_term >= 0, sp_term**2 / 4 + sp_term, torch.exp(torch.clamp(sp_term, max=1.0)) - 1)
    loss = V_c + alpha * torch.mean(f_star)
    loss.backward()
    dice_optimizer_c.step()

with torch.no_grad():
    sp_term = (rewards_learned_c - V_c) / alpha
    w_star_c = torch.where(sp_term >= 0, sp_term / 2 + 1, torch.exp(sp_term))
    w_star_c = w_star_c / w_star_c.mean()

# Guidance Network (IGL) on Cluster Dataset
guidance_net_c = GuidanceNet()
guidance_optimizer_c = optim.Adam(guidance_net_c.parameters(), lr=1e-3)
for epoch in range(2000):
    guidance_optimizer_c.zero_grad()
    t = torch.randint(0, num_steps, (1500,))
    t_float = t.float().unsqueeze(1) / num_steps
    noise = torch.randn_like(dataset_actions_c_t)
    xt = q_sample(dataset_actions_c_t, t, noise)
    g_val = guidance_net_c(xt, t_float)
    loss = torch.mean(w_star_c * torch.exp(-g_val) + g_val)
    loss.backward()
    guidance_optimizer_c.step()

# Sampling for Cluster Dataset
print("Sampling IDQL (unguided) on 4-Cluster...")
idql_candidates_c = sample_diffusion(guidance_type="none", num_samples=500, model=diff_model_c, reward_fn=learned_reward_fn_c, g_net=guidance_net_c)
idql_candidates_c_t = torch.FloatTensor(idql_candidates_c)
with torch.no_grad():
    idql_rewards_c = learned_reward_fn_c(idql_candidates_c_t).squeeze().numpy()
idql_best_indices_c = np.argsort(idql_rewards_c)[-64:]
idql_actions_c = idql_candidates_c[idql_best_indices_c]

print("Sampling QGPO (guide-based) on 4-Cluster...")
qgpo_actions_c = sample_diffusion(guidance_type="qgpo", guidance_scale=0.5, num_samples=64, model=diff_model_c, reward_fn=learned_reward_fn_c, g_net=guidance_net_c)

print("Sampling Diffusion-DICE on 4-Cluster...")
dice_candidates_c = sample_diffusion(guidance_type="dice", guidance_scale=1.0, num_samples=500, model=diff_model_c, reward_fn=learned_reward_fn_c, g_net=guidance_net_c)
dice_candidates_c_t = torch.FloatTensor(dice_candidates_c)
with torch.no_grad():
    dice_rewards_c = learned_reward_fn_c(dice_candidates_c_t).squeeze().numpy()
dice_best_indices_c = np.argsort(dice_rewards_c)[-64:]
dice_actions_c = dice_candidates_c[dice_best_indices_c]

# Plotting Cluster Dataset Results
print("Generating 4-Cluster visualization plot...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Dataset actions
axes[0, 0].scatter(dataset_actions_c[:, 0], dataset_actions_c[:, 1], c='blue', alpha=0.3, s=5)
axes[0, 0].set_title(r"Dataset Actions $\pi^{\mathcal{D}}$")
axes[0, 0].set_xlim(-4, 4)
axes[0, 0].set_ylim(-4, 4)
axes[0, 0].set_aspect('equal')

# Plot 2: Ground truth reward
Z_gt_c = ground_truth_reward_cluster(grid_actions)
im1 = axes[0, 1].contourf(X, Y, Z_gt_c, levels=50, cmap='viridis')
axes[0, 1].set_title("Ground Truth Reward R")
axes[0, 1].set_xlim(-4, 4)
axes[0, 1].set_ylim(-4, 4)
axes[0, 1].set_aspect('equal')
fig.colorbar(im1, ax=axes[0, 1])

# Plot 3: Learned reward with overestimation peak in center
with torch.no_grad():
    Z_learned_c = learned_reward_fn_c(grid_actions_t).numpy().reshape(100, 100)
im2 = axes[0, 2].contourf(X, Y, Z_learned_c, levels=50, cmap='viridis')
axes[0, 2].set_title(r"Learned Reward $\hat{R}$ (with OOD Overestimation)")
axes[0, 2].set_xlim(-4, 4)
axes[0, 2].set_ylim(-4, 4)
axes[0, 2].set_aspect('equal')
fig.colorbar(im2, ax=axes[0, 2])

# Plot 4: QGPO
axes[1, 0].scatter(qgpo_actions_c[:, 0], qgpo_actions_c[:, 1], c='red', alpha=0.7, s=15)
axes[1, 0].set_title("QGPO (Guide-only)")
axes[1, 0].set_xlim(-4, 4)
axes[1, 0].set_ylim(-4, 4)
axes[1, 0].set_aspect('equal')

# Plot 5: IDQL
axes[1, 1].scatter(idql_actions_c[:, 0], idql_actions_c[:, 1], c='purple', alpha=0.7, s=15)
axes[1, 1].set_title("IDQL (Select-only)")
axes[1, 1].set_xlim(-4, 4)
axes[1, 1].set_ylim(-4, 4)
axes[1, 1].set_aspect('equal')

# Plot 6: Diffusion-DICE
axes[1, 2].scatter(dice_actions_c[:, 0], dice_actions_c[:, 1], c='green', alpha=0.7, s=15)
axes[1, 2].set_title("Diffusion-DICE (Ours)")
axes[1, 2].set_xlim(-4, 4)
axes[1, 2].set_ylim(-4, 4)
axes[1, 2].set_aspect('equal')

plt.tight_layout()
output_fig_path_c = "demo/toycase_cluster_results.png"
plt.savefig(output_fig_path_c, dpi=300)
plt.close()
print(f"4-Cluster results saved to {output_fig_path_c}")

# ==============================================================================
# EXTENSION 2: GUIDANCE SCALE HYPERPARAMETER SWEEP
# ==============================================================================
print("\n--- Running Extension 2: Guidance Scale Sweep ---")

scales = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
qgpo_id_ratios = []
qgpo_true_rewards = []
dice_id_ratios = []
dice_true_rewards = []

for scale in scales:
    print(f"  Sweeping scale = {scale}...")
    # 1. QGPO
    qgpo_sweep_actions = sample_diffusion(guidance_type="qgpo", guidance_scale=scale, num_samples=200, model=diff_model, reward_fn=learned_reward_fn, g_net=guidance_net)
    norms_qgpo = np.linalg.norm(qgpo_sweep_actions, axis=-1)
    # Annular bounds are 1.5 <= ||a||_2 <= 2.5. We use 1.4 <= ||a||_2 <= 2.6 to allow slight tolerance.
    qgpo_id = np.mean((norms_qgpo >= 1.4) & (norms_qgpo <= 2.6))
    qgpo_reward = np.mean(ground_truth_reward(qgpo_sweep_actions))
    qgpo_id_ratios.append(qgpo_id)
    qgpo_true_rewards.append(qgpo_reward)
    
    # 2. Diffusion-DICE
    dice_sweep_candidates = sample_diffusion(guidance_type="dice", guidance_scale=scale, num_samples=500, model=diff_model, reward_fn=learned_reward_fn, g_net=guidance_net)
    dice_sweep_candidates_t = torch.FloatTensor(dice_sweep_candidates)
    with torch.no_grad():
        dice_sweep_rewards = learned_reward_fn(dice_sweep_candidates_t).squeeze().numpy()
    dice_best_indices = np.argsort(dice_sweep_rewards)[-200:]
    dice_sweep_actions = dice_sweep_candidates[dice_best_indices]
    
    norms_dice = np.linalg.norm(dice_sweep_actions, axis=-1)
    dice_id = np.mean((norms_dice >= 1.4) & (norms_dice <= 2.6))
    dice_reward = np.mean(ground_truth_reward(dice_sweep_actions))
    dice_id_ratios.append(dice_id)
    dice_true_rewards.append(dice_reward)

# Plotting Sweep curves
print("Generating tuning sweep plots...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot Left: ID Ratio
axes[0].plot(scales, qgpo_id_ratios, label="QGPO (Guide-only)", marker='o', color='red', linestyle='--')
axes[0].plot(scales, dice_id_ratios, label="Diffusion-DICE (Ours)", marker='s', color='green')
axes[0].set_xlabel("Guidance Scale $\eta$")
axes[0].set_ylabel("In-Distribution Ratio")
axes[0].set_title("Safety (In-Distribution Ratio)")
axes[0].grid(True)
axes[0].legend()

# Plot Right: True Reward
axes[1].plot(scales, qgpo_true_rewards, label="QGPO (Guide-only)", marker='o', color='red', linestyle='--')
axes[1].plot(scales, dice_true_rewards, label="Diffusion-DICE (Ours)", marker='s', color='green')
axes[1].set_xlabel("Guidance Scale $\eta$")
axes[1].set_ylabel("Average True Reward")
axes[1].set_title("Performance (True Reward)")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
output_fig_path_t = "demo/toycase_tuning.png"
plt.savefig(output_fig_path_t, dpi=300)
plt.close()
print(f"Guidance sweep plot saved to {output_fig_path_t}")

# ==============================================================================
# COPYING ALL OUTPUTS TO THE REPORT IMAGE DIRECTORY
# ==============================================================================
print("\n--- Synchronizing all figures with report/images/ ---")
copy_to_report("toycase_results.png")
copy_to_report("toycase_cluster_results.png")
copy_to_report("toycase_tuning.png")

print("\nAll demo components executed and synchronized successfully!")
