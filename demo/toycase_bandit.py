import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create demo directory if it doesn't exist
os.makedirs("demo", exist_ok=True)

# 1. Dataset Generation
# Annular region: 1.5 <= ||a||_2 <= 2.5
def generate_annular_data(num_samples=1500):
    samples = []
    while len(samples) < num_samples:
        # Sample from bivariate standard normal
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

# 2. Reward MLP Training
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

reward_mlp = RewardMLP()
reward_optimizer = optim.Adam(reward_mlp.parameters(), lr=1e-3)

# Train Reward MLP on the dataset
for epoch in range(1000):
    reward_optimizer.zero_grad()
    pred = reward_mlp(dataset_actions_t)
    loss = nn.MSELoss()(pred, dataset_rewards_t)
    loss.backward()
    reward_optimizer.step()

# Wrap reward MLP to explicitly incorporate overestimation in the OOD center
def learned_reward_fn(a_tensor):
    with torch.no_grad():
        pred = reward_mlp(a_tensor)
    # Add a synthetic overestimation peak in the center OOD region (r < 1.0)
    r = torch.norm(a_tensor, p=2, dim=-1, keepdim=True)
    overestimate = 1.5 * torch.clamp(1.0 - r, min=0.0)
    return pred + overestimate

# 3. Behavior Diffusion Model
class SimpleScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Score network predicts the noise epsilon
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

diff_model = SimpleScoreNet()
diff_optimizer = optim.Adam(diff_model.parameters(), lr=2e-3)

# Simple Beta Schedule (DDPM)
num_steps = 16
betas = torch.linspace(1e-4, 0.02, num_steps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_t = alphas_cumprod[t].unsqueeze(1)
    return torch.sqrt(alpha_t) * x0 + torch.sqrt(1.0 - alpha_t) * noise

# Train diffusion model
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

# 4. DICE Optimization for weights w*(a)
# We optimize a single scalar V to minimize the DICE dual objective
V = torch.zeros(1, requires_grad=True)
alpha = 0.5
dice_optimizer = optim.Adam([V], lr=1e-2)

with torch.no_grad():
    rewards_learned = learned_reward_fn(dataset_actions_t).squeeze()

for epoch in range(500):
    dice_optimizer.zero_grad()
    # bellman residual for bandit is simply reward - V
    sp_term = (rewards_learned - V) / alpha
    # Piecewise f-divergence conjugate
    # f*(x) = x^2 / 4 + x (for x >= 0), exp(x) - 1 (for x < 0)
    f_star = torch.where(sp_term >= 0, sp_term**2 / 4 + sp_term, torch.exp(torch.clamp(sp_term, max=1.0)) - 1)
    loss = V + alpha * torch.mean(f_star)
    loss.backward()
    dice_optimizer.step()

# Compute learned weights w*(a)
with torch.no_grad():
    sp_term = (rewards_learned - V) / alpha
    # (f')^-1(x) = x/2 + 1 (for x >= 0), exp(x) (for x < 0)
    w_star = torch.where(sp_term >= 0, sp_term / 2 + 1, torch.exp(sp_term))
    # Normalize weights
    w_star = w_star / w_star.mean()

# 5. In-sample Guidance Network Training (IGL)
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

guidance_net = GuidanceNet()
guidance_optimizer = optim.Adam(guidance_net.parameters(), lr=1e-3)

for epoch in range(2000):
    guidance_optimizer.zero_grad()
    t = torch.randint(0, num_steps, (1500,))
    t_float = t.float().unsqueeze(1) / num_steps
    noise = torch.randn_like(dataset_actions_t)
    xt = q_sample(dataset_actions_t, t, noise)
    
    g_val = guidance_net(xt, t_float)
    # Guidance loss: E [ w*(a) * exp(-g) + g ]
    loss = torch.mean(w_star * torch.exp(-g_val) + g_val)
    loss.backward()
    guidance_optimizer.step()

# 6. Sampling procedures
def sample_diffusion(guidance_type="none", guidance_scale=1.0, num_samples=200):
    # Start with standard normal noise
    xt = torch.randn(num_samples, 2)
    
    for t_idx in reversed(range(num_steps)):
        t_tensor = torch.full((num_samples,), t_idx, dtype=torch.long)
        t_float = t_tensor.float().unsqueeze(1) / num_steps
        
        # Predict behavior noise
        with torch.no_grad():
            pred_noise = diff_model(xt, t_float)
        
        # Calculate guidance gradient
        if guidance_type == "none":
            guided_noise = pred_noise
        elif guidance_type == "qgpo":
            # Guided by gradient of reward MLP
            xt_grad = xt.clone().detach().requires_grad_(True)
            rewards = learned_reward_fn(xt_grad)
            grad = torch.autograd.grad(torch.sum(rewards), xt_grad)[0]
            # Guided noise: epsilon = epsilon_b - eta * sqrt(1 - cumprod) * grad
            std_t = torch.sqrt(1.0 - alphas_cumprod[t_idx])
            guided_noise = pred_noise - guidance_scale * std_t * grad
        elif guidance_type == "dice":
            # Guided by IGL gradient
            xt_grad = xt.clone().detach().requires_grad_(True)
            g_vals = guidance_net(xt_grad, t_float)
            grad = torch.autograd.grad(torch.sum(g_vals), xt_grad)[0]
            std_t = torch.sqrt(1.0 - alphas_cumprod[t_idx])
            guided_noise = pred_noise - guidance_scale * std_t * grad
            
        # DDPM step
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

# Run the 3 algorithms
print("Sampling IDQL (unguided)...")
# IDQL: Sample 500 candidates, then select 64 times with learned_reward_fn
idql_candidates = sample_diffusion(guidance_type="none", num_samples=500)
idql_candidates_t = torch.FloatTensor(idql_candidates)
with torch.no_grad():
    idql_rewards = learned_reward_fn(idql_candidates_t).squeeze().numpy()
# Select 64 actions with highest reward
idql_best_indices = np.argsort(idql_rewards)[-64:]
idql_actions = idql_candidates[idql_best_indices]

print("Sampling QGPO (guide-based)...")
# QGPO: Guided by MLP gradient, guidance scale = 0.5
qgpo_actions = sample_diffusion(guidance_type="qgpo", guidance_scale=0.5, num_samples=64)

print("Sampling Diffusion-DICE...")
# Diffusion-DICE: Guided by IGL gradient, guidance scale = 1.0
dice_candidates = sample_diffusion(guidance_type="dice", guidance_scale=1.0, num_samples=500)
dice_candidates_t = torch.FloatTensor(dice_candidates)
with torch.no_grad():
    dice_rewards = learned_reward_fn(dice_candidates_t).squeeze().numpy()
# Select 64 actions with highest reward
dice_best_indices = np.argsort(dice_rewards)[-64:]
dice_actions = dice_candidates[dice_best_indices]

# 7. Plotting and Visualizations
print("Generating visualization plots...")
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

# Plot 4: QGPO actions (stuck in center)
axes[1, 0].scatter(qgpo_actions[:, 0], qgpo_actions[:, 1], c='red', alpha=0.7, s=15)
axes[1, 0].set_title("QGPO (Guide-only)")
axes[1, 0].set_xlim(-4, 4)
axes[1, 0].set_ylim(-4, 4)
axes[1, 0].set_aspect('equal')

# Plot 5: IDQL actions (mostly in center due to select-only errors)
axes[1, 1].scatter(idql_actions[:, 0], idql_actions[:, 1], c='purple', alpha=0.7, s=15)
axes[1, 1].set_title("IDQL (Select-only)")
axes[1, 1].set_xlim(-4, 4)
axes[1, 1].set_ylim(-4, 4)
axes[1, 1].set_aspect('equal')

# Plot 6: Diffusion-DICE actions (successfully on the outer ring peaks)
axes[1, 2].scatter(dice_actions[:, 0], dice_actions[:, 1], c='green', alpha=0.7, s=15)
axes[1, 2].set_title("Diffusion-DICE (Ours)")
axes[1, 2].set_xlim(-4, 4)
axes[1, 2].set_ylim(-4, 4)
axes[1, 2].set_aspect('equal')

plt.tight_layout()
output_fig_path = "demo/toycase_results.png"
plt.savefig(output_fig_path, dpi=300)
plt.close()

print(f"Visualization saved to {output_fig_path}")
print("Demo execution finished successfully!")
