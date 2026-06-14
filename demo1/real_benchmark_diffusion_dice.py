import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class TrainConfig:
    seed: int = 42
    reward_epochs: int = 700
    diffusion_epochs: int = 900
    dice_epochs: int = 300
    guidance_epochs: int = 700
    num_steps: int = 16
    num_candidates: int = 600
    num_actions: int = 64
    hidden_size: int = 64
    grid_size: int = 120


@dataclass
class RealDatasetProblem:
    name: str
    display_name: str
    actions: np.ndarray
    labels: np.ndarray
    target_label: int
    target_name: str
    support_radius: float
    reward_bandwidth: float
    guidance_bandwidth: float
    xlim: tuple
    ylim: tuple
    pca_explained_variance: float


@dataclass
class TrainedContext:
    problem: RealDatasetProblem
    config: TrainConfig
    reward_mlp: nn.Module
    diff_model: nn.Module
    guidance_net: nn.Module
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    anchor_actions: torch.Tensor = None
    anchor_log_weights: torch.Tensor = None


class RewardMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SimpleScoreNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        self.net = nn.Sequential(
            nn.Linear(18, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1))


class GuidanceNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        self.net = nn.Sequential(
            nn.Linear(18, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1)).squeeze(-1)


def load_real_dataset(dataset_name, seed):
    if dataset_name == "breast_cancer":
        raw = load_breast_cancer()
        target_label = 1
        display_name = "Breast Cancer Wisconsin Diagnostic"
    elif dataset_name == "wine":
        raw = load_wine()
        target_label = 0
        display_name = "Wine Recognition"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    x_scaled = StandardScaler().fit_transform(raw.data)
    pca = PCA(n_components=2, random_state=seed)
    x_pca = pca.fit_transform(x_scaled)
    actions = StandardScaler().fit_transform(x_pca).astype(np.float32)
    labels = raw.target.astype(np.int64)

    nn_model = NearestNeighbors(n_neighbors=2).fit(actions)
    distances, _ = nn_model.kneighbors(actions)
    nearest_other = distances[:, 1]
    support_radius = float(np.quantile(nearest_other, 0.95) * 2.0)
    support_radius = max(support_radius, 0.25)
    reward_bandwidth = max(support_radius * 1.15, 0.35)
    guidance_bandwidth = max(support_radius * 0.9, 0.30)

    pad = 0.8
    xlim = (float(actions[:, 0].min() - pad), float(actions[:, 0].max() + pad))
    ylim = (float(actions[:, 1].min() - pad), float(actions[:, 1].max() + pad))

    return RealDatasetProblem(
        name=dataset_name,
        display_name=display_name,
        actions=actions,
        labels=labels,
        target_label=target_label,
        target_name=str(raw.target_names[target_label]),
        support_radius=support_radius,
        reward_bandwidth=reward_bandwidth,
        guidance_bandwidth=guidance_bandwidth,
        xlim=xlim,
        ylim=ylim,
        pca_explained_variance=float(np.sum(pca.explained_variance_ratio_)),
    )


def target_vector(problem):
    return (problem.labels == problem.target_label).astype(np.float32)


def nearest_distances(problem, actions):
    diff = actions[:, None, :] - problem.actions[None, :, :]
    return np.sqrt(np.min(np.sum(diff**2, axis=-1), axis=1))


def support_mask(problem, actions):
    return nearest_distances(problem, actions) <= problem.support_radius


def knn_oracle_reward(problem, actions):
    diff = actions[:, None, :] - problem.actions[None, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    weights = np.exp(-dist_sq / (2.0 * problem.reward_bandwidth**2))
    y_target = target_vector(problem)
    return (weights @ y_target) / (np.sum(weights, axis=1) + 1e-8)


def make_schedule(num_steps):
    betas = torch.linspace(1e-4, 0.02, num_steps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def q_sample(x0, t, alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_t = alphas_cumprod[t].unsqueeze(1)
    return torch.sqrt(alpha_t) * x0 + torch.sqrt(1.0 - alpha_t) * noise


def critic_probability(ctx, a_tensor, detach=False):
    if detach:
        with torch.no_grad():
            return torch.sigmoid(ctx.reward_mlp(a_tensor))
    return torch.sigmoid(ctx.reward_mlp(a_tensor))


def critic_logit(ctx, a_tensor, detach=False):
    if detach:
        with torch.no_grad():
            return ctx.reward_mlp(a_tensor)
    return ctx.reward_mlp(a_tensor)


def train_problem(problem, config):
    print(f"\nTraining on real dataset: {problem.display_name}")
    print(
        f"Samples={len(problem.actions)}, target='{problem.target_name}', "
        f"PCA variance={problem.pca_explained_variance:.3f}"
    )

    actions_t = torch.tensor(problem.actions, dtype=torch.float32)
    y_t = torch.tensor(target_vector(problem), dtype=torch.float32)

    reward_mlp = RewardMLP(config.hidden_size)
    reward_optimizer = optim.Adam(reward_mlp.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(config.reward_epochs):
        reward_optimizer.zero_grad()
        loss = bce(reward_mlp(actions_t), y_t)
        loss.backward()
        reward_optimizer.step()

    diff_model = SimpleScoreNet(config.hidden_size)
    diff_optimizer = optim.Adam(diff_model.parameters(), lr=2e-3)
    betas, alphas, alphas_cumprod = make_schedule(config.num_steps)

    ctx = TrainedContext(
        problem=problem,
        config=config,
        reward_mlp=reward_mlp,
        diff_model=diff_model,
        guidance_net=None,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
    )

    for _ in range(config.diffusion_epochs):
        diff_optimizer.zero_grad()
        t = torch.randint(0, config.num_steps, (len(actions_t),))
        noise = torch.randn_like(actions_t)
        xt = q_sample(actions_t, t, alphas_cumprod, noise)
        t_float = t.float().unsqueeze(1) / config.num_steps
        loss = nn.MSELoss()(diff_model(xt, t_float), noise)
        loss.backward()
        diff_optimizer.step()

    with torch.no_grad():
        learned_rewards = critic_probability(ctx, actions_t, detach=True)

    v_value = torch.zeros(1, requires_grad=True)
    alpha = 0.5
    dice_optimizer = optim.Adam([v_value], lr=1e-2)

    for _ in range(config.dice_epochs):
        dice_optimizer.zero_grad()
        sp_term = (learned_rewards - v_value) / alpha
        f_star = torch.where(
            sp_term >= 0,
            sp_term**2 / 4 + sp_term,
            torch.exp(torch.clamp(sp_term, max=1.0)) - 1,
        )
        loss = v_value + alpha * torch.mean(f_star)
        loss.backward()
        dice_optimizer.step()

    with torch.no_grad():
        sp_term = (learned_rewards - v_value) / alpha
        w_star = torch.where(sp_term >= 0, sp_term / 2 + 1, torch.exp(sp_term))
        w_star = w_star / w_star.mean()
        ctx.anchor_actions = actions_t.clone()
        ctx.anchor_log_weights = torch.log(torch.clamp(w_star, min=1e-6))

    guidance_net = GuidanceNet(config.hidden_size)
    guidance_optimizer = optim.Adam(guidance_net.parameters(), lr=1e-3)
    ctx.guidance_net = guidance_net

    for _ in range(config.guidance_epochs):
        guidance_optimizer.zero_grad()
        t = torch.randint(0, config.num_steps, (len(actions_t),))
        noise = torch.randn_like(actions_t)
        xt = q_sample(actions_t, t, alphas_cumprod, noise)
        t_float = t.float().unsqueeze(1) / config.num_steps
        g_val = guidance_net(xt, t_float)
        loss = torch.mean(w_star * torch.exp(-g_val) + g_val)
        loss.backward()
        guidance_optimizer.step()

    return ctx


def in_sample_guidance_score(ctx, a_tensor, bandwidth=None):
    if bandwidth is None:
        bandwidth = ctx.problem.guidance_bandwidth
    diff = a_tensor[:, None, :] - ctx.anchor_actions[None, :, :]
    dist_sq = torch.sum(diff**2, dim=-1)
    log_kernel = ctx.anchor_log_weights[None, :] - dist_sq / (2.0 * bandwidth**2)
    return torch.logsumexp(log_kernel, dim=1)


def sample_diffusion(ctx, guidance_type="none", guidance_scale=1.0, num_samples=200):
    xt = torch.randn(num_samples, 2)
    config = ctx.config

    for t_idx in reversed(range(config.num_steps)):
        t_tensor = torch.full((num_samples,), t_idx, dtype=torch.long)
        t_float = t_tensor.float().unsqueeze(1) / config.num_steps

        with torch.no_grad():
            pred_noise = ctx.diff_model(xt, t_float)

        guided_noise = pred_noise
        if guidance_type == "qgpo":
            xt_grad = xt.clone().detach().requires_grad_(True)
            scores = critic_logit(ctx, xt_grad, detach=False)
            grad = torch.autograd.grad(torch.sum(scores), xt_grad)[0]
            std_t = torch.sqrt(1.0 - ctx.alphas_cumprod[t_idx])
            guided_noise = pred_noise - guidance_scale * std_t * grad
        elif guidance_type == "dice":
            xt_grad = xt.clone().detach().requires_grad_(True)
            bandwidth = ctx.problem.guidance_bandwidth + 0.15 * (
                t_idx / max(1, config.num_steps - 1)
            )
            g_vals = ctx.guidance_net(xt_grad, t_float)
            g_vals = g_vals + in_sample_guidance_score(ctx, xt_grad, bandwidth=bandwidth)
            grad = torch.autograd.grad(torch.sum(g_vals), xt_grad)[0]
            std_t = torch.sqrt(1.0 - ctx.alphas_cumprod[t_idx])
            guided_noise = pred_noise - guidance_scale * std_t * grad

        alpha_t = ctx.alphas[t_idx]
        alpha_t_cumprod = ctx.alphas_cumprod[t_idx]
        beta_t = ctx.betas[t_idx]
        noise = torch.randn_like(xt) if t_idx > 0 else torch.zeros_like(xt)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / torch.sqrt(1.0 - alpha_t_cumprod)) * guided_noise
        )
        xt = mean + torch.sqrt(beta_t) * noise

    return xt.detach().numpy()


def select_top_by_critic(ctx, candidates, num_actions, support_gate=False):
    candidate_t = torch.tensor(candidates, dtype=torch.float32)
    critic_scores = critic_probability(ctx, candidate_t, detach=True).numpy()
    pool_indices = np.arange(len(candidates))

    if support_gate:
        with torch.no_grad():
            support_scores = in_sample_guidance_score(ctx, candidate_t).numpy()
        keep_count = max(num_actions, len(candidates) // 2)
        pool_indices = np.argsort(support_scores)[-keep_count:]

    selected_pool = pool_indices[np.argsort(critic_scores[pool_indices])[-num_actions:]]
    return candidates[selected_pool]


def run_methods(ctx, qgpo_eta=2.0, dice_eta=1.0):
    config = ctx.config

    print("Sampling IDQL-style select-only baseline...")
    idql_candidates = sample_diffusion(
        ctx,
        guidance_type="none",
        guidance_scale=0.0,
        num_samples=config.num_candidates,
    )
    idql_actions = select_top_by_critic(ctx, idql_candidates, config.num_actions)

    print("Sampling QGPO-style guide-only baseline...")
    qgpo_actions = sample_diffusion(
        ctx,
        guidance_type="qgpo",
        guidance_scale=qgpo_eta,
        num_samples=config.num_actions,
    )

    print("Sampling Diffusion-DICE guide-then-select method...")
    dice_candidates = sample_diffusion(
        ctx,
        guidance_type="dice",
        guidance_scale=dice_eta,
        num_samples=config.num_candidates,
    )
    dice_actions = select_top_by_critic(
        ctx,
        dice_candidates,
        config.num_actions,
        support_gate=True,
    )

    return {
        "QGPO-style guide-only": qgpo_actions,
        "IDQL-style select-only": idql_actions,
        "Diffusion-DICE": dice_actions,
    }


def evaluate_actions(ctx, method, actions, guidance_scale=""):
    problem = ctx.problem
    action_t = torch.tensor(actions, dtype=torch.float32)
    oracle_rewards = knn_oracle_reward(problem, actions)
    critic_rewards = critic_probability(ctx, action_t, detach=True).numpy()
    nearest = nearest_distances(problem, actions)
    in_support = nearest <= problem.support_radius

    return {
        "dataset": problem.name,
        "target_class": problem.target_name,
        "method": method,
        "guidance_scale": guidance_scale,
        "num_actions": len(actions),
        "id_ratio": float(np.mean(in_support)),
        "ood_ratio": float(1.0 - np.mean(in_support)),
        "oracle_reward_mean": float(np.mean(oracle_rewards)),
        "critic_reward_mean": float(np.mean(critic_rewards)),
        "target_region_ratio": float(np.mean(oracle_rewards >= 0.5)),
        "mean_nearest_distance": float(np.mean(nearest)),
    }


def save_metrics_csv(rows, output_path):
    fieldnames = [
        "dataset",
        "target_class",
        "method",
        "guidance_scale",
        "num_actions",
        "id_ratio",
        "ood_ratio",
        "oracle_reward_mean",
        "critic_reward_mean",
        "target_region_ratio",
        "mean_nearest_distance",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_metrics(rows):
    print("\nMetrics:")
    for row in rows:
        print(
            f"- {row['method']}: "
            f"ID={row['id_ratio']:.3f}, "
            f"OOD={row['ood_ratio']:.3f}, "
            f"OracleR={row['oracle_reward_mean']:.3f}, "
            f"CriticR={row['critic_reward_mean']:.3f}, "
            f"TargetRegion={row['target_region_ratio']:.3f}"
        )


def plot_experiment(ctx, actions_by_method, output_path):
    problem = ctx.problem
    config = ctx.config
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    target_mask = problem.labels == problem.target_label
    axes[0, 0].scatter(
        problem.actions[~target_mask, 0],
        problem.actions[~target_mask, 1],
        c="tab:gray",
        alpha=0.45,
        s=12,
        label="other class",
    )
    axes[0, 0].scatter(
        problem.actions[target_mask, 0],
        problem.actions[target_mask, 1],
        c="tab:blue",
        alpha=0.65,
        s=14,
        label=f"target: {problem.target_name}",
    )
    axes[0, 0].set_title("Real dataset support after PCA")
    axes[0, 0].legend(loc="best", fontsize=8)

    x = np.linspace(problem.xlim[0], problem.xlim[1], config.grid_size)
    y = np.linspace(problem.ylim[0], problem.ylim[1], config.grid_size)
    x_grid, y_grid = np.meshgrid(x, y)
    grid_actions = np.stack([x_grid, y_grid], axis=-1)
    flat_grid = grid_actions.reshape(-1, 2).astype(np.float32)
    flat_grid_t = torch.tensor(flat_grid, dtype=torch.float32)

    z_oracle = knn_oracle_reward(problem, flat_grid).reshape(config.grid_size, config.grid_size)
    im1 = axes[0, 1].contourf(x_grid, y_grid, z_oracle, levels=50, cmap="viridis")
    axes[0, 1].set_title("KNN oracle reward from real labels")
    fig.colorbar(im1, ax=axes[0, 1])

    z_critic = critic_probability(ctx, flat_grid_t, detach=True).numpy()
    z_critic = z_critic.reshape(config.grid_size, config.grid_size)
    im2 = axes[0, 2].contourf(x_grid, y_grid, z_critic, levels=50, cmap="viridis")
    axes[0, 2].set_title("Learned critic score")
    fig.colorbar(im2, ax=axes[0, 2])

    plot_order = [
        ("QGPO-style guide-only", "tab:red"),
        ("IDQL-style select-only", "tab:purple"),
        ("Diffusion-DICE", "tab:green"),
    ]
    for ax, (method, color) in zip(axes[1], plot_order):
        ax.scatter(
            problem.actions[:, 0],
            problem.actions[:, 1],
            c="lightgray",
            alpha=0.35,
            s=8,
        )
        actions = actions_by_method[method]
        ax.scatter(actions[:, 0], actions[:, 1], c=color, alpha=0.80, s=22)
        ax.set_title(method)

    for ax in axes.ravel():
        ax.set_xlim(*problem.xlim)
        ax.set_ylim(*problem.ylim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)

    title = (
        f"{problem.display_name} real-data benchmark "
        f"(target: {problem.target_name}, PCA variance: {problem.pca_explained_variance:.2f})"
    )
    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def run_policy_comparison(ctx, figure_path, metrics_path, qgpo_eta=2.0, dice_eta=1.0):
    actions_by_method = run_methods(ctx, qgpo_eta=qgpo_eta, dice_eta=dice_eta)
    rows = []
    for method, actions in actions_by_method.items():
        scale = qgpo_eta if method.startswith("QGPO") else dice_eta if method == "Diffusion-DICE" else ""
        rows.append(evaluate_actions(ctx, method, actions, guidance_scale=scale))

    plot_experiment(ctx, actions_by_method, figure_path)
    save_metrics_csv(rows, metrics_path)
    print_metrics(rows)
    print(f"Saved figure: {figure_path}")
    print(f"Saved metrics: {metrics_path}")
    return rows


def run_guidance_sweep(ctx, figure_path, metrics_path):
    print("\nRunning real-data guidance-scale sweep...")
    eta_values = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    rows = []

    idql_candidates = sample_diffusion(
        ctx,
        guidance_type="none",
        guidance_scale=0.0,
        num_samples=ctx.config.num_candidates,
    )
    idql_actions = select_top_by_critic(ctx, idql_candidates, ctx.config.num_actions)
    idql_row = evaluate_actions(ctx, "IDQL-style select-only", idql_actions, guidance_scale="")

    for eta in eta_values:
        qgpo_actions = sample_diffusion(
            ctx,
            guidance_type="qgpo",
            guidance_scale=eta,
            num_samples=ctx.config.num_actions,
        )
        rows.append(evaluate_actions(ctx, "QGPO-style guide-only", qgpo_actions, eta))

        dice_candidates = sample_diffusion(
            ctx,
            guidance_type="dice",
            guidance_scale=eta,
            num_samples=ctx.config.num_candidates,
        )
        dice_actions = select_top_by_critic(
            ctx,
            dice_candidates,
            ctx.config.num_actions,
            support_gate=True,
        )
        rows.append(evaluate_actions(ctx, "Diffusion-DICE", dice_actions, eta))

    rows.append(idql_row)
    save_metrics_csv(rows, metrics_path)
    plot_guidance_sweep(rows, eta_values, idql_row, figure_path)
    print_metrics(rows)
    print(f"Saved sweep figure: {figure_path}")
    print(f"Saved sweep metrics: {metrics_path}")
    return rows


def plot_guidance_sweep(rows, eta_values, idql_row, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = ["QGPO-style guide-only", "Diffusion-DICE"]
    colors = {
        "QGPO-style guide-only": "tab:red",
        "Diffusion-DICE": "tab:green",
    }

    for metric, ax, title in [
        ("id_ratio", axes[0], "In-distribution ratio"),
        ("oracle_reward_mean", axes[1], "Mean oracle reward from real labels"),
    ]:
        for method in methods:
            values = [
                row[metric]
                for row in rows
                if row["method"] == method and row["guidance_scale"] != ""
            ]
            ax.plot(
                eta_values,
                values,
                marker="o",
                linewidth=2,
                label=method,
                color=colors[method],
            )
        ax.axhline(
            idql_row[metric],
            linestyle="--",
            color="tab:purple",
            label="IDQL-style select-only",
        )
        ax.set_xlabel("Guidance scale eta")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("Hyperparameter sweep on real dataset")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def config_from_args(args):
    if not args.quick:
        return TrainConfig(seed=args.seed)
    return TrainConfig(
        seed=args.seed,
        reward_epochs=120,
        diffusion_epochs=160,
        dice_epochs=80,
        guidance_epochs=160,
        num_steps=8,
        num_candidates=200,
        num_actions=40,
        hidden_size=48,
        grid_size=70,
    )


def output_prefix(dataset_name):
    return f"real_{dataset_name}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-dataset Diffusion-DICE demo for CSC14005.",
    )
    parser.add_argument(
        "--dataset",
        choices=["breast_cancer", "wine", "all"],
        default="all",
        help="Real dataset to run.",
    )
    parser.add_argument(
        "--experiment",
        choices=["comparison", "sweep", "all"],
        default="all",
        help="Experiment type.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast smoke-test configuration.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix inserted before output file extensions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = config_from_args(args)
    set_seed(config.seed)
    SCRIPT_DIR.mkdir(exist_ok=True)

    dataset_names = ["breast_cancer", "wine"] if args.dataset == "all" else [args.dataset]

    for dataset_name in dataset_names:
        problem = load_real_dataset(dataset_name, seed=config.seed)
        ctx = train_problem(problem, config)
        prefix = output_prefix(dataset_name)
        suffix = args.output_suffix

        if args.experiment in {"comparison", "all"}:
            run_policy_comparison(
                ctx,
                SCRIPT_DIR / f"{prefix}_results{suffix}.png",
                SCRIPT_DIR / f"{prefix}_metrics{suffix}.csv",
            )

        if args.experiment in {"sweep", "all"}:
            run_guidance_sweep(
                ctx,
                SCRIPT_DIR / f"{prefix}_tuning{suffix}.png",
                SCRIPT_DIR / f"{prefix}_tuning_metrics{suffix}.csv",
            )

    print("\nReal benchmark demo finished successfully.")


if __name__ == "__main__":
    main()
