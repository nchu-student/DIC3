"""
Comparing 6 Bandit Strategies: Cumulative Regret over 10,000 Steps
===================================================================
Strategies: A/B Testing, Optimistic Initial Values, ε-Greedy,
            Softmax (Boltzmann), UCB, Thompson Sampling

Arms:
  A  →  true mean = 0.8
  B  →  true mean = 0.7
  C  →  true mean = 0.5

Each arm is Bernoulli (reward ∈ {0, 1}) with the given mean.
Budget = 10,000 pulls.  Results averaged over multiple runs.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
TRUE_MEANS = np.array([0.8, 0.7, 0.5])  # Arms A, B, C
N_ARMS = len(TRUE_MEANS)
BUDGET = 10_000
N_RUNS = 1000          # Monte-Carlo runs for averaging
OPTIMAL_MEAN = TRUE_MEANS.max()  # 0.8
ARM_LABELS = ["A", "B", "C"]

np.random.seed(42)


# ── Helper: pull an arm (Bernoulli reward) ─────────────────────────────────────
def pull(arm: int) -> int:
    return int(np.random.rand() < TRUE_MEANS[arm])


# ── 1. A/B Testing ────────────────────────────────────────────────────────────
def ab_testing(budget: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Phase 1 (first 2000): allocate equally among all 3 arms.
    Phase 2 (remaining 8000): exploit the arm with the highest observed mean.
    """
    rewards = np.zeros(budget)
    arms_chosen = np.zeros(budget, dtype=int)
    explore_budget = 2000
    per_arm = explore_budget // N_ARMS  # ~666 each

    counts = np.zeros(N_ARMS)
    totals = np.zeros(N_ARMS)
    t = 0

    # Exploration: round-robin
    for arm in range(N_ARMS):
        for _ in range(per_arm):
            r = pull(arm)
            rewards[t] = r
            arms_chosen[t] = arm
            totals[arm] += r
            counts[arm] += 1
            t += 1

    # Remaining exploration pulls (if 2000 not divisible by 3)
    while t < explore_budget:
        arm = t % N_ARMS
        r = pull(arm)
        rewards[t] = r
        arms_chosen[t] = arm
        totals[arm] += r
        counts[arm] += 1
        t += 1

    # Exploitation
    best_arm = np.argmax(totals / np.maximum(counts, 1))
    for _ in range(t, budget):
        rewards[t] = pull(best_arm)
        arms_chosen[t] = best_arm
        t += 1

    return rewards, arms_chosen


# ── 2. Optimistic Initial Values ──────────────────────────────────────────────
def optimistic_initial_values(budget: int, init_value: float = 1.0,
                               alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Start each arm's estimated value at `init_value` (optimistic).
    Use incremental sample-average update.  Always pick the arm with
    the highest current estimate (greedy w.r.t. optimistic values).
    """
    rewards = np.zeros(budget)
    arms_chosen = np.zeros(budget, dtype=int)
    Q = np.full(N_ARMS, init_value)  # optimistic initialisation
    counts = np.zeros(N_ARMS)

    for t in range(budget):
        arm = np.argmax(Q)
        r = pull(arm)
        rewards[t] = r
        arms_chosen[t] = arm
        counts[arm] += 1
        # Incremental mean update
        Q[arm] += (r - Q[arm]) / counts[arm]

    return rewards, arms_chosen


# ── 3. ε-Greedy ───────────────────────────────────────────────────────────────
def epsilon_greedy(budget: int, epsilon: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.zeros(budget)
    arms_chosen = np.zeros(budget, dtype=int)
    Q = np.zeros(N_ARMS)
    counts = np.zeros(N_ARMS)

    for t in range(budget):
        if np.random.rand() < epsilon:
            arm = np.random.randint(N_ARMS)
        else:
            arm = np.argmax(Q)
        r = pull(arm)
        rewards[t] = r
        arms_chosen[t] = arm
        counts[arm] += 1
        Q[arm] += (r - Q[arm]) / counts[arm]

    return rewards, arms_chosen


# ── 4. Softmax (Boltzmann) ────────────────────────────────────────────────────
def softmax_boltzmann(budget: int, tau: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Select arms with probability proportional to exp(Q/τ).
    τ controls exploration: small τ → more greedy.
    """
    rewards = np.zeros(budget)
    arms_chosen = np.zeros(budget, dtype=int)
    Q = np.zeros(N_ARMS)
    counts = np.zeros(N_ARMS)

    for t in range(budget):
        # Numerically stable softmax
        logits = Q / tau
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        arm = np.random.choice(N_ARMS, p=probs)

        r = pull(arm)
        rewards[t] = r
        arms_chosen[t] = arm
        counts[arm] += 1
        Q[arm] += (r - Q[arm]) / counts[arm]

    return rewards, arms_chosen


# ── 5. Upper Confidence Bound (UCB1) ─────────────────────────────────────────
def ucb1(budget: int, c: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.zeros(budget)
    arms_chosen = np.zeros(budget, dtype=int)
    Q = np.zeros(N_ARMS)
    counts = np.zeros(N_ARMS)

    # Pull each arm once
    for arm in range(N_ARMS):
        r = pull(arm)
        rewards[arm] = r
        arms_chosen[arm] = arm
        counts[arm] = 1
        Q[arm] = r

    for t in range(N_ARMS, budget):
        ucb_values = Q + c * np.sqrt(np.log(t) / counts)
        arm = np.argmax(ucb_values)
        r = pull(arm)
        rewards[t] = r
        arms_chosen[t] = arm
        counts[arm] += 1
        Q[arm] += (r - Q[arm]) / counts[arm]

    return rewards, arms_chosen


# ── 6. Thompson Sampling (Beta–Bernoulli) ─────────────────────────────────────
def thompson_sampling(budget: int) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.zeros(budget)
    arms_chosen = np.zeros(budget, dtype=int)
    alpha = np.ones(N_ARMS)  # successes + 1
    beta = np.ones(N_ARMS)   # failures + 1

    for t in range(budget):
        samples = np.random.beta(alpha, beta)
        arm = np.argmax(samples)
        r = pull(arm)
        rewards[t] = r
        arms_chosen[t] = arm
        alpha[arm] += r
        beta[arm] += (1 - r)

    return rewards, arms_chosen


# ── Run all strategies and collect cumulative regret ──────────────────────────
def run_all(n_runs: int = N_RUNS) -> dict:
    strategies = {
        "A/B Testing":              ab_testing,
        "Optimistic Initial Values": optimistic_initial_values,
        "ε-Greedy (ε=0.1)":        epsilon_greedy,
        "Softmax (τ=0.1)":         softmax_boltzmann,
        "UCB1":                     ucb1,
        "Thompson Sampling":        thompson_sampling,
    }

    cum_regret = {name: np.zeros(BUDGET) for name in strategies}

    for run in range(n_runs):
        if (run + 1) % 100 == 0:
            print(f"  Run {run + 1}/{n_runs}")
        for name, fn in strategies.items():
            rewards, arms_chosen = fn(BUDGET)
            # Per-step regret = μ* − μ_{chosen arm}  (deterministic, always ≥ 0)
            regret = OPTIMAL_MEAN - TRUE_MEANS[arms_chosen]
            cum_regret[name] += np.cumsum(regret)

    # Average over runs
    for name in cum_regret:
        cum_regret[name] /= n_runs

    return cum_regret


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_regret(cum_regret: dict, save_path: str = "regret_comparison.png"):
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6", "#1abc9c"]
    linestyles = ["-", "--", "-.", ":", "-", "--"]

    for (name, regret), color, ls in zip(cum_regret.items(), colors, linestyles):
        ax.plot(range(BUDGET), regret, label=name, color=color,
                linestyle=ls, linewidth=2)

    ax.set_yscale("log")
    ax.set_xlabel("Step (Budget Spent)", fontsize=14)
    ax.set_ylabel("Cumulative Regret (log scale)", fontsize=14)
    ax.set_title("Cumulative Regret Comparison of 6 Bandit Strategies\n"
                 "(Arms: A=0.8, B=0.7, C=0.5  |  Budget=10,000  |  "
                 f"Averaged over {N_RUNS} runs)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\n✅ Plot saved to: {save_path}")
    plt.close()


# ── Summary table ─────────────────────────────────────────────────────────────
def print_summary(cum_regret: dict):
    print("\n" + "=" * 75)
    print(f"{'Method':<30} {'Total Reward':>14} {'Total Regret':>14}")
    print("=" * 75)
    optimal_reward = OPTIMAL_MEAN * BUDGET  # 8000
    for name, regret in cum_regret.items():
        total_regret = regret[-1]
        total_reward = optimal_reward - total_regret
        print(f"{name:<30} {total_reward:>14.1f} {total_regret:>14.1f}")
    print("=" * 75)
    print(f"{'Optimal (always A)':<30} {optimal_reward:>14.1f} {'0.0':>14}")
    print("=" * 75)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running 6 bandit strategies comparison...")
    print(f"  Arms: {list(zip(ARM_LABELS, TRUE_MEANS))}")
    print(f"  Budget: {BUDGET:,}")
    print(f"  Runs: {N_RUNS:,}\n")

    cum_regret = run_all()
    plot_regret(cum_regret)
    print_summary(cum_regret)
