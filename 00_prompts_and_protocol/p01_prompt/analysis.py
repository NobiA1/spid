#!/usr/bin/env python3
"""
P01: Analysis & Figure Generation.

Computes all metrics (Spearman rho, Kendall tau, adjacent inversion rate, JSD)
and generates Figures P01-1 through P01-4.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[3]
RESULTS_DIR = ROOT_DIR / "data" / "controllability" / "P01" / "results" / "main"
RAW_DIR = RESULTS_DIR / "raw_responses"
FIG_DIR = RESULTS_DIR / "figures"

sys.path.insert(0, str(BASE_DIR))
from config import ACTION_SPACE

# ── Helpers ───────────────────────────────────────────────────────────────────

def action_distribution(actions: list[int], epsilon: float = 1e-8) -> np.ndarray:
    """Empirical distribution over ACTION_SPACE with additive smoothing."""
    counts = np.array([actions.count(a) for a in ACTION_SPACE], dtype=float)
    counts += epsilon
    return counts / counts.sum()


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    return float(jensenshannon(p, q) ** 2)


def adjacent_inversion_rate(level_means: dict, strict: bool = True) -> float:
    """Fraction of adjacent pairs with reversal.

    strict=True counts only M_{l+1} < M_l (true reversals).
    strict=False also counts ties M_{l+1} = M_l.
    """
    levels = sorted(level_means.keys())
    if strict:
        inversions = sum(
            1 for i in range(len(levels) - 1)
            if level_means[levels[i + 1]] < level_means[levels[i]]
        )
    else:
        inversions = sum(
            1 for i in range(len(levels) - 1)
            if level_means[levels[i + 1]] <= level_means[levels[i]]
        )
    return inversions / (len(levels) - 1)


def adjacent_tie_rate(level_means: dict) -> float:
    """Fraction of adjacent pairs where M_{l+1} == M_l."""
    levels = sorted(level_means.keys())
    ties = sum(
        1 for i in range(len(levels) - 1)
        if level_means[levels[i + 1]] == level_means[levels[i]]
    )
    return ties / (len(levels) - 1)


# ── MFT Analysis ─────────────────────────────────────────────────────────────

def analyze_mft(data: list) -> dict:
    results = {}
    for scenario_id in ["mft_bullish_01", "mft_bearish_01"]:
        subset = [r for r in data if r["scenario_id"] == scenario_id and r.get("action") is not None]
        by_agent = {}
        for r in subset:
            by_agent.setdefault(r["agent_id"], []).append(r["action"])
        agent_means = {aid: np.mean(acts) for aid, acts in by_agent.items()}
        direction = "bullish" if "bullish" in scenario_id else "bearish"
        if direction == "bullish":
            pass_count = sum(1 for m in agent_means.values() if m > 0)
        else:
            pass_count = sum(1 for m in agent_means.values() if m < 0)
        results[scenario_id] = {
            "agent_means": {k: float(v) for k, v in agent_means.items()},
            "pass_count": pass_count,
            "total_agents": len(agent_means),
            "all_pass": pass_count == len(agent_means),
        }
    return results


# ── DIR-A Analysis ────────────────────────────────────────────────────────────

def analyze_dir_a(data: list) -> dict:
    valid = [r for r in data if r.get("action") is not None]

    by_agent = {}
    for r in valid:
        by_agent.setdefault(r["agent_id"], []).append(r["action"])

    agent_means = {aid: float(np.mean(acts)) for aid, acts in by_agent.items()}

    agent_levels = {}
    for r in valid:
        agent_levels[r["agent_id"]] = r["level"]

    level_means = {}
    for level in range(10):
        agents_at_level = [aid for aid, lv in agent_levels.items() if lv == level]
        if agents_at_level:
            level_means[level] = float(np.mean([agent_means[a] for a in agents_at_level]))

    levels = sorted(level_means.keys())
    means = [level_means[l] for l in levels]

    rho, rho_p = stats.spearmanr(levels, means)
    tau, tau_p = stats.kendalltau(levels, means)
    inv_rate = adjacent_inversion_rate(level_means)

    agent_distributions = {}
    for aid, acts in by_agent.items():
        agent_distributions[aid] = action_distribution(acts)

    return {
        "level_means": level_means,
        "agent_means": agent_means,
        "agent_levels": agent_levels,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "adjacent_inversion_rate": float(inv_rate),
        "agent_distributions": {k: v.tolist() for k, v in agent_distributions.items()},
        "by_agent_actions": {k: v for k, v in by_agent.items()},
    }


# ── INV Analysis ──────────────────────────────────────────────────────────────

def analyze_inv(dir_a: dict) -> dict:
    distributions = {k: np.array(v) for k, v in dir_a["agent_distributions"].items()}
    agent_levels = dir_a["agent_levels"]
    by_agent = dir_a["by_agent_actions"]

    # Split-half JSD for tau_beh
    split_jsds = []
    for aid, acts in by_agent.items():
        if len(acts) < 10:
            continue
        acts_arr = np.array(acts)
        for _ in range(100):
            np.random.shuffle(acts_arr)
            half = len(acts_arr) // 2
            p1 = action_distribution(acts_arr[:half].tolist())
            p2 = action_distribution(acts_arr[half:2 * half].tolist())
            split_jsds.append(jsd(p1, p2))

    tau_beh = float(np.percentile(split_jsds, 95)) if split_jsds else 0.0

    agent_ids = sorted(distributions.keys())
    within, adjacent, far = [], [], []
    for i, j in combinations(range(len(agent_ids)), 2):
        ai, aj = agent_ids[i], agent_ids[j]
        li, lj = agent_levels[ai], agent_levels[aj]
        d = jsd(distributions[ai], distributions[aj])
        diff = abs(li - lj)
        if diff == 0:
            within.append(d)
        elif diff == 1:
            adjacent.append(d)
        elif diff >= 3:
            far.append(d)

    within_violation = sum(1 for d in within if d > tau_beh) / max(len(within), 1)

    return {
        "tau_beh": tau_beh,
        "within_level_jsd_median": float(np.median(within)) if within else None,
        "adjacent_level_jsd_median": float(np.median(adjacent)) if adjacent else None,
        "far_level_jsd_median": float(np.median(far)) if far else None,
        "within_level_violation_rate": float(within_violation),
        "within_jsds": within,
        "adjacent_jsds": adjacent,
        "far_jsds": far,
        "split_half_jsds": split_jsds,
    }


# ── DIR-B Analysis ────────────────────────────────────────────────────────────

def analyze_dir_b(data: list) -> dict:
    valid = [r for r in data if r.get("action") is not None]

    by_rt = {}
    for r in valid:
        rt = round(r["risk_tolerance"], 1)
        by_rt.setdefault(rt, []).append(r["action"])

    rt_means = {rt: float(np.mean(acts)) for rt, acts in sorted(by_rt.items())}
    rts = sorted(rt_means.keys())
    means = [rt_means[r] for r in rts]

    rho, rho_p = stats.spearmanr(rts, means)
    tau, tau_p = stats.kendalltau(rts, means)

    level_means_int = {int(round(rt * 10)): m for rt, m in rt_means.items()}
    inv_rate = adjacent_inversion_rate(level_means_int)

    return {
        "rt_means": rt_means,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "adjacent_inversion_rate": float(inv_rate),
    }


# ── DIR-C Analysis ────────────────────────────────────────────────────────────

def analyze_dir_c(data: dict) -> dict:
    actions_dict = data["actions"]
    levels = sorted(int(k) for k in actions_dict.keys())
    actions = [actions_dict[str(l)]["a_star"] for l in levels]

    if len(set(actions)) <= 1:
        return {
            "actions": {l: a for l, a in zip(levels, actions)},
            "spearman_rho": float("nan"),
            "kendall_tau": float("nan"),
            "adjacent_inversion_rate": 0.0,
            "adjacent_tie_rate": 1.0,
            "unique_bins": len(set(actions)),
            "note": "All actions identical; rho/tau undefined.",
        }

    rho, rho_p = stats.spearmanr(levels, actions)
    tau, tau_p = stats.kendalltau(levels, actions)
    level_means = {l: float(a) for l, a in zip(levels, actions)}
    inv_rate = adjacent_inversion_rate(level_means, strict=True)
    tie_rate = adjacent_tie_rate(level_means)

    return {
        "actions": {l: a for l, a in zip(levels, actions)},
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "adjacent_inversion_rate": float(inv_rate),
        "adjacent_tie_rate": float(tie_rate),
        "unique_bins": len(set(actions)),
        "alpha_scale": data.get("alpha_scale"),
    }


# ── Semantic-Behavioral Scatter ───────────────────────────────────────────────

def analyze_semantic_behavioral(dir_a: dict, embeddings: dict) -> dict:
    agent_ids = sorted(set(dir_a["agent_distributions"].keys()) & set(embeddings.keys()))
    if len(agent_ids) < 2:
        return {"note": "Insufficient embeddings for scatter analysis."}

    emb_matrix = np.array([embeddings[aid] for aid in agent_ids])
    dist_matrix = np.array([dir_a["agent_distributions"][aid] for aid in agent_ids])

    sem_dists, beh_dists = [], []
    for i, j in combinations(range(len(agent_ids)), 2):
        cos_sim = np.dot(emb_matrix[i], emb_matrix[j]) / (
            np.linalg.norm(emb_matrix[i]) * np.linalg.norm(emb_matrix[j]) + 1e-12
        )
        sem_dists.append(1.0 - cos_sim)
        beh_dists.append(jsd(dist_matrix[i], dist_matrix[j]))

    pearson_r, pearson_p = stats.pearsonr(sem_dists, beh_dists)
    spearman_rho, spearman_p = stats.spearmanr(sem_dists, beh_dists)

    return {
        "sem_dists": sem_dists,
        "beh_dists": beh_dists,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "n_pairs": len(sem_dists),
    }


# ── Figure Generation ─────────────────────────────────────────────────────────

def generate_figures(dir_a: dict, inv: dict, dir_b: dict, dir_c: dict,
                     scatter: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import blended_transform_factory

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "figure.dpi": 300})

    # ── Figure P01-1: NL distributional gradient ──────────────────────────
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    all_data, positions = [], []
    level_means = dir_a["level_means"]
    by_agent = dir_a["by_agent_actions"]
    agent_levels = dir_a["agent_levels"]

    for level in range(10):
        agents_at_level = [aid for aid, lv in agent_levels.items() if lv == level]
        all_actions = []
        for aid in agents_at_level:
            all_actions.extend(by_agent.get(aid, []))
        all_data.append(all_actions)
        positions.append(level)

    parts = ax.violinplot(all_data, positions=positions, showmeans=False,
                          showmedians=False, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("#7fbfff")
        pc.set_alpha(0.6)

    means_x = sorted(level_means.keys())
    means_y = [level_means[l] for l in means_x]
    ax.plot(means_x, means_y, "o-", color="#d62728", linewidth=1.5, markersize=3,
            label="Level Mean", zorder=5)

    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(increasing=True)
        iso_y = iso.fit_transform(means_x, means_y)
        ax.plot(means_x, iso_y, "--", color="#d62728", alpha=0.5,
                label="Isotonic Fit")
    except ImportError:
        pass

    ax.set_xlabel("Intended Risk Level")
    ax.set_ylabel("Target Position Change (%)")
    ax.set_ylim(-110, 110)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "P01_1_nl_gradient.png")
    plt.close(fig)

    # ── Figure P01-2: Three-condition comparison ──────────────────────────
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    nl_x = sorted(level_means.keys())
    nl_y = [level_means[l] for l in nl_x]
    ax.plot(nl_x, nl_y, "o-", color="gray", linewidth=1.5, markersize=3,
            label="NL")

    num_x = sorted(float(k) for k in dir_b["rt_means"].keys())
    num_y = [dir_b["rt_means"][k] for k in sorted(dir_b["rt_means"].keys(), key=float)]
    num_x_scaled = [x * 10 for x in num_x]
    ax.plot(num_x_scaled, num_y, "s-", color="#1f77b4", linewidth=1.5, markersize=3,
            label="Numeric")

    util_x = sorted(int(k) for k in dir_c["actions"].keys())
    util_y = [dir_c["actions"][l] for l in util_x]
    ax.plot(util_x, util_y, "D-", color="black", linewidth=1.5, markersize=3,
            label="Utility")

    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Mean Action")
    ax.set_ylim(-110, 110)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "P01_2_three_condition.png")
    plt.close(fig)

    # ── Figure P01-3: INV JSD boxplot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    box_data = [inv["within_jsds"], inv["adjacent_jsds"], inv["far_jsds"],
                inv["split_half_jsds"][:500]]
    labels = ["Within", "Adjacent", "Far (≥3)", "Bootstrap"]
    box_data_clean = []
    labels_clean = []
    for d, l in zip(box_data, labels):
        if d:
            box_data_clean.append(d)
            labels_clean.append(l)

    if box_data_clean:
        n_box = len(box_data_clean)
        x_spread = 1.42
        x_pos = [1.0 + i * x_spread for i in range(n_box)]
        bp = ax.boxplot(
            box_data_clean,
            positions=x_pos,
            tick_labels=labels_clean,
            widths=min(0.62, 0.75 * x_spread),
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=3),
        )
        ax.set_xlim(x_pos[0] - 0.58, x_pos[-1] + 0.58)
        colors = ["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f"]
        for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
            patch.set_facecolor(color)
        if inv["tau_beh"] > 0:
            tau_y = inv["tau_beh"]
            ax.axhline(tau_y, color="red", linestyle="--", alpha=0.7)
            trans_xy = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(
                0.98,
                tau_y,
                f"{tau_y:.4f}",
                transform=trans_xy,
                va="center",
                ha="right",
                fontsize=6,
                color="red",
                clip_on=True,
            )

    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylabel("Action Distribution", labelpad=1)
    plt.setp(
        ax.get_xticklabels(),
        rotation=42,
        ha="right",
        rotation_mode="anchor",
    )
    fig.tight_layout(rect=(0.038, 0.24, 0.99, 0.97))
    fig.savefig(FIG_DIR / "P01_3_inv_jsd.png")
    plt.close(fig)

    # ── Figure P01-4: Semantic vs Behavioral scatter ──────────────────────
    if scatter.get("sem_dists"):
        fig, ax = plt.subplots(figsize=(3.3, 3.0))
        ax.scatter(scatter["sem_dists"], scatter["beh_dists"],
                   s=4, alpha=0.15, color="#333333")
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            xy = lowess(scatter["beh_dists"], scatter["sem_dists"], frac=0.3)
            ax.plot(xy[:, 0], xy[:, 1], color="black", linewidth=2)
        except ImportError:
            z = np.polyfit(scatter["sem_dists"], scatter["beh_dists"], 3)
            p = np.poly1d(z)
            xs = np.linspace(min(scatter["sem_dists"]), max(scatter["sem_dists"]), 200)
            ax.plot(xs, p(xs), color="black", linewidth=2)

        ax.set_xlabel("Semantic Distance")
        ax.set_ylabel("Action Distribution")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "P01_4_sem_beh_scatter.png")
        plt.close(fig)


# ── Main Analysis Entry ──────────────────────────────────────────────────────

def run_analysis():
    print("  Loading raw results …")

    with open(RAW_DIR / "mft_responses.json") as f:
        mft_data = json.load(f)
    with open(RAW_DIR / "dir_a_nl.json") as f:
        dir_a_data = json.load(f)
    with open(RAW_DIR / "dir_b_numeric.json") as f:
        dir_b_data = json.load(f)
    with open(RAW_DIR / "dir_c_utility.json") as f:
        dir_c_data = json.load(f)

    emb_file = RESULTS_DIR / "embeddings" / "attribute_embeddings.json"
    embeddings = {}
    if emb_file.exists():
        with open(emb_file) as f:
            embeddings = json.load(f)

    print("  Analyzing MFT …")
    mft = analyze_mft(mft_data)

    print("  Analyzing DIR-A (NL) …")
    dir_a = analyze_dir_a(dir_a_data)

    print("  Analyzing INV …")
    inv = analyze_inv(dir_a)

    print("  Analyzing DIR-B (Numeric) …")
    dir_b = analyze_dir_b(dir_b_data)

    print("  Analyzing DIR-C (Utility) …")
    dir_c = analyze_dir_c(dir_c_data)

    print("  Analyzing Semantic-Behavioral scatter …")
    scatter = analyze_semantic_behavioral(dir_a, embeddings)

    # Summary
    metrics = {
        "mft": mft,
        "dir_a_nl": {
            "level_means": dir_a["level_means"],
            "spearman_rho": dir_a["spearman_rho"],
            "kendall_tau": dir_a["kendall_tau"],
            "adjacent_inversion_rate": dir_a["adjacent_inversion_rate"],
        },
        "inv": {
            "tau_beh": inv["tau_beh"],
            "within_level_jsd_median": inv["within_level_jsd_median"],
            "adjacent_level_jsd_median": inv["adjacent_level_jsd_median"],
            "far_level_jsd_median": inv["far_level_jsd_median"],
            "within_level_violation_rate": inv["within_level_violation_rate"],
        },
        "dir_b_numeric": {
            "rt_means": dir_b["rt_means"],
            "spearman_rho": dir_b["spearman_rho"],
            "kendall_tau": dir_b["kendall_tau"],
            "adjacent_inversion_rate": dir_b["adjacent_inversion_rate"],
        },
        "dir_c_utility": dir_c,
        "semantic_behavioral": {
            "pearson_r": scatter.get("pearson_r"),
            "spearman_rho": scatter.get("spearman_rho"),
            "n_pairs": scatter.get("n_pairs"),
        },
    }

    with open(RESULTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n  ── P01 Metrics Summary ──")
    print(f"  MFT Bullish pass: {mft.get('mft_bullish_01', {}).get('all_pass')}")
    print(f"  MFT Bearish pass: {mft.get('mft_bearish_01', {}).get('all_pass')}")
    print(f"  DIR-A NL:   ρ = {dir_a['spearman_rho']:.4f}, τ = {dir_a['kendall_tau']:.4f}, "
          f"inv = {dir_a['adjacent_inversion_rate']:.3f}")
    print(f"  DIR-B Num:  ρ = {dir_b['spearman_rho']:.4f}, τ = {dir_b['kendall_tau']:.4f}, "
          f"inv = {dir_b['adjacent_inversion_rate']:.3f}")
    rho_c = dir_c.get("spearman_rho", float("nan"))
    tau_c = dir_c.get("kendall_tau", float("nan"))
    inv_c = dir_c.get("adjacent_inversion_rate", 0.0)
    tie_c = dir_c.get("adjacent_tie_rate", 0.0)
    print(f"  DIR-C Util: ρ = {rho_c:.4f}, τ = {tau_c:.4f}, inv = {inv_c:.3f}, ties = {tie_c:.3f}")
    print(f"  INV:        within_viol = {inv['within_level_violation_rate']:.3f}, "
          f"τ_beh = {inv['tau_beh']:.5f}")
    if scatter.get("pearson_r") is not None:
        print(f"  Scatter:    Pearson r = {scatter['pearson_r']:.4f}, "
              f"Spearman ρ = {scatter['spearman_rho']:.4f}")

    print("\n  Generating figures …")
    generate_figures(dir_a, inv, dir_b, dir_c, scatter)
    print(f"  Figures saved to {FIG_DIR}")
    print("  Analysis complete.")


if __name__ == "__main__":
    run_analysis()
