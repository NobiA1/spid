"""
Full fidelity metric computation for the SUM paper.
Computes:
  A. Market-level: RMSE, MAPE, skewness, direction accuracy,
     daily return distribution (mean/std/kurt/skew), Kolmogorov-Smirnov vs real
  B. Stylized facts: volatility clustering (ACF of |r|), leverage effect
     (corr(r_t, σ²_{t+h})), fat tails (excess kurtosis), return autocorrelation
  C. Micro-level (from retail_decisions.jsonl):
     turnover, holding period, active ratio (analogous to E22-E24),
     average position, HHI (E25-E26), momentum bias (E28), disposition (E33),
     γ-return and σ-return correlation (E36-E37)
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[3]  # repo root
ROOT = WORKSPACE / "data/fidelity/full_year_msci_a50_2025/runs"
SUM_DIR = ROOT / "sum_msci_a50_2025_full_N1000_seed42_full_decisions_20260502_120345"
RB_DIR  = ROOT / "rb_msci_a50_2025_week1_N1000_seed42_20260526_150632"
OUT     = WORKSPACE / "outputs/fidelity/fidelity_metrics_results.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_col(path, col):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            first_key = list(row.keys())[0]
            date_val = row.get("date", row.get(first_key, ""))
            if date_val == "init":
                continue
            rows.append(float(row[col]))
    return np.array(rows)


def returns(prices):
    return np.diff(prices) / prices[:-1]


def acf(x, lag):
    n = len(x)
    mu = x.mean()
    c0 = np.sum((x - mu) ** 2) / n
    cl = np.sum((x[:-lag] - mu) * (x[lag:] - mu)) / n
    return cl / c0 if c0 != 0 else 0.0


def volatility_clustering_score(ret, max_lag=10):
    """Mean ACF of |returns| over lags 1..max_lag (should be positive for clustering)."""
    abs_r = np.abs(ret)
    return np.mean([acf(abs_r, k) for k in range(1, max_lag + 1)])


def leverage_effect(ret, max_lag=5):
    """Corr(r_t, σ²_{t:t+h}) — should be negative for leverage effect."""
    vol2 = ret ** 2
    corrs = []
    for h in range(1, max_lag + 1):
        r_past = ret[:-h]
        v_future = vol2[h:]
        if len(r_past) < 10:
            continue
        corr = np.corrcoef(r_past, v_future)[0, 1]
        corrs.append(corr)
    return float(np.mean(corrs)) if corrs else 0.0


def hill_tail_index(returns_abs, frac=0.10):
    """Hill estimator for tail index (larger = thinner tails)."""
    x = np.sort(returns_abs)[::-1]
    k = max(2, int(frac * len(x)))
    if x[k - 1] <= 0:
        return float("nan")
    logs = np.log(x[:k]) - np.log(x[k - 1])
    return float(k / np.sum(logs))


def ks_stat(a, b):
    """KS statistic between two samples (manual)."""
    combined = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), combined, side="right") / len(a)
    cdf_b = np.searchsorted(np.sort(b), combined, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def rmse(sim, real):
    return float(np.sqrt(np.mean((sim - real) ** 2)))


def mape(sim, real):
    return float(np.mean(np.abs((sim - real) / real)) * 100)


def direction_accuracy(sim_ret, real_ret):
    """Fraction of days where sign(sim_ret) == sign(real_ret)."""
    correct = np.sign(sim_ret) == np.sign(real_ret)
    return float(correct.mean())


def pearson_corr(x, y):
    if len(x) < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    num = np.sum((x - mx) * (y - my))
    den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
    return float(num / den) if den != 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# A. Market-level metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_market_metrics(index_csv, label="SUM"):
    sim  = read_csv_col(index_csv, "equal_weight_sim")
    real = read_csv_col(index_csv, "equal_weight_real")
    n = min(len(sim), len(real))
    sim, real = sim[:n], real[:n]

    # Prices (add base 1000)
    sim_prices  = np.concatenate([[1000.0], sim])
    real_prices = np.concatenate([[1000.0], real])

    sim_ret  = returns(sim_prices)
    real_ret = returns(real_prices)

    metrics = {
        "label": label,
        "n_days": int(n),
        "final_sim_return_pct":  float((sim[-1] / 1000 - 1) * 100),
        "final_real_return_pct": float((real[-1] / 1000 - 1) * 100),
        "ew_rmse":  rmse(sim, real),
        "ew_mape":  mape(sim, real),
        # Return distribution
        "sim_mean_daily_ret":  float(sim_ret.mean() * 100),
        "real_mean_daily_ret": float(real_ret.mean() * 100),
        "sim_std_daily_ret":   float(sim_ret.std() * 100),
        "real_std_daily_ret":  float(real_ret.std() * 100),
        "sim_skewness":        float(_skewness(sim_ret)),
        "real_skewness":       float(_skewness(real_ret)),
        "sim_kurtosis":        float(_kurtosis(sim_ret)),
        "real_kurtosis":       float(_kurtosis(real_ret)),
        # Direction accuracy
        "direction_accuracy":  direction_accuracy(sim_ret, real_ret),
        # KS distance on daily returns
        "ks_daily_returns":    ks_stat(sim_ret, real_ret),
        # Volatility comparison
        "sim_annualized_vol":  float(sim_ret.std() * math.sqrt(243) * 100),
        "real_annualized_vol": float(real_ret.std() * math.sqrt(243) * 100),
        # Stylized facts
        "volatility_clustering_mean_acf_lag110": volatility_clustering_score(sim_ret),
        "lag1_abs_ret_acf_sim":   float(acf(np.abs(sim_ret), 1)),
        "lag1_abs_ret_acf_real":  float(acf(np.abs(real_ret), 1)),
        "lag1_ret_acf_sim":       float(acf(sim_ret, 1)),
        "lag1_ret_acf_real":      float(acf(real_ret, 1)),
        "leverage_effect":        leverage_effect(sim_ret),
        "leverage_effect_real":   leverage_effect(real_ret),
        "excess_kurtosis_sim":    float(_kurtosis(sim_ret) - 3),
        "excess_kurtosis_real":   float(_kurtosis(real_ret) - 3),
        "hill_tail_index_sim":    hill_tail_index(np.abs(sim_ret)),
        "hill_tail_index_real":   hill_tail_index(np.abs(real_ret)),
    }
    return metrics


def _skewness(x):
    n = len(x)
    mu, sigma = x.mean(), x.std()
    if sigma == 0:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def _kurtosis(x):
    n = len(x)
    mu, sigma = x.mean(), x.std()
    if sigma == 0:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4))


# ─────────────────────────────────────────────────────────────────────────────
# C. Micro-level metrics from retail_decisions.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl_parts(base_dir, prefix):
    """Load all .jsonl.partNNN files with a given prefix, yield records."""
    parts = sorted(base_dir.glob(f"{prefix}.part*"))
    for p in parts:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except Exception:
                        pass


def compute_micro_metrics(sim_dir):
    """
    Compute E22-E37 style micro metrics from retail_decisions.jsonl
    Returns dict of results.
    """
    # --- collect per-agent per-step data ---
    # agent_id -> list of {date, step, stock, direction, wealth, gamma_t, sigma, deal_amount}
    agent_records = defaultdict(list)

    for rec in load_jsonl_parts(sim_dir, "retail_decisions.jsonl"):
        aid  = rec.get("agent_id", "")
        style = rec.get("style", "unknown")
        date  = rec.get("date", "")
        step  = rec.get("step", 0)
        deal  = rec.get("deal_amount", 0.0) or 0.0
        gamma = rec.get("gamma_t", float("nan"))
        sigma = rec.get("sigma", float("nan"))
        wealth = rec.get("wealth", float("nan"))
        direction = rec.get("direction", "HOLD")
        stock = rec.get("stock", "")
        r_exec = rec.get("R_exec", float("nan"))
        r_hat  = rec.get("R_hat", float("nan"))

        agent_records[aid].append({
            "style": style,
            "date": date,
            "step": step,
            "stock": stock,
            "direction": direction,
            "wealth": wealth,
            "deal_amount": deal,
            "gamma_t": gamma,
            "sigma": sigma,
            "R_exec": r_exec,
            "R_hat": r_hat,
        })

    # --- per-agent aggregate stats ---
    agent_stats = {}
    for aid, recs in agent_records.items():
        style = recs[0]["style"]
        total_steps = len(set(r["step"] for r in recs))
        active_steps = len([r for r in recs if r["direction"] != "HOLD"])

        # turnover: total deal_amount / mean_wealth
        total_deal = sum(r["deal_amount"] for r in recs if r["deal_amount"])
        wealths = [r["wealth"] for r in recs if r["wealth"] and not math.isnan(r["wealth"])]
        mean_wealth = float(np.mean(wealths)) if wealths else float("nan")
        turnover = total_deal / mean_wealth if mean_wealth > 0 else 0.0

        # holding period: approximate by intervals between direction-changes
        holding_periods = _estimate_holding_period(recs)

        # active ratio
        active_ratio = active_steps / total_steps if total_steps > 0 else 0.0

        # gamma (use mean of all recorded gamma_t)
        gammas = [r["gamma_t"] for r in recs if not math.isnan(r.get("gamma_t", float("nan")))]
        mean_gamma = float(np.mean(gammas)) if gammas else float("nan")

        # sigma
        sigmas = [r["sigma"] for r in recs if not math.isnan(r.get("sigma", float("nan")))]
        mean_sigma = float(np.mean(sigmas)) if sigmas else float("nan")

        # wealth return: use first/last wealth
        first_w = next((r["wealth"] for r in recs if r["wealth"] and not math.isnan(r["wealth"])), float("nan"))
        last_w  = wealths[-1] if wealths else float("nan")
        wealth_return = (last_w / first_w - 1) * 100 if (first_w and not math.isnan(first_w)) else float("nan")

        # momentum bias: corr(direction_numeric, past R_hat)
        momentum_bias = _compute_momentum_bias(recs)

        # disposition effect: PGR - PLR
        disposition = _compute_disposition(recs)

        agent_stats[aid] = {
            "style": style,
            "turnover": turnover,
            "mean_holding_period": float(np.mean(holding_periods)) if holding_periods else float("nan"),
            "active_ratio": active_ratio,
            "mean_gamma": mean_gamma,
            "mean_sigma": mean_sigma,
            "wealth_return_pct": wealth_return,
            "momentum_bias": momentum_bias,
            "disposition": disposition,
            "n_steps": total_steps,
            "n_active_steps": active_steps,
        }

    # --- aggregate by style ---
    styles = ["value", "momentum", "noise", "institution"]
    by_style = {}
    for st in styles:
        agents = [v for v in agent_stats.values() if v["style"] == st]
        if not agents:
            continue
        def _agg(key):
            vals = [a[key] for a in agents if not math.isnan(a.get(key, float("nan")))]
            return {
                "count": len(vals),
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "median": float(np.median(vals)) if vals else float("nan"),
                "std": float(np.std(vals)) if vals else float("nan"),
            }
        by_style[st] = {
            "turnover": _agg("turnover"),
            "holding_period": _agg("mean_holding_period"),
            "active_ratio": _agg("active_ratio"),
            "wealth_return_pct": _agg("wealth_return_pct"),
            "momentum_bias": _agg("momentum_bias"),
            "disposition": _agg("disposition"),
            "mean_gamma": _agg("mean_gamma"),
            "mean_sigma": _agg("mean_sigma"),
        }

    # --- gamma/sigma vs return correlations (E36-E37 analog) ---
    all_gamma = np.array([v["mean_gamma"] for v in agent_stats.values()
                          if not math.isnan(v.get("mean_gamma", float("nan")))])
    all_sigma = np.array([v["mean_sigma"] for v in agent_stats.values()
                          if not math.isnan(v.get("mean_sigma", float("nan")))])
    all_ret   = np.array([v["wealth_return_pct"] for v in agent_stats.values()
                          if not math.isnan(v.get("wealth_return_pct", float("nan")))])

    # align lengths
    n_gamma = min(len(all_gamma), len(all_ret))
    n_sigma = min(len(all_sigma), len(all_ret))

    gamma_ret_corr = pearson_corr(all_gamma[:n_gamma], all_ret[:n_gamma])
    sigma_ret_corr = pearson_corr(all_sigma[:n_sigma], all_ret[:n_sigma])

    # quintile analysis for gamma
    gamma_quintiles = _quintile_analysis(
        [v["mean_gamma"] for v in agent_stats.values()],
        [v["wealth_return_pct"] for v in agent_stats.values()]
    )
    sigma_quintiles = _quintile_analysis(
        [v["mean_sigma"] for v in agent_stats.values()],
        [v["wealth_return_pct"] for v in agent_stats.values()]
    )

    return {
        "n_agents": len(agent_stats),
        "by_style": by_style,
        "gamma_return_corr_overall": gamma_ret_corr,
        "sigma_return_corr_overall": sigma_ret_corr,
        "gamma_quintiles": gamma_quintiles,
        "sigma_quintiles": sigma_quintiles,
    }


def _estimate_holding_period(recs):
    """Estimate holding periods from buy→sell sequences."""
    # Sort by step
    sorted_recs = sorted(recs, key=lambda r: r["step"])
    periods = []
    open_positions = {}  # stock -> step when opened
    for r in sorted_recs:
        s = r["stock"]
        d = r["direction"]
        step = r["step"]
        if d == "BUY":
            open_positions[s] = step
        elif d == "SELL" and s in open_positions:
            periods.append(step - open_positions[s])
            del open_positions[s]
    # Close any remaining positions at end
    last_step = sorted_recs[-1]["step"] if sorted_recs else 0
    for s, open_step in open_positions.items():
        periods.append(last_step - open_step)
    return periods


def _compute_momentum_bias(recs):
    """
    Corr(buy_probability, past R_hat) — momentum bias.
    Returns positive if agents chase past returns.
    """
    buy_flag  = []
    r_hat_vals = []
    for r in recs:
        rh = r.get("R_hat", float("nan"))
        if math.isnan(rh):
            continue
        direction = r["direction"]
        if direction == "BUY":
            buy_flag.append(1.0)
        elif direction == "SELL":
            buy_flag.append(-1.0)
        else:
            buy_flag.append(0.0)
        r_hat_vals.append(rh)
    if len(buy_flag) < 3:
        return float("nan")
    return pearson_corr(np.array(r_hat_vals), np.array(buy_flag))


def _compute_disposition(recs):
    """
    Simplified disposition effect: mean(realized_gain_sell) - mean(realized_gain_hold_on_loss).
    We approximate using R_exec on SELL orders: PGR = fraction of sells with positive R_exec.
    Returns PGR - 0.5 as a proxy.
    """
    sell_recs = [r for r in recs if r["direction"] == "SELL"]
    if not sell_recs:
        return float("nan")
    gains = [r["R_exec"] for r in sell_recs
             if r["R_exec"] is not None and not math.isnan(r.get("R_exec", float("nan")))]
    if not gains:
        return float("nan")
    pgr = sum(1 for g in gains if g > 0) / len(gains)
    return pgr - 0.5  # positive → disposition effect (too eager to realize gains)


def _quintile_analysis(param_vals, return_vals):
    """Sort by param, split into 5 quintiles, compute mean return per quintile."""
    pairs = [(p, r) for p, r in zip(param_vals, return_vals)
             if not math.isnan(p) and not math.isnan(r)]
    if len(pairs) < 5:
        return []
    pairs.sort(key=lambda x: x[0])
    n = len(pairs)
    q_size = n // 5
    result = []
    for i in range(5):
        start = i * q_size
        end = (i + 1) * q_size if i < 4 else n
        q_pairs = pairs[start:end]
        q_params = [p[0] for p in q_pairs]
        q_returns = [p[1] for p in q_pairs]
        result.append({
            "quintile": i + 1,
            "param_range": f"{min(q_params):.4f} - {max(q_params):.4f}",
            "param_mean": float(np.mean(q_params)),
            "return_mean": float(np.mean(q_returns)),
            "return_std": float(np.std(q_returns)),
            "count": len(q_pairs),
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    results = {}

    print("=== A. Market-level metrics (SUM full-year) ===")
    sum_market = compute_market_metrics(SUM_DIR / "index_series.csv", "SUM_full_year")
    results["market_sum"] = sum_market
    for k, v in sum_market.items():
        print(f"  {k}: {v}")

    print("\n=== A. Market-level metrics (RB week-1) ===")
    rb_market = compute_market_metrics(RB_DIR / "index_series.csv", "RB_week1")
    results["market_rb"] = rb_market
    for k, v in rb_market.items():
        print(f"  {k}: {v}")

    print("\n=== C. Micro-level metrics (SUM full-year, from retail_decisions.jsonl) ===")
    micro = compute_micro_metrics(SUM_DIR)
    results["micro_sum"] = micro

    print("  By-style turnover / holding / active_ratio:")
    for st, data in micro["by_style"].items():
        t = data["turnover"]["mean"]
        h = data["holding_period"]["mean"]
        a = data["active_ratio"]["mean"]
        mb = data["momentum_bias"]["mean"]
        d  = data["disposition"]["mean"]
        wr = data["wealth_return_pct"]["mean"]
        print(f"    {st:12s}: turnover={t:.3f}, holding={h:.1f}d, active={a:.3f}, "
              f"mom_bias={mb:.4f}, disp={d:.4f}, return={wr:.2f}%")

    print(f"\n  γ-return corr:  {micro['gamma_return_corr_overall']:.4f}")
    print(f"  σ-return corr:  {micro['sigma_return_corr_overall']:.4f}")

    print("\n  γ quintile analysis:")
    for q in micro["gamma_quintiles"]:
        print(f"    Q{q['quintile']} γ∈[{q['param_range']}]: return={q['return_mean']:.2f}%")

    print("\n  σ quintile analysis:")
    for q in micro["sigma_quintiles"]:
        print(f"    Q{q['quintile']} σ∈[{q['param_range']}]: return={q['return_mean']:.2f}%")

    # Save
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
