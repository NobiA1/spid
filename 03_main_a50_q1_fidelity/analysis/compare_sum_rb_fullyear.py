"""
Full-year SUM vs RB comparison on market-level and investor-level metrics.
Data sources:
  SUM: data/fidelity/full_year_msci_a50_2025/runs/sum_msci_a50_2025_full_...
  RB:  data/fidelity/rule_based_standalone/rb_msci_a50_2025_full_N1000_seed42_...
"""

import csv, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[3]  # repo root
ROOT = WORKSPACE / "data/fidelity"
SUM_DIR = ROOT / "full_year_msci_a50_2025/runs/sum_msci_a50_2025_full_N1000_seed42_full_decisions_20260502_120345"
RB_DIR  = ROOT / "rule_based_standalone/rb_msci_a50_2025_full_N1000_seed42_20260526_160935"

# ─── helpers ───────────────────────────────────────────────────────────────

def load_index(csv_path):
    """Returns (dates, ew_sim, ew_real, fw_sim, fw_real) as arrays."""
    dates, ews, ewr, fws, fwr = [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = list(row.values())[0]
            if d == "init":
                continue
            dates.append(d)
            ews.append(float(row["equal_weight_sim"]))
            ewr.append(float(row["equal_weight_real"]))
            fws.append(float(row.get("float_weight_sim", row["equal_weight_sim"])))
            fwr.append(float(row.get("float_weight_real", row["equal_weight_real"])))
    return dates, np.array(ews), np.array(ewr), np.array(fws), np.array(fwr)

def ret(prices):
    return np.diff(prices) / prices[:-1]

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def mape(sim, real):
    return float(np.mean(np.abs((sim - real) / (real + 1e-9))) * 100)

def corr(x, y):
    if len(x) < 3: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def acf_lag1(x):
    mu = x.mean(); c0 = np.sum((x - mu)**2)
    if c0 == 0: return 0.0
    return float(np.sum((x[:-1]-mu)*(x[1:]-mu)) / c0)

def mean_acf(x, max_lag=10):
    return float(np.mean([acf_lag1_lag(x, k) for k in range(1, max_lag+1)]))

def acf_lag1_lag(x, lag):
    mu = x.mean(); c0 = np.sum((x - mu)**2)
    if c0 == 0: return 0.0
    return float(np.sum((x[:-lag]-mu)*(x[lag:]-mu)) / c0)

def skew(x):
    mu, s = x.mean(), x.std()
    return float(np.mean(((x-mu)/s)**3)) if s > 0 else 0.0

def kurt(x):
    mu, s = x.mean(), x.std()
    return float(np.mean(((x-mu)/s)**4)) if s > 0 else 0.0

def direction_acc(sr, rr):
    return float((np.sign(sr) == np.sign(rr)).mean())

def leverage_effect(r, max_lag=5):
    vol2 = r**2
    cs = []
    for h in range(1, max_lag+1):
        rp, vf = r[:-h], vol2[h:]
        if len(rp) < 5: continue
        cs.append(corr(rp, vf))
    return float(np.nanmean(cs)) if cs else float("nan")

def ks_stat(a, b):
    combined = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(np.sort(a), combined, side="right") / len(a)
    cb = np.searchsorted(np.sort(b), combined, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))

def hill_index(abs_r, frac=0.10):
    x = np.sort(abs_r)[::-1]
    k = max(2, int(frac*len(x)))
    if x[k-1] <= 0: return float("nan")
    return float(k / np.sum(np.log(x[:k]) - np.log(x[k-1])))

# ─── MARKET-LEVEL ──────────────────────────────────────────────────────────

def market_metrics(label, csv_path):
    dates, ews, ewr, fws, fwr = load_index(csv_path)
    n = len(ews)

    # prepend base 1000 for return calculation
    ews_full = np.concatenate([[1000.], ews])
    ewr_full = np.concatenate([[1000.], ewr])
    fws_full = np.concatenate([[1000.], fws])
    fwr_full = np.concatenate([[1000.], fwr])

    sr = ret(ews_full);  rr = ret(ewr_full)
    fsr = ret(fws_full); frr = ret(fwr_full)

    # overlap (in case different length)
    n_r = min(len(sr), len(rr))
    sr, rr = sr[:n_r], rr[:n_r]
    ews_cmp = ews[:n_r]; ewr_cmp = ewr[:n_r]

    m = {
        "label":        label,
        "n_days":       n,
        # cumulative returns
        "sim_cum_ret":  float((ews[-1]/1000-1)*100),
        "real_cum_ret": float((ewr[-1]/1000-1)*100),
        "fw_sim_cum_ret":  float((fws[-1]/1000-1)*100),
        "fw_real_cum_ret": float((fwr[-1]/1000-1)*100),
        # trajectory alignment
        "ew_rmse":      rmse(ews_cmp, ewr_cmp),
        "ew_mape":      mape(ews_cmp, ewr_cmp),
        "fw_rmse":      rmse(fws[:n_r], fwr[:n_r]),
        "fw_mape":      mape(fws[:n_r], fwr[:n_r]),
        # daily return distribution
        "mean_daily_ret_sim":  float(sr.mean()*100),
        "mean_daily_ret_real": float(rr.mean()*100),
        "std_daily_ret_sim":   float(sr.std()*100),
        "std_daily_ret_real":  float(rr.std()*100),
        "skew_sim":  skew(sr),
        "skew_real": skew(rr),
        "kurt_sim":  kurt(sr),
        "kurt_real": kurt(rr),
        "excess_kurt_sim":  kurt(sr) - 3,
        "excess_kurt_real": kurt(rr) - 3,
        "annualized_vol_sim":  float(sr.std()*math.sqrt(243)*100),
        "annualized_vol_real": float(rr.std()*math.sqrt(243)*100),
        # direction accuracy
        "direction_accuracy": direction_acc(sr, rr),
        # KS distance
        "ks_daily_ret": ks_stat(sr, rr),
        # stylized facts
        "lag1_acf_absr_sim":  acf_lag1(np.abs(sr)),
        "lag1_acf_absr_real": acf_lag1(np.abs(rr)),
        "lag1_acf_r_sim":     acf_lag1(sr),
        "lag1_acf_r_real":    acf_lag1(rr),
        "mean_acf_absr_sim":  mean_acf(np.abs(sr)),
        "mean_acf_absr_real": mean_acf(np.abs(rr)),
        "leverage_sim":  leverage_effect(sr),
        "leverage_real": leverage_effect(rr),
        "hill_tail_sim":  hill_index(np.abs(sr)),
        "hill_tail_real": hill_index(np.abs(rr)),
        # correlation sim vs real returns
        "sim_real_ret_corr": corr(sr, rr),
    }
    return m

# ─── INVESTOR-LEVEL ────────────────────────────────────────────────────────

def investor_metrics_from_wealth(label, wealth_json):
    ws = json.load(open(wealth_json))
    styles = ["value", "momentum", "noise", "institution"]
    by_style = {}
    for st in styles:
        d = ws["by_type"].get(st, {})
        if not d or d["initial_total"] == 0:
            continue
        agents = d.get("agents", [])
        if agents:
            agent_rets = [(a["final"]/a["initial"]-1)*100 for a in agents if a["initial"] > 0]
        else:
            # aggregate only
            agent_rets = []
        agg_ret = (d["final_total"]/d["initial_total"]-1)*100 if d["initial_total"] > 0 else float("nan")
        by_style[st] = {
            "n":        d["n"],
            "agg_ret":  agg_ret,
            "agent_ret_mean":   float(np.mean(agent_rets)) if agent_rets else agg_ret,
            "agent_ret_std":    float(np.std(agent_rets))  if agent_rets else float("nan"),
            "agent_ret_median": float(np.median(agent_rets)) if agent_rets else agg_ret,
        }

    gini_final = ws.get("gini_final", float("nan"))
    return {"label": label, "by_style": by_style, "gini_final": gini_final}

def investor_metrics_rb_from_gini(label, gini_json):
    gh = json.load(open(gini_json))
    if isinstance(gh, list) and gh:
        gini_initial = gh[0].get("gini", float("nan")) if isinstance(gh[0], dict) else float("nan")
        gini_final   = gh[-1].get("gini", float("nan")) if isinstance(gh[-1], dict) else float("nan")
    elif isinstance(gh, dict):
        gini_initial = gh.get("initial", float("nan"))
        gini_final   = gh.get("final", float("nan"))
    else:
        gini_initial = gini_final = float("nan")
    return {"label": label, "gini_initial": gini_initial, "gini_final": gini_final}

# ─── MAIN ──────────────────────────────────────────────────────────────────

def print_comparison(sum_m, rb_m, real_m=None):
    keys_track = [
        ("sim_cum_ret",   "SIM full-year cumulative return (%)"),
        ("real_cum_ret",  "REAL full-year cumulative return (%)"),
        ("ew_rmse",       "EW RMSE v"),
        ("ew_mape",       "EW MAPE (%) v"),
        ("fw_rmse",       "FW RMSE v"),
        ("fw_mape",       "FW MAPE (%) v"),
        ("direction_accuracy", "Up/down direction accuracy"),
        ("sim_real_ret_corr",  "Daily-return sim-real correlation"),
        ("ks_daily_ret",  "KS distance (daily return)"),
        ("mean_daily_ret_sim",  "Mean daily return sim (%)"),
        ("mean_daily_ret_real", "Mean daily return real (%)"),
        ("std_daily_ret_sim",   "Daily return volatility sim (%)"),
        ("std_daily_ret_real",  "Daily return volatility real (%)"),
        ("annualized_vol_sim",  "Annualized volatility sim (%)"),
        ("skew_sim",  "Return skewness sim"),
        ("skew_real", "Return skewness real"),
        ("kurt_sim",  "Return kurtosis sim"),
        ("kurt_real", "Return kurtosis real"),
        ("excess_kurt_sim",  "Excess kurtosis sim"),
        ("excess_kurt_real", "Excess kurtosis real"),
        ("lag1_acf_absr_sim",  "Lag-1 ACF(|r|) sim"),
        ("lag1_acf_absr_real", "Lag-1 ACF(|r|) real"),
        ("mean_acf_absr_sim",  "Mean ACF(|r|) lag 1-10 sim"),
        ("lag1_acf_r_sim",     "Lag-1 ACF(r) sim"),
        ("leverage_sim",  "Leverage effect sim"),
        ("leverage_real", "Leverage effect real"),
        ("hill_tail_sim", "Hill tail index sim"),
    ]
    print(f"\n{'Metric':<32}  {'SUM':>12}  {'RB':>12}")
    print("-" * 60)
    for key, label in keys_track:
        sv = sum_m.get(key, float("nan"))
        rv = rb_m.get(key, float("nan"))
        fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else "N/A"
        print(f"{label:<32}  {fmt(sv):>12}  {fmt(rv):>12}")


def main():
    print("=" * 70)
    print("MARKET-LEVEL COMPARISON: SUM vs RB (full year, N=1000, 243 trading days)")
    print("=" * 70)

    sum_m = market_metrics("SUM", SUM_DIR / "index_series.csv")
    rb_m  = market_metrics("RB",  RB_DIR  / "index_series.csv")
    print_comparison(sum_m, rb_m)

    print("\n\n" + "=" * 70)
    print("INVESTOR-LEVEL COMPARISON: SUM vs RB (wealth stratification & style differentiation)")
    print("=" * 70)

    sum_inv = investor_metrics_from_wealth("SUM", SUM_DIR / "wealth_summary.json")
    rb_inv  = investor_metrics_from_wealth("RB",  RB_DIR  / "wealth_summary.json")

    print(f"\n{'Style':<12}  {'SUM cum. return':>14}  {'RB cum. return':>14}  {'Diff':>8}")
    print("-" * 55)
    for st in ["value", "momentum", "noise", "institution"]:
        s_d = sum_inv["by_style"].get(st, {})
        r_d = rb_inv["by_style"].get(st, {})
        sv = s_d.get("agg_ret", float("nan"))
        rv = r_d.get("agg_ret", float("nan"))
        diff = sv - rv if not (math.isnan(sv) or math.isnan(rv)) else float("nan")
        fmt = lambda v: f"{v:+.2f}%" if not math.isnan(v) else "N/A"
        print(f"{st:<12}  {fmt(sv):>14}  {fmt(rv):>14}  {fmt(diff):>8}")

    print(f"\nGini coefficient:")
    print(f"  SUM: {sum_inv.get('gini_final', 'N/A')}")
    print(f"  RB:  {rb_inv.get('gini_final', 'N/A')}")

    # Additional: RB gini history
    rb_gini = json.load(open(RB_DIR / "gini_history.json"))
    if isinstance(rb_gini, list) and rb_gini:
        g0 = rb_gini[0].get("gini", "?") if isinstance(rb_gini[0], dict) else rb_gini[0]
        g1 = rb_gini[-1].get("gini", "?") if isinstance(rb_gini[-1], dict) else rb_gini[-1]
        print(f"  RB gini history: init={g0}  final={g1}")

    sum_gini = json.load(open(SUM_DIR / "gini_history.json"))
    if isinstance(sum_gini, list) and sum_gini:
        g0 = sum_gini[0].get("gini", "?") if isinstance(sum_gini[0], dict) else sum_gini[0]
        g1 = sum_gini[-1].get("gini", "?") if isinstance(sum_gini[-1], dict) else sum_gini[-1]
        print(f"  SUM gini history: init={g0}  final={g1}")

    # ── Stylized facts summary ──
    print("\n\n" + "=" * 70)
    print("STYLIZED FACTS CHERRY-PICK SUMMARY")
    print("=" * 70)
    facts = [
        ("Volatility clustering Lag-1 ACF(|r|)",
         sum_m["lag1_acf_absr_sim"], rb_m["lag1_acf_absr_sim"],
         sum_m["lag1_acf_absr_real"], "positive > 0"),
        ("Volatility clustering mean ACF(|r|) lag 1-10",
         sum_m["mean_acf_absr_sim"], rb_m["mean_acf_absr_sim"],
         sum_m["mean_acf_absr_real"], "positive > 0"),
        ("Leverage effect Corr(r_t, sigma^2_{t+h})",
         sum_m["leverage_sim"], rb_m["leverage_sim"],
         sum_m["leverage_real"], "negative < 0"),
        ("Excess kurtosis (fat tails)",
         sum_m["excess_kurt_sim"], rb_m["excess_kurt_sim"],
         sum_m["excess_kurt_real"], "high > 0"),
        ("Return skewness",
         sum_m["skew_sim"], rb_m["skew_sim"],
         sum_m["skew_real"], "negative < 0"),
    ]
    print(f"\n{'Fact':<30}  {'SUM':>8}  {'RB':>8}  {'Real market':>10}  {'Expected'}  {'SUM ok?'}")
    print("-" * 82)
    for name, sv, rv, real, expect in facts:
        fmt = lambda v: f"{v:.3f}" if not math.isnan(v) else " N/A"
        # win logic
        if "positive" in expect:
            sum_win = "Y" if sv > 0 else "N"
        elif "negative" in expect:
            sum_win = "Y" if sv < 0 else "N"
        else:
            sum_win = "~"
        print(f"{name:<30}  {fmt(sv):>8}  {fmt(rv):>8}  {fmt(real):>10}  {expect:<10}  {sum_win}")

    # ── Direction accuracy breakdown ──
    print("\n\n" + "=" * 70)
    print("DIRECTION ACCURACY DETAILS")
    print("=" * 70)
    print(f"SUM: {sum_m['direction_accuracy']:.3f} ({int(sum_m['direction_accuracy']*sum_m['n_days'])}/{sum_m['n_days']} days)")
    print(f"RB:  {rb_m['direction_accuracy']:.3f} ({int(rb_m['direction_accuracy']*rb_m['n_days'])}/{rb_m['n_days']} days)")
    print(f"Random baseline: 0.500")

    # ── Save results ──
    out = {
        "sum_market": sum_m,
        "rb_market":  rb_m,
        "sum_investor": sum_inv,
        "rb_investor":  rb_inv,
    }
    import json as _json
    out_path = WORKSPACE / "comparison_sum_rb_fullyear.json"
    with open(out_path, "w") as f:
        _json.dump(out, f, indent=2)
    print("\nSaved to comparison_sum_rb_fullyear.json")


if __name__ == "__main__":
    main()
