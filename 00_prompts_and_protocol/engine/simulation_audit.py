"""
Simulation audit helpers.

Tracks float-market-cap weighted drift, weekly supervision notes, and
correction biases that can be fed back into the next week.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        n = max(len(weights), 1)
        return {k: 1.0 / n for k in weights}
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def _weighted_return(prev_prices: Dict[str, float], curr_prices: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for stock, weight in weights.items():
        p_prev = prev_prices.get(stock, 0.0)
        p_curr = curr_prices.get(stock, 0.0)
        if p_prev > 0:
            total += weight * (p_curr / p_prev - 1.0)
    return total


@dataclass
class DailyAuditRecord:
    date: str
    sim_index: float
    real_index: float
    sim_return: float
    real_return: float
    gap: float
    mean_value_signal: float = 0.0
    mean_momentum_signal: float = 0.0
    mean_noise_signal: float = 0.0
    mean_obi: float = 0.0
    order_count: int = 0
    trade_count: int = 0


@dataclass
class WeeklyAuditRecord:
    week_key: str
    start_date: str
    end_date: str
    sim_return: float
    real_return: float
    gap: float
    abs_gap: float
    mean_daily_gap: float
    root_cause: str
    correction_strength: float
    stock_bias_after: Dict[str, float] = field(default_factory=dict)


class WeeklyMarketSupervisor:
    def __init__(self, stock_weights: Dict[str, float], base_index: float = 1000.0):
        self.weights = _normalize_weights(stock_weights)
        self.base_index = base_index
        self.stock_bias: Dict[str, float] = {s: 0.0 for s in self.weights}
        self.daily_records: List[DailyAuditRecord] = []
        self.weekly_records: List[WeeklyAuditRecord] = []
        self._daily_snapshots: List[Dict] = []

    def record_day(
        self,
        date: str,
        sim_prices: Dict[str, float],
        real_prices: Dict[str, float],
        day_stats: Optional[Dict] = None,
    ) -> DailyAuditRecord:
        day_stats = day_stats or {}
        if self._daily_snapshots:
            prev_sim = self._daily_snapshots[-1]["sim_prices"]
            prev_real = self._daily_snapshots[-1]["real_prices"]
            sim_ret = _weighted_return(prev_sim, sim_prices, self.weights)
            real_ret = _weighted_return(prev_real, real_prices, self.weights)
        else:
            sim_ret = 0.0
            real_ret = 0.0

        prev_sim_index = self.daily_records[-1].sim_index if self.daily_records else self.base_index
        prev_real_index = self.daily_records[-1].real_index if self.daily_records else self.base_index
        sim_index = prev_sim_index * (1.0 + sim_ret)
        real_index = prev_real_index * (1.0 + real_ret)
        gap = real_ret - sim_ret

        record = DailyAuditRecord(
            date=date,
            sim_index=sim_index,
            real_index=real_index,
            sim_return=sim_ret,
            real_return=real_ret,
            gap=gap,
            mean_value_signal=float(day_stats.get("mean_value_signal", 0.0)),
            mean_momentum_signal=float(day_stats.get("mean_momentum_signal", 0.0)),
            mean_noise_signal=float(day_stats.get("mean_noise_signal", 0.0)),
            mean_obi=float(day_stats.get("mean_obi", 0.0)),
            order_count=int(day_stats.get("order_count", 0)),
            trade_count=int(day_stats.get("trade_count", 0)),
        )
        self.daily_records.append(record)
        self._daily_snapshots.append({
            "date": date,
            "sim_prices": dict(sim_prices),
            "real_prices": dict(real_prices),
        })
        return record

    def close_week(self, week_key: str, week_dates: List[str]) -> Optional[WeeklyAuditRecord]:
        if not week_dates:
            return None

        week_days = [r for r in self.daily_records if r.date in week_dates]
        if len(week_days) < 2:
            return None

        start_date = week_days[0].date
        end_date = week_days[-1].date
        start_snap = next(s for s in self._daily_snapshots if s["date"] == start_date)
        end_snap = next(s for s in self._daily_snapshots if s["date"] == end_date)

        sim_ret = _weighted_return(start_snap["sim_prices"], end_snap["sim_prices"], self.weights)
        real_ret = _weighted_return(start_snap["real_prices"], end_snap["real_prices"], self.weights)
        gap = real_ret - sim_ret
        abs_gap = abs(gap)
        mean_daily_gap = sum(r.gap for r in week_days) / len(week_days)

        root_cause = self._infer_root_cause(week_days, gap)
        correction_strength = self._derive_correction_strength(gap)
        stock_corrections = self._derive_stock_corrections(start_snap["sim_prices"], end_snap["sim_prices"], start_snap["real_prices"], end_snap["real_prices"])
        self._update_bias(stock_corrections, correction_strength)

        record = WeeklyAuditRecord(
            week_key=week_key,
            start_date=start_date,
            end_date=end_date,
            sim_return=sim_ret,
            real_return=real_ret,
            gap=gap,
            abs_gap=abs_gap,
            mean_daily_gap=mean_daily_gap,
            root_cause=root_cause,
            correction_strength=correction_strength,
            stock_bias_after=dict(self.stock_bias),
        )
        self.weekly_records.append(record)
        return record

    def get_bias(self, stock: str) -> float:
        return float(self.stock_bias.get(stock, 0.0))

    def export_daily(self) -> List[Dict]:
        return [asdict(r) for r in self.daily_records]

    def export_weekly(self) -> List[Dict]:
        return [asdict(r) for r in self.weekly_records]

    def _infer_root_cause(self, week_days: List[DailyAuditRecord], gap: float) -> str:
        mean_value = sum(r.mean_value_signal for r in week_days) / len(week_days)
        mean_momentum = sum(r.mean_momentum_signal for r in week_days) / len(week_days)
        mean_noise = sum(r.mean_noise_signal for r in week_days) / len(week_days)
        mean_obi = sum(r.mean_obi for r in week_days) / len(week_days)
        mean_trades = sum(r.trade_count for r in week_days) / len(week_days)

        if abs(gap) < 0.003:
            return "weekly_trend_aligned"
        if abs(mean_momentum) >= abs(mean_value) and gap * mean_momentum < 0:
            return "momentum_signal_misalignment"
        if abs(mean_noise) >= max(abs(mean_value), abs(mean_momentum)) and gap * mean_noise < 0:
            return "noise_sentiment_overreach"
        if abs(mean_obi) > 0.15 and mean_trades > 0:
            return "liquidity_orderbook_dislocation"
        if gap > 0:
            return "simulated_trend_too_weak"
        return "simulated_trend_too_strong"

    def _derive_correction_strength(self, gap: float) -> float:
        strength = gap * 0.7
        return max(-0.03, min(0.03, strength))

    def _derive_stock_corrections(
        self,
        start_sim: Dict[str, float],
        end_sim: Dict[str, float],
        start_real: Dict[str, float],
        end_real: Dict[str, float],
    ) -> Dict[str, float]:
        corrections: Dict[str, float] = {}
        for stock, weight in self.weights.items():
            p0s = start_sim.get(stock, 0.0)
            p1s = end_sim.get(stock, 0.0)
            p0r = start_real.get(stock, 0.0)
            p1r = end_real.get(stock, 0.0)
            sim_ret = (p1s / p0s - 1.0) if p0s > 0 else 0.0
            real_ret = (p1r / p0r - 1.0) if p0r > 0 else 0.0
            corrections[stock] = max(-0.03, min(0.03, (real_ret - sim_ret) * (0.35 + 0.65 * weight)))
        return corrections

    def _update_bias(self, stock_corrections: Dict[str, float], correction_strength: float) -> None:
        for stock, corr in stock_corrections.items():
            self.stock_bias[stock] = 0.8 * self.stock_bias.get(stock, 0.0) + 0.2 * (corr + correction_strength)

