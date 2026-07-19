"""Stress and implementation validation for updated_algorithm.py."""

from pathlib import Path
import time

import numpy as np
import pandas as pd

import updated_algorithm as algorithm
from pair_universe_research import (
    CombinedBook,
    install_research_coint_cache,
    official_score,
)


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "pair_research_results"
RESULTS_DIR.mkdir(exist_ok=True)
PRICES = pd.read_csv(HERE / "prices.txt", sep=r"\s+").values.T

COMMISSIONS = np.full(algorithm.N_INST, 0.0001)
COMMISSIONS[0] = 0.00002
DOLLAR_LIMITS = np.full(algorithm.N_INST, 10_000.0)
DOLLAR_LIMITS[0] = 100_000.0


def run_backtest(step, history, start_day, fee_multiplier=1.0, keep_positions=False):
    """Exact eval.py accounting with a configurable commission multiplier."""
    n_instruments, n_days = history.shape
    cash = value = previous_commission = 0.0
    current = np.zeros(n_instruments, dtype=int)
    pnl = []
    positions = []
    turnover = total_commission = 0.0

    for day in range(start_day, n_days + 1):
        seen = history[:, :day]
        prices = seen[:, -1]
        if day < n_days:
            proposed = np.asarray(step(seen))
            limits = (DOLLAR_LIMITS / prices).astype(int)
            new_position = np.clip(proposed, -limits, limits).astype(int)
        else:
            new_position = current.copy()
        change = new_position - current
        cash -= prices.dot(change) + previous_commission
        dollar_volume = prices * np.abs(change)
        turnover += dollar_volume.sum()
        previous_commission = (
            np.sum(dollar_volume * COMMISSIONS) * fee_multiplier
        )
        total_commission += previous_commission
        current = new_position
        new_value = cash + current.dot(prices)
        daily_pnl = new_value - value
        value = new_value
        if day > start_day:
            pnl.append(daily_pnl)
            if keep_positions:
                positions.append(current.copy())
    return np.asarray(pnl), turnover, total_commission, np.asarray(positions)


def summarize(label, pnl, turnover, commission):
    mean = pnl.mean()
    std = pnl.std()
    equity = pnl.cumsum()
    padded = np.r_[0.0, equity]
    drawdown = padded - np.maximum.accumulate(padded)
    return {
        "label": label,
        "days": len(pnl),
        "total_pnl": pnl.sum(),
        "mean": mean,
        "std": std,
        "sharpe": np.sqrt(250) * mean / std if std else 0.0,
        "score": official_score(mean, std),
        "hit_rate": np.mean(pnl > 0),
        "max_drawdown": drawdown.min(),
        "turnover": turnover,
        "commission": commission,
    }


def block_bootstrap(pnl, samples=5_000, block_length=10, seed=20260719):
    rng = np.random.default_rng(seed)
    n_days = len(pnl)
    n_blocks = int(np.ceil(n_days / block_length))
    boot = np.empty((samples, n_days))
    offsets = np.arange(block_length)
    for sample in range(samples):
        starts = rng.integers(0, n_days, size=n_blocks)
        indices = (starts[:, None] + offsets) % n_days
        boot[sample] = pnl[indices.ravel()[:n_days]]
    means = boot.mean(axis=1)
    stds = boot.std(axis=1)
    sharpes = np.sqrt(250) * means / np.maximum(stds, 1e-12)
    scores = means * sharpes**2 / (sharpes**2 + 1.0)
    scores[means <= 0] = means[means <= 0]
    return pd.DataFrame(
        {
            "metric": ("mean", "sharpe", "score", "total_pnl"),
            "q025": (
                np.quantile(means, 0.025),
                np.quantile(sharpes, 0.025),
                np.quantile(scores, 0.025),
                np.quantile(means * n_days, 0.025),
            ),
            "median": (
                np.median(means),
                np.median(sharpes),
                np.median(scores),
                np.median(means * n_days),
            ),
            "q975": (
                np.quantile(means, 0.975),
                np.quantile(sharpes, 0.975),
                np.quantile(scores, 0.975),
                np.quantile(means * n_days, 0.975),
            ),
        }
    )


def main():
    install_research_coint_cache()
    validation_rows = []

    windows = (
        ("days_251_333", 333, 250),
        ("days_334_416", 416, 333),
        ("days_417_500", 500, 416),
        ("days_501_625", 625, 500),
        ("days_626_750", 750, 625),
        ("days_501_750", 750, 500),
    )
    for label, end, start in windows:
        pnl, turnover, commission, _ = run_backtest(
            algorithm.UpdatedStrategy().step, PRICES[:, :end], start
        )
        validation_rows.append(summarize(label, pnl, turnover, commission))

    fee_rows = []
    latest_pnl = None
    latest_positions = None
    for multiplier in (0.0, 1.0, 2.0, 5.0):
        pnl, turnover, commission, positions = run_backtest(
            algorithm.UpdatedStrategy().step,
            PRICES,
            500,
            fee_multiplier=multiplier,
            keep_positions=True,
        )
        fee_rows.append(
            summarize(f"fee_{multiplier:.0f}x", pnl, turnover, commission)
        )
        if multiplier == 1.0:
            latest_pnl = pnl
            latest_positions = positions

    pair_names = ("AENO-NWIG", "HUXZ-ACAC", "EORC-NGTE", "SMAH-ILVX")
    ablation_rows = []
    specifications = list(algorithm.PAIR_SPECS)
    variants = {"all_four": specifications}
    for index, pair_name in enumerate(pair_names):
        variants[f"without_{pair_name}"] = [
            specification
            for specification_index, specification in enumerate(specifications)
            if specification_index != index
        ]
    for label, variant_specs in variants.items():
        pnl, turnover, commission, _ = run_backtest(
            CombinedBook(variant_specs).step, PRICES, 500
        )
        ablation_rows.append(summarize(label, pnl, turnover, commission))

    validation = pd.DataFrame(validation_rows)
    fees = pd.DataFrame(fee_rows)
    ablations = pd.DataFrame(ablation_rows)
    bootstrap = block_bootstrap(latest_pnl)

    # Implementation invariants.
    latest_prices = PRICES[:, 500:750].T
    caps = (DOLLAR_LIMITS[None, :] / latest_prices).astype(int)
    # eval.py enforces caps on trading days but deliberately does not call the
    # strategy or rebalance on the final mark-to-market day.
    cap_ok = bool(np.all(np.abs(latest_positions[:-1]) <= caps[:-1]))
    deterministic_a = run_backtest(algorithm.UpdatedStrategy().step, PRICES, 500)[0]
    deterministic_b = run_backtest(algorithm.UpdatedStrategy().step, PRICES, 500)[0]
    deterministic = bool(np.array_equal(deterministic_a, deterministic_b))

    validation.to_csv(RESULTS_DIR / "shifted_window_validation.csv", index=False)
    fees.to_csv(RESULTS_DIR / "commission_stress.csv", index=False)
    ablations.to_csv(RESULTS_DIR / "leave_one_pair_out.csv", index=False)
    bootstrap.to_csv(RESULTS_DIR / "block_bootstrap.csv", index=False)

    print("SHIFTED WINDOWS")
    print(validation.to_string(index=False))
    print("\nCOMMISSION STRESS")
    print(fees.to_string(index=False))
    print("\nLEAVE-ONE-PAIR-OUT")
    print(ablations.to_string(index=False))
    print("\n10-DAY BLOCK BOOTSTRAP")
    print(bootstrap.to_string(index=False))
    print(f"\ncap_ok={cap_ok} deterministic={deterministic}")
    if not cap_ok or not deterministic:
        raise AssertionError("Implementation invariant failed")


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"Validation runtime: {time.time() - started:.2f}s")
