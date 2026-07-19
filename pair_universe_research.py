"""Reproducible full-universe cointegration scan and portfolio simulation.

The research protocol deliberately separates:

* discovery: non-overlapping days 1-250 and 251-500;
* parameter selection: three rolling-origin folds contained in days 1-500;
* holdout: days 501-750, never used to choose a pair or parameter;
* final eligibility: raw Engle-Granger p < 0.05 in both directions in every
  250-day block, plus the algorithm's online rolling ADF health guard.

Run from any directory with the exploration virtual environment:
    .venv/bin/python pair_universe_research.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint

import updated_algorithm as algorithm


HERE = Path(__file__).resolve().parent
PRICES_FILE = HERE / "prices.txt"
RESULTS_DIR = HERE / "pair_research_results"
RESULTS_DIR.mkdir(exist_ok=True)

SEGMENTS = ((0, 250), (250, 500), (500, 750))
SELECTION_FOLDS = (
    ("selection_251_333", 333, 250),
    ("selection_334_416", 416, 333),
    ("selection_417_500", 500, 416),
)
ALL_WINDOWS = SELECTION_FOLDS + (("holdout_501_750", 750, 500),)
LOOKBACKS = (40, 60, 80, 100, 125, 150)
ENTRY_ZS = (0.8, 1.0, 1.25, 1.5)
RNG_SEED = 20260719
PERSISTENCE_PERMUTATIONS = 5_000

DEFAULT_COMMISSION = 0.0001
ALGO_COMMISSION = 0.00002
DEFAULT_LIMIT = 10_000.0
ALGO_LIMIT = 100_000.0


def official_score(mean, std):
    """Official risk-adjusted competition score."""
    if mean <= 0 or std < 1e-10:
        return mean
    sharpe = np.sqrt(250) * mean / std
    return mean * sharpe**2 / (sharpe**2 + 1.0)


def run_eval(step, price_history, start_day):
    """Reproduce eval.py accounting, including its one-day commission lag."""
    n_instruments, n_days = price_history.shape
    commission_rates = np.full(n_instruments, DEFAULT_COMMISSION)
    commission_rates[0] = ALGO_COMMISSION
    dollar_limits = np.full(n_instruments, DEFAULT_LIMIT)
    dollar_limits[0] = ALGO_LIMIT
    cash = value = previous_commission = 0.0
    current = np.zeros(n_instruments, dtype=int)
    pnl = []
    turnover = total_commission = 0.0

    for day in range(start_day, n_days + 1):
        seen = price_history[:, :day]
        prices = seen[:, -1]
        if day < n_days:
            proposed = np.asarray(step(seen))
            limits = (dollar_limits / prices).astype(int)
            new_position = np.clip(proposed, -limits, limits).astype(int)
        else:
            new_position = current.copy()
        change = new_position - current
        cash -= prices.dot(change) + previous_commission
        dollar_volume = prices * np.abs(change)
        turnover += dollar_volume.sum()
        previous_commission = np.sum(dollar_volume * commission_rates)
        total_commission += previous_commission
        current = new_position
        new_value = cash + current.dot(prices)
        daily_pnl = new_value - value
        value = new_value
        if day > start_day:
            pnl.append(daily_pnl)
    return np.asarray(pnl), turnover, total_commission


def metrics(pnl, turnover, commission):
    mean = pnl.mean()
    std = pnl.std()
    sharpe = np.sqrt(250) * mean / std if std > 0 else 0.0
    equity = pnl.cumsum()
    padded = np.r_[0.0, equity]
    drawdown = padded - np.maximum.accumulate(padded)
    return {
        "total_pnl": pnl.sum(),
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "score": official_score(mean, std),
        "turnover": turnover,
        "commission": commission,
        "max_drawdown": drawdown.min(),
    }


def install_research_coint_cache():
    """Cache repeated formal health tests across parameter simulations.

    Hundreds of strategy configurations see the identical rolling arrays.
    Caching is confined to this research process; updated_algorithm.py itself
    continues to calculate health directly from each evaluator input.
    """
    uncached_coint = algorithm.coint
    cache = {}

    def cached_coint(y, x, trend="c", autolag="aic"):
        key = (
            len(y),
            float(y[0]),
            float(y[-1]),
            float(x[0]),
            float(x[-1]),
            float(np.sum(y)),
            float(np.sum(x)),
            trend,
            autolag,
        )
        if key not in cache:
            cache[key] = uncached_coint(y, x, trend=trend, autolag=autolag)
        return cache[key]

    algorithm.coint = cached_coint
    return cache


def scan_all_pairs(log_prices, names):
    pair_ids = list(combinations(range(len(names)), 2))
    p_values = np.empty((len(pair_ids), len(SEGMENTS)))
    direction_p_values = np.empty((len(pair_ids), len(SEGMENTS), 2))

    for row, (left, right) in enumerate(pair_ids):
        for segment, (start, end) in enumerate(SEGMENTS):
            x = log_prices[left, start:end]
            y = log_prices[right, start:end]
            p_y_on_x = coint(y, x, trend="c", autolag="aic")[1]
            p_x_on_y = coint(x, y, trend="c", autolag="aic")[1]
            direction_p_values[row, segment] = (p_y_on_x, p_x_on_y)
            # Requiring both directions is conservative and avoids choosing the
            # regression direction by looking at the holdout p-value.
            p_values[row, segment] = max(p_y_on_x, p_x_on_y)

    q_values = np.column_stack(
        [
            multipletests(p_values[:, segment], method="fdr_bh")[1]
            for segment in range(len(SEGMENTS))
        ]
    )
    result = pd.DataFrame(
        {
            "left": [names[left] for left, _ in pair_ids],
            "right": [names[right] for _, right in pair_ids],
            "left_id": [left for left, _ in pair_ids],
            "right_id": [right for _, right in pair_ids],
        }
    )
    for segment in range(len(SEGMENTS)):
        result[f"p_block_{segment + 1}"] = p_values[:, segment]
        result[f"q_block_{segment + 1}"] = q_values[:, segment]
        result[f"p_right_on_left_{segment + 1}"] = direction_p_values[:, segment, 0]
        result[f"p_left_on_right_{segment + 1}"] = direction_p_values[:, segment, 1]
    result["max_discovery_p"] = p_values[:, :2].max(axis=1)
    result["max_all_p"] = p_values.max(axis=1)
    result["max_all_q"] = q_values.max(axis=1)
    result["discovery_pass"] = result["max_discovery_p"] < 0.05
    result["full_period_pass"] = result["max_all_p"] < 0.05
    return result, p_values, pair_ids


def persistence_permutation_test(p_values, pair_ids, n_instruments):
    """Empirical multiplicity check for repeated significance by pair name.

    Instrument labels are independently permuted in blocks 2 and 3. This keeps
    each block's complete p-value structure while destroying persistence of a
    particular named pair across blocks.
    """
    matrices = []
    for segment in range(p_values.shape[1]):
        matrix = np.ones((n_instruments, n_instruments))
        for row, (left, right) in enumerate(pair_ids):
            matrix[left, right] = matrix[right, left] = p_values[row, segment]
        matrices.append(matrix)

    left_ids = np.array([left for left, _ in pair_ids])
    right_ids = np.array([right for _, right in pair_ids])
    first_pass = matrices[0][left_ids, right_ids] < 0.05
    rng = np.random.default_rng(RNG_SEED)
    null_counts = np.empty(PERSISTENCE_PERMUTATIONS, dtype=int)
    for iteration in range(PERSISTENCE_PERMUTATIONS):
        permutation_2 = rng.permutation(n_instruments)
        permutation_3 = rng.permutation(n_instruments)
        pass_2 = (
            matrices[1][permutation_2[left_ids], permutation_2[right_ids]] < 0.05
        )
        pass_3 = (
            matrices[2][permutation_3[left_ids], permutation_3[right_ids]] < 0.05
        )
        null_counts[iteration] = np.sum(first_pass & pass_2 & pass_3)
    observed = np.sum((p_values < 0.05).all(axis=1))
    empirical_p = (1 + np.sum(null_counts >= observed)) / (
        PERSISTENCE_PERMUTATIONS + 1
    )
    return observed, null_counts, empirical_p


def evaluate_pair_config(prices, x_id, y_id, lookback, entry_z):
    rows = []
    for window_name, history_end, score_start in ALL_WINDOWS:
        strategy = algorithm.PairState(x_id, y_id, lookback, entry_z)
        pnl, turnover, commission = run_eval(
            strategy.step, prices[:, :history_end], score_start
        )
        rows.append(
            {
                "window": window_name,
                "x_id": x_id,
                "y_id": y_id,
                "lookback": lookback,
                "entry_z": entry_z,
                **metrics(pnl, turnover, commission),
            }
        )
    return rows


def tune_discovery_pairs(prices, names, scan):
    all_rows = []
    selected_rows = []
    candidates = scan.loc[scan["discovery_pass"]]
    selection_names = [name for name, _, _ in SELECTION_FOLDS]

    for candidate in candidates.itertuples(index=False):
        pair_label = f"{candidate.left}-{candidate.right}"
        candidate_rows = []
        # Direction selection uses only the three selection folds.
        for x_id, y_id in (
            (candidate.left_id, candidate.right_id),
            (candidate.right_id, candidate.left_id),
        ):
            for lookback in LOOKBACKS:
                for entry_z in ENTRY_ZS:
                    rows = evaluate_pair_config(
                        prices, x_id, y_id, lookback, entry_z
                    )
                    for row in rows:
                        row["pair"] = pair_label
                        row["x_name"] = names[x_id]
                        row["y_name"] = names[y_id]
                    candidate_rows.extend(rows)

        candidate_frame = pd.DataFrame(candidate_rows)
        all_rows.extend(candidate_rows)
        selection = candidate_frame[
            candidate_frame["window"].isin(selection_names)
        ]
        grouped = selection.groupby(
            ["pair", "x_id", "y_id", "x_name", "y_name", "lookback", "entry_z"],
            as_index=False,
        ).agg(
            worst_selection_score=("score", "min"),
            mean_selection_score=("score", "mean"),
            worst_selection_mean=("mean", "min"),
        )
        eligible = grouped[grouped["worst_selection_mean"] > 0]
        ranking = eligible if not eligible.empty else grouped
        best = ranking.sort_values(
            ["worst_selection_score", "mean_selection_score"], ascending=False
        ).iloc[0]
        holdout = candidate_frame[
            (candidate_frame["window"] == "holdout_501_750")
            & (candidate_frame["x_id"] == best.x_id)
            & (candidate_frame["y_id"] == best.y_id)
            & (candidate_frame["lookback"] == best.lookback)
            & (candidate_frame["entry_z"] == best.entry_z)
        ].iloc[0]
        selected_rows.append(
            {
                **best.to_dict(),
                "holdout_mean": holdout["mean"],
                "holdout_std": holdout["std"],
                "holdout_score": holdout["score"],
                "holdout_total_pnl": holdout["total_pnl"],
                "full_period_significant": bool(candidate.full_period_pass),
                "holdout_max_p": candidate.p_block_3,
            }
        )
    return pd.DataFrame(all_rows), pd.DataFrame(selected_rows)


class PairBook:
    def __init__(self, specifications):
        self.pairs = [algorithm.PairState(*specification) for specification in specifications]

    def step(self, history):
        return sum(
            (pair.step(history) for pair in self.pairs),
            np.zeros(algorithm.N_INST, dtype=int),
        )


class ConfigurableBreadth:
    """Exact breadth sleeve with a configurable instrument exclusion set."""

    def __init__(self, exclude):
        self.exclude = set(exclude) | {algorithm.ALGO_INDEX}
        self.position = np.zeros(algorithm.N_INST, dtype=int)

    def step(self, history):
        if history.shape[1] < algorithm.BREADTH_LOOKBACK + 2:
            return np.zeros(algorithm.N_INST, dtype=int)
        log_prices = np.log(history[:, -algorithm.BREADTH_LOOKBACK :])
        demeaned = log_prices - log_prices.mean(axis=1, keepdims=True)
        residual = demeaned - demeaned.mean(axis=0, keepdims=True)
        residual_std = np.maximum(residual.std(axis=1), 1e-8)
        z_score = (residual[:, -1] - residual.mean(axis=1)) / residual_std
        dollars = (
            -np.clip(z_score / algorithm.BREADTH_Z_CAP, -1.0, 1.0)
            * algorithm.DEFAULT_DOLLAR_LIMIT
        )
        for instrument in self.exclude:
            dollars[instrument] = 0.0
        prices = history[:, -1]
        new_position = np.trunc(dollars / prices).astype(int)
        current_dollars = self.position * prices
        keep = (
            np.abs(dollars - current_dollars) < algorithm.BREADTH_DEADBAND
        )
        new_position[keep] = self.position[keep]
        limits = (algorithm.DEFAULT_DOLLAR_LIMIT / prices).astype(int)
        self.position = np.clip(new_position, -limits, limits)
        return self.position


class CombinedBook:
    def __init__(self, specifications):
        pair_instruments = {
            instrument
            for x_id, y_id, _, _ in specifications
            for instrument in (x_id, y_id)
        }
        self.breadth = ConfigurableBreadth(pair_instruments)
        self.pairs = PairBook(specifications)

    def step(self, history):
        return self.breadth.step(history) + self.pairs.step(history)


def simulate_subsets(prices, selected):
    pair_specs = {}
    for row in selected.itertuples(index=False):
        pair_specs[row.pair] = (
            int(row.x_id),
            int(row.y_id),
            int(row.lookback),
            float(row.entry_z),
        )

    rows = []
    labels = sorted(pair_specs)
    for subset_size in range(1, len(labels) + 1):
        for subset in combinations(labels, subset_size):
            specifications = [pair_specs[label] for label in subset]
            subset_label = " + ".join(subset)
            for window_name, history_end, score_start in ALL_WINDOWS:
                for strategy_name, strategy in (
                    ("pairs_only", PairBook(specifications)),
                    ("combined", CombinedBook(specifications)),
                ):
                    pnl, turnover, commission = run_eval(
                        strategy.step, prices[:, :history_end], score_start
                    )
                    rows.append(
                        {
                            "subset": subset_label,
                            "pair_count": subset_size,
                            "strategy": strategy_name,
                            "window": window_name,
                            **metrics(pnl, turnover, commission),
                        }
                    )
    return pd.DataFrame(rows)


def main():
    data = pd.read_csv(PRICES_FILE, sep=r"\s+")
    names = list(data.columns)
    prices = data.values.T
    log_prices = np.log(prices)

    print("Scanning all 1,275 pairs in all three 250-day blocks...")
    scan, p_values, pair_ids = scan_all_pairs(log_prices, names)
    scan.to_csv(RESULTS_DIR / "full_pair_scan.csv", index=False)

    observed, null_counts, empirical_p = persistence_permutation_test(
        p_values, pair_ids, len(names)
    )
    permutation_summary = pd.DataFrame(
        {
            "observed_persistent_pairs": [observed],
            "null_mean": [null_counts.mean()],
            "null_95_percentile": [np.quantile(null_counts, 0.95)],
            "empirical_p_value": [empirical_p],
            "permutations": [PERSISTENCE_PERMUTATIONS],
        }
    )
    permutation_summary.to_csv(
        RESULTS_DIR / "persistence_permutation_test.csv", index=False
    )

    print(
        f"Discovery pass: {scan.discovery_pass.sum()} pairs; "
        f"full-period pass: {scan.full_period_pass.sum()} pairs; "
        f"FDR-5% full-period pass: {(scan.max_all_q < 0.05).sum()} pairs."
    )
    print(permutation_summary.to_string(index=False))

    print("Selecting pair direction/parameters using days 1-500 only...")
    health_cache = install_research_coint_cache()
    parameter_results, selected = tune_discovery_pairs(prices, names, scan)
    parameter_results.to_csv(
        RESULTS_DIR / "pair_parameter_simulations.csv", index=False
    )
    selected.to_csv(RESULTS_DIR / "selected_pair_configs.csv", index=False)
    print(selected.to_string(index=False))

    print("Simulating every subset of the discovery candidates...")
    subset_results = simulate_subsets(prices, selected)
    subset_results.to_csv(
        RESULTS_DIR / "pair_subset_simulations.csv", index=False
    )

    selection_windows = [name for name, _, _ in SELECTION_FOLDS]
    selection = subset_results[
        (subset_results.strategy == "combined")
        & subset_results.window.isin(selection_windows)
    ]
    ranking = selection.groupby(
        ["subset", "pair_count"], as_index=False
    ).agg(
        worst_selection_score=("score", "min"),
        mean_selection_score=("score", "mean"),
        worst_selection_mean=("mean", "min"),
    )
    holdout = subset_results[
        (subset_results.strategy == "combined")
        & (subset_results.window == "holdout_501_750")
    ][["subset", "score", "mean", "std", "total_pnl", "max_drawdown"]].rename(
        columns={
            "score": "holdout_score",
            "mean": "holdout_mean",
            "std": "holdout_std",
            "total_pnl": "holdout_total_pnl",
            "max_drawdown": "holdout_max_drawdown",
        }
    )
    ranking = ranking.merge(holdout, on="subset").sort_values(
        ["worst_selection_score", "mean_selection_score"], ascending=False
    )
    ranking.to_csv(RESULTS_DIR / "subset_selection_ranking.csv", index=False)
    print("Top subset rankings (selected without holdout):")
    print(ranking.head(15).to_string(index=False))
    print(f"Cached formal health tests: {len(health_cache):,}")
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
