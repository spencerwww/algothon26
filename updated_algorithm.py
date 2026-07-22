"""Standalone Algothon strategy: ridge forecast + residuals + four pairs.

This file is submission-compatible: the evaluator only needs to import
``getMyPosition(prcSoFar)``. NumPy and statsmodels are both included in the
official development environment.

Portfolio construction
----------------------
1. Four disjoint, historically stable pairs trade rolling OLS log-price
   spreads with Bollinger-style z-score entries and a mean-crossing exit.
2. A ridge model forecasts next-day returns and overrides the 20 strongest
   non-pair forecasts at the maximum legal position size.
3. ULXY five-day momentum replaces HETT's residual-level position.
4. The remaining instruments trade a continuous residual-level
   mean-reversion signal. Pair instruments and ALGO are excluded from this
   sleeve so per-instrument clipping cannot damage the pair hedge ratios.

The constants are deliberately a small set of choices from the historical
regime audit, rather than a dense optimisation of the final 250 days.
"""

import numpy as np
from statsmodels.tsa.stattools import coint


N_INST = 51
DEFAULT_DOLLAR_LIMIT = 10_000.0
ALGO_INDEX = 0

# (x index, y index, lookback, entry |z|), where y is regressed on x.
# AENO-NWIG, HUXZ-ACAC, EORC-NGTE, SMAH-ILVX.
PAIR_SPECS = (
    (1, 20, 100, 1.0),
    (8, 27, 150, 0.8),
    (13, 45, 40, 1.0),
    (10, 46, 125, 0.8),
)

# Rolling cointegration health guard. The formal two-direction Engle-Granger
# test uses enough history to avoid the very low power of 40-150 day tests.
# New entries require p < 0.05 in both regression directions; an existing
# trade is closed immediately on failure. The short-window ADF t-statistic is
# retained as a diagnostic but is not presented as a formal p-value.
COINT_HEALTH_LOOKBACK = 250
COINT_P_THRESHOLD = 0.05
COINT_ADF_LAGS = 1
COINT_FAILURE_CONFIRMATIONS = 1

BREADTH_LOOKBACK = 100
BREADTH_Z_CAP = 0.5
BREADTH_DEADBAND = 1_000.0

# Cross-forecast portfolio. Alpha and top-k were selected through time-ordered
# folds on the first 375 price observations. Top-k 10 through 35 formed a
# stable positive plateau; 20 had the best worst-period performance. The model
# is fitted once, on the history supplied by the evaluator's first call, so no
# future observation is used in its coefficients.
RIDGE_ALPHA = 123.285
RIDGE_TOP_K = 20
RIDGE_MIN_TRAIN_DAYS = 250
RIDGE_TRAIN_START = 31

# Confirmed cross-momentum rule: ULXY's five-day return predicts HETT's
# next-day return with the same positive direction.
ULXY_INDEX = 40
HETT_INDEX = 7
CROSS_MOMENTUM_LOOKBACK = 5

PAIR_INSTRUMENTS = frozenset(
    instrument
    for x_index, y_index, _, _ in PAIR_SPECS
    for instrument in (x_index, y_index)
)
BREADTH_EXCLUDE = PAIR_INSTRUMENTS | {ALGO_INDEX}
RIDGE_ELIGIBLE = np.array(
    sorted(set(range(N_INST)) - PAIR_INSTRUMENTS - {HETT_INDEX}),
    dtype=int,
)


class PairState:
    """Stateful rolling-OLS mean-reversion strategy for one pair."""

    def __init__(self, x_index, y_index, lookback, entry_z):
        self.x_index = x_index
        self.y_index = y_index
        self.ids = np.array([y_index, x_index])
        self.lookback = lookback
        self.entry_z = entry_z
        self.direction = 0
        self.entry_beta = None
        self.failed_health_checks = 0
        self.last_adf_t = np.inf
        self.last_coint_p = 1.0
        self.blocked_signal_days = 0
        self.health_exit_count = 0
        self.position = np.zeros(N_INST, dtype=int)

    @staticmethod
    def _beta(log_prices):
        """OLS slope for log(y) = intercept + beta * log(x)."""
        y, x = log_prices
        x_centered = x - x.mean()
        denominator = np.dot(x_centered, x_centered)
        if denominator < 1e-14:
            return 0.0
        return np.dot(x_centered, y - y.mean()) / denominator

    @staticmethod
    def _zscore(log_prices, beta):
        y, x = log_prices
        spread = y - beta * x
        standard_deviation = spread.std()
        if standard_deviation < 1e-12:
            return 0.0
        return (spread[-1] - spread.mean()) / standard_deviation

    @staticmethod
    def _adf_t_stat(residual):
        """Return the ADF t-statistic for the lagged residual-level term.

        Regression: delta(e_t) = c + gamma*e_(t-1)
                                  + lagged delta(e_t) terms + error.
        A more-negative gamma t-statistic is stronger evidence that the
        residual is mean reverting. Implemented with NumPy to keep this file
        standalone and deterministic in the grading environment.
        """
        delta = np.diff(residual)
        lag_count = COINT_ADF_LAGS
        if delta.size <= lag_count + 3:
            return np.inf

        response = delta[lag_count:]
        columns = [
            np.ones(response.size),
            residual[lag_count:-1],
        ]
        for lag in range(1, lag_count + 1):
            columns.append(delta[lag_count - lag : -lag])
        design = np.column_stack(columns)

        coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
        errors = response - design @ coefficients
        degrees_of_freedom = response.size - design.shape[1]
        if degrees_of_freedom <= 0:
            return np.inf

        error_variance = np.dot(errors, errors) / degrees_of_freedom
        covariance = error_variance * np.linalg.pinv(design.T @ design)
        gamma_variance = covariance[1, 1]
        if not np.isfinite(gamma_variance) or gamma_variance <= 1e-18:
            return np.inf
        return coefficients[1] / np.sqrt(gamma_variance)

    def _cointegration_is_healthy(self, price_history, signal_log_prices):
        """Run a formal rolling two-direction Engle-Granger health check."""
        beta = self._beta(signal_log_prices)
        y, x = signal_log_prices
        intercept = np.mean(y - beta * x)
        residual = y - (intercept + beta * x)
        self.last_adf_t = self._adf_t_stat(residual)

        if price_history.shape[1] < COINT_HEALTH_LOOKBACK:
            self.last_coint_p = 1.0
            return False
        health_log_prices = np.log(
            price_history[self.ids, -COINT_HEALTH_LOOKBACK:]
        )
        health_y, health_x = health_log_prices
        p_y_on_x = coint(
            health_y, health_x, trend="c", autolag="aic"
        )[1]
        p_x_on_y = coint(
            health_x, health_y, trend="c", autolag="aic"
        )[1]
        self.last_coint_p = max(p_y_on_x, p_x_on_y)
        return np.isfinite(self.last_coint_p) and (
            self.last_coint_p < COINT_P_THRESHOLD
        )

    def _shares(self, direction, beta, prices):
        # Log-spread weights are (1, -beta). Scale the whole hedge so its
        # largest leg is exactly the $10k per-instrument limit.
        weights = np.array([1.0, -beta])
        largest_weight = np.max(np.abs(weights))
        if largest_weight < 1e-12:
            return np.zeros(2, dtype=int)
        dollars = direction * DEFAULT_DOLLAR_LIMIT * weights / largest_weight
        # Truncate toward zero: rounding can exceed a hard dollar limit by one
        # share, after which eval.py would hold a different position from this
        # object's internal state.
        return np.trunc(dollars / prices).astype(int)

    def step(self, price_history):
        if price_history.shape[1] < self.lookback + 1:
            return np.zeros(N_INST, dtype=int)

        current_prices = price_history[self.ids, -1]
        log_prices = np.log(price_history[self.ids, -self.lookback :])
        new_position = np.zeros(N_INST, dtype=int)
        cointegration_healthy = self._cointegration_is_healthy(
            price_history, log_prices
        )

        if self.direction == 0:
            self.failed_health_checks = 0
            beta = self._beta(log_prices)
            z_score = self._zscore(log_prices, beta)
            entry_signal = abs(z_score) > self.entry_z
            if entry_signal and not cointegration_healthy:
                self.blocked_signal_days += 1
            if cointegration_healthy and entry_signal:
                # Positive spread: short y/long x. Negative: long y/short x.
                self.direction = -1 if z_score > 0 else 1
                self.entry_beta = beta
                new_position[self.ids] = self._shares(
                    self.direction, beta, current_prices
                )
        else:
            if cointegration_healthy:
                self.failed_health_checks = 0
            else:
                self.failed_health_checks += 1

            z_score = self._zscore(log_prices, self.entry_beta)
            crossed_mean = (
                (self.direction < 0 and z_score <= 0)
                or (self.direction > 0 and z_score >= 0)
            )
            cointegration_failed = (
                self.failed_health_checks >= COINT_FAILURE_CONFIRMATIONS
            )
            if crossed_mean or cointegration_failed:
                if cointegration_failed and not crossed_mean:
                    self.health_exit_count += 1
                self.direction = 0
                self.entry_beta = None
                self.failed_health_checks = 0
            else:
                # Hold to avoid commission. If price movement has pushed a leg
                # above its dollar cap, resize both legs together.
                held = self.position[self.ids]
                if np.any(np.abs(held * current_prices) > DEFAULT_DOLLAR_LIMIT):
                    new_position[self.ids] = self._shares(
                        self.direction, self.entry_beta, current_prices
                    )
                else:
                    new_position[self.ids] = held

        self.position = new_position
        return new_position


class ResidualLevelSleeve:
    """Continuous cross-sectional residual-level mean reversion."""

    def __init__(self):
        self.position = np.zeros(N_INST, dtype=int)

    def step(self, price_history):
        if price_history.shape[1] < BREADTH_LOOKBACK + 2:
            return np.zeros(N_INST, dtype=int)

        log_prices = np.log(price_history[:, -BREADTH_LOOKBACK:])
        demeaned = log_prices - log_prices.mean(axis=1, keepdims=True)
        residual = demeaned - demeaned.mean(axis=0, keepdims=True)
        residual_std = np.maximum(residual.std(axis=1), 1e-8)
        z_score = (
            residual[:, -1] - residual.mean(axis=1)
        ) / residual_std

        dollars = (
            -np.clip(z_score / BREADTH_Z_CAP, -1.0, 1.0)
            * DEFAULT_DOLLAR_LIMIT
        )
        for instrument in BREADTH_EXCLUDE:
            dollars[instrument] = 0.0

        current_prices = price_history[:, -1]
        new_position = np.trunc(dollars / current_prices).astype(int)

        # Do not pay commission for small desired changes.
        current_dollars = self.position * current_prices
        keep = np.abs(dollars - current_dollars) < BREADTH_DEADBAND
        new_position[keep] = self.position[keep]

        # Keep internal state identical to the evaluator's capped state even
        # when an adverse price move makes a previously legal holding too big.
        share_limits = (DEFAULT_DOLLAR_LIMIT / current_prices).astype(int)
        new_position = np.clip(new_position, -share_limits, share_limits)

        self.position = new_position
        return new_position


class RidgeForecastSleeve:
    """One-step multi-output ridge forecast with max-size top-k positions."""

    def __init__(self):
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.coefficients = None

    def _fit_once(self, price_history):
        n_days = price_history.shape[1]
        if self.coefficients is not None or n_days < RIDGE_MIN_TRAIN_DAYS:
            return

        log_prices = np.log(price_history.T)
        returns = np.diff(log_prices, axis=0)
        # For decision d, return[d-1] is known and predicts return[d].
        decisions = np.arange(RIDGE_TRAIN_START, n_days - 1)
        features = returns[decisions - 1]
        targets = returns[decisions]

        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0)
        self.feature_std[self.feature_std < 1e-10] = 1.0
        standardized = (
            features - self.feature_mean
        ) / self.feature_std
        self.target_mean = targets.mean(axis=0)

        gram = standardized.T @ standardized
        rhs = standardized.T @ (targets - self.target_mean)
        self.coefficients = np.linalg.solve(
            gram + RIDGE_ALPHA * np.eye(N_INST), rhs
        )

    def step(self, price_history):
        self._fit_once(price_history)
        positions = np.zeros(N_INST, dtype=int)
        selected = np.zeros(N_INST, dtype=bool)
        if self.coefficients is None:
            return positions, selected

        current_return = np.log(
            price_history[:, -1] / price_history[:, -2]
        )
        standardized = (
            current_return - self.feature_mean
        ) / self.feature_std
        forecast = self.target_mean + standardized @ self.coefficients

        eligible_forecasts = np.abs(forecast[RIDGE_ELIGIBLE])
        selected_ids = RIDGE_ELIGIBLE[
            np.argsort(eligible_forecasts)[-RIDGE_TOP_K:]
        ]
        current_prices = price_history[:, -1]
        dollar_limits = np.full(N_INST, DEFAULT_DOLLAR_LIMIT)
        dollar_limits[ALGO_INDEX] = 100_000.0
        positions[selected_ids] = np.trunc(
            np.sign(forecast[selected_ids])
            * dollar_limits[selected_ids]
            / current_prices[selected_ids]
        ).astype(int)
        selected[selected_ids] = True
        return positions, selected


class HettCrossMomentum:
    """Max-size HETT position from ULXY's trailing five-day direction."""

    def step(self, price_history):
        if price_history.shape[1] <= CROSS_MOMENTUM_LOOKBACK:
            return 0
        ulxy_move = np.log(
            price_history[ULXY_INDEX, -1]
            / price_history[ULXY_INDEX, -1 - CROSS_MOMENTUM_LOOKBACK]
        )
        direction = np.sign(ulxy_move)
        return int(
            direction
            * DEFAULT_DOLLAR_LIMIT
            / price_history[HETT_INDEX, -1]
        )


class UpdatedStrategy:
    """Combine pair, ridge, cross-momentum and residual sleeves legally."""

    def __init__(self):
        self.pairs = [PairState(*spec) for spec in PAIR_SPECS]
        self.breadth = ResidualLevelSleeve()
        self.ridge = RidgeForecastSleeve()
        self.hett_cross_momentum = HettCrossMomentum()

    def step(self, price_history):
        n_instruments = price_history.shape[0]
        if n_instruments != N_INST:
            raise ValueError(f"Expected {N_INST} instruments, got {n_instruments}")

        total = self.breadth.step(price_history)
        for pair in self.pairs:
            total = total + pair.step(price_history)

        # Ridge forecasts never touch pair instruments, preserving hedge
        # ratios. They override rather than add to residual positions because
        # each instrument has only one legal dollar-position cap.
        ridge_positions, ridge_selected = self.ridge.step(price_history)
        total[ridge_selected] = ridge_positions[ridge_selected]

        # HETT's residual attribution was negative, while this cross-momentum
        # rule confirmed in discovery, validation and both final half-windows.
        total[HETT_INDEX] = self.hett_cross_momentum.step(price_history)

        # Defensive pre-clipping. The official evaluator repeats this check.
        dollar_limits = np.full(N_INST, DEFAULT_DOLLAR_LIMIT)
        dollar_limits[ALGO_INDEX] = 100_000.0
        share_limits = (dollar_limits / price_history[:, -1]).astype(int)
        return np.clip(total, -share_limits, share_limits).astype(int)


_STRATEGY = UpdatedStrategy()


def getMyPosition(prcSoFar):
    """Required Algothon evaluator entry point."""
    return _STRATEGY.step(prcSoFar)
