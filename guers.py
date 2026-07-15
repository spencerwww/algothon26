import numpy as np

nInst = 51

# ---------------------------------------------------------------------------
# Basket mean-reversion strategy (correlated baskets, column indices in
# prices.txt)
# ---------------------------------------------------------------------------
BASKETS = [
    [41, 49, 50],   # BLBT, MHRM, EAFC
    [9, 38],        # DUCT, HRND
    [1, 20],        # AENO, NWIG
    [10, 46],       # SMAH, ILVX
    [13, 45],       # EORC, NGTE
    [8, 27],        # HUXZ, ACAC
    [31, 43],       # ACIX, ITPA
]

BASKET_LOOKBACK = 36
BASKET_ENTRY_Z = 1.0
BASKET_EXIT_Z = 0.2
BASKET_DLR_POS_LIMIT = 10_000

currentPos = np.zeros(nInst, dtype=int)


def _trade_basket(prcSoFar, basket, pos):
    """Update positions for one correlated basket via z-score mean reversion."""
    logP = np.log(prcSoFar[basket, -BASKET_LOOKBACK:])
    demeaned = logP - logP.mean(axis=1, keepdims=True)
    spread = demeaned - demeaned.mean(axis=0, keepdims=True)

    mu = spread.mean(axis=1)
    sd = spread.std(axis=1)
    sd[sd < 1e-8] = 1e-8
    z = (spread[:, -1] - mu) / sd

    for k, inst in enumerate(basket):
        maxShares = int(BASKET_DLR_POS_LIMIT / prcSoFar[inst, -1])
        if z[k] > BASKET_ENTRY_Z:
            pos[inst] = -maxShares  # rich vs basket -> short
        elif z[k] < -BASKET_ENTRY_Z:
            pos[inst] = maxShares  # cheap vs basket -> long
        elif abs(z[k]) < BASKET_EXIT_Z:
            pos[inst] = 0
        # otherwise hold existing position to avoid churn


# ---------------------------------------------------------------------------
# Bollinger strategy on ALGO (column 0 in prices.txt — the SPECIAL instrument:
# 0.2bp commission, $100k position limit). ALGO is not in any basket, so the
# two strategies trade disjoint instruments.
# ---------------------------------------------------------------------------
BOLL_INST = 0

BOLL_LOOKBACK = 30
BOLL_ENTRY_Z = 0.75
BOLL_DIRECTION = -1  # +1 breakout/momentum, -1 mean reversion (fade the break)
# Kaufman Efficiency Ratio gate: ER < threshold = choppy regime (fade the band
# break), ER > threshold = trending regime (follow it). None disables the gate
# and BOLL_DIRECTION alone decides. Random-walk baseline ER is ~1/sqrt(N).
BOLL_ER_THRESHOLD = 0.3
BOLL_DLR_POS_LIMIT = 100_000


class BollingerStrategy:
    """Bollinger bands with an ER regime gate, exit when price crosses the
    middle band (= (upper + lower) / 2 = the SMA, since the bands are
    symmetric).

    Decision tree while price is outside a band (re-evaluated daily, so a
    position flips if the regime flips while still outside the band):
      ER < er_threshold  -> fade the break (short above upper, long below lower)
      ER > er_threshold  -> follow the trend (long above upper, short below lower)
    Inside the bands no new position is opened; an open position is held until
    price crosses the middle band against its entry side.

    er_threshold=None reverts to a fixed direction: +1 breakout/momentum,
    -1 mean reversion."""

    def __init__(self, lookback=BOLL_LOOKBACK, entry_z=BOLL_ENTRY_Z,
                 direction=BOLL_DIRECTION, er_threshold=BOLL_ER_THRESHOLD,
                 er_n=None, target_dlr=BOLL_DLR_POS_LIMIT):
        self.lookback = lookback
        self.entry_z = entry_z
        self.direction = direction
        self.er_threshold = er_threshold
        self.er_n = er_n if er_n is not None else lookback
        self.target_dlr = target_dlr
        self.currentPos = np.zeros(nInst, dtype=int)
        self.tradeDir = 0  # +1 long, -1 short, 0 flat
        self.entrySide = 0  # band the position entered through: +1 upper, -1 lower
        self.lastBands = (np.nan, np.nan, np.nan)  # (upper, middle, lower)
        self.lastER = np.nan

    def _bands(self, prcSoFar):
        """Bands over the past `lookback` days, excluding today — the breakout
        is judged against yesterday's channel, as in the Donchian strategy."""
        window = prcSoFar[BOLL_INST, -self.lookback - 1:-1]
        mid = window.mean()
        sd = window.std()
        return mid + self.entry_z * sd, mid, mid - self.entry_z * sd

    def _er(self, prcSoFar):
        """Kaufman Efficiency Ratio over the last er_n moves, including today:
        |net change| / sum(|daily changes|). 1 = perfectly straight trend,
        ~1/sqrt(er_n) = random walk, 0 = pure whipsaw."""
        window = prcSoFar[BOLL_INST, -self.er_n - 1:]
        denom = np.abs(np.diff(window)).sum()
        if denom < 1e-12:
            return 0.0
        return abs(window[-1] - window[0]) / denom

    def step(self, prcSoFar):
        nins, nt = prcSoFar.shape

        if nt < max(self.lookback, self.er_n) + 1:
            return np.zeros(nins, dtype=int)

        price = prcSoFar[BOLL_INST, -1]
        upper, middle, lower = self._bands(prcSoFar)
        self.lastBands = (upper, middle, lower)
        er = self._er(prcSoFar)
        self.lastER = er

        outSide = 1 if price > upper else (-1 if price < lower else 0)

        newPos = np.zeros(nins, dtype=int)

        if outSide != 0:
            if self.er_threshold is None:
                desired = outSide * self.direction
            else:
                desired = outSide if er > self.er_threshold else -outSide
            if desired != self.tradeDir:
                self.tradeDir = desired
                self.entrySide = outSide
                newPos[BOLL_INST] = desired * int(self.target_dlr / price)
            else:
                newPos[BOLL_INST] = self.currentPos[BOLL_INST]
        elif self.tradeDir != 0:
            if self.entrySide * (price - middle) < 0:
                # price crossed the middle band against the side the trade
                # entered through: exit (newPos stays flat)
                self.tradeDir = 0
                self.entrySide = 0
            else:
                newPos[BOLL_INST] = self.currentPos[BOLL_INST]

        self.currentPos = newPos
        return self.currentPos


_bollingerStrategy = BollingerStrategy()


# ---------------------------------------------------------------------------
# OLS pairs strategy on RRES/NAYO (columns 35 and 24 in prices.txt) — disjoint
# from the baskets and from ALGO.
# ---------------------------------------------------------------------------
PAIR = [35, 24]  # [y, x] = [RRES, NAYO]

PAIRS_LOOKBACK = 75
PAIRS_ENTRY_Z = 0.7
PAIRS_DLR_POS_LIMIT = 10_000


def _estimateBeta(logP):
    """OLS slope of log(RRES) on log(NAYO) over the window."""
    y, x = logP
    slope, _ = np.polyfit(x, y, 1)
    return slope


def _zscore(logP, beta):
    y, x = logP
    spread = y - beta * x
    sd = spread.std()
    if sd < 1e-12:
        return 0.0
    return (spread[-1] - spread.mean()) / sd


class PairsStrategy:
    def __init__(self, lookback=PAIRS_LOOKBACK, entry_z=PAIRS_ENTRY_Z,
                 target_dlr=PAIRS_DLR_POS_LIMIT):
        self.lookback = lookback
        self.entry_z = entry_z
        self.target_dlr = target_dlr
        self.currentPos = np.zeros(nInst, dtype=int)
        self.tradeDir = 0  # +1 long spread, -1 short spread, 0 flat
        self.entryBeta = None  # hedge ratio frozen at entry
        self.lastZ = np.nan  # most recent z-score, for diagnostics/plots

    def _pairShares(self, direction, beta, prices):
        """Spread weights are (1, -beta) on log prices, so each leg's dollar
        notional is proportional to its weight; scale so the largest leg is
        target_dlr."""
        weights = np.array([1.0, -beta])
        dollars = direction * self.target_dlr * weights / np.max(np.abs(weights))
        return np.round(dollars / prices).astype(int)

    def step(self, prcSoFar):
        nins, nt = prcSoFar.shape

        if nt < self.lookback + 1:
            return np.zeros(nins, dtype=int)

        prices = prcSoFar[PAIR, -1]
        logP = np.log(prcSoFar[PAIR, -self.lookback:])

        newPos = np.zeros(nins, dtype=int)

        if self.tradeDir == 0:
            beta = _estimateBeta(logP)
            z = _zscore(logP, beta)
            self.lastZ = z
            if abs(z) > self.entry_z:
                self.tradeDir = -1 if z > 0 else 1
                self.entryBeta = beta
                newPos[PAIR] = self._pairShares(self.tradeDir, beta, prices)
        else:
            z = _zscore(logP, self.entryBeta)
            self.lastZ = z
            if (self.tradeDir == -1 and z <= 0) or (self.tradeDir == 1 and z >= 0):
                # spread crossed its rolling mean: exit (newPos stays flat)
                self.tradeDir = 0
                self.entryBeta = None
            else:
                maxShares = (PAIRS_DLR_POS_LIMIT / prices).astype(int)
                if np.any(np.abs(self.currentPos[PAIR]) > maxShares):
                    # a leg would hit the $10k limit and get force-trimmed
                    # alone; rescale the whole pair to keep the hedge ratio
                    # intact
                    newPos[PAIR] = self._pairShares(
                        self.tradeDir, self.entryBeta, prices
                    )
                else:
                    newPos[PAIR] = self.currentPos[PAIR]

        self.currentPos = newPos
        return self.currentPos


_pairsStrategy = PairsStrategy()


def getMyPosition(prcSoFar):
    global currentPos
    nins, nt = prcSoFar.shape

    bollPos = _bollingerStrategy.step(prcSoFar)
    pairsPos = _pairsStrategy.step(prcSoFar)

    if nt < BASKET_LOOKBACK + 1:
        newPos = np.asarray(bollPos, dtype=int).copy()
        newPos[PAIR] = pairsPos[PAIR]
        currentPos = newPos
        return currentPos

    newPos = currentPos.copy()
    for basket in BASKETS:
        _trade_basket(prcSoFar, basket, newPos)

    newPos[BOLL_INST] = bollPos[BOLL_INST]
    newPos[PAIR] = pairsPos[PAIR]
    currentPos = newPos
    return currentPos
