import numpy as np

nInst = 51

# instrument traded: ALGO (column 0 in prices.txt — the SPECIAL instrument:
# 0.2bp commission, $100k position limit)
INST = 0

LOOKBACK = 30
ENTRY_Z = 0.75
DIRECTION = -1  # +1 breakout/momentum, -1 mean reversion (fade the break)
# Kaufman Efficiency Ratio gate: ER < threshold = choppy regime (fade the band
# break), ER > threshold = trending regime (follow it). None disables the gate
# and DIRECTION alone decides. Random-walk baseline ER is ~1/sqrt(N).
ER_THRESHOLD = 0.3
DLR_POS_LIMIT = 100_000


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

    def __init__(self, lookback=LOOKBACK, entry_z=ENTRY_Z, direction=DIRECTION,
                 er_threshold=ER_THRESHOLD, er_n=None, target_dlr=DLR_POS_LIMIT):
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
        window = prcSoFar[INST, -self.lookback - 1:-1]
        mid = window.mean()
        sd = window.std()
        return mid + self.entry_z * sd, mid, mid - self.entry_z * sd

    def _er(self, prcSoFar):
        """Kaufman Efficiency Ratio over the last er_n moves, including today:
        |net change| / sum(|daily changes|). 1 = perfectly straight trend,
        ~1/sqrt(er_n) = random walk, 0 = pure whipsaw."""
        window = prcSoFar[INST, -self.er_n - 1:]
        denom = np.abs(np.diff(window)).sum()
        if denom < 1e-12:
            return 0.0
        return abs(window[-1] - window[0]) / denom

    def step(self, prcSoFar):
        nins, nt = prcSoFar.shape

        if nt < max(self.lookback, self.er_n) + 1:
            return np.zeros(nins, dtype=int)

        price = prcSoFar[INST, -1]
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
                newPos[INST] = desired * int(self.target_dlr / price)
            else:
                newPos[INST] = self.currentPos[INST]
        elif self.tradeDir != 0:
            if self.entrySide * (price - middle) < 0:
                # price crossed the middle band against the side the trade
                # entered through: exit (newPos stays flat)
                self.tradeDir = 0
                self.entrySide = 0
            else:
                newPos[INST] = self.currentPos[INST]

        self.currentPos = newPos
        return self.currentPos


_defaultStrategy = BollingerStrategy()


def getMyPosition(prcSoFar):
    return _defaultStrategy.step(prcSoFar)
