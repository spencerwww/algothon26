import numpy as np

nInst = 51

# instrument traded: ALGO (column 0 in prices.txt — the SPECIAL instrument:
# 0.2bp commission, $100k position limit)
INST = 0

LOOKBACK = 20
DLR_POS_LIMIT = 100_000


class DonchianStrategy:
    def __init__(self, lookback=LOOKBACK, target_dlr=DLR_POS_LIMIT):
        self.lookback = lookback
        self.target_dlr = target_dlr
        self.currentPos = np.zeros(nInst, dtype=int)
        self.tradeDir = 0  # +1 long, -1 short, 0 flat
        self.lastBands = (np.nan, np.nan, np.nan)  # (upper, middle, lower)

    def _bands(self, prcSoFar):
        """Channel over the past `lookback` days, excluding today — otherwise
        today's price could never break above its own high."""
        window = prcSoFar[INST, -self.lookback - 1:-1]
        upper, lower = window.max(), window.min()
        return upper, (upper + lower) / 2, lower

    def step(self, prcSoFar):
        nins, nt = prcSoFar.shape

        if nt < self.lookback + 1:
            return np.zeros(nins, dtype=int)

        price = prcSoFar[INST, -1]
        upper, middle, lower = self._bands(prcSoFar)
        self.lastBands = (upper, middle, lower)

        newPos = np.zeros(nins, dtype=int)

        if self.tradeDir == 0:
            if price > upper:
                self.tradeDir = 1
            elif price < lower:
                self.tradeDir = -1
            if self.tradeDir != 0:
                newPos[INST] = self.tradeDir * int(self.target_dlr / price)
        else:
            crossed = (self.tradeDir == 1 and price < middle) or (
                self.tradeDir == -1 and price > middle
            )
            if crossed:
                # price crossed the middle band: exit (newPos stays flat)
                self.tradeDir = 0
            else:
                newPos[INST] = self.currentPos[INST]

        self.currentPos = newPos
        return self.currentPos


_defaultStrategy = DonchianStrategy()


def getMyPosition(prcSoFar):
    return _defaultStrategy.step(prcSoFar)
