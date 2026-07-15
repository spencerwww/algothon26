import numpy as np

nInst = 51

# pair: NWIG regressed on AENO (columns in prices.txt)
PAIR = [35, 24]  # [y, x] = [RRES, NAYO]

LOOKBACK = 75
ENTRY_Z = 0.7
DLR_POS_LIMIT = 10_000


def _estimateBeta(logP):
    """OLS slope of log(NWIG) on log(AENO) over the window."""
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
    def __init__(self, lookback=LOOKBACK, entry_z=ENTRY_Z, target_dlr=DLR_POS_LIMIT):
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
                maxShares = (DLR_POS_LIMIT / prices).astype(int)
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


_defaultStrategy = PairsStrategy()


def getMyPosition(prcSoFar):
    return _defaultStrategy.step(prcSoFar)
