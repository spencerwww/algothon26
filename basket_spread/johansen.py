import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

nInst = 51

# basket: AENO, DUCT, MHRM, NWIG (columns in prices.txt)
BASKET = [1, 9, 49, 20]

LOOKBACK = 35
ENTRY_Z = 0.75
DLR_POS_LIMIT = 10_000


def _estimateBeta(logP):
    """First Johansen eigenvector over the window, scaled to max|component| = 1.

    The eigenvector's sign is arbitrary, so pin the first component positive
    to keep the z-score's sign consistent from day to day.
    """
    res = coint_johansen(logP.T, 0, 1)
    beta = res.evec[:, 0]
    beta = beta / np.max(np.abs(beta))
    if beta[0] < 0:
        beta = -beta
    return beta


def _zscore(logP, beta):
    spread = beta @ logP
    sd = spread.std()
    if sd < 1e-12:
        return 0.0
    return (spread[-1] - spread.mean()) / sd


class SpreadStrategy:
    def __init__(self, lookback=LOOKBACK, entry_z=ENTRY_Z, target_dlr=DLR_POS_LIMIT):
        self.lookback = lookback
        self.entry_z = entry_z
        self.target_dlr = target_dlr
        self.currentPos = np.zeros(nInst, dtype=int)
        self.tradeDir = 0  # +1 long spread, -1 short spread, 0 flat
        self.entryBeta = None  # cointegrating vector frozen at entry

    def _basketShares(self, direction, beta, prices):
        """Betas weight log prices, so each leg's dollar notional is
        proportional to beta; scale so the largest leg is target_dlr."""
        dollars = direction * self.target_dlr * beta
        return np.round(dollars / prices).astype(int)

    def step(self, prcSoFar):
        nins, nt = prcSoFar.shape

        if nt < self.lookback + 1:
            return np.zeros(nins, dtype=int)

        prices = prcSoFar[BASKET, -1]
        logP = np.log(prcSoFar[BASKET, -self.lookback:])

        newPos = np.zeros(nins, dtype=int)

        if self.tradeDir == 0:
            beta = _estimateBeta(logP)
            z = _zscore(logP, beta)
            if abs(z) > self.entry_z:
                self.tradeDir = -1 if z > 0 else 1
                self.entryBeta = beta
                newPos[BASKET] = self._basketShares(self.tradeDir, beta, prices)
        else:
            z = _zscore(logP, self.entryBeta)
            if (self.tradeDir == -1 and z <= 0) or (self.tradeDir == 1 and z >= 0):
                # spread crossed its rolling mean: exit (newPos stays flat)
                self.tradeDir = 0
                self.entryBeta = None
            else:
                maxShares = (DLR_POS_LIMIT / prices).astype(int)
                if np.any(np.abs(self.currentPos[BASKET]) > maxShares):
                    # a leg would hit the $10k limit and get force-trimmed
                    # alone; rescale the whole basket to keep the hedge
                    # ratios intact
                    newPos[BASKET] = self._basketShares(
                        self.tradeDir, self.entryBeta, prices
                    )
                else:
                    newPos[BASKET] = self.currentPos[BASKET]

        self.currentPos = newPos
        return self.currentPos


_defaultStrategy = SpreadStrategy()


def getMyPosition(prcSoFar):
    return _defaultStrategy.step(prcSoFar)
