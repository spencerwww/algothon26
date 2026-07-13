"""Parameter sweep of the Johansen spread strategy on the train set.

Train set = first 70% of prices.txt (350 of 500 days). The last 150 days are
held out and must only be touched once, for final validation of the chosen
parameters. All configs are scored over the same fixed window (day
MAX_LOOKBACK+1 onward) so cells are comparable regardless of their own
lookback.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb

from basket_spread.johansen import SpreadStrategy

PRICES_FILE = "./prices.txt"
TRAIN_FRAC = 0.7

LOOKBACKS = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
ENTRY_ZS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]

# scoring-function parameter and fee/limit setup copied from eval.py
SCORE_PARAM = 1.0
DEFAULT_COMM = 0.0001
INST0_COMM = 0.00002
DEFAULT_LIMIT = 10_000
INST0_LIMIT = 100_000


def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T


def score(mu, sigma, param=SCORE_PARAM):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    return mu * sr**2 / (sr**2 + param**2)


def backtest(strategy, prcHist, startDay):
    """eval.py's PnL loop against a SpreadStrategy instance.

    startDay is the first day getMyPosition-equivalent is called; PnL is
    scored for days startDay+1 .. nt, matching eval.py.
    """
    nInst, nt = prcHist.shape
    commRate = np.full(nInst, DEFAULT_COMM)
    commRate[0] = INST0_COMM
    dlrPosLimit = np.full(nInst, float(DEFAULT_LIMIT))
    dlrPosLimit[0] = float(INST0_LIMIT)

    cash, value, comm = 0.0, 0.0, 0.0
    curPos = np.zeros(nInst)
    todayPLL = []
    entries = 0
    prevDir = 0

    for t in range(startDay, nt + 1):
        prcSoFar = prcHist[:, :t]
        curPrices = prcSoFar[:, -1]
        if t < nt:
            newPosOrig = strategy.step(prcSoFar)
            if prevDir == 0 and strategy.tradeDir != 0:
                entries += 1
            prevDir = strategy.tradeDir
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)

        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos)
        comm = np.sum(dvolumes * commRate)
        curPos = np.array(newPos)
        todayPL = cash + curPos.dot(curPrices) - value
        value = cash + curPos.dot(curPrices)
        if t > startDay:
            todayPLL.append(todayPL)

    pll = np.array(todayPLL)
    mu, sd = pll.mean(), pll.std()
    sharpe = np.sqrt(250) * mu / sd if sd > 0 else 0.0
    return {
        "meanPL": mu,
        "stdPL": sd,
        "sharpe": sharpe,
        "score": score(mu, sd),
        "entries": entries,
    }


def runSweep():
    prcAll = loadPrices(PRICES_FILE)
    nTrainDays = int(prcAll.shape[1] * TRAIN_FRAC)
    prcTrain = prcAll[:, :nTrainDays]
    startDay = max(LOOKBACKS) + 1  # same scoring window for every config

    results = {}
    for lb in LOOKBACKS:
        for ez in ENTRY_ZS:
            strat = SpreadStrategy(lookback=lb, entry_z=ez)
            results[(lb, ez)] = backtest(strat, prcTrain, startDay)
            r = results[(lb, ez)]
            print(
                f"LOOKBACK={lb:3d} ENTRY_Z={ez:4.2f}  score={r['score']:7.2f}  "
                f"sharpe={r['sharpe']:5.2f}  meanPL={r['meanPL']:7.2f}  "
                f"entries={r['entries']:2d}"
            )
    return results, nTrainDays, startDay


# ---------------------------------------------------------------------------
# heatmap rendering (light mode, dataviz reference palette)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"

# diverging blue <-> red with neutral gray midpoint (polarity around 0)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_red_blue",
    ["#8f2f2e", "#e34948", "#f0efec", "#3987e5", "#0d366b"],
)
# sequential single-hue blue ramp (magnitude)
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#5598e7", "#256abf", "#0d366b"],
)


def _cellInk(rgba):
    r, g, b = to_rgb(rgba)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#ffffff" if lum < 0.45 else INK


def _drawPanel(ax, grid, title, cmap, norm, fmt):
    nRows, nCols = grid.shape
    ax.set_facecolor(SURFACE)
    ax.pcolormesh(
        grid, cmap=cmap, norm=norm, edgecolors=SURFACE, linewidth=2
    )
    for i in range(nRows):
        for j in range(nCols):
            color = cmap(norm(grid[i, j]))
            ax.text(
                j + 0.5, i + 0.5, fmt(grid[i, j]),
                ha="center", va="center", fontsize=9,
                color=_cellInk(color),
            )
    ax.set_title(title, color=INK, fontsize=11, pad=10, loc="left")
    ax.set_xticks(np.arange(nCols) + 0.5, [str(lb) for lb in LOOKBACKS])
    ax.set_yticks(np.arange(nRows) + 0.5, [f"{z:g}" for z in ENTRY_ZS])
    ax.set_xlabel("LOOKBACK (days)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("ENTRY_Z", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plotHeatmaps(results, nTrainDays, startDay, outfile="param_heatmap.png"):
    shape = (len(ENTRY_ZS), len(LOOKBACKS))
    scores = np.zeros(shape)
    sharpes = np.zeros(shape)
    entries = np.zeros(shape)
    for i, ez in enumerate(ENTRY_ZS):
        for j, lb in enumerate(LOOKBACKS):
            r = results[(lb, ez)]
            scores[i, j] = r["score"]
            sharpes[i, j] = r["sharpe"]
            entries[i, j] = r["entries"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.patch.set_facecolor(SURFACE)

    sLim = max(abs(scores).max(), 1e-9)
    _drawPanel(
        axes[0], scores, "Competition score",
        DIV_CMAP, TwoSlopeNorm(vcenter=0.0, vmin=-sLim, vmax=sLim),
        lambda v: f"{v:.1f}",
    )
    shLim = max(abs(sharpes).max(), 1e-9)
    _drawPanel(
        axes[1], sharpes, "Annualised Sharpe",
        DIV_CMAP, TwoSlopeNorm(vcenter=0.0, vmin=-shLim, vmax=shLim),
        lambda v: f"{v:.2f}",
    )
    _drawPanel(
        axes[2], entries, "Round-trip entries",
        SEQ_CMAP, plt.Normalize(vmin=0, vmax=max(entries.max(), 1)),
        lambda v: f"{int(v)}",
    )

    scoredDays = nTrainDays - startDay
    fig.suptitle(
        f"Johansen spread strategy — train set (first {nTrainDays} days, "
        f"{scoredDays} scored)",
        color=INK, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(outfile, dpi=200, facecolor=SURFACE)
    print(f"\nheatmap saved to {outfile}")


def plotSurface(results, nTrainDays, startDay, outfile="param_surface.html"):
    """Interactive 3D score surface (rotate/zoom/hover), self-contained HTML."""
    import plotly.graph_objects as go

    shape = (len(ENTRY_ZS), len(LOOKBACKS))
    scores = np.zeros(shape)
    custom = np.zeros(shape + (3,))  # sharpe, meanPL, entries per cell
    for i, ez in enumerate(ENTRY_ZS):
        for j, lb in enumerate(LOOKBACKS):
            r = results[(lb, ez)]
            scores[i, j] = r["score"]
            custom[i, j] = (r["sharpe"], r["meanPL"], r["entries"])

    sLim = max(abs(scores).max(), 1e-9)
    fig = go.Figure(
        go.Surface(
            x=LOOKBACKS,
            y=ENTRY_ZS,
            z=scores,
            customdata=custom,
            colorscale=[
                [0.0, "#8f2f2e"], [0.25, "#e34948"], [0.5, "#f0efec"],
                [0.75, "#3987e5"], [1.0, "#0d366b"],
            ],
            cmin=-sLim, cmax=sLim,  # symmetric so the gray midpoint sits at 0
            colorbar=dict(title="score", tickfont=dict(color=INK_MUTED)),
            contours={"z": {"show": True, "usecolormap": True,
                            "project_z": True, "highlightcolor": "#0b0b0b"}},
            hovertemplate=(
                "LOOKBACK %{x}<br>ENTRY_Z %{y}<br>"
                "score %{z:.2f}<br>sharpe %{customdata[0]:.2f}<br>"
                "meanPL %{customdata[1]:.2f}<br>"
                "entries %{customdata[2]:.0f}<extra></extra>"
            ),
        )
    )
    scoredDays = nTrainDays - startDay
    fig.update_layout(
        title=dict(
            text=(
                f"Johansen spread strategy — competition score, train set "
                f"(first {nTrainDays} days, {scoredDays} scored)"
            ),
            font=dict(color=INK, size=15),
        ),
        scene=dict(
            xaxis_title="LOOKBACK (days)",
            yaxis_title="ENTRY_Z",
            zaxis_title="score",
            xaxis=dict(color=INK_SECONDARY),
            yaxis=dict(color=INK_SECONDARY),
            zaxis=dict(color=INK_SECONDARY),
        ),
        paper_bgcolor=SURFACE,
        font=dict(color=INK_SECONDARY),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(outfile, include_plotlyjs=True)
    print(f"3D surface saved to {outfile}")


if __name__ == "__main__":
    results, nTrainDays, startDay = runSweep()

    print("\n=== ranked by score ===")
    ranked = sorted(results.items(), key=lambda kv: kv[1]["score"], reverse=True)
    for (lb, ez), r in ranked:
        print(
            f"LOOKBACK={lb:3d} ENTRY_Z={ez:4.2f}  score={r['score']:7.2f}  "
            f"sharpe={r['sharpe']:5.2f}  meanPL={r['meanPL']:7.2f}  "
            f"entries={r['entries']:2d}"
        )

    rows = [
        {"lookback": lb, "entry_z": ez, **r}
        for (lb, ez), r in sorted(results.items())
    ]
    pd.DataFrame(rows).to_csv("param_sweep_results.csv", index=False)
    print("results saved to param_sweep_results.csv")

    plotHeatmaps(results, nTrainDays, startDay)
    plotSurface(results, nTrainDays, startDay)
