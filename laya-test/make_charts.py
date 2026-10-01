"""
Draw the README charts for the tournament and the move-quality evaluation.

    python make_charts.py                       # uses the results below, writes images/*.png
    python make_charts.py --summaries runs/tournament --out images
    python make_charts.py --no-highlight        # without the circled results

To circle a different result, edit HIGHLIGHT below.

With --summaries, results are read from the tournament's summary.json files instead of
the numbers typed in below.
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Patch

# ------------------------------------------------------------------ data
# (first-named, second-named, wins, draws, losses, unfinished, {termination: count})
MATCHES = [
    ("Jev", "Random", 9, 1, 0, 0, {"checkmate": 9, "insufficient_material": 1}),
    ("Laya (base)", "Random", 3, 3, 0, 4,
     {"move limit": 4, "checkmate": 3, "insufficient_material": 2, "stalemate": 1}),
    ("CLM (base)", "Random", 0, 2, 4, 4,
     {"checkmate": 4, "move limit": 4, "stalemate": 1, "threefold_repetition": 1}),
    ("Jev", "Laya (base)", 10, 0, 0, 0, {"checkmate": 10}),
    ("Jev", "CLM (base)", 10, 0, 0, 0, {"checkmate": 10}),
    ("Laya (base)", "CLM (base)", 7, 3, 0, 0, {"checkmate": 7, "threefold_repetition": 3}),
    ("Jev", "Laya v2", 3, 6, 1, 0,
     {"insufficient_material": 5, "checkmate": 4, "fifty_moves": 1}),
    ("Jev", "CLM (fine-tuned)", 1, 5, 4, 0,
     {"checkmate": 5, "insufficient_material": 4, "fifty_moves": 1}),
    ("Laya v2", "CLM (fine-tuned)", 1, 8, 1, 0,
     {"threefold_repetition": 7, "checkmate": 2, "insufficient_material": 1}),
]
# winning chance lost per move on the same 300 held-out positions (lower is better)
EVAL = [("Random move", 0.23087), ("CLM (base)", 0.21924), ("Laya, 1st fine-tune", 0.14596),
        ("Laya v2", 0.10330), ("CLM (fine-tuned)", 0.09837)]

# ------------------------------------------------------------------ highlights
# The result to circle in each chart (set any entry to None to skip it). Names must match
# the player names used above.
HIGHLIGHT = {
    # head-to-head grid: (row player, column player, note)
    "head_to_head": ("CLM (fine-tuned)", "Jev", "Fine-tuned CLM\nbeats Jev"),
    # tournament bars: ((first-named, second-named), note)
    "results": (("Jev", "CLM (fine-tuned)"),
                "Highlighted: fine-tuned CLM scored 65% against Jev (4 wins, 5 draws, 1 loss)"),
    # move-quality bars: (model, note)
    "move_quality": ("CLM (fine-tuned)", "best"),
}
HIGHLIGHT_COLOR = "#a8322a"

NAMES = {"jev": "Jev", "random": "Random", "laya_base": "Laya (base)", "laya_v2": "Laya v2",
         "clm_base": "CLM (base)", "clm_ft": "CLM (fine-tuned)"}

# ------------------------------------------------------------------ style (vintage, readable on light and dark pages)
PAPER, INK, MUTED, GRID = "#f4ecdb", "#33261a", "#7a6650", "#e2d5bb"
WIN, DRAW, LOSS, UNF = "#5d8a68", "#c3b393", "#b4543f", "#e6dccb"
END_COLORS = {"checkmate": "#8a3b2e", "threefold_repetition": "#c99a4a",
              "insufficient_material": "#9fb39a", "stalemate": "#7d8fa8",
              "fifty_moves": "#b39bc0", "move limit": "#d8cdb7"}
END_LABELS = {"checkmate": "Checkmate", "threefold_repetition": "Threefold repetition",
              "insufficient_material": "Insufficient material", "stalemate": "Stalemate",
              "fifty_moves": "Fifty-move rule", "move limit": "Move limit (unfinished)"}


def style():
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": INK, "font.size": 11,
        "font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.titlesize": 14,
    })


def load_summaries(folder):
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*", "summary.json"))):
        s = json.load(open(path))
        name = os.path.basename(os.path.dirname(path)).split("_", 1)[1]
        a, b = name.split("_vs_")
        t = s["tally"]
        unf = s["games_finished"] - t["win"] - t["draw"] - t["loss"]
        ends = {}
        for g in s["games"]:
            ends[g["termination"]] = ends.get(g["termination"], 0) + 1
        out.append((NAMES.get(a, a), NAMES.get(b, b), t["win"], t["draw"], t["loss"], unf, ends))
    return out


# ------------------------------------------------------------------ charts
def chart_results(matches, path):
    """Stacked bars: wins / draws / losses / unfinished for each match, with the score."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    labels = [f"{a}  vs  {b}" for a, b, *_ in matches][::-1]
    rows = matches[::-1]
    for i, (a, b, w, d, l, u, _) in enumerate(rows):
        left = 0
        for value, color in ((w, WIN), (d, DRAW), (l, LOSS), (u, UNF)):
            if value:
                ax.barh(i, value, left=left, color=color, height=0.62, edgecolor=PAPER, linewidth=1.5)
                ax.text(left + value / 2, i, str(value), ha="center", va="center", fontsize=10,
                        color="white" if color in (WIN, LOSS) else INK, fontweight="bold")
            left += value
        decided = w + d + l
        score = f"{(w + 0.5 * d) / decided:.0%}" if decided else "–"
        ax.text(10.35, i, score, va="center", ha="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("games (10 per match)", color=MUTED)
    h = HIGHLIGHT.get("results") if HIGHLIGHT else None
    if h:
        pair, note = h
        for i, (a, b, *_) in enumerate(rows):
            if (a, b) == tuple(pair):
                ax.add_patch(FancyBboxPatch((-0.05, i - 0.43), 11.3, 0.86,
                                            boxstyle="round,pad=0,rounding_size=0.35",
                                            fill=False, edgecolor=HIGHLIGHT_COLOR, linewidth=2.6,
                                            clip_on=False, zorder=5, mutation_aspect=0.4))
                fig.text(0.01, 0.012, note, fontsize=10, color=HIGHLIGHT_COLOR, fontweight="bold")
    ax.text(10.35, len(rows) - 0.35, "score", fontsize=10, color=MUTED, va="bottom")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Tournament results (from the first-named side's point of view)", loc="left", pad=28)
    ax.legend(handles=[Patch(color=WIN, label="Win"), Patch(color=DRAW, label="Draw"),
                       Patch(color=LOSS, label="Loss"), Patch(color=UNF, label="Unfinished")],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_endings(matches, path):
    """Stacked bars: how each match's games ended."""
    order = ["checkmate", "threefold_repetition", "insufficient_material", "stalemate",
             "fifty_moves", "move limit"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    rows = matches[::-1]
    for i, (a, b, *_, ends) in enumerate(rows):
        left = 0
        for key in order:
            n = ends.get(key, 0)
            if n:
                ax.barh(i, n, left=left, color=END_COLORS[key], height=0.62, edgecolor=PAPER, linewidth=1.5)
                ax.text(left + n / 2, i, str(n), ha="center", va="center", fontsize=10,
                        color="white" if key in ("checkmate", "stalemate") else INK, fontweight="bold")
                left += n
    ax.set_yticks(range(len(rows)), [f"{a}  vs  {b}" for a, b, *_ in rows])
    ax.set_xlim(0, 10)
    ax.set_xticks(range(0, 11, 2))
    ax.set_xlabel("games", color=MUTED)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("How the games ended", loc="left", pad=46)
    used = [k for k in order if any(m[-1].get(k) for m in matches)]
    ax.legend(handles=[Patch(color=END_COLORS[k], label=END_LABELS[k]) for k in used],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_crosstable(matches, path):
    """Head-to-head grid: each cell is the row player's score against the column player."""
    players = ["Jev", "CLM (fine-tuned)", "Laya v2", "Laya (base)", "CLM (base)", "Random"]
    idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    grid = [[None] * n for _ in range(n)]
    detail = [[""] * n for _ in range(n)]
    for a, b, w, d, l, u, _ in matches:
        decided = w + d + l
        if not decided or a not in idx or b not in idx:
            continue
        s = (w + 0.5 * d) / decided
        grid[idx[a]][idx[b]], grid[idx[b]][idx[a]] = s, 1 - s
        detail[idx[a]][idx[b]] = f"{w}-{d}-{l}"
        detail[idx[b]][idx[a]] = f"{l}-{d}-{w}"
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    cmap = plt.get_cmap("BrBG")
    for r in range(n):
        for c in range(n):
            if r == c:
                color, txt, sub = GRID, "", ""
            elif grid[r][c] is None:
                color, txt, sub = PAPER, "·", ""
            else:
                v = grid[r][c]
                color = cmap(0.18 + 0.64 * v)
                txt, sub = f"{v:.0%}", detail[r][c]
            ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1, facecolor=color, edgecolor=PAPER, lw=3))
            rgb = plt.matplotlib.colors.to_rgb(color)
            dark = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] < 0.56   # dark cell -> white text
            ax.text(c + 0.5, n - 1 - r + 0.56, txt, ha="center", va="center", fontsize=13,
                    fontweight="bold", color="white" if dark else INK)
            ax.text(c + 0.5, n - 1 - r + 0.27, sub, ha="center", va="center", fontsize=8.5,
                    color="#f4ecdb" if dark else MUTED)
    h = HIGHLIGHT.get("head_to_head") if HIGHLIGHT else None
    if h and h[0] in idx and h[1] in idx:
        r, c = idx[h[0]], idx[h[1]]
        cx, cy = c + 0.5, n - 1 - r + 0.5
        ax.add_patch(Ellipse((cx, cy), 1.04, 1.0, fill=False, edgecolor=HIGHLIGHT_COLOR,
                             linewidth=3, clip_on=False, zorder=5))
        # put the note in the emptiest cells of the same row, pointing back at the circle
        empty = [cc for cc in range(n) if cc != r and grid[r][cc] is None]
        tx = (min(empty) + max(empty) + 1) / 2 if empty else n + 0.6
        ax.annotate(h[2], xy=(cx + 0.53, cy + 0.12), xytext=(tx, cy), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=HIGHLIGHT_COLOR, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.45", fc=PAPER, ec=HIGHLIGHT_COLOR, lw=1.5),
                    arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT_COLOR, lw=2,
                                    shrinkA=4, shrinkB=2, connectionstyle="arc3,rad=0.15"))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([c + 0.5 for c in range(n)], players, rotation=30, ha="right")
    ax.set_yticks([n - 1 - r + 0.5 for r in range(n)], players)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Head to head: row player's score vs column player", loc="left", pad=12)
    ax.text(0, -1.25, "Cell text: score, then wins-draws-losses for the row player.  "
                      "· = not played.", fontsize=9, color=MUTED, transform=ax.transData)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def chart_eval(path):
    """Winning chance lost per move on the same 300 positions."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = [n for n, _ in EVAL][::-1]
    vals = [v for _, v in EVAL][::-1]
    colors = [MUTED if n == "Random move" else (WIN if v < 0.12 else "#c99a4a") for n, v in zip(names, vals)]
    ax.barh(range(len(vals)), vals, color=colors, height=0.6)
    for i, v in enumerate(vals):
        cut = 1 - v / EVAL[0][1]
        note = "" if names[i] == "Random move" else f"   ({cut:.0%} better than random)"
        ax.text(v + 0.004, i, f"{v:.3f}{note}", va="center", fontsize=10, color=INK)
    ax.axvline(EVAL[0][1], color=MUTED, linestyle="--", linewidth=1)
    h = HIGHLIGHT.get("move_quality") if HIGHLIGHT else None
    if h and h[0] in names:
        i = names.index(h[0])
        ax.add_patch(FancyBboxPatch((-0.005, i - 0.42), vals[i] + 0.114, 0.84,
                                    boxstyle="round,pad=0,rounding_size=0.012",
                                    fill=False, edgecolor=HIGHLIGHT_COLOR, linewidth=2.6,
                                    clip_on=False, zorder=5, mutation_aspect=18))
    ax.set_yticks(range(len(vals)), names)
    ax.set_xlim(0, 0.33)
    ax.set_xlabel("winning chance lost per move vs Stockfish's best (lower is better)", color=MUTED)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="y", pad=14)
    ax.set_title("Move quality on the same 300 held-out positions", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summaries", default=None, help="tournament folder with NN_a_vs_b/summary.json")
    ap.add_argument("--out", default="images")
    ap.add_argument("--no-highlight", action="store_true", help="draw the charts without circles")
    args = ap.parse_args()
    if args.no_highlight:
        HIGHLIGHT.clear()
    style()
    matches = load_summaries(args.summaries) if args.summaries else MATCHES
    os.makedirs(args.out, exist_ok=True)
    chart_results(matches, os.path.join(args.out, "tournament_results.png"))
    chart_crosstable(matches, os.path.join(args.out, "head_to_head.png"))
    chart_endings(matches, os.path.join(args.out, "how_games_ended.png"))
    chart_eval(os.path.join(args.out, "move_quality.png"))
    print(f"Charts written to {args.out}/")


if __name__ == "__main__":
    main()