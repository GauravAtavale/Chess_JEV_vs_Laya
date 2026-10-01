"""
Round-robin of chess players (Jev, base and fine-tuned Laya and CLM, random), for Colab.

    import tournament as T
    T.run_all()                                        # plays every match (skips finished ones)
    T.show_table()                                     # results of all matches
    picks = T.list_games(termination="checkmate")      # find games (also: game=3 or game=[3, 7]) ...
    T.export_games(picks)                              # ... and save them to one PGN to replay locally

Each match is saved in runs/tournament/NN_<a>_vs_<b>/ (games.pgn, moves.jsonl, results.jsonl,
console.txt, summary.json). Per-move output is not printed, only one line per game.
"""
import glob
import json
import os
import subprocess
import sys
import urllib.request

import chess.pgn

OUT = "runs/tournament"          # all matches are saved here, one folder each
GAMES = 10
RULES = True                     # give rule facts to Jev and base Laya too (fine-tuned models always get them)
LAYA_V2 = "runs/laya_chess_v2"
MAX_REQUESTS = 30000             # safety cap on Jev API requests per match (about $1 at most)
CLM_URL = os.environ.get("CLM_URL", "http://127.0.0.1:8700")
JEV_PRICE = 0.042 / 1e6          # dollars per Jev input token

# name: (kind, args when playing as --player, args when playing as --opponent)
PLAYERS = {
    "jev":       ("jev",    [], []),
    "random":    ("random", [], []),
    "laya_base": ("laya",   ["--player-laya-path", "base"], ["--opponent-laya-path", "base"]),
    "laya_v2":   ("laya",   ["--player-laya-path", LAYA_V2], ["--opponent-laya-path", LAYA_V2]),
    "clm_base":  ("clm",    [], []),
    "clm_ft":    ("clm",    ["--player-clm-model", "chess"], ["--opponent-clm-model", "chess"]),
}
MATCHES = [("jev", "random"), ("laya_base", "random"), ("clm_base", "random"),
           ("jev", "laya_base"), ("jev", "clm_base"), ("laya_base", "clm_base"),
           ("jev", "laya_v2"), ("jev", "clm_ft"), ("laya_v2", "clm_ft")]


def server_up(url):
    try:
        urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5)
        return True
    except Exception:
        return False


def run_all(matches=None, games=None):
    """Play every match in MATCHES (or the given list). Finished matches are skipped, so after a
    disconnect you can simply run this again."""
    matches = matches or MATCHES
    games = games or GAMES
    clm_ok = server_up(CLM_URL)
    jev_ok = bool(os.environ.get("TYPESAFE_API_KEY"))
    print(f"CLM server: {'running' if clm_ok else 'NOT running'} | Jev key: {'loaded' if jev_ok else 'MISSING'}\n")
    for i, (a, b) in enumerate(matches, 1):
        folder = f"{OUT}/{MATCHES.index((a, b)) + 1 if (a, b) in MATCHES else i:02d}_{a}_vs_{b}"
        title = f"[{i}/{len(matches)}] {a} vs {b}"
        kinds = {PLAYERS[a][0], PLAYERS[b][0]}
        if os.path.exists(f"{folder}/summary.json"):
            print(f"{title}: already done, skipping")
            continue
        if "clm" in kinds and not clm_ok:
            print(f"{title}: SKIPPED (start the CLM servers first)")
            continue
        if "jev" in kinds and not jev_ok:
            print(f"{title}: SKIPPED (load TYPESAFE_API_KEY first)")
            continue
        cmd = [sys.executable, "laya_arena.py", "--games", str(games),
               "--player", PLAYERS[a][0], *PLAYERS[a][1],
               "--opponent", PLAYERS[b][0], *PLAYERS[b][2],
               "--quiet", "--log-dir", folder, "--max-requests", str(MAX_REQUESTS),
               "--clm-url", CLM_URL] + (["--rules"] if RULES else [])
        print(f"{title} ...", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail = []
        for line in proc.stdout:   # one line per game + the summary; everything is in console.txt
            tail = (tail + [line])[-25:]
            if line.startswith(("Game ", "Wins", "How games", "Jev API", "Stopped")):
                print("   " + line.rstrip(), flush=True)
        if proc.wait() != 0:
            print("   FAILED. Last lines:\n" + "".join("     " + l for l in tail))
    print("\nAll done. Results are in", OUT)


def show_table():
    """One row per match. W/D/L, score and material are from the first-named side's view."""
    print(f"{'match':26s} {'W':>3s} {'D':>3s} {'L':>3s} {'unf':>4s} {'score':>6s} {'material':>9s} "
          f"{'Jev $':>6s}  how games ended")
    for folder in sorted(glob.glob(f"{OUT}/*/")):
        path = os.path.join(folder, "summary.json")
        name = os.path.basename(folder.rstrip("/"))
        if not os.path.exists(path):
            print(f"{name:26s} (not finished)")
            continue
        s = json.load(open(path))
        t, n = s["tally"], s["games_finished"]
        decided = t["win"] + t["draw"] + t["loss"]
        unfinished = n - decided
        score = f"{(t['win'] + 0.5 * t['draw']) / decided:6.0%}" if decided else f"{'-':>6s}"
        ends = {}
        for g in s["games"]:
            ends[g["termination"]] = ends.get(g["termination"], 0) + 1
        ends_txt = ", ".join(f"{k} {v}" for k, v in sorted(ends.items(), key=lambda kv: -kv[1]))
        cost = s.get("api_input_tokens", 0) * JEV_PRICE
        print(f"{name:26s} {t['win']:3d} {t['draw']:3d} {t['loss']:3d} {unfinished:4d} {score} "
              f"{s['average_material_edge']:+9.1f} {cost:6.2f}  {ends_txt}")
    print("\nW/D/L, score and material are from the first-named side's point of view.\n"
          "unf = games that reached the move limit (not counted in the score).")


def list_games(match="", result=None, termination=None, game=None):
    """match: part of a folder name ("jev_vs_clm_ft", "07"); result: "win" / "draw" / "loss"
    (for the first-named side); termination: e.g. "checkmate", "repetition", "stalemate";
    game: one game number (3) or several ([3, 7]) within the match."""
    games = None if game is None else ({game} if isinstance(game, int) else set(game))
    found = []
    for folder in sorted(glob.glob(f"{OUT}/*{match}*/")):
        name = os.path.basename(folder.rstrip("/"))
        if not os.path.exists(folder + "results.jsonl"):
            continue
        for r in map(json.loads, open(folder + "results.jsonl")):
            if result and r["result_for_player"] != result:
                continue
            if termination and termination not in r["termination"]:
                continue
            if games is not None and r["game"] not in games:
                continue
            found.append((name, r["game"]))
            print(f"{name:26s} game {r['game']:2d}: {r['white']} vs {r['black']} -> {r['pgn_result']} "
                  f"({r['termination']}, {r['plies']} half-moves, material {r['material_edge_for_player']:+d})")
    print(f"\n{len(found)} game(s)")
    return found


def export_games(picks, out=f"{OUT}/picked_games.pgn"):
    """picks: list of (match folder name, game number), e.g. what list_games returns.
    Writes them into one PGN file; replay locally with: python replay_game.py picked_games.pgn"""
    with open(out, "w") as f:
        for name, number in picks:
            with open(f"{OUT}/{name}/games.pgn") as src:
                for _ in range(number):
                    game = chess.pgn.read_game(src)
            game.headers["Event"] = f"{name} game {number}"
            print(game, file=f, end="\n\n")
    print(f"Saved {len(picks)} game(s) to {out}")
