"""
Generate Stockfish-labelled training data for fine-tuning Laya on chess.

Positions come from games that Stockfish plays against itself with added
randomness (a few random opening moves, occasional random moves, and sampling
among good moves), so the data covers openings, middlegames and endgames,
including the messy positions a weak opponent creates.

For every recorded position, Stockfish evaluates EVERY legal move (MultiPV)
and converts each score to an expected score for the side to move:
    1.0 = certain win, 0.5 = draw, 0.0 = certain loss.
That number is the training target for Laya's yes/no question
"Will the side to move win the game after playing the candidate move?".

    brew install stockfish
    python make_chess_dataset.py --positions 20000
    python make_chess_dataset.py --positions 200 --out smoke.jsonl     # quick test

The output is JSON lines, one position per line. Re-running with the same --out
resumes where it stopped. Games are split 90/5/5 into train/val/test by game,
so positions from one game never land in two splits.
"""
import argparse
import atexit
import collections
import json
import math
import multiprocessing as mp
import os
import random
import shutil
import time

import chess
import chess.engine

_engine = None
_cfg = None


def find_stockfish(path=None):
    for c in (path, shutil.which("stockfish"), "/opt/homebrew/bin/stockfish",
              "/usr/local/bin/stockfish", "/usr/games/stockfish"):
        if c and os.path.exists(c):
            return c
    raise SystemExit("Stockfish not found. On macOS run: brew install stockfish "
                     "(or pass --stockfish /path/to/stockfish)")


def split_for(game_id):
    r = game_id % 20
    return "val" if r == 0 else "test" if r == 1 else "train"


def _init_worker(stockfish_path, cfg):
    global _engine, _cfg
    _cfg = cfg
    _engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    _engine.configure({"Threads": 1, "Hash": cfg["hash_mb"]})
    atexit.register(_engine.quit)


def evaluate_all_moves(board, depth):
    """Expected score (side to move's view) after each legal move, best first."""
    n = board.legal_moves.count()
    infos = _engine.analyse(board, chess.engine.Limit(depth=depth), multipv=n)
    ply = board.ply()
    out = {}
    for info in infos:
        if "pv" not in info or not info["pv"]:
            continue
        move = info["pv"][0]
        wdl = info["score"].pov(board.turn).wdl(model="sf", ply=ply)
        out[move.uci()] = round(wdl.expectation(), 4)
    # any move MultiPV did not report (rare): score it on its own
    for move in board.legal_moves:
        if move.uci() not in out:
            info = _engine.analyse(board, chess.engine.Limit(depth=depth), root_moves=[move])
            wdl = info["score"].pov(board.turn).wdl(model="sf", ply=ply)
            out[move.uci()] = round(wdl.expectation(), 4)
    return sorted(out.items(), key=lambda kv: -kv[1])


def play_and_label(game_id):
    """Play one randomised game; return labelled records for a sample of its positions."""
    cfg = _cfg
    rng = random.Random(cfg["seed"] * 1_000_003 + game_id)
    board = chess.Board()
    sans, records = [], []

    for _ in range(rng.randint(0, cfg["max_random_opening"])):  # diversify openings
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        sans.append(board.san(move))
        board.push(move)

    while not board.is_game_over(claim_draw=True) and board.ply() < cfg["max_plies"]:
        scored = evaluate_all_moves(board, cfg["depth"])
        if len(scored) > 1 and rng.random() < cfg["sample_rate"]:
            records.append({
                "game": game_id,
                "split": split_for(game_id),
                "fen": board.fen(),
                "recent": sans[-8:],
                "moves": [[uci, e] for uci, e in scored],   # best first
                "depth": cfg["depth"],
            })
        # choose the next move: sometimes random, otherwise sample among strong moves
        if rng.random() < cfg["random_move_rate"]:
            move = rng.choice(list(board.legal_moves))
        else:
            best = scored[0][1]
            weights = [math.exp((e - best) / cfg["temperature"]) for _, e in scored]
            uci = rng.choices([u for u, _ in scored], weights=weights)[0]
            move = chess.Move.from_uci(uci)
        sans.append(board.san(move))
        board.push(move)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--positions", type=int, default=20000, help="positions to collect in total")
    ap.add_argument("--out", default="chess_positions.jsonl")
    ap.add_argument("--depth", type=int, default=10, help="Stockfish search depth per position")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--stockfish", default=None, help="path to the Stockfish binary")
    ap.add_argument("--sample-rate", type=float, default=0.3,
                    help="fraction of positions in each game to record")
    ap.add_argument("--random-move-rate", type=float, default=0.12)
    ap.add_argument("--temperature", type=float, default=0.04,
                    help="how adventurous the self-play is (higher = more varied moves)")
    ap.add_argument("--max-random-opening", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=220)
    ap.add_argument("--hash-mb", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    stockfish_path = find_stockfish(args.stockfish)
    cfg = dict(depth=args.depth, sample_rate=args.sample_rate, random_move_rate=args.random_move_rate,
               temperature=args.temperature, max_random_opening=args.max_random_opening,
               max_plies=args.max_plies, hash_mb=args.hash_mb, seed=args.seed)

    done_games, have = set(), 0
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                if line.strip():
                    done_games.add(json.loads(line)["game"])
                    have += 1
        print(f"Resuming: {have} positions from {len(done_games)} games already in {args.out}")
    if have >= args.positions:
        print("Target already reached.")
        return

    print(f"Stockfish: {stockfish_path} | depth {args.depth} | {args.workers} workers | "
          f"target {args.positions} positions")
    todo = (g for g in range(10**9) if g not in done_games)
    t0, new = time.time(), 0
    with open(args.out, "a") as f, mp.Pool(args.workers, _init_worker, (stockfish_path, cfg)) as pool:
        # keep a bounded number of games in flight
        pending = collections.deque(pool.apply_async(play_and_label, (next(todo),))
                                    for _ in range(args.workers * 2))
        while pending:
            records = pending.popleft().get()
            pending.append(pool.apply_async(play_and_label, (next(todo),)))
            for r in records:
                f.write(json.dumps(r) + "\n")
            f.flush()
            new += len(records)
            total = have + new
            rate = new / max(1e-9, time.time() - t0)
            eta = (args.positions - total) / max(rate, 1e-9) / 60
            print(f"\r{total}/{args.positions} positions | {rate:.1f}/s | ETA {eta:.0f} min   ",
                  end="", flush=True)
            if total >= args.positions:
                pool.terminate()
                break
    print(f"\nDone. {have + new} positions in {args.out}")


if __name__ == "__main__":
    main()
