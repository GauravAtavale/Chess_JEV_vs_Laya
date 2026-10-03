"""
CLM chess helpers: one input format shared by training and play, a converter from our
Stockfish datasets to CLM's fine-tuning format, and an evaluator.

Each position becomes ONE choice question whose options are the legal moves (all of
them; CLM embeds each option separately, so there is no option-count limit):

  state    position (FEN, board diagram, recent moves) + rule facts about the position
  options  "Nf3 (g1f3): White knight moves from g1 to f3; result if played: ..."
  target   a probability for every move from Stockfish (best moves high, bad moves ~0)

Convert (writes <out>/all/train-00000.parquet and <out>/all/test-00000.parquet):
    python clm_chess.py convert --data chess_20k.jsonl chess_extra_10k.jsonl --out clm_chess_data

Evaluate a served model on held-out positions (same metrics as train_laya_chess.py):
    python clm_chess.py evaluate --data chess_extra_10k.jsonl --url http://127.0.0.1:8700 --model chess

Evaluate a Laya checkpoint on exactly the same positions (for a like-for-like comparison):
    python clm_chess.py evaluate --data chess_20k.jsonl chess_extra_10k.jsonl --laya-path laya_chess_20k

Needs laya_player.py next to this file.
"""
import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess  # noqa: E402

from laya_player import describe_move, position_rule_facts, rule_facts_for_move  # noqa: E402

QUESTION_KEY = "move"
INSTRUCTIONS = ("Which move should the side to move play? Your goal is to win the game; a draw "
                "is not a win. Avoid moves that give away material, allow checkmate, or throw "
                "away a winning position by allowing a draw.")


# ------------------------------------------------------------------ shared input format
def clm_state(board, recent_moves, rule_facts=True):
    state = {
        "position_fen": board.fen(),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "recent_moves": " ".join(recent_moves) or "none (start of game)",
        # 8 text rows, rank 8 at the top; uppercase = White, lowercase = Black, '.' = empty
        "board": str(board),
    }
    if rule_facts:
        state.update(position_rule_facts(board))
    return state


def move_option_text(board, move, rule_facts=True):
    text = f"{board.san(move)} ({move.uci()}): {describe_move(board, move)}"
    if rule_facts:
        text += "; " + rule_facts_for_move(board, move)
    return text


def clm_options(board, moves=None, rule_facts=True):
    """{uci: option text} for the given moves (default: all legal moves)."""
    moves = list(board.legal_moves) if moves is None else moves
    return {m.uci(): move_option_text(board, m, rule_facts) for m in moves}


def clm_question(board, moves=None, rule_facts=True):
    return {"type": "choice", "instructions": INSTRUCTIONS,
            "criteria": clm_options(board, moves, rule_facts)}


# ------------------------------------------------------------------ data helpers
def move_value(m):
    """[uci, expected score(, centipawns)] -> value in [-1, 1] (log-scaled centipawns if present)."""
    if len(m) >= 3 and m[2] is not None:
        cp = m[2]
        return math.copysign(math.log1p(abs(cp) / 100) / math.log1p(100), cp)
    return 2.0 * m[1] - 1.0


def board_for(record):
    if record.get("history"):
        board = chess.Board()
        for uci in record["history"]:
            board.push_uci(uci)
        return board
    return chess.Board(record["fen"])


def read_records(paths):
    for path in paths:
        with open(path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    r = json.loads(line)
                    r["_id"] = f"{os.path.basename(path)}:{r['game']}:{i}"
                    yield r


def informative(record):
    values = [move_value(m) for m in record["moves"]]
    return max(values) - min(values) >= 0.02


# ------------------------------------------------------------------ convert
def convert(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = random.Random(args.seed)
    rows = {"train": [], "test": []}
    skipped = 0
    t0 = time.time()
    for rec in read_records(args.data):
        if not informative(rec):
            skipped += 1
            continue
        split = "test" if rec["split"] == "test" else "train"   # CLM carves its own val set
        board = board_for(rec)
        values = {m[0]: move_value(m) for m in rec["moves"]}
        ranked = sorted(values, key=values.get, reverse=True)
        if split == "train" and args.max_options and len(ranked) > args.max_options:
            keep = ranked[:4] + rng.sample(ranked[4:], args.max_options - 4)
        else:
            keep = ranked                                           # test: every legal move
        best = max(values[u] for u in keep)
        weights = {u: math.exp(-(best - values[u]) / args.tau) for u in keep}
        z = sum(weights.values())
        moves = [chess.Move.from_uci(u) for u in keep]
        rows[split].append({
            "id": rec["_id"],
            "workflow": "chess",
            "state": json.dumps(clm_state(board, rec["recent"], args.rule_facts)),
            "questions": json.dumps({QUESTION_KEY: clm_question(board, moves, args.rule_facts)}),
            "gold": json.dumps({QUESTION_KEY: {
                "label": max(keep, key=values.get),
                "probabilities": {u: round(w / z, 6) for u, w in weights.items()}}}),
        })
    folder = os.path.join(args.out, "all")
    os.makedirs(folder, exist_ok=True)
    for split, r in rows.items():
        rng.shuffle(r)
        pq.write_table(pa.Table.from_pylist(r), os.path.join(folder, f"{split}-00000.parquet"))
    print(f"Wrote {len(rows['train'])} train and {len(rows['test'])} test questions to {folder}/ "
          f"({skipped} positions skipped: all moves equal) in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------ evaluate
def ask_server(url, model, state, question, api_key=None, timeout=120):
    body = {"state": state, "questions": {QUESTION_KEY: question}}
    if model:
        body["model"] = model
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", json.dumps(body).encode(),
                                 headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        answer = json.loads(resp.read())["answers"][QUESTION_KEY]
    return answer["probabilities"]


def laya_scorer(path):
    """Score every legal move with a Laya checkpoint, using the input format it expects."""
    from laya_player import QUESTION_KEY as LAYA_KEY, LayaPlayer, build_state
    player = LayaPlayer(local_path=None if path == "base" else path, verbose=False)

    def score(board, record):
        moves = list(board.legal_moves)
        pos = position_rule_facts(board) if player.rule_facts else None
        states = [build_state(board, m, record["recent"], player.include_diagram, player.rule_facts, pos)
                  for m in moves]
        results = player.agent.predict_batch(states, player.questions, batch_size=player.batch_size)
        return {m.uci(): float(r["answers"][LAYA_KEY]["noul"]) for m, r in zip(moves, results)}
    return player.name, score


def evaluate(args):
    records = [r for r in read_records(args.data) if r["split"] == args.split and informative(r)]
    random.Random(args.seed).shuffle(records)
    records = records[:args.positions]
    if args.laya_path:
        name, laya_score = laya_scorer(args.laya_path)
    top1 = loss = loss_rand = vloss = vloss_rand = 0.0
    t0 = time.time()
    for i, rec in enumerate(records):
        board = board_for(rec)
        if args.laya_path:
            probs = laya_score(board, rec)
        else:
            probs = ask_server(args.url, args.model, clm_state(board, rec["recent"], args.rule_facts),
                               clm_question(board, None, args.rule_facts))
        chosen = max(probs, key=probs.get)
        e = {m[0]: m[1] for m in rec["moves"]}
        v = {m[0]: move_value(m) for m in rec["moves"]}
        best_e, best_v = max(e.values()), max(v.values())
        top1 += e[chosen] >= best_e - 1e-9
        loss += best_e - e[chosen]
        loss_rand += best_e - sum(e.values()) / len(e)
        vloss += best_v - v[chosen]
        vloss_rand += best_v - sum(v.values()) / len(v)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(records)} positions, {time.time() - t0:.0f}s", flush=True)
    n = max(1, len(records))
    label = name if args.laya_path else f"CLM ({args.model or 'server default'})"
    report = {"model": label, "positions": len(records),
              "top1_match": top1 / n, "avg_expected_score_lost": loss / n,
              "random_move_expected_score_lost": loss_rand / n,
              "avg_value_lost": vloss / n, "random_move_value_lost": vloss_rand / n}
    print(json.dumps(report, indent=2))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert", help="our .jsonl datasets -> CLM fine-tuning parquet")
    c.add_argument("--data", nargs="+", required=True)
    c.add_argument("--out", default="clm_chess_data")
    c.add_argument("--tau", type=float, default=0.1,
                   help="how sharply target probabilities favour the best move")
    c.add_argument("--max-options", type=int, default=16,
                   help="training questions keep the top 4 moves plus random others up to this "
                        "many (limits GPU memory); test questions keep every legal move")
    c.add_argument("--no-rule-facts", dest="rule_facts", action="store_false")
    c.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate", help="score a served CLM model on held-out positions")
    e.add_argument("--data", nargs="+", required=True)
    e.add_argument("--url", default="http://127.0.0.1:8700")
    e.add_argument("--model", default=None, help="served model name (e.g. chess); default = server default")
    e.add_argument("--laya-path", default=None,
                   help="evaluate this Laya checkpoint folder instead of CLM ('base' = original Laya)")
    e.add_argument("--split", default="test", choices=["test", "val", "train"])
    e.add_argument("--positions", type=int, default=300)
    e.add_argument("--no-rule-facts", dest="rule_facts", action="store_false")
    e.add_argument("--report", default=None, help="save the results to this .json file")
    e.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    convert(args) if args.cmd == "convert" else evaluate(args)


if __name__ == "__main__":
    main()
