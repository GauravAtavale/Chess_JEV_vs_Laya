"""
Headless arena: play many games between two bots and report the results.

    python laya_arena.py --games 10                     # Laya (English) vs Random
    python laya_arena.py --games 10 --player random     # control: Random vs Random
    python laya_arena.py --games 10 --player jev        # Jev vs Random
    python laya_arena.py --games 20 --player laya --player-laya-path laya_chess_20k --opponent laya
    python laya_arena.py --games 10 --player laya --player-laya-path laya_chess_20k --opponent jev --max-requests 40000
    python laya_arena.py --games 20 --player laya --opponent jev --temperature 0.1   # sampling on
    python laya_arena.py --games 10 --player laya --opponent jev --rules            # rule facts on
    python laya_arena.py --games 10 --player clm --player-clm-model chess --opponent laya --opponent-laya-path laya_chess_20k

Every run is logged to its own folder, arena_runs/<date-time>_<player>-vs-<opponent>/:
    console.txt    everything printed on screen (including any error)
    moves.jsonl    one line per move: who moved, the move, its probability, the model's
                   top-5 alternatives, thinking time, and the position before the move
    results.jsonl  one line per finished game (written as soon as each game ends)
    games.pgn      all games; each move carries the model's scores as a comment, so
                   you can step through them in any PGN viewer (e.g. lichess.org/paste)
    summary.json   settings, per-game results, totals and API usage

Draws: a game ends by threefold repetition or the fifty-move rule once that has
actually happened (not merely when the side to move could claim it).

Colors alternate every game. When neither side is the random bot, each game
starts with a few random moves (--random-opening) so the games differ; otherwise
two deterministic players would repeat the same two games forever. Games that
hit --max-plies are adjudicated by material (pawn 1, knight/bishop 3, rook 5,
queen 9) and reported separately.
"""
import argparse
import datetime
import json
import os
import random
import sys
import time

import chess
import chess.pgn

from laya_player import game_finished

try:
    from chess_game import RandomPlayer
except ImportError:  # e.g. on a server without pygame
    class RandomPlayer:
        name = "Random"

        def choose_move(self, board):
            return random.choice(list(board.legal_moves))

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


class Tee:
    """Write to the terminal and a log file at the same time."""

    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile

    def write(self, text):
        self.stream.write(text)
        self.logfile.write(text)

    def flush(self):
        self.stream.flush()
        self.logfile.flush()

    def isatty(self):
        return False


def make_player(kind, args, path=None, seed=None, clm_model=None):
    sampling = dict(temperature=args.temperature, top_k=args.top_k, seed=seed,
                    rule_facts=args.rules)
    if kind == "clm":
        from clm_player import CLMPlayer
        # CLM always gets rule facts: the fine-tuned heads are trained with them
        return CLMPlayer(url=args.clm_url, model=clm_model or args.clm_model, verbose=not args.quiet,
                         temperature=args.temperature, top_k=args.top_k, seed=seed)
    if kind == "laya":
        from laya_player import LayaPlayer
        path = path or args.laya_path
        if path == "base":
            path = None
        return LayaPlayer(model=args.laya_model, local_path=path,
                          use_ab_labels=args.ab_labels, verbose=not args.quiet, **sampling)
    if kind == "jev":
        from jev_player import JevPlayer
        return JevPlayer(provider=args.jev_provider, use_ab_labels=args.ab_labels,
                         verbose=not args.quiet, max_calls=args.max_requests, **sampling)
    return RandomPlayer()


def material(board, color):
    return sum(v * (len(board.pieces(p, color)) - len(board.pieces(p, not color)))
               for p, v in VALUES.items())


def api_usage(*players):
    """Total Jev requests and input tokens used so far (0 for non-API players)."""
    seen, calls, tokens = set(), 0, 0
    for p in players:
        if id(p) in seen:
            continue
        seen.add(id(p))
        calls += getattr(p, "calls", 0)
        tokens += getattr(p, "input_tokens", 0)
    return calls, tokens


def play_game(white, black, max_plies, opening_plies, rng, game_no, move_log):
    """Play one game. Returns (board, per-move comments for the PGN)."""
    board = chess.Board()
    comments = []
    for _ in range(opening_plies):  # shared random opening so games differ
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        record = {"game": game_no, "ply": board.ply() + 1, "move_number": board.fullmove_number,
                  "side": "white" if board.turn else "black", "player": "random opening",
                  "san": board.san(move), "uci": move.uci(), "fen_before": board.fen(),
                  "legal_moves": board.legal_moves.count()}
        move_log.write(json.dumps(record) + "\n")
        comments.append("random opening move")
        board.push(move)

    players = {chess.WHITE: white, chess.BLACK: black}
    while not game_finished(board) and len(board.move_stack) < max_plies:
        mover = players[board.turn]
        fen_before, n_legal = board.fen(), board.legal_moves.count()
        t0 = time.time()
        move = mover.choose_move(board.copy())
        think = time.time() - t0
        if move not in board.legal_moves:
            raise RuntimeError(f"{mover.name} returned illegal move {move}")
        san = board.san(move)
        scores = getattr(mover, "last_scores", None) if not isinstance(mover, RandomPlayer) else None
        p_chosen = next((p for s, p in scores if s == san), None) if scores else None
        record = {"game": game_no, "ply": board.ply() + 1, "move_number": board.fullmove_number,
                  "side": "white" if board.turn else "black", "player": mover.name,
                  "san": san, "uci": move.uci(), "fen_before": fen_before,
                  "legal_moves": n_legal, "think_seconds": round(think, 3)}
        if scores:
            record["was_top_move"] = san == scores[0][0]
            record["probability"] = round(p_chosen, 4) if p_chosen is not None else None
            record["top5"] = [[s, round(p, 4)] for s, p in scores[:5]]
            record["spread"] = round(scores[0][1] - scores[-1][1], 4)
            alts = ", ".join(f"{s} {p:.3f}" for s, p in scores[1:4])
            comments.append(f"{mover.name}: p={p_chosen:.3f}; next best: {alts}; time {think:.2f}s"
                            if p_chosen is not None else f"{mover.name}; time {think:.2f}s")
        else:
            comments.append(f"{mover.name}; time {think:.2f}s")
        move_log.write(json.dumps(record) + "\n")
        move_log.flush()
        board.push(move)
    return board, comments


def build_pgn(board, comments, headers):
    game = chess.pgn.Game()
    game.headers.update(headers)
    node = game
    for move, comment in zip(board.move_stack, comments):
        node = node.add_variation(move)
        node.comment = comment
    return game


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--player", default="laya", choices=["laya", "jev", "clm", "random"])
    ap.add_argument("--opponent", default="random", choices=["laya", "jev", "clm", "random"])
    ap.add_argument("--clm-url", default=None, help="CLM server (default CLM_URL or http://127.0.0.1:8700)")
    ap.add_argument("--clm-model", default=None, help="served CLM model for every CLM player, e.g. chess")
    ap.add_argument("--player-clm-model", default=None, help="CLM model for --player only")
    ap.add_argument("--opponent-clm-model", default=None, help="CLM model for --opponent only")
    ap.add_argument("--jev-provider", default=None, choices=["typesafe", "openrouter"])
    ap.add_argument("--max-requests", type=int, default=5000,
                    help="safety cap on paid Jev API requests for this run")
    ap.add_argument("--laya-model", default="english",
                    choices=["english", "multilingual", "typed-decisions"])
    ap.add_argument("--laya-path", default=None, help="local checkpoint folder for every Laya player")
    ap.add_argument("--player-laya-path", default=None,
                    help="checkpoint for --player only ('base' = original Laya)")
    ap.add_argument("--opponent-laya-path", default=None,
                    help="checkpoint for --opponent only ('base' = original Laya)")
    ap.add_argument("--random-opening", type=int, default=None,
                    help="random moves played at the start of each game "
                         "(default: 4 if neither side is random, else 0)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = models always play their top move; e.g. 0.1 = sometimes play one of "
                         "their near-best moves, which breaks repetition loops")
    ap.add_argument("--top-k", type=int, default=3,
                    help="with --temperature, only the model's top-k moves can be sampled")
    ap.add_argument("--rules", action="store_true",
                    help="give the models rule facts (draw rules, material, repetition) and "
                         "the objective to win")
    ap.add_argument("--ab-labels", action="store_true")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="don't print per-move model scores on screen")
    ap.add_argument("--log-dir", default=None,
                    help="folder for this run's logs (default: arena_runs/<date-time>_<matchup>)")
    args = ap.parse_args()
    random.seed(args.seed)

    # ---- log folder and console capture
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.log_dir or os.path.join("arena_runs", f"{stamp}_{args.player}-vs-{args.opponent}")
    os.makedirs(run_dir, exist_ok=True)
    console = open(os.path.join(run_dir, "console.txt"), "w", buffering=1)
    sys.stdout = Tee(sys.stdout, console)
    sys.stderr = Tee(sys.stderr, console)
    print(f"Logging this run to {run_dir}/")
    print("Command: python " + " ".join(sys.argv))

    player = make_player(args.player, args, args.player_laya_path, seed=args.seed + 11,
                         clm_model=args.player_clm_model)
    opponent = make_player(args.opponent, args, args.opponent_laya_path, seed=args.seed + 23,
                           clm_model=args.opponent_clm_model)
    if args.temperature > 0:
        print(f"Sampling on: temperature {args.temperature}, top {args.top_k} moves")
    if args.random_opening is None:
        args.random_opening = 4 if "random" not in (args.player, args.opponent) else 0
    if args.random_opening:
        print(f"Each game starts with {args.random_opening} random moves")
    opening_rng = random.Random(args.seed + 1)
    if args.player == "random":
        player.name = "Random (control)"

    tally = {"win": 0, "draw": 0, "loss": 0, "unfinished_ahead": 0,
             "unfinished_level": 0, "unfinished_behind": 0}
    games, stopped_reason = [], None
    t_start = time.time()
    move_log = open(os.path.join(run_dir, "moves.jsonl"), "w")
    results_log = open(os.path.join(run_dir, "results.jsonl"), "w")
    pgn_path = os.path.join(run_dir, "games.pgn")

    with open(pgn_path, "w") as pgn_file:
        for g in range(args.games):
            player_is_white = g % 2 == 0
            white, black = (player, opponent) if player_is_white else (opponent, player)
            color = chess.WHITE if player_is_white else chess.BLACK
            t0 = time.time()
            calls_before, tokens_before = api_usage(player, opponent)
            # the same opening is used for both colors of each pair of games, for fairness
            if g % 2 == 0:
                pair_seed = opening_rng.random()
            try:
                board, comments = play_game(white, black, args.max_plies, args.random_opening,
                                            random.Random(pair_seed), g + 1, move_log)
            except Exception as e:
                if "request cap" not in str(e):
                    raise
                stopped_reason = str(e)
                print(f"\nStopped during game {g + 1}: {e}. That game is not counted. "
                      f"Raise the limit with --max-requests to play more games.")
                break

            outcome = board.outcome(claim_draw=True) if game_finished(board) else None
            edge = material(board, color)
            if outcome is not None:
                reason = outcome.termination.name.lower()
                if outcome.winner is None:
                    key, text = "draw", f"draw ({reason})"
                elif outcome.winner == color:
                    key, text = "win", f"WIN ({reason})"
                else:
                    key, text = "loss", f"loss ({reason})"
            else:
                reason = "move limit"
                key = ("unfinished_ahead" if edge > 0 else
                       "unfinished_behind" if edge < 0 else "unfinished_level")
                text = f"unfinished after {args.max_plies} plies"
            tally[key] += 1
            calls_after, tokens_after = api_usage(player, opponent)
            seconds = time.time() - t0
            print(f"Game {g + 1}/{args.games}: {player.name} as "
                  f"{'White' if player_is_white else 'Black'} -> {text}, "
                  f"material {edge:+d}, {len(board.move_stack)} plies, {seconds:.0f}s")

            result = {"game": g + 1, "white": white.name, "black": black.name,
                      "player_color": "white" if player_is_white else "black",
                      "result_for_player": key, "termination": reason,
                      "pgn_result": board.result(claim_draw=True) if outcome else "*",
                      "material_edge_for_player": edge, "plies": len(board.move_stack),
                      "seconds": round(seconds, 1),
                      "api_requests": calls_after - calls_before,
                      "api_input_tokens": tokens_after - tokens_before,
                      "final_fen": board.fen()}
            games.append(result)
            results_log.write(json.dumps(result) + "\n")
            results_log.flush()

            headers = {"Event": "Laya arena", "Round": str(g + 1),
                       "Date": datetime.date.today().strftime("%Y.%m.%d"),
                       "White": white.name, "Black": black.name,
                       "Result": result["pgn_result"], "Termination": reason}
            print(build_pgn(board, comments, headers), file=pgn_file, end="\n\n")
            pgn_file.flush()

    move_log.close()
    results_log.close()
    n = len(games)
    calls, tokens = api_usage(player, opponent)
    ahead = sum(1 for r in games if r["result_for_player"] == "draw"
                and r["material_edge_for_player"] >= 3)
    behind = sum(1 for r in games if r["result_for_player"] == "draw"
                 and r["material_edge_for_player"] <= -3)
    summary = {"run_dir": run_dir, "draws_player_ahead_3plus": ahead,
               "draws_player_behind_3plus": behind, "command": " ".join(sys.argv), "settings": vars(args),
               "player": player.name, "opponent": opponent.name, "games_finished": n,
               "stopped_early": stopped_reason, "tally": tally,
               "average_material_edge": (sum(r["material_edge_for_player"] for r in games) / n) if n else None,
               "total_seconds": round(time.time() - t_start, 1),
               "api_requests": calls, "api_input_tokens": tokens, "games": games}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if n == 0:
        print("No games finished.")
    else:
        print(f"\n=== {player.name} vs {opponent.name}, {n} games, {time.time() - t_start:.0f}s ===")
        print(f"Wins {tally['win']}  Draws {tally['draw']}  Losses {tally['loss']}")
        print(f"Unfinished: ahead on material {tally['unfinished_ahead']}, level "
              f"{tally['unfinished_level']}, behind {tally['unfinished_behind']}")
        print(f"Average final material edge: {summary['average_material_edge']:+.1f} pawns")
        print(f"Draws with {player.name} ahead by 3+ points: {ahead}; behind by 3+: {behind}")
        terms = {}
        for r in games:
            terms[r["termination"]] = terms.get(r["termination"], 0) + 1
        print("How games ended: " + ", ".join(f"{k} {v}" for k, v in sorted(terms.items())))
        if calls:
            print(f"Jev API usage: {calls} requests, {tokens:,} input tokens")
    print(f"Logs saved in {run_dir}/ (console.txt, moves.jsonl, results.jsonl, games.pgn, summary.json)")


if __name__ == "__main__":
    main()