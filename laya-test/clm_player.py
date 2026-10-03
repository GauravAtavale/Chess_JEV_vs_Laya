"""
CLM chess player: asks a served CLM model ONE choice question per turn, with every legal
move as an option, and plays the move with the highest probability.

The server is CLM's `clm-serve` (TypeSafe-compatible /v1/systemone API), running wherever
the GPU is (e.g. Colab). Settings, as arguments or environment variables:
    url    CLM_URL    default http://127.0.0.1:8700
    model  CLM_MODEL  served model name, e.g. "chess" for the fine-tuned heads
                      (default: the server's default model = the original CLM heads)
"""
import os
import random
import time

import chess

from clm_chess import ask_server, clm_question, clm_state
from laya_player import recent_san, sample_move


class CLMPlayer:
    def __init__(self, url=None, model=None, rule_facts=True, verbose=True,
                 temperature=0.0, top_k=3, seed=None):
        self.url = url or os.environ.get("CLM_URL", "http://127.0.0.1:8700")
        self.model = model or os.environ.get("CLM_MODEL") or None
        self.api_key = os.environ.get("CLM_API_KEY")
        self.rule_facts = rule_facts
        self.verbose = verbose
        self.temperature, self.top_k = temperature, top_k
        self.rng = random.Random(seed)
        self.name = f"CLM ({self.model or 'original'})"
        self.last_scores = []
        self.requests = 0
        print(f"[clm] using {self.url} (model: {self.model or 'server default'})")

    def score_moves(self, board):
        state = clm_state(board, recent_san(board), self.rule_facts)
        probs = ask_server(self.url, self.model, state, clm_question(board, None, self.rule_facts),
                           self.api_key)
        self.requests += 1
        scored = [(chess.Move.from_uci(u), float(p)) for u, p in probs.items()]
        return sorted(scored, key=lambda x: -x[1])

    def choose_move(self, board):
        t0 = time.time()
        scored = self.score_moves(board)
        best_move, best_p = scored[0]
        self.last_scores = [(board.san(m), p) for m, p in scored]
        move = sample_move(scored, self.temperature, self.top_k, self.rng)
        if self.verbose:
            top = ", ".join(f"{san} {p:.3f}" for san, p in self.last_scores[:5])
            note = f" | sampled {board.san(move)}" if move != best_move else ""
            print(f"   [{self.name}] {len(scored)} moves in {time.time() - t0:.2f}s | "
                  f"top: {top} | spread {best_p - scored[-1][1]:.3f}{note}")
        return move
