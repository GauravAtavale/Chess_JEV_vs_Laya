"""
Zero-shot Laya chess player.

No search and no pruning: for every legal move we build one state
(the position + that candidate move) and ask Laya a yes/no (`noul`) question,
"Is this a strong move?". All moves are scored in one batched call and the
move with the highest P(yes) is played.

The code only supplies the rules of chess (which moves are legal, and what a
move literally does: which piece, captures, check). Every judgment is Laya's.
"""
import math
import os
import random
import time

import chess

REPO = "convaiinnovations/laya"
SUBFOLDERS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}

QUESTION_KEY = "good_move"


# "strong": the zero-shot question.
# "win":    the question the fine-tuned model is trained on; its target is Stockfish's
#           expected score after the move (win = 1, draw = 0.5, loss = 0).
QUESTION_VARIANTS = {
    "strong": {
        "instructions": ("Is the candidate move a strong move for the side to move "
                         "in this chess position?"),
        "criteria": {
            "true": "yes, it is a strong move that improves the position",
            "false": "no, it is a weak move or a blunder",
        },
    },
    "win": {
        "instructions": ("Will the side to move win the game after playing the "
                         "candidate move?"),
        "criteria": {
            "true": "yes, the side to move goes on to win",
            "false": "no, the side to move draws or loses",
        },
    },
}

# Variants used with rule facts (--rules): same questions plus the objective and the draw rules.
_OBJECTIVE = ("Your goal is to win the game; a draw is not a win. Use the rule facts: "
              "a move that ends the game in a draw (threefold repetition, stalemate, the "
              "fifty-move rule, insufficient material) or lets the opponent force a draw by "
              "repetition throws away winning chances, especially when the side to move is ahead. ")
QUESTION_VARIANTS["strong_rules"] = {
    "instructions": _OBJECTIVE + ("Is the candidate move a strong move that helps the side "
                                  "to move win?"),
    "criteria": {
        "true": "yes, it is a strong move that improves the side to move's chances of winning",
        "false": "no, it is weak, a blunder, or it gives away a win by allowing a draw",
    },
}
QUESTION_VARIANTS["win_rules"] = {
    "instructions": _OBJECTIVE + ("Will the side to move win the game after playing the "
                                  "candidate move?"),
    "criteria": dict(QUESTION_VARIANTS["win"]["criteria"]),
}

QUESTION_VARIANTS["best"] = {
    "instructions": ("Is the candidate move one of the best moves available to the side to "
                     "move in this chess position?"),
    "criteria": {
        "true": "yes, it is the best move or nearly as good as the best move",
        "false": "no, clearly better moves are available",
    },
}
QUESTION_VARIANTS["best_rules"] = {
    "instructions": _OBJECTIVE + QUESTION_VARIANTS["best"]["instructions"],
    "criteria": dict(QUESTION_VARIANTS["best"]["criteria"]),
}

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_for(board, color):
    return sum(v * (len(board.pieces(p, color)) - len(board.pieces(p, not color)))
               for p, v in PIECE_VALUES.items())


def game_finished(board):
    """True when the game is over under the rules a player cannot avoid.

    Checkmate, stalemate, insufficient material and the automatic 75-move / fivefold
    rules, plus a threefold repetition or fifty-move draw once it has actually happened.
    (A draw that the side to move merely *could* claim with its next move does not end
    the game here, so a player who is ahead is never forced into it.)
    """
    return board.is_game_over() or board.is_repetition(3) or board.halfmove_clock >= 100


def occurrences(board):
    """How many times the current position has occurred in this game (including now)."""
    n = 1
    while n < 5 and board.is_repetition(n + 1):
        n += 1
    return n


def rule_facts_for_move(board, move):
    """Rules-only facts about what the move does to the game's result (no evaluation)."""
    after = board.copy()
    after.push(move)
    facts = []
    if after.is_checkmate():
        return "result if played: checkmate, the side to move wins immediately"
    if after.is_stalemate():
        return "result if played: stalemate, the game ends in a draw"
    if after.is_insufficient_material():
        return ("result if played: draw by insufficient material "
                "(neither side can checkmate)")
    n = occurrences(after)
    if n >= 3:
        return ("result if played: draw by threefold repetition "
                f"(this position will have occurred {n} times)")
    if after.halfmove_clock >= 100:
        return "result if played: draw by the fifty-move rule"
    if n == 2:
        facts.append("repeats an earlier position (second occurrence; a third is a draw)")
    # can the opponent answer with a move that completes a threefold repetition?
    # (impossible unless at least 4 reversible plies lead up to the reply)
    for reply in (after.legal_moves if after.halfmove_clock >= 3 else ()):
        after.push(reply)
        forced = after.is_repetition(3)
        after.pop()
        if forced:
            facts.append("lets the opponent force a draw by repetition with the next move")
            break
    if board.halfmove_clock >= 20:
        if after.halfmove_clock == 0:
            facts.append("resets the fifty-move counter (capture or pawn move)")
        else:
            left = (100 - after.halfmove_clock + 1) // 2
            facts.append(f"does not reset the fifty-move counter ({left} moves each left "
                         "before a draw)")
    facts.insert(0, "result if played: the game continues")
    return "; ".join(facts)


def position_rule_facts(board):
    """Rules-only facts about the current position, from the side to move's view."""
    edge = material_for(board, board.turn)
    side = "white" if board.turn == chess.WHITE else "black"
    if edge > 0:
        mat = f"the side to move ({side}) is ahead by {edge} points of material"
    elif edge < 0:
        mat = f"the side to move ({side}) is behind by {-edge} points of material"
    else:
        mat = "material is level"
    left = (100 - board.halfmove_clock + 1) // 2
    return {
        "material_balance": mat + " (pawn 1, knight 3, bishop 3, rook 5, queen 9)",
        "fifty_move_rule": (f"{left} moves each without a capture or pawn move until a draw"),
        "current_position_occurrences": occurrences(board),
    }


def make_questions(use_ab_labels=False, variant="strong"):
    v = QUESTION_VARIANTS[variant]
    q = {"type": "noul", "instructions": v["instructions"], "criteria": dict(v["criteria"])}
    if use_ab_labels:
        # Laya README (#156): the English checkpoint can follow the literal
        # false/true labels instead of the state; neutral labels are a workaround.
        q["labels"] = {"true": "A", "false": "B"}
    return {QUESTION_KEY: q}


def describe_move(board, move):
    """Plain-English, rules-only description of what the move does (no evaluation)."""
    color = "White" if board.turn == chess.WHITE else "Black"
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) > 4 else "queenside"
        text = f"{color} castles {side}"
    else:
        piece = chess.piece_name(board.piece_type_at(move.from_square))
        text = (f"{color} {piece} moves from {chess.square_name(move.from_square)} "
                f"to {chess.square_name(move.to_square)}")
    captured = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        text += ", capturing a pawn en passant"
    elif captured:
        text += f", capturing a {chess.piece_name(captured.piece_type)}"
    if move.promotion:
        text += f", promoting to a {chess.piece_name(move.promotion)}"
    if board.gives_check(move):
        after = board.copy(stack=False)
        after.push(move)
        text += ", giving checkmate" if after.is_checkmate() else ", giving check"
    return text


def build_state(board, move, recent_moves, include_diagram=True, rule_facts=False, position_facts=None):
    """The input for one candidate move. `board` must carry the game's move history
    (board.copy() keeps it) so repetition facts are correct."""
    state = {
        "position_fen": board.fen(),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "recent_moves": " ".join(recent_moves) or "none (start of game)",
        "candidate_move": f"{board.san(move)} ({move.uci()})",
        "move_description": describe_move(board, move),
    }
    if rule_facts:
        state.update(position_facts or position_rule_facts(board))
        state["move_rule_facts"] = rule_facts_for_move(board, move)
    if include_diagram:
        # 8 text rows, rank 8 at the top; uppercase = White, lowercase = Black, '.' = empty
        state["board"] = str(board)
    return state


def sample_move(scored, temperature=0.0, top_k=3, rng=random):
    """Pick a move from [(move, P(yes))] sorted best first.

    temperature 0 = always the top move (deterministic). Above 0, one of the top_k moves is
    sampled with weight exp(-(p_best - p) / (temperature * spread)), where spread is the gap
    between the best and worst move in this position. Scaling by the spread makes the same
    temperature behave similarly for models whose probabilities sit close together (original
    Laya) or far apart (fine-tuned Laya, Jev). Moves nearly as good as the top move get picked
    often; clearly worse ones almost never.
    """
    if temperature <= 0 or len(scored) == 1:
        return scored[0][0]
    cands = scored[:max(1, top_k)]
    spread = max(scored[0][1] - scored[-1][1], 1e-6)
    best = cands[0][1]
    weights = [math.exp(-(best - p) / (temperature * spread)) for _, p in cands]
    return rng.choices([m for m, _ in cands], weights=weights)[0]


def recent_san(board, n=8):
    replay = board.root()
    sans = []
    for m in board.move_stack:
        sans.append(replay.san(m))
        replay.push(m)
    return sans[-n:]


class LayaPlayer:
    """Scores every legal move with Laya and plays the highest P(strong move)."""

    def __init__(self, model="english", local_path=None, device=None,
                 batch_size=16, use_ab_labels=False, include_diagram=True, verbose=True,
                 question=None, temperature=0.0, top_k=3, seed=None, rule_facts=False):
        import laya
        import torch

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.name = f"Laya ({model})"
        self.batch_size = batch_size
        self.include_diagram = include_diagram
        self.verbose = verbose
        self.last_scores = []   # [(san, probability)] for the most recent decision
        self._length_checked = False
        self.temperature, self.top_k = temperature, top_k
        self.rng = random.Random(seed)

        t0 = time.time()
        if local_path:
            self.agent = laya.load(os.path.expanduser(local_path), device=device)
        else:
            self.agent = laya.load(REPO, subfolder=SUBFOLDERS[model], device=device)
        # A checkpoint fine-tuned by train_laya_chess.py records which question it learned.
        cfg = getattr(self.agent, "cfg", {}) or {}
        # a model trained with rule facts always gets them
        self.rule_facts = rule_facts or bool(cfg.get("chess_rule_facts"))
        self.question = question or cfg.get("chess_question", "strong")
        if self.rule_facts and not self.question.endswith("_rules"):
            self.question += "_rules"
            if cfg.get("model_name") == "laya-chess":
                print("[laya] note: this fine-tuned model never saw rule facts in training; "
                      "they are new input for it")
        self.questions = make_questions(use_ab_labels, self.question)
        if cfg.get("model_name") == "laya-chess":
            folder = os.path.basename(os.path.normpath(local_path)) if local_path else ""
            self.name = f"Laya fine-tuned ({folder})" if folder else "Laya (chess fine-tuned)"
        if self.rule_facts:
            self.name += " +rules"
        print(f"[laya] loaded {self.name} on {device} in {time.time() - t0:.1f}s "
              f"(question: {self.question!r})")

    def _check_length(self, state):
        """Warn if the input is longer than the checkpoint's window (Laya would truncate it)."""
        self._length_checked = True
        try:
            from laya.common import build_sequence
            internal = self.agent._to_internal(self.questions[QUESTION_KEY])
            seq, _ = build_sequence(self.agent.tok, state, internal, 100_000,
                                    self.agent.cfg.get("head_max_len", 256))
            limit = self.agent.cfg.get("max_len")
            print(f"[laya] input is {len(seq)} tokens (checkpoint limit {limit})")
            if limit and len(seq) > limit:
                print("[laya] WARNING: input exceeds the limit and will be truncated. "
                      "Try include_diagram=False.")
        except Exception as e:  # internal API; may differ between Laya versions
            print(f"[laya] could not check input length ({type(e).__name__}); continuing")

    def score_moves(self, board):
        """Return [(move, P(strong))] for every legal move, highest first."""
        moves = list(board.legal_moves)
        recent = recent_san(board)
        pos = position_rule_facts(board) if self.rule_facts else None
        states = [build_state(board, m, recent, self.include_diagram, self.rule_facts, pos)
                  for m in moves]
        if not self._length_checked:
            self._check_length(states[0])
        results = self.agent.predict_batch(states, self.questions, batch_size=self.batch_size)
        probs = [float(r["answers"][QUESTION_KEY]["noul"]) for r in results]
        return sorted(zip(moves, probs), key=lambda x: -x[1])

    def choose_move(self, board):
        t0 = time.time()
        scored = self.score_moves(board)
        best_move, best_p = scored[0]
        self.last_scores = [(board.san(m), p) for m, p in scored]
        move = sample_move(scored, self.temperature, self.top_k, self.rng)
        if self.verbose:
            top = ", ".join(f"{san} {p:.3f}" for san, p in self.last_scores[:5])
            spread = best_p - scored[-1][1]
            note = f" | sampled {board.san(move)}" if move != best_move else ""
            print(f"   [{self.name}] {len(scored)} moves in {time.time() - t0:.2f}s | "
                  f"top: {top} | spread {spread:.3f}{note}")
        return move
