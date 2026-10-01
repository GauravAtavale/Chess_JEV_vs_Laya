"""
Playable chess game (pygame + python-chess) with pluggable players.

    python chess_game.py                          # you (White) vs Random
    python chess_game.py --white random --black human
    python chess_game.py --white random --black random   # watch two bots
    python chess_game.py --black laya                      # you vs Laya
    python chess_game.py --black jev                       # you vs Jev (needs an API key)
    python chess_game.py --white laya --black jev          # watch Laya vs Jev
    python chess_game.py --white laya --black laya --black-laya-path laya_chess_20k
                                                           # original vs fine-tuned

Controls: click a piece, then click a highlighted square.
          U = undo your last move   N = new game   Esc = quit
Pawns promote to a queen automatically.

A player is any object with a `name` and a `choose_move(board) -> chess.Move`
method. Register new players in PLAYERS near the top of this file
(this is where Laya will plug in).
"""
import argparse
import os
import random
import threading

import chess
import pygame

from laya_player import game_finished

# ---------------------------------------------------------------- players


class RandomPlayer:
    """Placeholder AI: picks a random legal move."""

    name = "Random"

    def choose_move(self, board: chess.Board) -> chess.Move:
        return random.choice(list(board.legal_moves))


def make_laya_player(args, side):
    from laya_player import LayaPlayer  # imported lazily so the game runs without Laya
    # a per-side path (--white-laya-path / --black-laya-path) overrides --laya-path;
    # the word "base" means the original checkpoint from Hugging Face
    path = getattr(args, f"{side}_laya_path", None) or args.laya_path
    if path == "base":
        path = None
    return LayaPlayer(model=args.laya_model, local_path=path, use_ab_labels=args.ab_labels,
                      temperature=args.temperature, top_k=args.top_k, rule_facts=args.rules)


def make_jev_player(args, side):
    from jev_player import JevPlayer  # imported lazily; needs an API key
    return JevPlayer(provider=args.jev_provider, use_ab_labels=args.ab_labels,
                     temperature=args.temperature, top_k=args.top_k, rule_facts=args.rules)


PLAYERS = {
    "human": None,          # None means "take clicks from the mouse"
    "random": lambda args, side: RandomPlayer(),
    "laya": make_laya_player,
    "jev": make_jev_player,
}

# ---------------------------------------------------------------- drawing

SQ = 80
BOARD_PX = SQ * 8
BAR_PX = 56
LIGHT, DARK = (240, 217, 181), (181, 136, 99)
SELECTED, LAST_MOVE, CHECK = (246, 246, 105), (205, 210, 106), (235, 97, 80)
DOT = (40, 40, 40)
BAR_BG, BAR_TEXT = (38, 36, 33), (235, 235, 235)

GLYPHS = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
          chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}
LETTERS = {chess.KING: "K", chess.QUEEN: "Q", chess.ROOK: "R",
           chess.BISHOP: "B", chess.KNIGHT: "N", chess.PAWN: "P"}

FONT_FILES = [
    "/System/Library/Fonts/Apple Symbols.ttf",                   # macOS
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",      # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",           # Linux
    "C:/Windows/Fonts/seguisym.ttf",                             # Windows
]


def load_piece_font(size, use_letters):
    if not use_letters:
        for path in FONT_FILES:
            if os.path.exists(path):
                return pygame.font.Font(path, size), GLYPHS
    return pygame.font.SysFont("arial", int(size * 0.7), bold=True), LETTERS


def render_piece(font, glyphs, piece):
    """Piece glyph filled white or black, with a contrasting outline."""
    text = glyphs[piece.piece_type]
    fill = (250, 250, 250) if piece.color == chess.WHITE else (20, 20, 20)
    edge = (20, 20, 20) if piece.color == chess.WHITE else (250, 250, 250)
    base = font.render(text, True, fill)
    outline = font.render(text, True, edge)
    w, h = base.get_size()
    surf = pygame.Surface((w + 4, h + 4), pygame.SRCALPHA)
    for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        surf.blit(outline, (2 + dx, 2 + dy))
    surf.blit(base, (2, 2))
    return surf


class ChessGame:
    def __init__(self, white, black, flipped, use_letters):
        pygame.init()
        pygame.display.set_caption("Chess")
        self.screen = pygame.display.set_mode((BOARD_PX, BOARD_PX + BAR_PX))
        self.clock = pygame.time.Clock()
        font, glyphs = load_piece_font(int(SQ * 0.8), use_letters)
        self.piece_images = {(pt, c): render_piece(font, glyphs, chess.Piece(pt, c))
                             for pt in GLYPHS for c in (chess.WHITE, chess.BLACK)}
        self.small_font = pygame.font.SysFont("arial", 14, bold=True)
        self.bar_font = pygame.font.SysFont("arial", 20)
        self.players = {chess.WHITE: white, chess.BLACK: black}
        self.flipped = flipped
        self.new_game()

    # ------------------------------------------------------------ state
    def new_game(self):
        self.board = chess.Board()
        self.selected = None
        self.ai_thread = None
        self.ai_result = None
        self.ai_error = None
        self.game_id = getattr(self, "game_id", 0) + 1
        print("\n=== New game ===")

    def is_human(self, color):
        return self.players[color] is None

    def player_name(self, color):
        p = self.players[color]
        return "You" if p is None else p.name

    def push(self, move):
        n = self.board.fullmove_number
        prefix = f"{n}." if self.board.turn == chess.WHITE else f"{n}..."
        print(f"{prefix} {self.board.san(move)}")
        self.board.push(move)
        if game_finished(self.board):
            print(f"Game over: {self.result_text()}")

    def undo(self):
        if self.ai_thread is not None:
            return  # don't undo while a bot is thinking
        if not any(self.is_human(c) for c in (chess.WHITE, chess.BLACK)):
            return
        # pop back to the most recent position where a human is to move
        while self.board.move_stack:
            self.board.pop()
            if self.is_human(self.board.turn):
                break
        self.selected = None
        print("(undo)")

    # ------------------------------------------------------------ geometry
    def square_at(self, pos):
        x, y = pos
        if not (0 <= x < BOARD_PX and 0 <= y < BOARD_PX):
            return None
        f, r = x // SQ, 7 - y // SQ
        if self.flipped:
            f, r = 7 - f, 7 - r
        return chess.square(f, r)

    def square_xy(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if self.flipped:
            f, r = 7 - f, 7 - r
        return f * SQ, (7 - r) * SQ

    # ------------------------------------------------------------ input
    def on_click(self, pos):
        if game_finished(self.board) or not self.is_human(self.board.turn):
            return
        sq = self.square_at(pos)
        if sq is None:
            return
        piece = self.board.piece_at(sq)
        if self.selected is None:
            if piece and piece.color == self.board.turn:
                self.selected = sq
            return
        move = chess.Move(self.selected, sq)
        moving = self.board.piece_at(self.selected)
        if moving and moving.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
            move.promotion = chess.QUEEN
        if move in self.board.legal_moves:
            self.push(move)
            self.selected = None
        elif piece and piece.color == self.board.turn:
            self.selected = sq
        else:
            self.selected = None

    # ------------------------------------------------------------ bots
    def maybe_start_ai(self):
        if game_finished(self.board) or self.is_human(self.board.turn) or self.ai_thread:
            return
        player, board_copy, game_id = self.players[self.board.turn], self.board.copy(), self.game_id

        def work():
            try:
                move = player.choose_move(board_copy)
                self.ai_result = (game_id, board_copy.fen(), move)
            except Exception as e:  # keep the window alive if a bot crashes
                self.ai_error = e

        self.ai_thread = threading.Thread(target=work, daemon=True)
        self.ai_thread.start()

    def collect_ai(self):
        if self.ai_thread is None or self.ai_thread.is_alive():
            return
        self.ai_thread = None
        if self.ai_error is not None:
            print(f"Bot error, playing a random move instead: {self.ai_error!r}")
            self.ai_error = None
            self.push(random.choice(list(self.board.legal_moves)))
            return
        result, self.ai_result = self.ai_result, None
        if result is None:
            return
        game_id, fen, move = result
        # ignore stale results (new game or undo happened while thinking)
        if game_id != self.game_id or fen != self.board.fen():
            return
        if move in self.board.legal_moves:
            self.push(move)
        else:
            print(f"Bot returned illegal move {move}, playing a random move instead")
            self.push(random.choice(list(self.board.legal_moves)))

    # ------------------------------------------------------------ render
    def result_text(self):
        outcome = self.board.outcome(claim_draw=True) or self.board.outcome()
        if outcome is None:
            return ""
        reason = outcome.termination.name.replace("_", " ").lower()
        if outcome.winner is None:
            return f"Draw ({reason})"
        winner = "White" if outcome.winner == chess.WHITE else "Black"
        return f"{winner} wins ({reason})"

    def status_text(self):
        if game_finished(self.board):
            return self.result_text() + "   —   N: new game"
        color = "White" if self.board.turn == chess.WHITE else "Black"
        check = "  Check!" if self.board.is_check() else ""
        if self.is_human(self.board.turn):
            return f"Your move ({color}){check}"
        return f"{self.player_name(self.board.turn)} ({color}) is thinking…{check}"

    def draw(self):
        last = self.board.peek() if self.board.move_stack else None
        king_in_check = self.board.king(self.board.turn) if self.board.is_check() else None
        targets = set()
        if self.selected is not None:
            targets = {m.to_square for m in self.board.legal_moves if m.from_square == self.selected}

        for sq in chess.SQUARES:
            x, y = self.square_xy(sq)
            light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            color = LIGHT if light else DARK
            if last and sq in (last.from_square, last.to_square):
                color = LAST_MOVE
            if sq == self.selected:
                color = SELECTED
            if sq == king_in_check:
                color = CHECK
            pygame.draw.rect(self.screen, color, (x, y, SQ, SQ))

            piece = self.board.piece_at(sq)
            if piece:
                img = self.piece_images[(piece.piece_type, piece.color)]
                self.screen.blit(img, img.get_rect(center=(x + SQ // 2, y + SQ // 2)))
            if sq in targets:
                if piece:  # capture: ring
                    pygame.draw.circle(self.screen, DOT, (x + SQ // 2, y + SQ // 2), SQ // 2 - 3, 4)
                else:
                    pygame.draw.circle(self.screen, DOT, (x + SQ // 2, y + SQ // 2), SQ // 8)

        # coordinates
        for i in range(8):
            f = 7 - i if self.flipped else i
            r = i if self.flipped else 7 - i
            fc = LIGHT if i % 2 == 0 else DARK
            self.screen.blit(self.small_font.render("abcdefgh"[f], True, fc),
                             (i * SQ + SQ - 12, BOARD_PX - 18))
            rc = DARK if i % 2 == 0 else LIGHT
            self.screen.blit(self.small_font.render(str(r + 1), True, rc), (3, i * SQ + 2))

        pygame.draw.rect(self.screen, BAR_BG, (0, BOARD_PX, BOARD_PX, BAR_PX))
        text = self.bar_font.render(self.status_text(), True, BAR_TEXT)
        self.screen.blit(text, text.get_rect(midleft=(14, BOARD_PX + BAR_PX // 2)))
        pygame.display.flip()

    # ------------------------------------------------------------ loop
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.on_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_n:
                        self.new_game()
                    elif event.key == pygame.K_u:
                        self.undo()
            self.collect_ai()
            self.maybe_start_ai()
            self.draw()
            self.clock.tick(30)
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Play chess against a pluggable bot.")
    parser.add_argument("--white", default="human", choices=PLAYERS)
    parser.add_argument("--black", default="random", choices=PLAYERS)
    parser.add_argument("--laya-model", default="english",
                        choices=["english", "multilingual", "typed-decisions"])
    parser.add_argument("--laya-path", default=None,
                        help="local checkpoint folder for every Laya player")
    parser.add_argument("--white-laya-path", default=None,
                        help="checkpoint for White only ('base' = original Laya)")
    parser.add_argument("--black-laya-path", default=None,
                        help="checkpoint for Black only ('base' = original Laya)")
    parser.add_argument("--ab-labels", action="store_true",
                        help="show the model neutral A/B labels instead of true/false")
    parser.add_argument("--jev-provider", default=None, choices=["typesafe", "openrouter"],
                        help="where to call Jev (default: whichever API key is set)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = bots always play their top move; e.g. 0.1 adds variety")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rules", action="store_true",
                        help="give the bots rule facts (draw rules, material) and the goal to win")
    parser.add_argument("--letters", action="store_true",
                        help="draw pieces as letters if the chess symbols don't render")
    args = parser.parse_args()

    make = lambda key, side: PLAYERS[key](args, side) if PLAYERS[key] else None
    white, black = make(args.white, "white"), make(args.black, "black")
    flipped = white is not None and black is None  # show Black at the bottom if you play Black
    ChessGame(white, black, flipped, args.letters).run()


if __name__ == "__main__":
    main()
