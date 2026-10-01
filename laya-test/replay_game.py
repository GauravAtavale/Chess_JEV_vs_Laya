"""
Replay saved games in a pygame window: player cards with avatars and last-decision
times, a head-to-head score, the move list, and each model's decision details with
probability bars. Two vintage colour themes: "walnut" (default) and "parchment".

Games play automatically, like a live match: each move comes after a short, slightly
random "thinking" pause, and when a game ends the result stays on screen for a few seconds
before the next game in the file starts.

    python replay_game.py games.pgn                 # play every game in the file, from game 1
    python replay_game.py games.pgn --game 7        # start at game 7
    python replay_game.py games.pgn --seconds 2     # slower thinking pauses
    python replay_game.py games.pgn --one-game      # stop after one game
    python replay_game.py games.pgn --paused        # start paused (step with the arrow keys)
    python replay_game.py games.pgn --flip          # Black at the bottom
    python replay_game.py games.pgn --scale 0.85    # smaller window
    python replay_game.py games.pgn --theme parchment   # light, old printed-score look
    python replay_game.py games.pgn --record match.mp4  # save a video instead of showing a window

Controls (all optional, the games play on their own):
    Space           pause / resume                Esc        quit
    Right / Left    next / previous move (pauses) Up / Down  previous / next game
    Home / End      start / end of the game       + / -      faster / slower
"""
import argparse
import math
import os
import random
import re
import time

import chess
import chess.pgn
import pygame

from chess_game import load_piece_font, render_piece

# ------------------------------------------------------------------ themes
# Every colour used in the window comes from the active theme (pick with --theme).
THEMES = {
    # Warm walnut and maple board, parchment text, brass accents: an old club-room feel.
    "walnut": dict(
        BG=(34, 26, 20), PANEL=(46, 36, 28), RAIL=(40, 31, 24), LINE=(80, 63, 47),
        MUTED=(178, 158, 130), TEXT=(241, 229, 205), ACCENT=(204, 163, 86),
        JEV=(201, 114, 78), LAYA=(126, 166, 152), CLM=(170, 134, 176), RANDOM=(150, 140, 125),
        LIGHT_SQ=(233, 212, 171), DARK_SQ=(152, 104, 66), LAST_LIGHT=(228, 199, 118),
        LAST_DARK=(186, 140, 70), CHECK_SQ=(192, 70, 54), FRAME=(94, 65, 41),
        DEEP=(29, 22, 17), ROW_ALT=(40, 31, 24), ACTIVE_ROW=(112, 85, 46),
        ACTIVE_TEXT=(250, 236, 200), THINK_BG=(88, 67, 36), THINK_BORDER=(172, 133, 70),
        THINK_TEXT=(244, 214, 150), CLOCK_BG=(38, 30, 23), PHASE_BG=(53, 41, 31),
        PHASE_TEXT=(214, 196, 166), PILL_BG=(60, 47, 35), PILL_ON=(92, 70, 38),
        BADGE=(72, 56, 40), TRACK=(68, 54, 40), ALT_BAR=(146, 118, 72), FAINT=(132, 114, 92),
        SCORE_TEXT=(236, 220, 188), COORD_ON_DARK=(238, 220, 186), COORD_ON_LIGHT=(128, 88, 56),
        NOTICE_BG=(30, 23, 18, 238), NOTICE_BORDER=(160, 124, 68)),
    # Old printed-score look: aged paper, ink, oxblood accents, tan and umber board.
    "parchment": dict(
        BG=(234, 224, 200), PANEL=(243, 235, 214), RAIL=(226, 214, 186), LINE=(201, 185, 153),
        MUTED=(118, 100, 78), TEXT=(50, 39, 28), ACCENT=(132, 44, 38),
        JEV=(170, 86, 50), LAYA=(52, 108, 96), CLM=(110, 72, 128), RANDOM=(120, 110, 96),
        LIGHT_SQ=(240, 226, 194), DARK_SQ=(170, 132, 96), LAST_LIGHT=(232, 206, 140),
        LAST_DARK=(190, 150, 88), CHECK_SQ=(196, 74, 60), FRAME=(122, 92, 62),
        DEEP=(236, 226, 204), ROW_ALT=(236, 227, 205), ACTIVE_ROW=(214, 186, 150),
        ACTIVE_TEXT=(60, 30, 20), THINK_BG=(236, 212, 196), THINK_BORDER=(170, 90, 70),
        THINK_TEXT=(132, 44, 38), CLOCK_BG=(238, 229, 207), PHASE_BG=(232, 220, 194),
        PHASE_TEXT=(84, 66, 48), PILL_BG=(226, 214, 190), PILL_ON=(236, 206, 196),
        BADGE=(222, 208, 178), TRACK=(222, 210, 184), ALT_BAR=(176, 140, 104), FAINT=(150, 132, 108),
        SCORE_TEXT=(60, 44, 30), COORD_ON_DARK=(246, 236, 212), COORD_ON_LIGHT=(140, 104, 70),
        NOTICE_BG=(246, 238, 218, 240), NOTICE_BORDER=(132, 44, 38)),
}


def use_theme(name):
    """Make the chosen theme's colours the module-level colour names used when drawing."""
    globals().update(THEMES[name])


use_theme("walnut")

W, H = 1060, 800
RAIL_W = 72
BOARD_X, BOARD_Y, SQ = 100, 186, 64
BOARD_PX = SQ * 8
PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 652, 124, 380, 666

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
START_COUNT = {chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 2, chess.QUEEN: 1}
TERMINATION_TEXT = {"checkmate": "Checkmate", "stalemate": "Stalemate",
                    "threefold_repetition": "Draw by threefold repetition",
                    "fifty_moves": "Draw by the fifty-move rule",
                    "insufficient_material": "Draw by insufficient material",
                    "fivefold_repetition": "Draw by fivefold repetition",
                    "seventyfive_moves": "Draw by the 75-move rule",
                    "move limit": "Stopped at the move limit"}


# ------------------------------------------------------------------ helpers
def font(size, bold=False, serif=False):
    names = "georgia,times" if serif else "helveticaneue,helvetica,segoeui,arial"
    return pygame.font.SysFont(names, size, bold=bold)


def load_games(path):
    games = []
    with open(path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    if not games:
        raise SystemExit(f"No games found in {path}")
    return games


def short_name(name):
    """'Laya fine-tuned (laya_chess_v2) +rules' -> 'Laya v2', 'CLM (chess)' -> 'CLM FT', ..."""
    n = name.replace(" +rules", "").strip()
    if n.startswith("Laya fine-tuned"):
        m = re.search(r"\(([^)]*)\)", n)
        folder = m.group(1) if m else ""
        tag = "v2" if "v2" in folder else ("FT" if not folder else folder.replace("laya_chess_", "").replace("_", " "))
        return f"Laya {tag}" if tag else "Laya FT"
    if n.startswith("Laya"):
        return "Laya"
    if n.startswith("CLM"):
        return "CLM" if "original" in n or "default" in n else "CLM FT"
    if n.startswith("Jev"):
        return "Jev"
    if n.startswith("Random"):
        return "Random"
    return n[:14]


def identity(name):
    """(avatar letter, accent colour, one-line description) for a player name."""
    n = name.lower()
    rules = " · rule facts" if "+rules" in n else ""
    if n.startswith("jev"):
        return "J", JEV, "TypeSafe · hosted API" + rules
    if n.startswith("laya fine-tuned"):
        return "L", LAYA, "Open weights · fine-tuned on Stockfish data" + rules
    if n.startswith("laya"):
        return "L", LAYA, "Open weights · original checkpoint" + rules
    if n.startswith("clm"):
        tuned = "original heads" if ("original" in n or "default" in n) else "fine-tuned heads"
        return "C", CLM, f"Qwen3-8B + {tuned} · rule facts"
    if n.startswith("random"):
        return "R", RANDOM, "Uniformly random legal moves"
    return name[:1].upper() or "?", ACCENT, ""


def parse_comment(comment):
    """Arena comments look like 'Name: p=0.482; next best: c4 0.462, d4 0.438; time 0.31s'."""
    info = {"p": None, "alts": [], "time": None, "random_opening": "random opening" in comment}
    m = re.search(r"p=([0-9.]+)", comment)
    if m:
        info["p"] = float(m.group(1))
    m = re.search(r"next best:\s*([^;]*)", comment)
    if m:
        for part in m.group(1).split(","):
            bits = part.strip().rsplit(" ", 1)
            if len(bits) == 2:
                try:
                    info["alts"].append((bits[0], float(bits[1])))
                except ValueError:
                    pass
    m = re.search(r"time ([0-9.]+)s", comment)
    if m:
        info["time"] = float(m.group(1))
    return info


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ------------------------------------------------------------------ viewer
class ReplayViewer:
    def __init__(self, games, index, flipped=False, use_letters=False, seconds=1.0, jitter=0.6,
                 end_pause=4.0, all_games=True, start_playing=True, scale=1.0, title="",
                 record=None, fps=30):
        # Recording renders on a virtual clock: frames are produced as fast as the computer
        # allows, and the video still plays at the normal pace.
        self.record, self.fps, self.frame = record, fps, 0
        if record:
            start_playing = True
        pygame.init()
        pygame.display.set_caption("Model Matches · Chess replay")
        self.scale = scale
        size = (int(W * scale) // 2 * 2, int(H * scale) // 2 * 2)   # even sizes for video
        self.window = pygame.display.set_mode(size)
        self.screen = pygame.Surface((W, H))
        self.clock = pygame.time.Clock()
        self.title = title
        self.games, self.flipped = games, flipped
        self.seconds, self.jitter, self.end_pause = seconds, jitter, end_pause
        self.all_games, self.playing = all_games, start_playing

        piece_font, glyphs = load_piece_font(int(SQ * 0.82), use_letters)
        self.pieces = {(pt, c): render_piece(piece_font, glyphs, chess.Piece(pt, c))
                       for pt in glyphs for c in (chess.WHITE, chess.BLACK)}
        small_font, small_glyphs = load_piece_font(17, use_letters)
        self.small_pieces = {(pt, c): render_piece(small_font, small_glyphs, chess.Piece(pt, c))
                             for pt in small_glyphs for c in (chess.WHITE, chess.BLACK)}
        brand_font, brand_glyphs = load_piece_font(46, use_letters)
        self.brand = brand_font.render(brand_glyphs[chess.KNIGHT], True, ACCENT)

        self.f = {k: font(*v) for k, v in {
            "eyebrow": (10, True), "tiny": (10,), "small": (12,), "body": (13,), "bold": (14, True),
            "name": (15, True), "h1": (26, True), "h2": (22,), "clock": (20,), "score": (27,),
            "avatar": (24, False, True), "notice": (30, False, True), "coord": (11, True),
            "result_big": (68, False, True), "result_head": (28, True), "result_reason": (19,),
            "result_next": (15, True),
            "move": (13, True), "h3": (17,)}.items()}
        self.load(index)

    def now(self):
        """Milliseconds: real time normally, frame-based time while recording."""
        if self.record:
            return int(self.frame * 1000 / self.fps)
        return pygame.time.get_ticks()

    # ------------------------------------------------------------ navigation
    def load(self, index):
        self.index = max(0, min(index, len(self.games) - 1))
        self.game = self.games[self.index]
        self.nodes = list(self.game.mainline())
        self.sans, b = [], self.game.board()
        for node in self.nodes:
            self.sans.append(b.san(node.move))
            b.push(node.move)
        self.infos = [parse_comment(n.comment or "") for n in self.nodes]
        self.goto(0)

    def goto(self, ply):
        self.ply = max(0, min(ply, len(self.nodes)))
        self.board = self.game.board()
        for node in self.nodes[:self.ply]:
            self.board.push(node.move)
        self.restart_timer()

    def restart_timer(self):
        self.last_step = self.now()
        if self.ply >= len(self.nodes):
            self.wait = self.end_pause
        else:
            self.wait = self.seconds + random.uniform(0, self.jitter * self.seconds)

    @property
    def finished(self):
        return self.ply >= len(self.nodes)

    # ------------------------------------------------------------ game facts
    def names(self):
        h = self.game.headers
        return h.get("White", "White"), h.get("Black", "Black")

    def result_text(self):
        h = self.game.headers
        result = h.get("Result", "*")
        term = h.get("Termination", "")
        nice = TERMINATION_TEXT.get(term, term.replace("_", " ").capitalize() if term else "")
        white, black = self.names()
        if result == "1-0":
            return "1–0", f"{nice} · {short_name(white)} wins"
        if result == "0-1":
            return "0–1", f"{nice} · {short_name(black)} wins"
        if result == "1/2-1/2":
            return "½–½", nice or "Draw"
        return "—", nice or "Unfinished"

    def series_score(self):
        """Score between the current pair over the games played so far in this file."""
        pair = sorted(short_name(n) for n in self.names())
        score = {pair[0]: 0.0, pair[1]: 0.0}
        played = 0
        for i, g in enumerate(self.games[:self.index + 1]):
            if i == self.index and not self.finished:
                break
            w, b = short_name(g.headers.get("White", "")), short_name(g.headers.get("Black", ""))
            if sorted([w, b]) != pair:
                continue
            played += 1
            r = g.headers.get("Result", "*")
            if r == "1-0":
                score[w] += 1
            elif r == "0-1":
                score[b] += 1
            elif r == "1/2-1/2":
                score[w] += 0.5
                score[b] += 0.5
        return pair, score, played

    def mover_of(self, i):
        """Colour that played move index i (0-based)."""
        start = self.game.board().turn
        return start if i % 2 == 0 else (not start)

    def last_decision(self, color):
        """Info about the most recent move made by `color` (up to the current position)."""
        for i in range(self.ply - 1, -1, -1):
            if self.mover_of(i) == color:
                return i, self.infos[i]
        return None, None

    def captured_by(self, color):
        """Opponent pieces that `color` has captured, and `color`'s material edge."""
        opp = not color
        lost = []
        for pt, n in START_COUNT.items():
            lost += [pt] * max(0, n - len(self.board.pieces(pt, opp)))
        edge = sum(v * (len(self.board.pieces(p, color)) - len(self.board.pieces(p, opp)))
                   for p, v in VALUES.items())
        return lost, edge

    # ------------------------------------------------------------ drawing primitives
    def text(self, s, key, color, pos, anchor="topleft"):
        surf = self.f[key].render(s, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def fit(self, s, key, max_w):
        while s and self.f[key].size(s)[0] > max_w:
            s = s[:-2] + "…" if len(s) > 2 else ""
        return s

    def box(self, rect, fill, border=None, radius=8, width=1):
        pygame.draw.rect(self.screen, fill, rect, border_radius=radius)
        if border:
            pygame.draw.rect(self.screen, border, rect, width, border_radius=radius)

    # ------------------------------------------------------------ sections
    def draw_rail(self):
        pygame.draw.rect(self.screen, RAIL, (0, 0, RAIL_W, H))
        pygame.draw.line(self.screen, LINE, (RAIL_W - 1, 0), (RAIL_W - 1, H))
        self.screen.blit(self.brand, self.brand.get_rect(center=(RAIL_W // 2, 50)))
        self.text("MODEL", "tiny", MUTED, (RAIL_W // 2, 88), "midtop")
        self.text("MATCHES", "eyebrow", TEXT, (RAIL_W // 2, 102), "midtop")
        pygame.draw.line(self.screen, LINE, (RAIL_W // 2 - 18, 132), (RAIL_W // 2 + 18, 132))
        pygame.draw.circle(self.screen, ACCENT, (RAIL_W // 2, H - 64), 3)
        self.text("REPLAY", "tiny", MUTED, (RAIL_W // 2, H - 52), "midtop")

    def draw_header(self):
        self.text("CHESS   /   REPLAY", "tiny", MUTED, (BOARD_X, 22))
        label = self.fit(self.title, "small", 380)
        r = self.text(label, "small", MUTED, (W - 28, 20), "topright")
        pygame.draw.circle(self.screen, ACCENT, (r.left - 10, r.centery), 3)
        pygame.draw.line(self.screen, LINE, (RAIL_W, 56), (W, 56))
        white, black = self.names()
        self.text(self.fit(f"{short_name(white)} vs {short_name(black)}", "h1", 540), "h1", TEXT,
                  (BOARD_X, 70))
        self.text("Decision models choosing every move · replayed from the saved game record",
                  "small", MUTED, (BOARD_X, 102))

    def draw_player(self, color, y):
        name = self.names()[0 if color == chess.WHITE else 1]
        letter, accent, desc = identity(name)
        thinking = (self.playing and not self.finished and self.board.turn == color)
        # avatar
        av = pygame.Rect(BOARD_X, y + 4, 43, 43)
        self.box(av, lerp(BG, accent, 0.22), lerp(BG, accent, 0.45), radius=9)
        self.text(letter, "avatar", accent, av.center, "center")
        # name + colour tag + description
        r = self.text(self.fit(short_name(name), "name", 200), "name", TEXT, (BOARD_X + 55, y + 3))
        self.text("WHITE" if color == chess.WHITE else "BLACK", "tiny", MUTED, (r.right + 9, y + 7))
        self.text(self.fit(desc, "tiny", 300), "tiny", MUTED, (BOARD_X + 55, y + 23))
        # captures + material edge
        lost, edge = self.captured_by(color)
        x = BOARD_X + 55
        for pt in sorted(lost, key=lambda p: -VALUES[p]):
            img = self.small_pieces[(pt, not color)]
            self.screen.blit(img, (x, y + 34))
            x += 11
        if edge > 0:
            self.text(f"+{edge}", "tiny", MUTED, (x + 12, y + 37))
        # last-decision clock
        clock = pygame.Rect(BOARD_X + BOARD_PX - 120, y + 4, 120, 44)
        if thinking:
            self.box(clock, THINK_BG, THINK_BORDER, radius=7)
        else:
            self.box(clock, CLOCK_BG, LINE, radius=7)
        label_color = THINK_TEXT if thinking else MUTED
        self.text("THINKING" if thinking else "LAST DECISION", "tiny", label_color,
                  (clock.right - 10, clock.top + 5), "topright")
        _, info = self.last_decision(color)
        if thinking:
            value = f"{(self.now() - self.last_step) / 1000:.1f}s"
        elif info is None or info["random_opening"]:
            value = "—"
        elif info["time"] is not None:
            value = f"{info['time']:.2f}s"
        elif info["p"] is not None:
            value = f"p {info['p']:.2f}"
        else:
            value = "—"
        self.text(value, "clock", THINK_TEXT if thinking else TEXT,
                  (clock.right - 10, clock.bottom - 4), "bottomright")

    def square_xy(self, sq):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if self.flipped:
            f, r = 7 - f, 7 - r
        return BOARD_X + f * SQ, BOARD_Y + (7 - r) * SQ

    def draw_board(self):
        frame = pygame.Rect(BOARD_X - 5, BOARD_Y - 5, BOARD_PX + 10, BOARD_PX + 10)
        self.box(frame, FRAME, radius=5)
        last = self.nodes[self.ply - 1].move if self.ply else None
        check = self.board.king(self.board.turn) if self.board.is_check() else None
        for sq in chess.SQUARES:
            x, y = self.square_xy(sq)
            light = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1
            color = LIGHT_SQ if light else DARK_SQ
            if last and sq in (last.from_square, last.to_square):
                color = LAST_LIGHT if light else LAST_DARK
            pygame.draw.rect(self.screen, color, (x, y, SQ, SQ))
            if sq == check:
                pygame.draw.circle(self.screen, CHECK_SQ, (x + SQ // 2, y + SQ // 2), SQ // 2 - 2)
            piece = self.board.piece_at(sq)
            if piece:
                img = self.pieces[(piece.piece_type, piece.color)]
                self.screen.blit(img, img.get_rect(center=(x + SQ // 2, y + SQ // 2 + 1)))
        # coordinates (rank top-left, file bottom-right, like the repo)
        left_file = 7 if self.flipped else 0
        bottom_rank = 7 if self.flipped else 0
        for i in range(8):
            f = 7 - i if self.flipped else i          # file shown in column i
            r = i if self.flipped else 7 - i          # rank shown in row i
            dark_left = (left_file + r) % 2 == 0      # a1 is a dark square
            self.text(str(r + 1), "coord", COORD_ON_DARK if dark_left else COORD_ON_LIGHT,
                      (BOARD_X + 4, BOARD_Y + i * SQ + 3))
            dark_bottom = (f + bottom_rank) % 2 == 0
            self.text("abcdefgh"[f], "coord", COORD_ON_DARK if dark_bottom else COORD_ON_LIGHT,
                      (BOARD_X + (i + 1) * SQ - 4, BOARD_Y + BOARD_PX - 2), "bottomright")
        if self.finished:
            self.draw_notice()

    def result_parts(self):
        """(score like '1–0', headline like 'Jev wins' or 'Draw', reason like 'Checkmate')."""
        h = self.game.headers
        result, term = h.get("Result", "*"), h.get("Termination", "")
        reason = {"checkmate": "Checkmate", "stalemate": "Stalemate",
                  "threefold_repetition": "Threefold repetition",
                  "fivefold_repetition": "Fivefold repetition",
                  "fifty_moves": "Fifty-move rule", "seventyfive_moves": "75-move rule",
                  "insufficient_material": "Insufficient material",
                  "move limit": "Move limit reached"}.get(term, term.replace("_", " ").capitalize())
        white, black = self.names()
        if result == "1-0":
            return "1–0", f"{short_name(white)} wins", reason
        if result == "0-1":
            return "0–1", f"{short_name(black)} wins", reason
        if result == "1/2-1/2":
            return "½–½", "Draw", reason
        return "—", "Unfinished", reason

    def draw_notice(self):
        score, headline, reason = self.result_parts()
        rect = pygame.Rect(0, 0, 440, 210)
        rect.center = (BOARD_X + BOARD_PX // 2, BOARD_Y + BOARD_PX // 2)
        shadow = rect.move(0, 6)
        pygame.draw.rect(self.screen, lerp(BG, (0, 0, 0), 0.45), shadow, border_radius=14)
        pygame.draw.rect(self.screen, NOTICE_BG[:3], rect, border_radius=14)   # solid card
        pygame.draw.rect(self.screen, NOTICE_BORDER, rect, 2, border_radius=14)
        self.text(score, "result_big", TEXT, (rect.centerx, rect.top + 14), "midtop")
        self.text(self.fit(headline, "result_head", 400), "result_head", ACCENT,
                  (rect.centerx, rect.top + 100), "midtop")
        if reason:
            self.text(self.fit(reason, "result_reason", 400), "result_reason", MUTED,
                      (rect.centerx, rect.top + 140), "midtop")
        if self.all_games and self.index < len(self.games) - 1 and self.playing:
            left = max(0, self.wait - (self.now() - self.last_step) / 1000)
            self.text(f"Next game in {left:.0f}s", "result_next", MUTED,
                      (rect.centerx, rect.top + 174), "midtop")

    def draw_toolbar(self):
        y = BOARD_Y + BOARD_PX + 64 + 10
        pygame.draw.line(self.screen, LINE, (BOARD_X, y), (BOARD_X + BOARD_PX, y))
        if self.finished:
            status = self.result_text()[1]
        elif self.ply == 0:
            status = "Starting position"
        else:
            side = "White" if self.board.turn == chess.WHITE else "Black"
            status = f"Move {self.board.fullmove_number} · {side} to move" + (" · check" if self.board.is_check() else "")
        pygame.draw.circle(self.screen, ACCENT, (BOARD_X + 3, y + 17), 3)
        self.text(self.fit(status, "small", 330), "small", MUTED, (BOARD_X + 12, y + 9))
        self.text(f"LEGAL MOVES  {self.board.legal_moves.count()}", "tiny", MUTED,
                  (BOARD_X + BOARD_PX, y + 11), "topright")

    def draw_panel(self):
        x, y, w = PANEL_X, PANEL_Y, PANEL_W
        self.box(pygame.Rect(x, y, w, PANEL_H), PANEL, LINE, radius=12)
        white, black = self.names()
        # heading + status pill
        self.text("HEAD TO HEAD", "eyebrow", ACCENT, (x + 22, y + 20))
        pair, score, played = self.series_score()
        heading = self.fit(f"{pair[0]}  vs  {pair[1]}", "h2", 250)
        self.text(heading, "h2", TEXT, (x + 22, y + 36))
        state = "FINISHED" if self.finished else ("PLAYING" if self.playing else "PAUSED")
        pill_bg, pill_fg = (PILL_ON, ACCENT) if state == "PLAYING" else (PILL_BG, MUTED)
        pill_w = self.f["tiny"].size(state)[0] + 18
        self.box(pygame.Rect(x + w - 22 - pill_w, y + 22, pill_w, 22), pill_bg, radius=4)
        self.text(state, "tiny", pill_fg, (x + w - 22 - pill_w // 2, y + 33), "center")
        # score box
        sb = pygame.Rect(x + 20, y + 76, w - 40, 64)
        self.box(sb, DEEP, LINE, radius=8)
        for i, name in enumerate(pair):
            cx = sb.left + 52 if i == 0 else sb.right - 52
            self.text(self.fit(name.upper(), "tiny", 100), "tiny", MUTED, (cx, sb.top + 10), "midtop")
            val = score[name]
            self.text(f"{val:g}", "score", SCORE_TEXT, (cx, sb.top + 24), "midtop")
        self.text(f"GAME {self.index + 1:02d}", "tiny", MUTED, (sb.centerx, sb.top + 18), "midtop")
        self.text(f"of {len(self.games)} in file", "tiny", FAINT, (sb.centerx, sb.top + 34), "midtop")
        # moves section
        ty = y + 156
        self.text("Moves", "body", TEXT, (x + 22, ty))
        cnt = self.f["tiny"].render(str(self.ply), True, TEXT)
        self.box(pygame.Rect(x + 72, ty + 1, cnt.get_width() + 10, 17), BADGE, radius=4)
        self.screen.blit(cnt, (x + 77, ty + 3))
        pygame.draw.line(self.screen, ACCENT, (x + 22, ty + 24), (x + 105, ty + 24), 2)
        pygame.draw.line(self.screen, LINE, (x, ty + 26), (x + w, ty + 26))
        self.draw_moves(x, ty + 27, w, 218)
        # decision details
        self.draw_details(x, ty + 252, w)
        # phase line
        py = y + PANEL_H - 74
        pygame.draw.rect(self.screen, PHASE_BG, (x + 1, py, w - 2, 36))
        pygame.draw.line(self.screen, LINE, (x, py), (x + w, py))
        pygame.draw.line(self.screen, LINE, (x, py + 36), (x + w, py + 36))
        if self.finished:
            phase = "Game over · " + self.result_text()[1]
        elif not self.playing:
            phase = "Paused · press Space to resume"
        else:
            mover = white if self.board.turn == chess.WHITE else black
            phase = f"{short_name(mover)} is choosing a move…"
        pulse = 0.5 + 0.5 * math.sin(self.now() / 160) if (self.playing and not self.finished) else 1
        pygame.draw.circle(self.screen, lerp(PHASE_BG, ACCENT, 0.25 + 0.75 * pulse), (x + 24, py + 18), 3)
        self.text(self.fit(phase, "small", w - 60), "small", PHASE_TEXT, (x + 36, py + 10))
        # footer: keys
        self.text("SPACE pause  ·  ←/→ step  ·  ↑/↓ game  ·  +/− speed", "tiny", MUTED,
                  (x + w // 2, y + PANEL_H - 24), "midtop")

    def draw_moves(self, x, y, w, h):
        pygame.draw.rect(self.screen, ROW_ALT, (x + 1, y, w - 2, 24))
        self.text("#", "tiny", MUTED, (x + 22, y + 6))
        self.text("WHITE", "tiny", MUTED, (x + 62, y + 6))
        self.text("BLACK", "tiny", MUTED, (x + 62 + (w - 80) // 2, y + 6))
        top, row_h = y + 24, 26
        rows = (h - 24) // row_h
        first_black = self.game.board().turn == chess.BLACK
        offset = 1 if first_black else 0
        total_rows = (self.ply + offset + 1) // 2
        cur_row = (self.ply - 1 + offset) // 2 if self.ply else 0
        start = max(0, min(cur_row - rows + 2, total_rows - rows))
        start_no = self.game.board().fullmove_number
        clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(x + 1, top, w - 2, rows * row_h))
        for r in range(start, min(total_rows, start + rows)):
            ry = top + (r - start) * row_h
            if (r - start) % 2 == 1:
                pygame.draw.rect(self.screen, ROW_ALT, (x + 1, ry, w - 2, row_h))
            self.text(f"{start_no + r}", "tiny", MUTED, (x + 22, ry + 7))
            for side in (0, 1):
                i = r * 2 + side - offset
                if 0 <= i < self.ply:
                    cx = x + 52 + side * ((w - 80) // 2)
                    if i == self.ply - 1:
                        self.box(pygame.Rect(cx, ry + 3, 92, row_h - 6), ACTIVE_ROW, radius=4)
                        color = ACTIVE_TEXT
                    else:
                        color = TEXT
                    self.text(self.sans[i], "move", color, (cx + 10, ry + 5))
        self.screen.set_clip(clip)
        if self.ply == 0:
            self.text("♙", "notice", FAINT, (x + w // 2, top + 30), "midtop")
            self.text("No moves yet.", "small", MUTED, (x + w // 2, top + 78), "midtop")

    def draw_details(self, x, y, w):
        pygame.draw.line(self.screen, LINE, (x, y - 6), (x + w, y - 6))
        self.text("LATEST MODEL CHOICE", "tiny", MUTED, (x + 22, y + 2))
        if self.ply == 0:
            self.text("Waiting for the first move", "h3", TEXT, (x + 22, y + 18))
            return
        i = self.ply - 1
        info, san = self.infos[i], self.sans[i]
        mover = self.names()[0 if self.mover_of(i) == chess.WHITE else 1]
        if info["random_opening"]:
            self.text(f"{san}  · random opening move", "h3", TEXT, (x + 22, y + 18))
            self.text("The first few moves are random so every game starts differently.",
                      "tiny", MUTED, (x + 22, y + 46))
            return
        self.text(self.fit(f"{san}  by {short_name(mover)}", "h3", w - 44), "h3", TEXT, (x + 22, y + 18))
        cands = ([(san, info["p"])] if info["p"] is not None else []) + info["alts"][:3]
        if not cands:
            self.text("No scores recorded for this move.", "tiny", MUTED, (x + 22, y + 46))
            return
        for k, (move, p) in enumerate(cands):
            ry = y + 48 + k * 22
            self.text(move, "small", TEXT if k == 0 else MUTED, (x + 22, ry))
            track = pygame.Rect(x + 96, ry + 6, w - 170, 5)
            pygame.draw.rect(self.screen, TRACK, track, border_radius=3)
            fill = track.copy()
            fill.width = max(2, int(track.width * max(0.0, min(1.0, p))))
            pygame.draw.rect(self.screen, ACCENT if k == 0 else ALT_BAR, fill, border_radius=3)
            self.text(f"{p:.2f}", "tiny", MUTED, (x + w - 22, ry + 1), "topright")
        self.text("Scores describe the model's move preferences, not its chance of winning.",
                  "tiny", FAINT, (x + 22, y + 52 + len(cands) * 22))

    def draw(self):
        self.screen.fill(BG)
        self.draw_rail()
        self.draw_header()
        top_color = chess.WHITE if self.flipped else chess.BLACK
        self.draw_player(top_color, BOARD_Y - 62)
        self.draw_board()
        self.draw_player(not top_color, BOARD_Y + BOARD_PX + 8)
        self.draw_toolbar()
        self.draw_panel()
        if self.scale == 1.0:
            self.window.blit(self.screen, (0, 0))
        else:
            self.window.blit(pygame.transform.smoothscale(self.screen, self.window.get_size()), (0, 0))
        pygame.display.flip()

    # ------------------------------------------------------------ loop
    def run(self):
        writer = None
        if self.record:
            import imageio
            writer = imageio.get_writer(self.record, fps=self.fps, codec="libx264",
                                        quality=8, macro_block_size=1, pixelformat="yuv420p")
            total = sum(len(list(g.mainline_moves())) for g in
                        (self.games[self.index:] if self.all_games else [self.game]))
            print(f"Recording to {self.record} ({total} moves). Ctrl+C stops early and keeps "
                  f"the video so far.")
            t0 = time.time()
        try:
            self._loop(writer)
        except KeyboardInterrupt:
            print("\nStopped early.")
        if writer:
            writer.close()
            print(f"Saved {self.record}: {self.frame / self.fps:.0f}s of video "
                  f"(rendered in {time.time() - t0:.0f}s)")
        pygame.quit()

    def _loop(self, writer):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k == pygame.K_ESCAPE:
                        running = False
                    elif k == pygame.K_RIGHT:
                        self.playing = False
                        self.goto(self.ply + 1)
                    elif k == pygame.K_LEFT:
                        self.playing = False
                        self.goto(self.ply - 1)
                    elif k == pygame.K_HOME:
                        self.goto(0)
                    elif k == pygame.K_END:
                        self.playing = False
                        self.goto(len(self.nodes))
                    elif k == pygame.K_SPACE:
                        self.playing = not self.playing
                        last_game = self.index == len(self.games) - 1 or not self.all_games
                        if self.playing and self.finished and last_game:
                            self.goto(0)          # finished: Space replays this game
                        else:
                            self.restart_timer()
                    elif k == pygame.K_DOWN:
                        self.load(self.index + 1)
                    elif k == pygame.K_UP:
                        self.load(self.index - 1)
                    elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.seconds = max(0.1, self.seconds / 1.5)
                        self.restart_timer()
                    elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.seconds = min(10.0, self.seconds * 1.5)
                        self.restart_timer()
            if self.playing and self.now() - self.last_step >= self.wait * 1000:
                if not self.finished:
                    self.goto(self.ply + 1)
                elif self.all_games and self.index < len(self.games) - 1:
                    self.load(self.index + 1)     # result shown; on to the next game
                else:
                    self.playing = False          # last game finished
                    if writer:
                        running = False           # recording: stop after the final result
            self.draw()
            if writer:
                frame = pygame.surfarray.array3d(self.window).swapaxes(0, 1)
                writer.append_data(frame)
                self.frame += 1
                if self.frame % (self.fps * 20) == 0:
                    print(f"  {self.frame / self.fps:5.0f}s of video | game {self.index + 1}, "
                          f"move {self.ply}/{len(self.nodes)}", flush=True)
            else:
                self.clock.tick(30)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pgn", help="a .pgn file (e.g. arena_runs/.../games.pgn)")
    ap.add_argument("--game", type=int, default=1, help="which game in the file to start with (1 = first)")
    ap.add_argument("--flip", action="store_true", help="show Black at the bottom")
    ap.add_argument("--seconds", type=float, default=1.0,
                    help="base 'thinking' pause per move, in seconds")
    ap.add_argument("--jitter", type=float, default=0.6,
                    help="random extra pause, as a fraction of --seconds (0 = fixed pace)")
    ap.add_argument("--end-pause", type=float, default=4.0,
                    help="seconds the final position stays on screen before the next game")
    ap.add_argument("--one-game", action="store_true", help="stop after the starting game")
    ap.add_argument("--paused", action="store_true", help="start paused")
    ap.add_argument("--scale", type=float, default=1.0, help="window size factor, e.g. 0.85")
    ap.add_argument("--theme", default="walnut", choices=sorted(THEMES),
                    help="colour theme: walnut (dark wood, default) or parchment (light paper)")
    ap.add_argument("--record", default=None, metavar="VIDEO.mp4",
                    help="save the replay as an MP4 video instead of showing a window")
    ap.add_argument("--fps", type=int, default=30, help="video frames per second (with --record)")
    ap.add_argument("--show", action="store_true",
                    help="with --record, also show the window while rendering")
    ap.add_argument("--letters", action="store_true", help="draw pieces as letters")
    args = ap.parse_args()
    use_theme(args.theme)
    if args.record and not args.show:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # render without opening a window
    games = load_games(args.pgn)
    print(f"{len(games)} game(s) in {args.pgn}")
    ReplayViewer(games, args.game - 1, args.flip, args.letters, args.seconds, args.jitter,
                 args.end_pause, all_games=not args.one_game, start_playing=not args.paused,
                 scale=args.scale, title=os.path.basename(args.pgn),
                 record=args.record, fps=args.fps).run()


if __name__ == "__main__":
    main()