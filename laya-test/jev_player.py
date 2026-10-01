"""
Jev (TypeSafe) chess player: the same design as LayaPlayer.

For every legal move we send one state (position + that candidate move) and the
same yes/no (`noul`) question Laya gets, then play the move with the highest
P(yes). Jev has no batch endpoint, so the requests run in parallel threads.

Two ways to reach Jev (pick with --jev-provider or JEV_PROVIDER):
  typesafe    https://api.typesafe.ai/v1/systemone    key in TYPESAFE_API_KEY
  openrouter  https://openrouter.ai/api/v1/systemone  key in OPENROUTER_API_KEY

The key can be an environment variable or a line in a .env file in the project
folder (or the folder you run from), e.g.   TYPESAFE_API_KEY=ts-...
"""
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from laya_player import (QUESTION_KEY, build_state, make_questions, position_rule_facts,
                         recent_san, sample_move)

def load_dotenv_files():
    """Read KEY=VALUE lines from .env files; real environment variables take precedence."""
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in dict.fromkeys([os.getcwd(), here]):
        path = os.path.join(folder, ".env")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                elif " #" in value:
                    value = value.split(" #", 1)[0].rstrip()
                os.environ.setdefault(key, value)


load_dotenv_files()

PROVIDERS = {
    "typesafe": dict(url="https://api.typesafe.ai/v1/systemone",
                     key_env="TYPESAFE_API_KEY", model="jev-latest"),
    "openrouter": dict(url="https://openrouter.ai/api/v1/systemone",
                       key_env="OPENROUTER_API_KEY", model="typesafe/jev-1.13"),
}


class JevError(RuntimeError):
    pass


class JevPlayer:
    def __init__(self, provider=None, model=None, workers=8, use_ab_labels=False,
                 include_diagram=True, verbose=True, max_calls=5000,
                 temperature=0.0, top_k=3, seed=None, rule_facts=False):
        provider = provider or os.environ.get("JEV_PROVIDER") or (
            "openrouter" if os.environ.get("OPENROUTER_API_KEY")
            and not os.environ.get("TYPESAFE_API_KEY") else "typesafe")
        if provider not in PROVIDERS:
            raise JevError(f"provider must be one of {list(PROVIDERS)}")
        cfg = PROVIDERS[provider]
        self.key = os.environ.get(cfg["key_env"])
        if not self.key:
            raise JevError(f"{cfg['key_env']} not found. Add a line {cfg['key_env']}=your-key "
                           f"to the .env file in your project folder, or set it in your environment.")
        self.url = os.environ.get("JEV_URL", cfg["url"])  # override only for testing
        self.model = model or os.environ.get("JEV_MODEL") or cfg["model"]
        self.name = f"Jev ({self.model})"
        self.workers = workers
        self.include_diagram = include_diagram
        self.verbose = verbose
        self.rule_facts = rule_facts
        self.questions = make_questions(use_ab_labels, "strong_rules" if rule_facts else "strong")
        if rule_facts:
            self.name += " +rules"
        self.max_calls = max_calls  # safety cap on paid requests per session
        self.calls = 0
        self.input_tokens = 0
        self._lock = threading.Lock()
        self.last_scores = []
        self.temperature, self.top_k = temperature, top_k
        self.rng = random.Random(seed)
        print(f"[jev] using {provider}: {self.url} (model {self.model}), "
              f"cap {max_calls} requests this session")

    # ------------------------------------------------------------ HTTP
    def _post(self, payload, attempts=4):
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        delay = 0.5
        for attempt in range(attempts):
            req = urllib.request.Request(self.url, body, headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                retryable = e.code == 429 or e.code >= 500
                if not retryable or attempt == attempts - 1:
                    raise JevError(f"HTTP {e.code}: {detail}") from None
                wait = float(e.headers.get("retry-after") or delay)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == attempts - 1:
                    raise JevError(f"connection failed: {e}") from None
                wait = delay
            time.sleep(wait + random.random() * 0.2)
            delay = min(delay * 2, 5)

    def _ask(self, state):
        with self._lock:
            if self.calls >= self.max_calls:
                raise JevError(f"request cap of {self.max_calls} reached")
            self.calls += 1
        payload = {"model": self.model, "state": state, "questions": self.questions}
        try:
            data = self._post(payload)
        except JevError as e:
            # If Jev rejects the Laya-style criteria/labels fields, fall back to a
            # plain noul question and keep using it for the rest of the session.
            q = payload["questions"][QUESTION_KEY]
            if "422" in str(e) and ("criteria" in q or "labels" in q):
                plain = {QUESTION_KEY: {"type": "noul", "instructions": q["instructions"]}}
                with self._lock:
                    if "criteria" in self.questions[QUESTION_KEY] or "labels" in self.questions[QUESTION_KEY]:
                        print(f"[jev] request rejected ({e}); using instructions only from now on")
                        self.questions = plain
                payload["questions"] = plain
                data = self._post(payload)
            else:
                raise
        usage = data.get("usage") or {}
        with self._lock:
            self.input_tokens += int(usage.get("input_tokens") or 0)
        answer = data["answers"][QUESTION_KEY]
        p = float(answer["noul"])
        if not 0.0 <= p <= 1.0:
            raise JevError(f"invalid probability {p}")
        return p

    # ------------------------------------------------------------ player
    def score_moves(self, board):
        moves = list(board.legal_moves)
        recent = recent_san(board)
        pos = position_rule_facts(board) if self.rule_facts else None
        states = [build_state(board, m, recent, self.include_diagram, self.rule_facts, pos)
                  for m in moves]
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            probs = list(pool.map(self._ask, states))
        return sorted(zip(moves, probs), key=lambda x: -x[1])

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
                  f"top: {top} | spread {best_p - scored[-1][1]:.3f}{note} | "
                  f"session: {self.calls} requests, {self.input_tokens:,} input tokens")
        return move
