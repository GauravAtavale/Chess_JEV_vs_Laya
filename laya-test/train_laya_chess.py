"""
Fine-tune Laya on Stockfish-labelled chess positions.

Each training example is one (position, candidate move) pair, rendered exactly
as laya_player.py renders it at play time, with the yes/no question
"Will the side to move win the game after playing the candidate move?".
The soft target is Stockfish's expected score after that move.

The training loop follows the official Laya fine-tuning notebook (RLCD:
policy gradient on a strictly proper scoring rule + soft cross-entropy).

    # 1) smoke test on your Mac first (a few minutes, checks everything runs)
    python train_laya_chess.py --data smoke.jsonl --smoke

    # 2) real run on Kaggle with two T4 GPUs
    torchrun --standalone --nproc_per_node=2 train_laya_chess.py \
        --data chess_positions.jsonl --out /kaggle/working/laya_chess

    # 3) single GPU (Colab, or Kaggle with one GPU)
    python train_laya_chess.py --data chess_positions.jsonl --out laya_chess

Needs laya_player.py next to this file (so training and play use identical inputs).
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chess  # noqa: E402
import torch  # noqa: E402

from laya_player import build_state, make_questions, QUESTION_KEY  # noqa: E402

QUESTION_VARIANT = "win"
BASE_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]


# ------------------------------------------------------------------ setup
def setup_distributed():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import datetime
        import torch.distributed as dist
        # generous timeout: rank 0 evaluates alone before and after training
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return dist, dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return None, 0, 1, device


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def get_base_model(base):
    if os.path.isdir(base):
        return base
    from huggingface_hub import snapshot_download
    return snapshot_download(base, allow_patterns=BASE_FILES)


# ------------------------------------------------------------------ data
def load_positions(path, max_positions=None):
    splits = {"train": [], "val": [], "test": []}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            # positions where every move scores the same carry no signal about move choice
            if r["moves"][0][1] - r["moves"][-1][1] < 0.01:
                continue
            splits[r["split"]].append(r)
    if max_positions:
        splits["train"] = splits["train"][:max_positions]
    return splits


def pick_moves(record, k, rng):
    """The top 2 moves plus a random sample of the rest (keeps good and bad moves)."""
    moves = record["moves"]
    if len(moves) <= k:
        return moves
    return moves[:2] + rng.sample(moves[2:], k - 2)


def make_item(tok, cfg, internal, record, uci, expected):
    from laya.common import QTYPES, build_sequence
    board = chess.Board(record["fen"])
    move = chess.Move.from_uci(uci)
    state = build_state(board, move, record["recent"])
    ids, markers = build_sequence(tok, state, internal, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != 2:
        return None
    e = min(1.0, max(0.0, float(expected)))
    return {"ids": ids, "markers": markers, "qtype": QTYPES["noul"],
            "target": [1.0 - e, e], "label": int(e >= 0.5)}


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
            "target": target, "qtype": torch.tensor([it["qtype"] for it in items])}


def forward(model, batch, device):
    use_amp = device.type == "cuda"
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                            batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                            batch["qtype"].to(device))
    return logits.float(), act


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate(model, tok, cfg, internal, positions, device, batch_size=32):
    """Score every legal move of each position; compare the chosen move with Stockfish's."""
    model.eval()
    top1, loss_model, loss_random = 0, 0.0, 0.0
    for rec in positions:
        items = [make_item(tok, cfg, internal, rec, uci, e) for uci, e in rec["moves"]]
        p_true = []
        for i in range(0, len(items), batch_size):
            logits, _ = forward(model, collate(items[i:i + batch_size], tok.pad_token_id), device)
            p_true += torch.softmax(logits[:, :2], -1)[:, 1].tolist()
        chosen = max(range(len(p_true)), key=lambda j: p_true[j])
        best = rec["moves"][0][1]
        top1 += int(rec["moves"][chosen][1] >= best - 1e-9)
        loss_model += best - rec["moves"][chosen][1]
        loss_random += best - sum(e for _, e in rec["moves"]) / len(rec["moves"])
    n = max(1, len(positions))
    model.train()
    return {"positions": len(positions), "top1_match": top1 / n,
            "avg_expected_score_lost": loss_model / n,
            "random_move_expected_score_lost": loss_random / n}


def fit_temperature(model, items, pad_id, device):
    """Fit one softmax temperature for noul answers on held-out items."""
    zs, ts = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(items), 32):
            b = collate(items[i:i + 32], pad_id)
            logits, _ = forward(model, b, device)
            zs.append(logits[:, :2].cpu())
            ts.append(b["target"][:, :2])
    model.train()
    if not zs:
        return 1.0
    Z, T = torch.cat(zs), torch.cat(ts)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.5, 5.0).item())


def save_model(model, tok, cfg, out_dir, extra_cfg=None):
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out_dir, "encoder"))
    tok.save_pretrained(os.path.join(out_dir, "tokenizer"))
    c = dict(cfg)
    c.update(extra_cfg or {})
    with open(os.path.join(out_dir, "rl_agent_config.json"), "w") as f:
        json.dump(c, f, indent=2)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="convaiinnovations/laya",
                    help="Hub id or local folder of the checkpoint to start from")
    ap.add_argument("--out", default="laya_chess")
    ap.add_argument("--moves-per-position", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--no-rl", action="store_true", help="soft cross-entropy only (faster)")
    ap.add_argument("--max-train-positions", type=int, default=None)
    ap.add_argument("--eval-positions", type=int, default=200)
    ap.add_argument("--skip-baseline-eval", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to check the pipeline works (use on your Mac)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        args.max_train_positions = args.max_train_positions or 30
        args.epochs, args.eval_positions, args.micro_batch, args.grad_accum = 1, 5, 4, 1
        args.moves_per_position = 4

    dist, rank, world, device = setup_distributed()
    torch.manual_seed(args.seed)

    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.agent import Agent, _fix_tokenizer_config
    from laya.common import build_model, proper_reward

    # ---- model and tokenizer (rank 0 downloads first)
    if rank == 0:
        model_dir = get_base_model(args.base)
    if dist:
        dist.barrier()
    if rank != 0:
        model_dir = get_base_model(args.base)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    log(rank, f"Base model: {model_dir} | max_len {cfg['max_len']} | head_max_len {cfg['head_max_len']}"
              f" | device {device} | {world} process(es)")

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    if args.smoke:
        # the smoke test only checks the pipeline: train the small decision head and keep the
        # encoder frozen, so it fits in a laptop's memory
        for p in model.encoder.parameters():
            p.requires_grad_(False)
    else:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    model.train()

    # ---- data
    question = make_questions(variant=QUESTION_VARIANT)[QUESTION_KEY]
    internal = Agent._to_internal(question)
    splits = load_positions(args.data, args.max_train_positions)
    log(rank, f"Positions: train {len(splits['train'])}, val {len(splits['val'])}, "
              f"test {len(splits['test'])}")
    if not splits["train"]:
        raise SystemExit("No training positions found in " + args.data)

    rng = random.Random(args.seed)   # same on every rank, so every rank builds the same list
    train_items = []
    for rec in splits["train"]:
        for uci, e in pick_moves(rec, args.moves_per_position, rng):
            it = make_item(tok, cfg, internal, rec, uci, e)
            if it:
                train_items.append(it)
    calib_items = []
    for rec in splits["val"][: max(50, args.eval_positions)]:
        for uci, e in pick_moves(rec, args.moves_per_position, rng):
            it = make_item(tok, cfg, internal, rec, uci, e)
            if it:
                calib_items.append(it)
    lengths = sorted(len(it["ids"]) for it in train_items)
    log(rank, f"Training items: {len(train_items)} | tokens per item: median "
              f"{lengths[len(lengths) // 2]}, max {lengths[-1]} (limit {cfg['max_len']})")
    if lengths[-1] >= cfg["max_len"]:
        log(rank, "WARNING: some inputs hit the length limit and were truncated")

    eval_set = (splits["val"] or splits["test"])[: args.eval_positions]
    if rank == 0 and eval_set and not args.skip_baseline_eval:
        t = time.time()
        base_metrics = evaluate(model, tok, cfg, internal, eval_set, device)
        log(rank, f"Before training: {base_metrics} ({time.time() - t:.0f}s)")
    else:
        base_metrics = None

    # ---- optimisation (same recipe as the official Laya notebook)
    train_model = model
    if dist:
        from torch.nn.parallel import DistributedDataParallel as DDP
        train_model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    my_items = train_items[rank::world]

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    groups = [{"params": head_params, "lr": args.lr_head}]
    if enc_params:
        groups.insert(0, {"params": enc_params, "lr": args.lr_encoder})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    total_updates = max(1, math.ceil(len(my_items) / (args.micro_batch * args.grad_accum)) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    GROUP, SIGMA_START, SIGMA_END = 4, 0.4, 0.1

    log(rank, f"Training {args.epochs} epoch(s), {len(my_items)} items per process, "
              f"{total_updates} optimizer steps")
    t0 = time.time()
    for epoch in range(args.epochs):
        random.Random(args.seed + 100 * epoch + rank).shuffle(my_items)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (epoch / max(1, args.epochs - 1))
        running, n_batches, accum = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        n_steps = math.ceil(len(my_items) / args.micro_batch)
        for b in range(0, len(my_items), args.micro_batch):
            batch = collate(my_items[b:b + args.micro_batch], tok.pad_token_id)
            logits, act = forward(train_model, batch, device)
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            qtype = batch["qtype"].to(device)
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            if args.no_rl:
                loss = loss_ce
            else:
                k = mask.sum(-1, keepdim=True).float()
                eps = torch.randn((GROUP,) + logits.shape, device=device) * sigma * mask
                eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
                z = logits.detach().unsqueeze(0) + eps
                q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
                with torch.no_grad():
                    r = proper_reward(q, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                    adv = (r - r.mean(0, keepdim=True)) / (r.std() + 1e-6)
                logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
                loss = -(adv * logp).mean() + loss_ce
            loss = loss / args.grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            accum += 1
            if accum % args.grad_accum == 0 or b + args.micro_batch >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss_ce.item()
            n_batches += 1
            if rank == 0 and (n_batches % 50 == 0 or args.smoke):
                el = time.time() - t0
                done = epoch * n_steps + n_batches
                eta = el / done * (args.epochs * n_steps - done) / 60
                print(f"  epoch {epoch + 1} step {n_batches}/{n_steps} | soft-CE {running / n_batches:.4f}"
                      f" | {el / 60:.1f} min elapsed, ETA {eta:.0f} min", flush=True)
        if dist:
            dist.barrier()
        if rank == 0:
            ckpt = os.path.join(args.out, "checkpoint_latest")
            save_model(model, tok, cfg, ckpt, {"model_name": "laya-chess", "chess_question": QUESTION_VARIANT})
            print(f"=== epoch {epoch + 1} done, soft-CE {running / max(1, n_batches):.4f}, "
                  f"checkpoint saved to {ckpt}", flush=True)

    # ---- calibration, final evaluation, save
    if rank == 0:
        del optimizer, scaler
        temp = fit_temperature(model, calib_items, tok.pad_token_id, device)
        temps = list(cfg.get("temperature", [1.0, 1.0, 1.0]))
        temps[2] = temp  # index 2 = noul
        extra = {"model_name": "laya-chess", "chess_question": QUESTION_VARIANT,
                 "fine_tuned": True, "temperature": temps}
        cfg.pop("temperature_by_options", None)  # would override the fitted noul temperature
        save_model(model, tok, cfg, args.out, extra)
        report = {"before": base_metrics}
        if eval_set:
            report["after_val"] = evaluate(model, tok, cfg, internal, eval_set, device)
        if splits["test"]:
            report["after_test"] = evaluate(model, tok, cfg, internal,
                                            splits["test"][: args.eval_positions], device)
        report["noul_temperature"] = temp
        with open(os.path.join(args.out, "eval_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("\nResults (top1_match = picked Stockfish's best move; "
              "expected_score_lost: 0 is perfect, lower is better):")
        print(json.dumps(report, indent=2))
        print(f"\nFine-tuned model saved to {args.out} ({(time.time() - t0) / 60:.0f} min)")
    if dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
