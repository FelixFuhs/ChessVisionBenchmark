import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import chess


def parse_fen64_to_board_array(fen64: str) -> List[str]:
    board = chess.Board(fen=f"{fen64} w - - 0 1")
    squares: List[str] = []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        squares.append(piece.symbol() if piece else ".")
    return squares


def square_accuracy(true_fen64: str, pred_fen64: str) -> Tuple[float, int]:
    try:
        B = parse_fen64_to_board_array(true_fen64)
        P = parse_fen64_to_board_array(pred_fen64)
        if len(B) != 64 or len(P) != 64:
            return 0.0, 64
        correct = sum(1 for b, p in zip(B, P) if b == p)
        acc = correct / 64.0
        hamming = 64 - correct
        return acc, hamming
    except Exception:
        return 0.0, 64


def clip_cp(x: int) -> int:
    return max(-1000, min(1000, int(x)))


@dataclass
class Scores:
    acc64: float
    hamming: float
    mae_cp: float
    composite: float


@dataclass
class ItemDetail:
    id: str
    acc64: float
    hamming: int
    abs_err: int
    fen_ok: bool
    eval_ok: bool


def _aggregate(accs: Iterable[float], hamms: Iterable[int], abs_errors: List[int]) -> Scores:
    # MAE defined as median in README
    abs_errors_sorted = sorted(abs_errors)
    n = len(abs_errors_sorted)
    if n == 0:
        mae_cp = 1000.0
    elif n % 2 == 1:
        mae_cp = float(abs_errors_sorted[n // 2])
    else:
        mae_cp = (abs_errors_sorted[n // 2 - 1] + abs_errors_sorted[n // 2]) / 2.0

    accs = list(accs)
    hamms = list(hamms)
    acc64 = sum(accs) / len(accs) if accs else 0.0
    hamming = sum(hamms) / len(hamms) if hamms else 64.0
    composite = 0.7 * acc64 + 0.3 * max(0.0, 1.0 - mae_cp / 1000.0)
    return Scores(acc64=acc64, hamming=hamming, mae_cp=mae_cp, composite=composite)


def evaluate_run(truths_path: Path, preds_path: Path) -> Scores:
    truths: Dict[str, Dict] = {}
    with open(truths_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            truths[obj["id"]] = obj

    preds: Dict[str, Dict] = {}
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            preds[obj["id"]] = obj

    ids = sorted(truths.keys())
    accs: List[float] = []
    hamms: List[int] = []
    abs_errors: List[int] = []

    for id_ in ids:
        t = truths[id_]
        p = preds.get(id_)
        # Task A
        if p and "fen64" in p:
            acc, h = square_accuracy(t["fen64"], p["fen64"])
        else:
            acc, h = 0.0, 64
        accs.append(acc)
        hamms.append(h)
        # Task B
        if p and isinstance(p.get("eval_cp_white"), (int, float, str)):
            try:
                pred = int(float(p["eval_cp_white"]))
            except Exception:
                pred = None
        else:
            pred = None
        if pred is None:
            abs_errors.append(1000)
        else:
            te = clip_cp(int(t["eval_cp_white"]))
            pe = clip_cp(int(pred))
            abs_errors.append(abs(pe - te))
    return _aggregate(accs, hamms, abs_errors)


def evaluate_run_details(truths_path: Path, preds_path: Path) -> Tuple[Scores, List[ItemDetail]]:
    truths: Dict[str, Dict] = {}
    with open(truths_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            truths[obj["id"]] = obj

    preds: Dict[str, Dict] = {}
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            preds[obj["id"]] = obj

    ids = sorted(truths.keys())
    details: List[ItemDetail] = []
    accs: List[float] = []
    hamms: List[int] = []
    abs_errors: List[int] = []

    for id_ in ids:
        t = truths[id_]
        p = preds.get(id_)
        # Task A
        fen_ok = False
        if p and "fen64" in p:
            try:
                # Will raise if unparsable
                parse_fen64_to_board_array(p["fen64"])  # type: ignore[arg-type]
                fen_ok = True
            except Exception:
                fen_ok = False
            acc, h = square_accuracy(t["fen64"], p["fen64"])  # returns (0,64) if parse fails
        else:
            acc, h = 0.0, 64
        # Task B
        eval_ok = False
        if p and isinstance(p.get("eval_cp_white"), (int, float, str)):
            try:
                pred = int(float(p["eval_cp_white"]))
                eval_ok = True
            except Exception:
                pred = None
        else:
            pred = None
        if pred is None:
            err = 1000
        else:
            te = clip_cp(int(t["eval_cp_white"]))
            pe = clip_cp(int(pred))
            err = abs(pe - te)
        details.append(ItemDetail(id=id_, acc64=acc, hamming=h, abs_err=err, fen_ok=fen_ok, eval_ok=eval_ok))
        accs.append(acc)
        hamms.append(h)
        abs_errors.append(err)

    scores = _aggregate(accs, hamms, abs_errors)
    return scores, details
