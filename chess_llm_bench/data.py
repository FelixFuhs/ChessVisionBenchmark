import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import chess

from .engine import EngineConfig, evaluate_board, stockfish_name
from .utils import only_fen64


@dataclass
class TruthItem:
    id: str
    image: str
    fen64: str
    eval_cp_white: int
    engine: dict
    seed: int


def _random_position(seed: int, min_plies: int = 6, max_plies: int = 40) -> chess.Board:
    rng = random.Random(seed)
    board = chess.Board()
    n = rng.randint(min_plies, max_plies)
    last_move = None
    for ply in range(n):
        moves = list(board.legal_moves)
        if not moves:
            break
        mv = rng.choice(moves)
        board.push(mv)
        last_move = mv
    # Ensure White to move
    if board.turn != chess.WHITE:
        moves = list(board.legal_moves)
        if moves:
            board.push(rng.choice(moves))
    return board


def generate_truths(
    out_dir: Path,
    n: int = 10,
    seed: int = 42,
    engine_cfg: Optional[EngineConfig] = None,
) -> List[TruthItem]:
    if engine_cfg is None:
        engine_cfg = EngineConfig()
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    truths: List[TruthItem] = []
    for i in range(1, n + 1):
        sid = f"{i:03d}"
        board_seed = seed + i * 7919  # simple decorrelated seeds
        board = _random_position(seed=board_seed)
        # Evaluate
        eval_cp = evaluate_board(board, cfg=engine_cfg)
        fen64 = only_fen64(board.fen())
        item = TruthItem(
            id=sid,
            image=str(images_dir / f"{sid}.png"),
            fen64=fen64,
            eval_cp_white=eval_cp,
            engine={
                "name": stockfish_name(),
                "options": {"Threads": engine_cfg.threads, "Hash": engine_cfg.hash_mb},
                "limit": (
                    {"movetime_ms": engine_cfg.movetime_ms}
                    if engine_cfg.movetime_ms
                    else {"depth": engine_cfg.depth}
                ),
                "syzygy": engine_cfg.syzygy,
            },
            seed=board_seed,
        )
        truths.append(item)
    return truths


def write_truths_jsonl(truths: List[TruthItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in truths:
            json.dump(asdict(t), f, ensure_ascii=False)
            f.write("\n")
