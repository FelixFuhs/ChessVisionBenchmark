import os
import platform
import shutil
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import requests


STOCKFISH_RELEASE_TAG = "sf_16.1"
GITHUB_API = (
    f"https://api.github.com/repos/official-stockfish/Stockfish/releases/tags/{STOCKFISH_RELEASE_TAG}"
)


def _bin_dir() -> Path:
    return Path("bin").absolute()


def _platform_asset_url() -> Optional[str]:
    """Pick a reasonable binary asset for the current platform."""
    try:
        resp = requests.get(GITHUB_API, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        assets = data.get("assets", [])
        system = platform.system()
        machine = platform.machine().lower()

        names = []
        if system == "Windows":
            # Prefer widely compatible SSE4.1 POPCNT, then AVX2, then generic
            names = [
                "stockfish-windows-x86-64-sse41-popcnt.zip",
                "stockfish-windows-x86-64-avx2.zip",
                "stockfish-windows-x86-64-bmi2.zip",
                "stockfish-windows-x86-64.zip",
            ]
        elif system == "Darwin":
            if machine in ("arm64", "aarch64"):
                names = ["stockfish-macos-m1-apple-silicon.tar"]
            else:
                names = [
                    "stockfish-macos-x86-64-sse41-popcnt.tar",
                    "stockfish-macos-x86-64-avx2.tar",
                    "stockfish-macos-x86-64.tar",
                ]
        else:  # Linux
            names = [
                "stockfish-ubuntu-x86-64-sse41-popcnt.tar",
                "stockfish-ubuntu-x86-64-avx2.tar",
                "stockfish-ubuntu-x86-64.tar",
            ]

        for n in names:
            for a in assets:
                if a.get("name") == n:
                    return a.get("browser_download_url")
    except Exception:
        return None
    return None


def _extract_engine(archive_path: Path) -> Optional[Path]:
    target_dir = _bin_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(target_dir)
    elif archive_path.suffix == ".tar":
        with tarfile.open(archive_path, "r") as t:
            t.extractall(target_dir)
    else:
        return None

    # Find the engine executable
    candidates = []
    for p in target_dir.rglob("*"):
        if p.is_file():
            name = p.name.lower()
            if platform.system() == "Windows":
                if name.startswith("stockfish") and name.endswith(".exe"):
                    candidates.append(p)
            else:
                if name.startswith("stockfish") and ".exe" not in name:
                    candidates.append(p)
    if not candidates:
        return None

    # Pick the largest (likely correct) binary
    engine_path = max(candidates, key=lambda x: x.stat().st_size)
    # Copy to stable name
    final = target_dir / ("stockfish.exe" if platform.system() == "Windows" else "stockfish")
    try:
        shutil.copy2(engine_path, final)
    except Exception:
        final = engine_path

    # Ensure executable permissions
    try:
        final.chmod(final.stat().st_mode | stat.S_IEXEC)
    except Exception:
        pass
    return final


def ensure_stockfish() -> Path:
    """Return a path to a Stockfish 16.1 binary, downloading if needed.

    Checks:
      - $STOCKFISH_PATH
      - bin/stockfish[.exe]
      - Downloads from GitHub release sf_16.1 if possible
    """
    # 1) Env var
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # 2) Local bin
    local = _bin_dir() / ("stockfish.exe" if platform.system() == "Windows" else "stockfish")
    if local.exists():
        return local

    # 3) Download
    url = _platform_asset_url()
    if not url:
        raise RuntimeError(
            "Could not determine a Stockfish binary for this platform. "
            "Please set STOCKFISH_PATH to a local engine executable."
        )
    archive_path = _bin_dir() / Path(url).name
    _bin_dir().mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(archive_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    engine_path = _extract_engine(archive_path)
    if not engine_path or not engine_path.exists():
        raise RuntimeError("Failed to extract Stockfish binary from archive.")
    return engine_path


@dataclass
class EngineConfig:
    threads: int = 4
    hash_mb: int = 256
    movetime_ms: Optional[int] = 500  # If None, use depth
    depth: Optional[int] = None
    syzygy: bool = False


def stockfish_name() -> str:
    return "Stockfish 16.1"


def evaluate_board(board: chess.Board, cfg: Optional[EngineConfig] = None) -> int:
    """Return mapped centipawn eval from White's perspective.

    Mate in n is mapped to sign * (10000 - n).
    """
    if cfg is None:
        cfg = EngineConfig()
    engine_path = ensure_stockfish()
    limit = (
        chess.engine.Limit(time=(cfg.movetime_ms or 0) / 1000.0)
        if cfg.movetime_ms
        else chess.engine.Limit(depth=cfg.depth or 18)
    )
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        engine.configure({"Threads": cfg.threads, "Hash": cfg.hash_mb})
        info = engine.analyse(board, limit)
        score = info["score"].pov(chess.WHITE)
        if score.is_mate():
            n = score.mate()
            assert n is not None
            sign = 1 if n > 0 else -1
            return int(sign * (10000 - abs(n)))
        else:
            # Score already oriented to White by pov().
            cp = score.score(mate_score=0)
            return int(cp)
