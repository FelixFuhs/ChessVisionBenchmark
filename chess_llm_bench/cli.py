import datetime as dt
import json
import os
from pathlib import Path
from typing import Optional, Annotated

import chess
import time
import typer
from dotenv import load_dotenv
from rich import print
from rich.table import Table
from rich.console import Console

from .data import TruthItem, generate_truths, write_truths_jsonl
from .engine import EngineConfig
from .metrics import Scores, evaluate_run, evaluate_run_details
from .providers import call_openai, call_openrouter
from .render import render_board_png
from .utils import ensure_dir, load_prompt, safe_filename


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def gen_data(
    n: int = typer.Option(10, help="Number of positions/images to generate"),
    seed: int = typer.Option(42, help="PRNG seed"),
    out: Path = typer.Option(Path("data"), help="Output data directory"),
    movetime_ms: int = typer.Option(500, help="Engine movetime in milliseconds"),
    threads: int = typer.Option(4, help="Stockfish Threads option"),
    hash_mb: int = typer.Option(256, help="Stockfish Hash (MB) option"),
):
    """Generate random positions, evaluate with Stockfish, render images, and write truths.jsonl."""
    load_dotenv(override=True)
    cfg = EngineConfig(threads=threads, hash_mb=hash_mb, movetime_ms=movetime_ms)
    truths = generate_truths(out_dir=out, n=n, seed=seed, engine_cfg=cfg)
    # Render images
    for t in truths:
        board = chess.Board(fen=f"{t.fen64} w - - 0 1")
        render_board_png(board, Path(t.image), size_px=512)
    # Write truths
    truths_path = out / "truths.jsonl"
    write_truths_jsonl(truths, truths_path)
    print(f"[green]Wrote[/green] {truths_path} and {len(truths)} PNGs under {out}/images/")


@app.command()
def prompt_file(
    out: Path = typer.Option(Path("prompts/fen_eval_prompt.txt"), help="Prompt file path"),
):
    """Create or overwrite the prompt file with the spec template."""
    ensure_dir(out.parent)
    content = (
        "You see a single chessboard image. Assume WHITE TO MOVE.\n"
        "Return ONLY valid JSON with these keys:\n"
        "- \"fen64\": the first FEN field (piece placement only).\n"
        "- \"eval_cp_white\": one integer centipawn evaluation from White's perspective. Positive means White is better.\n\n"
        "Rules:\n"
        "- JSON only. No extra text.\n"
        "- fen64 must use standard FEN piece letters and digit-run syntax for empty squares.\n"
        "- Do not include side-to-move, castling, en passant, halfmove or fullmove fields.\n\n"
        "Example:\n"
        '{"fen64":"rnbqkbnr/pp1ppppp/2p5/8/8/8/PPPPPPPP/RNBQKBNR","eval_cp_white":-15}'
        "\n"
    )
    out.write_text(content, encoding="utf-8")
    print(f"[green]Wrote[/green] {out}")


@app.command()
def run_model(
    provider: str = typer.Option(
        "openai", help="Provider: 'openai' or 'openrouter'"
    ),
    model: str = typer.Option(
        "gpt-4o-mini", help="Model name for the provider"
    ),
    truths: Path = typer.Option(Path("data/truths.jsonl"), help="Truths JSONL path"),
    prompt_path: Path = typer.Option(
        Path("prompts/fen_eval_prompt.txt"), help="Prompt file path"
    ),
    out_dir: Path = typer.Option(Path("runs"), help="Directory to write run file"),
    run_name: Optional[str] = typer.Option(None, help="Override run filename base"),
    request_timeout: int = typer.Option(120, help="Provider request timeout (seconds)"),
    max_items: Optional[int] = typer.Option(None, help="Limit to first N items"),
    # Use Annotated so the Python default is an int when called programmatically
    sleep_ms: Annotated[int, typer.Option(help="Sleep between requests (ms), helps with rate limits")]= 0,
):
    """Run a vision-capable model over images and save JSONL predictions."""
    load_dotenv(override=True)
    ensure_dir(out_dir)
    prompt = load_prompt(str(prompt_path))

    # Load truths to get ids and image paths
    items = []
    with open(truths, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            items.append(obj)

    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = run_name or f"{provider}-{model}_{ts}"
    base = safe_filename(base)
    out_path = out_dir / f"{base}.jsonl"

    if max_items:
        items = items[: max_items]
    with open(out_path, "w", encoding="utf-8") as outf:
        for obj in items:
            img = Path(obj["image"]) if os.path.isabs(obj["image"]) else Path(obj["image"])  # relative
            try:
                if provider.lower() == "openai":
                    resp = call_openai(model=model, prompt=prompt, image_path=img, timeout=request_timeout)
                elif provider.lower() == "openrouter":
                    resp = call_openrouter(model=model, prompt=prompt, image_path=img, timeout=request_timeout)
                else:
                    raise typer.BadParameter("provider must be 'openai' or 'openrouter'")
                parsed = resp.parsed or {}
                status = "ok"
            except Exception as e:
                print(f"[red]Error processing id={obj['id']}[/red]: {e}")
                parsed = {}
                status = "infra_error"
            rec = {
                "id": obj["id"],
                "fen64": parsed.get("fen64"),
                "eval_cp_white": parsed.get("eval_cp_white"),
                "status": status,
            }
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Processed image id={obj['id']} -> {rec}")
            if sleep_ms > 0:
                time.sleep(sleep_ms / 1000.0)

    print(f"[green]Wrote predictions[/green] to {out_path}")


@app.command()
def evaluate(
    truths: Path = typer.Option(Path("data/truths.jsonl"), help="Truths JSONL path"),
    preds: Path = typer.Option(..., help="Predictions JSONL path"),
):
    """Compute Acc64, Hamming, MAE_cp, and composite score for a run."""
    s: Scores = evaluate_run(truths, preds)
    table = Table(title="Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Acc64", f"{s.acc64:.3f}")
    table.add_row("Hamming", f"{s.hamming:.1f}")
    table.add_row("MAE_cp", f"{s.mae_cp:.1f}")
    table.add_row("Composite", f"{s.composite:.3f}")
    print(table)


@app.command()
def bench(
    provider: str = typer.Option("openrouter", help="Provider: 'openai' or 'openrouter'"),
    model: str = typer.Option(..., help="Model slug/name for provider"),
    truths: Path = typer.Option(Path("data/truths.jsonl"), help="Truths JSONL path"),
    prompt_path: Path = typer.Option(Path("prompts/fen_eval_prompt.txt"), help="Prompt file path"),
    out_dir: Path = typer.Option(Path("runs"), help="Directory for predictions and leaderboard"),
    run_name: Optional[str] = typer.Option(None, help="Override run filename base"),
    request_timeout: int = typer.Option(120, help="Request timeout seconds"),
    max_items: Optional[int] = typer.Option(None, help="Limit to first N items"),
    leaderboard_csv: Path = typer.Option(Path("runs/leaderboard.csv"), help="Append results here"),
    bootstrap: int = typer.Option(0, help="Bootstrap samples for CI (0 to skip)"),
    sleep_ms: int = typer.Option(0, help="Sleep between provider requests (ms), helps with rate limits"),
):
    """Run a model then evaluate and append to leaderboard.csv."""
    # 1) Run model
    run_model(
        provider=provider,
        model=model,
        truths=truths,
        prompt_path=prompt_path,
        out_dir=out_dir,
        run_name=run_name,
        request_timeout=request_timeout,
        max_items=max_items,
        sleep_ms=sleep_ms,
    )
    # Find latest run file in out_dir
    preds_list = sorted(out_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not preds_list:
        raise typer.Exit(code=1)
    latest = preds_list[0]
    # 2) Evaluate with details
    scores, details = evaluate_run_details(truths, latest)
    # 3) Append to leaderboard
    ensure_dir(leaderboard_csv.parent)
    header = (
        "timestamp,provider,model,dataset,n,acc64,hamming,mae_cp,composite,run_file,n_valid_fen,n_valid_eval"
    )
    if not leaderboard_csv.exists():
        leaderboard_csv.write_text(header + "\n", encoding="utf-8")
    n = len(details)
    n_fen_ok = sum(1 for d in details if d.fen_ok)
    n_eval_ok = sum(1 for d in details if d.eval_ok)
    # Count infrastructure/API errors from the predictions file (if it has 'status')
    n_infra_err = 0
    try:
        with open(latest, "r", encoding="utf-8") as _pf:
            for _line in _pf:
                if not _line.strip():
                    continue
                try:
                    _obj = json.loads(_line)
                except Exception:
                    continue
                if _obj.get("status") == "infra_error":
                    n_infra_err += 1
    except Exception:
        pass
    import datetime as _dt

    with open(leaderboard_csv, "a", encoding="utf-8") as f:
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        f.write(
            f"{ts},{provider},{model},{truths},{n},{scores.acc64:.6f},{scores.hamming:.6f},{scores.mae_cp:.6f},{scores.composite:.6f},{latest},{n_fen_ok},{n_eval_ok}\n"
        )

    # 4) Optionally compute bootstrap CI (printed only)
    if bootstrap and n > 1:
        import random
        accs = [d.acc64 for d in details]
        hamms = [d.hamming for d in details]
        errs = [d.abs_err for d in details]
        composites = []
        for _ in range(bootstrap):
            idxs = [random.randrange(n) for _ in range(n)]
            acc_b = [accs[i] for i in idxs]
            hamm_b = [hamms[i] for i in idxs]
            err_b = [errs[i] for i in idxs]
            from .metrics import _aggregate as _agg

            sc = _agg(acc_b, hamm_b, err_b)
            composites.append(sc.composite)
        composites.sort()
        lo = composites[int(0.025 * bootstrap)]
        hi = composites[int(0.975 * bootstrap)]
        print(f"Composite 95% CI (bootstrap {bootstrap}): [{lo:.3f}, {hi:.3f}]")

    # 5) Print results table like `evaluate`
    cov_line = f"Coverage: FEN ok {n_fen_ok}/{n}, Eval ok {n_eval_ok}/{n}"
    if n_infra_err:
        cov_line += f" (API errors: {n_infra_err})"
    print(cov_line)
    table = Table(title="Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Acc64", f"{scores.acc64:.3f}")
    table.add_row("Hamming", f"{scores.hamming:.1f}")
    table.add_row("MAE_cp", f"{scores.mae_cp:.1f}")
    table.add_row("Composite", f"{scores.composite:.3f}")
    print(table)
    print(f"Saved predictions: {latest}")
    print(f"Appended to leaderboard: {leaderboard_csv}")
    # Diagnostic: valid-only composite (excludes invalid/missing per-item outputs)
    valid = [d for d in details if d.fen_ok and d.eval_ok]
    if valid:
        from .metrics import _aggregate as _agg
        sc_v = _agg([d.acc64 for d in valid], [d.hamming for d in valid], [d.abs_err for d in valid])
        print(f"Valid-only Composite (diagnostic): {sc_v.composite:.3f} over {len(valid)}/{n}")


@app.command()
def plot(
    leaderboard_csv: Path = typer.Option(Path("runs/leaderboard.csv"), help="Leaderboard CSV"),
    out_png: Path = typer.Option(Path("runs/leaderboard.png"), help="Output plot image"),
    bootstrap: int = typer.Option(0, help="Bootstrap samples per run for CI (0 to skip)"),
    style: str = typer.Option("seaborn-v0_8-darkgrid", help="Matplotlib style (e.g., 'ggplot')"),
    label_mode: str = typer.Option("coverage", help="Y-labels: 'total', 'valid', or 'coverage'"),
):
    """Render a simple bar chart of Composite scores across runs.

    If --bootstrap > 0, computes 95% bootstrap CI per run using stored run files.
    """
    import csv
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[red]matplotlib not installed. Install with:[/red] pip install matplotlib")
        raise typer.Exit(code=1)

    rows = []
    with open(leaderboard_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        print("[yellow]No rows found in leaderboard.[/yellow]")
        raise typer.Exit()
    # Sort ascending by composite so lowest is left, highest right
    try:
        rows.sort(key=lambda x: float(x["composite"]))
    except Exception:
        pass
    def short_model(full: str) -> str:
        # keep portion after first '/' and show ':suffix' if any
        if "/" in full:
            full = full.split("/", 1)[1]
        return full

    # Coverage-aware labels: n_valid (both tasks) over total n for each run
    def _cov_label(row):
        prov = row.get('provider', '')
        m = short_model(row.get('model', ''))
        try:
            n = int(row.get('n', '0'))
        except Exception:
            n = 0
        try:
            n_f = int(row.get('n_valid_fen', '0'))
            n_e = int(row.get('n_valid_eval', '0'))
            cov = min(n_f, n_e) if n and (n_f or n_e) else None
        except Exception:
            cov = None
        if cov is not None and n:
            return f"{prov} • {m} (n={cov}/{n})"
        return f"{prov} • {m} (n={n})"
    labels_cov = [_cov_label(row) for row in rows]
    # Rebuild labels based on requested label_mode; defaults to coverage
    def _label(row):
        prov = row.get('provider', '')
        m = short_model(row.get('model', ''))
        try:
            n = int(row.get('n', '0'))
        except Exception:
            n = 0
        try:
            n_f = int(row.get('n_valid_fen', '0'))
            n_e = int(row.get('n_valid_eval', '0'))
            n_valid = min(n_f, n_e)
        except Exception:
            n_valid = None
        mode = (label_mode or 'coverage').lower()
        if mode == 'valid' and n_valid is not None:
            return f"{prov} | {m} (n={n_valid})"
        if mode == 'coverage' and n_valid is not None and n:
            return f"{prov} | {m} (n={n_valid}/{n})"
        return f"{prov} | {m} (n={n})"
    cov_labels = [_label(row) for row in rows]

    labels = [f"{row['provider']} • {short_model(row['model'])} (n={row['n']})" for row in rows]
    composites = [float(row["composite"]) for row in rows]
    # Coverage-aware labels: show n_valid (both tasks) over total n
    def _cov_label(row):
        prov = row.get('provider', '')
        m = short_model(row.get('model', ''))
        try:
            n = int(row.get('n', '0'))
        except Exception:
            n = 0
        try:
            n_f = int(row.get('n_valid_fen', '0'))
            n_e = int(row.get('n_valid_eval', '0'))
            cov = min(n_f, n_e) if n and (n_f or n_e) else None
        except Exception:
            cov = None
        if cov is not None and n:
            return f"{prov} • {m} (n={cov}/{n})"
        return f"{prov} • {m} (n={n})"
    cov_labels = [_cov_label(row) for row in rows]
    # Ensure coverage labels work even if CSV lacks coverage fields by recomputing
    def _cov_label_dyn(row):
        prov = row.get('provider', '')
        # Short model name
        m = short_model(row.get('model', ''))
        # total n
        try:
            n = int(row.get('n', '0'))
        except Exception:
            n = 0
        n_f = row.get('n_valid_fen')
        n_e = row.get('n_valid_eval')
        try:
            n_f = int(n_f) if n_f is not None else None
            n_e = int(n_e) if n_e is not None else None
        except Exception:
            n_f, n_e = None, None
        if (n_f is None or n_e is None) and row.get('dataset') and row.get('run_file'):
            try:
                from .metrics import evaluate_run_details as _erd
                _, _details = _erd(Path(row['dataset']), Path(row['run_file']))
                n_f = sum(1 for d in _details if d.fen_ok)
                n_e = sum(1 for d in _details if d.eval_ok)
            except Exception:
                n_f, n_e = None, None
        cov = min(n_f, n_e) if n and (n_f is not None and n_e is not None) else None
        if cov is not None and n:
            return f"{prov} • {m} (n={cov}/{n})"
        return f"{prov} • {m} (n={n})"
    cov_labels = [_cov_label_dyn(row) for row in rows]
    xerr = None

    if bootstrap > 0:
        lows, highs = [], []
        for row in rows:
            from .metrics import evaluate_run_details as _erd
            scores, details = _erd(Path(row["dataset"]), Path(row["run_file"]))
            import random
            n = len(details)
            comps = []
            accs = [d.acc64 for d in details]
            hamms = [d.hamming for d in details]
            errs = [d.abs_err for d in details]
            from .metrics import _aggregate as _agg
            for _ in range(bootstrap):
                idxs = [random.randrange(n) for _ in range(n)]
                sc = _agg([accs[i] for i in idxs], [hamms[i] for i in idxs], [errs[i] for i in idxs])
                comps.append(sc.composite)
            comps.sort()
            lo = comps[int(0.025 * bootstrap)]
            hi = comps[int(0.975 * bootstrap)]
            lows.append(lo)
            highs.append(hi)
        xerr = [[c - l for c, l in zip(composites, lows)], [h - c for c, h in zip(composites, highs)]]

    # Plot
    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("ggplot")
    height = max(5, 0.6 * len(labels_cov) + 1)
    plt.figure(figsize=(9, height))
    colors = plt.get_cmap("tab20")(range(len(labels)))
    y = list(range(len(labels)))
    bars = plt.barh(y, composites, xerr=xerr, capsize=5, color=colors)
    plt.yticks(y, cov_labels)
    plt.xlabel("Composite Score")
    plt.xlim(0, 1.0)
    plt.title("Chess LLM Mini-Benchmark — Leaderboard")
    # Annotate values at end of bars
    for i, b in enumerate(bars):
        val = composites[i]
        plt.text(b.get_width() + 0.01, b.get_y() + b.get_height()/2, f"{val:.3f}", va="center")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # Ensure ASCII-friendly title regardless of earlier title call
    try:
        plt.gca().set_title("Chess LLM Mini-Benchmark - Leaderboard")
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[green]Wrote plot[/green] to {out_png}")


@app.command()
def plot_box(
    leaderboard_csv: Path = typer.Option(Path("runs/leaderboard.csv"), help="Leaderboard CSV"),
    out_png: Path = typer.Option(Path("runs/leaderboard_box.png"), help="Output boxplot image"),
    style: str = typer.Option("seaborn-v0_8-darkgrid", help="Matplotlib style (e.g., 'ggplot')"),
    label_mode: str = typer.Option("coverage", help="Y-labels: 'total', 'valid', or 'coverage'"),
):
    """Boxplots of per-run item metrics (Acc64 and abs eval error).

    Sorts runs by Composite ascending using leaderboard.csv, then loads the
    referenced run files to compute distributions.
    """
    import csv
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[red]matplotlib not installed. Install with:[/red] pip install matplotlib")
        raise typer.Exit(code=1)

    rows = []
    with open(leaderboard_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        print("[yellow]No rows found in leaderboard.[/yellow]")
        raise typer.Exit()
    try:
        rows.sort(key=lambda x: float(x["composite"]))
    except Exception:
        pass

    # Prepare labels and distributions
    def short_model(full: str) -> str:
        if "/" in full:
            full = full.split("/", 1)[1]
        return full

    labels = [f"{row['provider']} • {short_model(row['model'])} (n={row['n']})" for row in rows]
    from .metrics import evaluate_run_details as _erd
    # Override labels to reflect requested label mode (valid-only, coverage, total)
    def _label(row):
        prov = row.get('provider', '')
        m = short_model(row.get('model', ''))
        try:
            n = int(row.get('n', '0'))
        except Exception:
            n = 0
        try:
            n_f = int(row.get('n_valid_fen', '0'))
            n_e = int(row.get('n_valid_eval', '0'))
            n_valid = min(n_f, n_e)
        except Exception:
            n_valid = None
        mode = (label_mode or 'coverage').lower()
        if mode == 'valid' and n_valid is not None:
            return f"{prov} | {m} (n={n_valid})"
        if mode == 'coverage' and n_valid is not None and n:
            return f"{prov} | {m} (n={n_valid}/{n})"
        return f"{prov} | {m} (n={n})"
    labels = [_label(row) for row in rows]
    acc_lists = []
    err_lists = []
    for row in rows:
        _, details = _erd(Path(row["dataset"]), Path(row["run_file"]))
        acc_lists.append([d.acc64 for d in details])
        err_lists.append([d.abs_err for d in details])

    try:
        plt.style.use(style)
    except Exception:
        plt.style.use("ggplot")
    height = max(5, 0.6 * len(labels) + 1)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, height))
    axs[0].boxplot(acc_lists, vert=False, labels=labels, showfliers=False)
    axs[0].set_xlim(0, 1)
    axs[0].set_title("Acc64 per run")
    axs[0].set_xlabel("Acc64")
    axs[1].boxplot(err_lists, vert=False, labels=labels, showfliers=False)
    axs[1].set_xlim(0, 1000)
    axs[1].set_title("Abs eval error (cp) per run")
    axs[1].set_xlabel("abs error (cp)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    print(f"[green]Wrote boxplots[/green] to {out_png}")


# Create a perfect baseline run by copying truths into a run file
@app.command()
def oracle_run(
    truths: Path = typer.Option(Path("data/truths.jsonl"), help="Truths JSONL path"),
    out_dir: Path = typer.Option(Path("runs"), help="Directory to write run file"),
    run_name: Optional[str] = typer.Option(None, help="Override run filename base"),
):
    """Copy truths to a predictions JSONL to validate the evaluator (should score 1.0)."""
    ensure_dir(out_dir)
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = run_name or f"oracle_{ts}"
    out_path = out_dir / f"{base}.jsonl"
    n = 0
    with open(truths, "r", encoding="utf-8") as inf, open(out_path, "w", encoding="utf-8") as outf:
        for line in inf:
            if not line.strip():
                continue
            obj = json.loads(line)
            rec = {"id": obj["id"], "fen64": obj["fen64"], "eval_cp_white": obj["eval_cp_white"]}
            outf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"[green]Wrote oracle predictions[/green] to {out_path} ({n} items)")

# Aliases for underscore command names (Typer defaults to hyphenated names)
app.command("gen_data")(gen_data)
app.command("prompt_file")(prompt_file)
app.command("run_model")(run_model)
app.command("oracle_run")(oracle_run)


def main():
    app()


if __name__ == "__main__":
    main()
