# Chess LLM Mini‑Benchmark (v0)

**Two tasks. Ten images. One number.**
This repo defines a tiny, reproducible benchmark that tests whether multimodal LLMs can (1) read a chessboard from an image and (2) understand the position’s strength.

## TL;DR

* **Task A — FEN‑64:** From a single chessboard image, output the **first FEN field only** (piece placement over 64 squares).
* **Task B — Eval:** Output a **centipawn evaluation from White’s perspective** under a fixed Stockfish reference.
* **Assumptions in v0:** White to move; ignore castling, en‑passant, halfmove, fullmove.
* **Scoring:** Square accuracy for Task A, MAE for Task B, combined into a single composite score.
* **Scope:** Fun, minimal leaderboard comparing a handful of LLMs. Start with \~10 positions.

---

## Why this benchmark?

It probes a simple “world‑model” loop: **vision → symbolic state → game semantics**. It is cheap to create, fully auto‑scored, and easy to extend with harder images later.

---

## Folder layout

```
chess-llm-bench/
├─ data/
│  ├─ images/                 # 10 PNGs (001.png ... 010.png)
│  └─ truths.jsonl            # ground truth (one JSON per image)
├─ prompts/
│  └─ fen_eval_prompt.txt     # single prompt template (strict JSON)
├─ runs/                      # model outputs per run (JSONL)
│  └─ gpt-xxx_2025-09-07.jsonl
└─ README.md                  # this file
```

---

## Tasks

### Task A — FEN‑64 (piece placement only)

Given an image, predict **fen64** — the first FEN field, e.g.,

```
r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR
```

Rules:

* Rank order is 8→1; file order a→h; use digit‑runs for empty squares.
* Use standard piece letters `{pnbrqk, PNBRQK}`; no extra fields.

### Task B — Evaluation (centipawns)

Predict **eval\_cp\_white** — a single integer centipawn score from White’s perspective. Positive means White is better.

**Reference engine settings (pinned):**

* `name`: Stockfish 16.1 (pin your exact build if you re‑generate data)
* `options`: `{Threads: 4, Hash: 256}`
* `limit`: either `movetime_ms: 500` *or* `depth: 18` (pick one and keep it fixed)
* `syzygy`: false (no tablebases) in v0

**Mate mapping:** if the engine reports mate in `n`, map to `sign * (10000 - n)` before clipping.

**Clipping:** clip engine truth and model predictions to the range `[-1000, 1000]` **for scoring only**.

---

## Data format

### `data/truths.jsonl` (one JSON object per line)

```json
{
  "id": "001",
  "image": "data/images/001.png",
  "fen64": "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
  "eval_cp_white": -34,
  "engine": {
    "name": "Stockfish 16.1",
    "options": {"Threads": 4, "Hash": 256},
    "limit": {"movetime_ms": 500},
    "syzygy": false
  }
}
```

### Model output (strict JSON only)

For each image you will prompt the model with the same template (below) and the image attached. The model must return **only** this JSON:

```json
{
  "fen64": "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
  "eval_cp_white": -20
}
```

If JSON is invalid or `fen64` is unparsable, treat Task A as zero credit for that item.

---

## Prompt template (copy into `prompts/fen_eval_prompt.txt`)

```
You see a single chessboard image. Assume WHITE TO MOVE.
Return ONLY valid JSON with these keys:
- "fen64": the first FEN field (piece placement only).
- "eval_cp_white": one integer centipawn evaluation from White’s perspective. Positive means White is better.

Rules:
- JSON only. No extra text.
- fen64 must use standard FEN piece letters and digit-run syntax for empty squares.
- Do not include side-to-move, castling, en passant, halfmove or fullmove fields.

Example:
{"fen64":"rnbqkbnr/pp1ppppp/2p5/8/8/8/PPPPPPPP/RNBQKBNR","eval_cp_white":-15}
```

---

## Scoring

Let `B` be the true board (64 squares) and `\hat B` the prediction derived from `fen64`.

### Task A — Board reconstruction

* **Square accuracy**: `Acc64 = (1/64) * sum( 1[ B_i == \hat B_i ] )`
* **Hamming distance**: `H = 64 * (1 - Acc64)` (reported for diagnostics)

### Task B — Eval closeness

* **MAE\_cp**: median absolute error between predicted and true `eval_cp_white`, after clipping both values to `[-1000, 1000]`.

### Composite score (single leaderboard number)

```
Score = 0.7 * Acc64 + 0.3 * max(0, 1 - MAE_cp / 1000)
```

Perfect FEN and zero cp error scores 1.0.

**Invalid output handling:**

* If JSON is invalid or `fen64` fails to parse to 64 squares, set `Acc64 = 0` for that item.
* If `eval_cp_white` is missing or non‑integer, use a worst‑case error of `MAE_cp_item = 1000` for that item.

All metrics are averaged over the dataset.

---

## How to generate the 10 positions (reference workflow)

This repo does not ship code; here is the minimal recipe you can implement with your favorite tools:

1. **Sample legal positions**

   * Start from the initial position.
   * Play a random legal move `N` times (e.g., sample `N` uniformly from `[6, 40]`).
   * Force **White to move** for v0 (if needed, make a final null tweak by adding a ply or mirror a move count).

2. **Evaluate with Stockfish**

   * Use pinned settings above.
   * Convert mate to large ± scores then clip to `[-1000, 1000]` **for scoring**.

3. **Render images**

   * 512×512 PNG, top‑down 2D, white at bottom, clearly visible pieces.
   * Keep v0 simple. Save to `data/images/` as `001.png` to `010.png`.

4. **Write `data/truths.jsonl`**

   * One JSON object per image, as shown in the schema.

5. **Prompt models**

   * Use `prompts/fen_eval_prompt.txt`.
   * For each image, attach the PNG and request **JSON only**.
   * Save each model’s responses as JSONL under `runs/` with fields `id, fen64, eval_cp_white`.

6. **Evaluate**

   * Parse truths and predictions, compute `Acc64`, `H`, `MAE_cp`, `Score`.
   * Produce a small table and a summary line.

**Implementation tip:** If you use Python, `python‑chess` can parse FEN and help convert to a 64‑square array.

---

## Leaderboard format (example)

| Model    | Acc64 | Hamming ↓ | MAE\_cp ↓ | Composite ↑ |
| -------- | ----: | --------: | --------: | ----------: |
| gpt‑X    | 0.812 |      12.0 |       145 |        0.73 |
| claude‑Y | 0.775 |      14.4 |       210 |        0.66 |
| gemini‑Z | 0.690 |      19.8 |       260 |        0.58 |

*Numbers above are placeholders.*

---

## Reproducibility checklist

* Pin engine version + options + time/depth.
* Store engine metadata in `truths.jsonl` for each item.
* Record the PRNG seed for move sampling.
* Keep the prompt file identical across models.
* Clip evals only for scoring; preserve raw engine outputs if you export them.

---

## Extensions (v1+ ideas)

* Orientation flips, rotations, perspective photos, shadows, partial occlusion.
* Larger dataset with ID vs OOD splits.
* Optional legality penalty for impossible boards.
* WDL probabilities or best‑move prediction as extra tracks.
* Private test set for public leaderboards.

---

## FAQ

**Q: Why not include castling/EP/halfmove/etc.?**
A: In a single static image you cannot reliably infer those history‑dependent fields. v0 focuses on what is visible.

**Q: Why clip at ±1000 cp?**
A: To prevent a few tactical blow‑outs from dominating MAE while preserving ranking differences.

**Q: Can I use tools or a chess engine inside the model call?**
A: This mini‑benchmark compares out‑of‑the‑box LLM behavior from an image. Please do **not** let the model call external engines.

---

## License and citation

* License: MIT (or choose your preference).
* If you use or extend this benchmark, please cite as:

  > *Chess LLM Mini‑Benchmark (v0). Felix et al., 2025. GitHub repository.*

---

## Changelog

* **v0 (2025-09-07):** Initial two-task spec, minimal dataset of 10 images, simple composite metric.

---

## Implementation (This Repo)

This repository now includes a minimal, reproducible Python implementation matching the spec above:

- Data generation: random legal positions, forced White-to-move, Stockfish 16.1 evals, 512×512 PNG renders, `data/truths.jsonl`.
- Prompting: single strict-JSON prompt in `prompts/fen_eval_prompt.txt`.
- Model runners: OpenAI and OpenRouter providers (vision models) with image attachment and JSON-only response.
- Scoring: Acc64 (square accuracy), Hamming distance, MAE_cp (median after clipping to ±1000), and composite score.

The CLI exposes three main commands: `gen_data`, `run_model`, and `evaluate`.

---

## Quickstart

### 1) Create a virtualenv and install deps

Windows (PowerShell):

```
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment

Copy `.env.example` to `.env` and fill the keys you need:

- `OPENAI_API_KEY` (for provider `openai`)
- `OPENROUTER_API_KEY` (for provider `openrouter`)
- Optional: `STOCKFISH_PATH` — path to a local Stockfish engine. If not set, the tooling will download Stockfish 16.1 for your platform into `bin/`.

```
copy .env.example .env     # Windows
cp .env.example .env       # macOS/Linux
```

### 3) Generate data (images + truths)

This downloads/uses Stockfish 16.1 with pinned options and generates 10 images + `data/truths.jsonl`:

```
python -m chess_llm_bench gen_data --n 10 --seed 1337 --movetime-ms 500 --threads 4 --hash-mb 256
```

Outputs:

- `data/images/001.png` … `010.png`
- `data/truths.jsonl`

### 4) Ensure the prompt file exists

The repo includes `prompts/fen_eval_prompt.txt`. To regenerate it:

```
python -m chess_llm_bench prompt_file
```

### 5) Run a model (OpenAI or OpenRouter)

OpenAI example (e.g., `gpt-4o-mini`):

```
python -m chess_llm_bench run_model --provider openai --model gpt-4o-mini --truths data/truths.jsonl --prompt-path prompts/fen_eval_prompt.txt --out-dir runs
```

OpenRouter example (pick a vision model that supports image input):

```
python -m chess_llm_bench run_model --provider openrouter --model openai/gpt-4o-mini --truths data/truths.jsonl --prompt-path prompts/fen_eval_prompt.txt --out-dir runs
```

This writes a JSONL file under `runs/` with records of the form:

```
{"id":"001","fen64":"...","eval_cp_white":-20}
```

### 6) Evaluate a run

```
python -m chess_llm_bench evaluate --truths data/truths.jsonl --preds runs/<your-run>.jsonl
```

You’ll get a small table with `Acc64`, `Hamming`, `MAE_cp`, and `Composite` score.

### 7) One-shot bench (run + eval + leaderboard)

```
python -m chess_llm_bench bench --provider openrouter --model "qwen/qwen2.5-vl-32b-instruct:free" --truths data/truths.jsonl --prompt-path prompts/fen_eval_prompt.txt --out-dir runs --leaderboard-csv results/leaderboard.csv --max-items 10 --bootstrap 0
```

- Appends a row to `results/leaderboard.csv` with metrics and the run file path.
- `--max-items` limits dataset size; `--bootstrap` prints a 95% CI for Composite when >0.

### 8) Plot the leaderboard

```
python -m chess_llm_bench plot --leaderboard-csv results/leaderboard.csv --out-png results/leaderboard.png --bootstrap 0
```

- Produces a sorted (ascending), horizontal bar chart of Composite scores with readable labels and coverage counts.
- Use `--bootstrap 500` to compute 95% CIs per bar (slower).

Boxplots of per-item metrics across runs:

```
python -m chess_llm_bench plot-box --leaderboard-csv results/leaderboard.csv --out-png results/leaderboard_box.png
```

- Left: Acc64 distribution per run; Right: absolute eval error (cp) per run.

---

## Cheat Sheet (OpenRouter)

1) Generate dataset (10 items):

```
python -m chess_llm_bench gen-data --n 10 --seed 1337 --movetime-ms 500 --threads 4 --hash-mb 256
```

2) Sanity check with oracle (expect perfect 1.0):

```
python -m chess_llm_bench oracle-run --truths data/truths.jsonl --out-dir runs
python -m chess_llm_bench evaluate --truths data/truths.jsonl --preds runs/oracle_*.jsonl
```

3) One-shot bench with a model (free example):

```
python -m chess_llm_bench bench --provider openrouter --model "qwen/qwen2.5-vl-32b-instruct:free" --truths data/truths.jsonl --prompt-path prompts/fen_eval_prompt.txt --out-dir runs --leaderboard-csv results/leaderboard.csv --sleep-ms 800
```

4) Plot the leaderboard:

```
python -m chess_llm_bench plot --leaderboard-csv results/leaderboard.csv --out-png results/leaderboard.png
```

---

## Details and Notes

- Engine: Stockfish 16.1, options `{Threads: 4, Hash: 256}`, limit `{movetime_ms: 500}` by default. You can switch to `--movetime-ms 0 --depth 18` to use a fixed depth instead.
- Mate mapping: Mate in `n` is mapped to `sign * (10000 - n)` before any clipping.
- Clipping: For scoring only, predictions and truths are clipped to `[-1000, 1000]`.
- White to move: The generator forces White to move in all positions.
- Rendering: PNG 512×512, 2D top-down, white at bottom, coordinates on.
- Prompt: Strict JSON only; no extra text. The CLI requests `response_format: json_object` when available.
- Seeds: Each truth record stores the `seed` used to sample the position for reproducibility.

Rendering note: Images use Unicode chess glyphs from a system font. On Windows, `Segoe UI Symbol` is used if available. If your system lacks these glyphs and pieces render as letters, set `CHESS_FONT_PATH` in `.env` to a TTF containing chess symbols (e.g., `C:\Windows\Fonts\seguisym.ttf`).

---

## Provider setup tips

- OpenAI: set `OPENAI_API_KEY` in `.env` or your environment; use a vision-capable model such as `gpt-4o-mini`.
- OpenRouter: set `OPENROUTER_API_KEY`; choose a model that supports image inputs (e.g., `openai/gpt-4o-mini`).
- OpenRouter free variants: many models expose a `:free` suffix (e.g., `qwen/qwen2.5-vl-32b-instruct:free`). Availability changes; if a run 404s or charges, pick another free variant from the catalog.
- Rate limits: If you see HTTP 429 errors, the runner retries with backoff. Tune via `.env`:
  - `OPENROUTER_MAX_RETRIES=3`
  - `OPENROUTER_RETRY_BASE=2.0`
  - Add `--sleep-ms 800` (or higher) to space requests when using free tiers.
  - Use `--max-items` for quick/cheap tests.
- Anthropic/Gemini: not wired-in in v0 of this code; can be added following the same pattern used for OpenAI/OpenRouter.

---

## Troubleshooting

- Stockfish download fails: set `STOCKFISH_PATH` to a locally installed engine and re-run.
- Model replies non-JSON: the runner attempts to parse JSON; invalid outputs yield zero credit for Task A and worst-case error for Task B, per spec.
- AVX/Instruction issues: if the downloaded engine crashes, set `STOCKFISH_PATH` to a compatible local binary, or switch to the `sse41-popcnt` build.
