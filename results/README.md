Results Artifacts

- Overview: Leaderboard results and charts for Chess LLM Mini‑Benchmark live here. Defaults for the CLI now write/read leaderboard artifacts from this folder.

- Files:
  - leaderboard.csv: Rolling leaderboard with one row per run. Columns: timestamp, provider, model, dataset, n, acc64, hamming, mae_cp, composite, run_file, n_valid_fen, n_valid_eval.
  - leaderboard.png: Bar chart of Composite scores per run (sorted ascending). Labels include coverage by default (n_valid/n), and optional 95% bootstrap CI if enabled.
  - leaderboard_box.png: Box plots showing per‑item distributions for Acc64 and abs eval error (centipawns) for each run.

- How To Regenerate:
  - Plot bars: python -m chess_llm_bench.cli plot
  - Plot box plots: python -m chess_llm_bench.cli plot-box
  - Both commands default to reading results/leaderboard.csv and writing images into results/.
  - To customize paths: add --leaderboard-csv and --out-png.

- Metrics
  - Acc64: Fraction of correctly reconstructed squares over 64 (Task A). Higher is better.
  - Hamming: Number of incorrect squares (diagnostic). Lower is better. Hamming = 64 * (1 − Acc64).
  - MAE_cp: Median absolute error of eval_cp_white in centipawns after clipping both truth and prediction to [−1000, 1000]. Lower is better.
  - Composite: Single leaderboard score combining both tasks:
    Score = 0.7 * Acc64 + 0.3 * max(0, 1 − MAE_cp / 1000)
    Range is approximately [0, 1]; higher is better.

- Coverage
  - For each run we also track validity:
    - n_valid_fen: Count of items with a parsable 64‑square FEN.
    - n_valid_eval: Count of items with a numeric eval_cp_white that can be parsed to int.
  - Coverage labels and some plots display n_valid/n (valid items over total), using min(n_valid_fen, n_valid_eval) for valid‑both.

- CSV Columns (leaderboard.csv)
  - timestamp: ISO timestamp when the row was appended.
  - provider: API/provider used (e.g., openai, openrouter).
  - model: Model identifier; may include a suffix like "(high)" if reasoning effort was used.
  - dataset: Path to the dataset truths.jsonl used.
  - n: Number of items evaluated for the run.
  - acc64: Average per‑square accuracy across items (0..1).
  - hamming: Average Hamming distance across items (0..64).
  - mae_cp: Median absolute centipawn error across items (0..1000).
  - composite: Composite score as defined above (0..1, higher is better).
  - run_file: Path to the predictions JSONL that was evaluated.
  - n_valid_fen: Count of items with valid FEN predictions.
  - n_valid_eval: Count of items with valid numeric evals.

