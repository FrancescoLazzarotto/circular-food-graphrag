"""Command-line interface of the evaluation toolkit.

Usage:
    graphrag-eval <subcommand> [options]
    python -m evalkit.cli <subcommand> [options]

Subcommands:
    build-dataset      Join run results with gold labels
    retrieval          Compute retrieval metrics
    text               Compute text similarity metrics
    judge              Run LLM-as-a-Judge
    judge-compare      Compare two judge runs
    ragas              Run RAGAS metrics
    kg                 Compute KG quality metrics
    gold-triples       Extract/apply gold triple candidates from Neo4j
    report-experiment  Full report for one experiment run
    report-project     Full report across all runs (project-level)
    baseline-update    Update baseline metrics file

Every ``cmd_*`` handler takes the parsed arguments and returns the process
exit code.
"""

from __future__ import annotations

# Ensure the evaluation/ directory is on sys.path so `evalkit` is importable
# as a top-level package regardless of how this module is invoked.
import sys as _sys
from pathlib import Path as _Path

_EVAL_DIR = _Path(__file__).resolve().parents[1]  # …/evaluation/
if str(_EVAL_DIR) not in _sys.path:
    _sys.path.insert(0, str(_EVAL_DIR))

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("graphrag")


def _setup_logging() -> None:
    """Log INFO and above to stderr."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")


# ─── Subcommand: build-dataset ───────────────────────────────────────────────

def cmd_build_dataset(args: argparse.Namespace) -> int:
    """``build-dataset``: join run results to the gold, save the rows as CSV."""
    from evalkit.io.dataset import build_dataset, rows_to_csv

    if args.smoke:
        _fixtures = Path(__file__).resolve().parents[1] / "fixtures"
        _smoke_results = _fixtures / "smoke_results.csv"
        gold_path = _fixtures / "smoke_gold.csv"
        output = Path(args.output) if args.output else (
            Path(__file__).resolve().parents[2] / "artifacts" / "evaluation" / "smoke_eval_dataset.csv"
        )
        # smoke_results.csv is a raw results file; wrap it in a temp dir as results.csv
        import shutil
        import tempfile

        _tmpdir = tempfile.mkdtemp()
        _tmp_run = Path(_tmpdir) / "smoke_run"
        _tmp_run.mkdir()
        shutil.copy(str(_smoke_results), str(_tmp_run / "results.csv"))
        run_dirs = [_tmp_run]
    else:
        if not args.input or not args.gold_file or not args.output:
            logger.error("--input, --gold-file and --output are required unless --smoke")
            return 1
        run_dirs = [Path(args.input)]
        gold_path = Path(args.gold_file)
        output = Path(args.output)

    rows = build_dataset(
        run_dirs=run_dirs,
        gold_path=gold_path,
        tag_contains=args.tag_contains or "",
    )

    if args.smoke and args.smoke_size > 0:
        rows = rows[: args.smoke_size]

    rows_to_csv(rows, output)
    logger.info("saved=%s rows=%d", output, len(rows))
    return 0


# ─── Subcommand: retrieval ────────────────────────────────────────────────────

def cmd_retrieval(args: argparse.Namespace) -> int:
    """``retrieval``: retrieval metrics per row and per group, with bootstrap CIs."""
    import csv

    from evalkit.io.dataset import rows_from_csv
    from evalkit.metrics.retrieval import RETRIEVAL_METRICS, compute_retrieval_row
    from evalkit.metrics.stats import aggregate

    if args.smoke:
        input_path = (
            Path(__file__).resolve().parents[2] / "artifacts" / "evaluation" / "smoke_eval_dataset.csv"
        )
        if not input_path.exists():
            logger.error("Smoke eval dataset not found at %s. Run build-dataset --smoke first.", input_path)
            return 1
    else:
        if not args.input:
            logger.error("--input required")
            return 1
        input_path = Path(args.input)

    rows = rows_from_csv(input_path)
    k = args.k if args.k and args.k > 0 else None
    row_metrics = [compute_retrieval_row(row, k=k) for row in rows]

    groups = aggregate(
        row_metrics=row_metrics,
        metric_names=RETRIEVAL_METRICS,
        n_bootstrap=args.n_bootstrap,
        ci=args.ci,
        seed=args.seed,
    )

    if args.save_json:
        import dataclasses
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([dataclasses.asdict(g) for g in groups], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("saved_json=%s", out)

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        import dataclasses

        flat = []
        for g in groups:
            d = dataclasses.asdict(g)
            row_flat: dict = {**d["keys"], "n_rows": d["n_rows"]}
            for m, stats in d["metrics"].items():
                for k2, v in stats.items():
                    row_flat[f"{m}_{k2}"] = v
            flat.append(row_flat)

        if flat:
            fieldnames = list(flat[0].keys())
            with out.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat)
        logger.info("saved_csv=%s", out)

    if args.save_row_csv:
        out = Path(args.save_row_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        if row_metrics:
            fieldnames = list(row_metrics[0].keys())
            with out.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(row_metrics)
        logger.info("saved_row_csv=%s", out)

    return 0


# ─── Subcommand: text ─────────────────────────────────────────────────────────

def cmd_text(args: argparse.Namespace) -> int:
    """``text``: text-similarity metrics per row and per group."""
    import csv

    from evalkit.io.dataset import rows_from_csv
    from evalkit.metrics.stats import aggregate
    from evalkit.metrics.text import TEXT_METRICS, compute_text_row

    rows = rows_from_csv(Path(args.input))
    row_metrics = [compute_text_row(row, bertscore=args.bertscore) for row in rows]

    groups = aggregate(row_metrics=row_metrics, metric_names=TEXT_METRICS)

    if args.save_json:
        import dataclasses

        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps([dataclasses.asdict(g) for g in groups], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.save_row_csv:
        import csv

        out = Path(args.save_row_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        if row_metrics:
            fieldnames = list(row_metrics[0].keys())
            with out.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(row_metrics)
    return 0


# ─── Subcommand: judge ────────────────────────────────────────────────────────

def cmd_judge(args: argparse.Namespace) -> int:
    """``judge``: score an evaluation dataset with the LLM judge."""
    from evalkit.io.dataset import rows_from_csv
    from evalkit.judge.backends import make_backend
    from evalkit.judge.llm_judge import LLMJudge

    if not args.model:
        logger.error("--model required for judge")
        return 1

    rows = rows_from_csv(Path(args.input))
    backend = make_backend(
        backend=args.backend,
        model_id=args.model,
        api_provider=args.provider,
        max_new_tokens=args.max_new_tokens,
        claude_code_bin=args.claude_bin,
    )
    rubric_names = [r.strip() for r in args.rubrics.split(",") if r.strip()]

    out_dir = Path(args.out) if args.out else None
    # The subscription backend (and any batch_size > 1) routes through the
    # batched, checkpointed path; batch_size == 1 scores row by row with LLMJudge.
    use_batched = args.backend == "claude_code" or args.batch_size > 1
    if use_batched:
        from evalkit.judge.batch import score_dataset_batched

        result = score_dataset_batched(
            rows,
            backend=backend,
            rubric_names=rubric_names,
            batch_size=args.batch_size,
            out_dir=out_dir,
            resume=args.resume,
        )
    else:
        judge = LLMJudge(backend=backend, rubric_names=rubric_names)
        result = judge.score_dataset(rows)

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "judge_summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("saved judge results to %s", out_dir)

    return 0


# ─── Subcommand: judge-compare ────────────────────────────────────────────────

def cmd_judge_compare(args: argparse.Namespace) -> int:
    """``judge-compare``: agreement between two judge runs."""
    from evalkit.judge.compare import compare_from_paths, render_markdown

    cmp = compare_from_paths(
        Path(args.a), Path(args.b), label_a=args.label_a, label_b=args.label_b
    )
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "judge_model_comparison.json").write_text(
            json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "judge_model_comparison.md").write_text(render_markdown(cmp), encoding="utf-8")
        logger.info("saved judge comparison to %s", out)
    else:
        print(render_markdown(cmp))
    return 0


# ─── Subcommand: ragas ────────────────────────────────────────────────────────

def cmd_ragas(args: argparse.Namespace) -> int:
    """``ragas``: score an evaluation dataset with RAGAS."""
    import csv

    from evalkit.io.dataset import rows_from_csv
    from evalkit.judge.ragas_backend import run_ragas

    rows = rows_from_csv(Path(args.input))
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]

    result = run_ragas(
        rows=rows,
        metric_names=metric_names,
        judge_model=args.judge_model or "",
        embed_model=args.embed_model,
        judge_backend=args.judge_backend,
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.vllm_model,
        vllm_api_key=args.vllm_api_key,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    if args.save_summary_json:
        out = Path(args.save_summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("saved_summary_json=%s", out)

    if args.save_row_csv and result.get("row_scores"):
        out = Path(args.save_row_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows_data = result["row_scores"]
        fieldnames = sorted({k for r in rows_data for k in r})
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_data)
        logger.info("saved_row_csv=%s", out)

    return 0


# ─── Subcommand: kg ───────────────────────────────────────────────────────────

def cmd_kg(args: argparse.Namespace) -> int:
    """``kg``: KG quality metrics, from a run's artifacts or from Neo4j."""
    import dataclasses

    from evalkit.kg.kg_quality import compute_from_artifacts, compute_from_neo4j

    if args.neo4j:
        import os

        result = compute_from_neo4j(
            neo4j_url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE") or "neo4j",
        )
    else:
        if not args.run_dir:
            logger.error("--run-dir required unless --neo4j")
            return 1
        result = compute_from_artifacts(Path(args.run_dir))

    payload = dataclasses.asdict(result)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("saved=%s", out)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    return 0


# ─── Subcommand: gold-triples ────────────────────────────────────────────────

def cmd_gold_triples(args: argparse.Namespace) -> int:
    """``gold-triples``: extract candidate gold triples, or apply a review."""
    import os

    from evalkit.kg.gold_triples import apply_review, extract_candidates

    gold_path = Path(args.gold)
    out_path = Path(args.out)

    if args.mode == "apply":
        if not args.candidates:
            logger.error("--candidates required in apply mode")
            return 1
        summary = apply_review(
            gold_path=gold_path,
            candidates_path=Path(args.candidates),
            out_path=out_path,
        )
    else:
        summary = extract_candidates(
            gold_path=gold_path,
            out_path=out_path,
            neo4j_url=os.getenv("NEO4J_URL", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE") or "neo4j",
            max_per_question=args.max_per_question,
            bridge=not args.no_bridge,
            min_score=args.min_score,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# ─── Subcommand: report-experiment ───────────────────────────────────────────

def cmd_report_experiment(args: argparse.Namespace) -> int:
    """``report-experiment``: full report of one run, Markdown and JSON."""
    from evalkit.config import EvalConfig
    from evalkit.report.aggregate import build_experiment_report
    from evalkit.report.json_report import write_json_report
    from evalkit.report.markdown import write_markdown
    from evalkit.report.plots import generate_plots

    if args.smoke:
        fixtures = Path(__file__).resolve().parents[1] / "fixtures"
        smoke_results = fixtures / "smoke_results.csv"
        gold_path = fixtures / "smoke_gold.csv"
        out_dir = Path(args.out) if args.out else Path("artifacts/tmp/evalkit_smoke")

        # Build eval dataset from smoke fixtures in-memory
        from evalkit.io.dataset import build_dataset, rows_to_csv
        from evalkit.report.aggregate import build_experiment_report as _build
        from evalkit.report.json_report import write_json_report
        from evalkit.report.markdown import write_markdown
        from evalkit.report.plots import generate_plots

        # smoke_results.csv is the raw results file; wrap it in a temp run dir
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_run = Path(tmpdir) / "smoke_run"
            tmp_run.mkdir()
            # symlink/copy smoke_results.csv as results.csv
            import shutil
            shutil.copy(str(smoke_results), str(tmp_run / "results.csv"))

            cfg = EvalConfig(k=args.k or None, bertscore=False)
            report = _build(run_dir=tmp_run, gold_path=gold_path, config=cfg)

        plots_dir: Path | None = None
        if not args.no_plots:
            plots_dir = out_dir / "plots"
            generate_plots(report, plots_dir)

        write_markdown(report, out_dir / "report.md", plots_dir=plots_dir)
        write_json_report(report, out_dir / "report.json")
        logger.info("Smoke report written to %s", out_dir)
        return 0
    else:
        if not args.run_dir:
            logger.error("--run-dir required")
            return 1
        run_dir = Path(args.run_dir)
        gold_path = Path(args.gold) if args.gold else None
        out_dir = Path(args.out) if args.out else run_dir / "evaluation"

    cfg = EvalConfig(
        k=args.k or None,
        bertscore=False,
    )

    kg_dir = Path(args.kg_dir) if args.kg_dir else None

    report = build_experiment_report(
        run_dir=run_dir,
        gold_path=gold_path,
        kg_artifacts_dir=kg_dir,
        config=cfg,
    )

    plots_dir: Path | None = None
    if not args.no_plots:
        plots_dir = out_dir / "plots"
        generate_plots(report, plots_dir)

    write_markdown(report, out_dir / "report.md", plots_dir=plots_dir)
    write_json_report(report, out_dir / "report.json")
    logger.info("Report written to %s", out_dir)
    return 0


# ─── Subcommand: report-project ──────────────────────────────────────────────

def cmd_report_project(args: argparse.Namespace) -> int:
    """``report-project``: report across all runs, Markdown and JSON."""
    from evalkit.config import EvalConfig
    from evalkit.report.aggregate import build_project_report
    from evalkit.report.json_report import write_json_report
    from evalkit.report.markdown import write_markdown
    from evalkit.report.plots import generate_plots

    experiments_root = Path(args.runs_root or "artifacts/experiments")
    gold_path = Path(args.gold) if args.gold else None
    out_dir = Path(args.out) if args.out else Path("artifacts/evaluation/project")

    cfg = EvalConfig(
        baselines_path=Path(args.baseline) if args.baseline else Path("evaluation/baselines/baseline_metrics.json"),
    )

    kg_dir = Path(args.kg_dir) if args.kg_dir else None

    report = build_project_report(
        experiments_root=experiments_root,
        gold_path=gold_path,
        kg_artifacts_dir=kg_dir,
        config=cfg,
        tag_contains=args.tag_contains or "",
    )

    plots_dir: Path | None = None
    if not args.no_plots:
        plots_dir = out_dir / "plots"
        generate_plots(report, plots_dir)

    write_markdown(report, out_dir / "project_report.md", plots_dir=plots_dir)
    write_json_report(report, out_dir / "project_report.json")
    logger.info("Project report written to %s", out_dir)
    return 0


# ─── Subcommand: baseline-update ─────────────────────────────────────────────

def cmd_baseline_update(args: argparse.Namespace) -> int:
    """``baseline-update``: store a report's global means as the baseline."""
    from evalkit.report.regression import update_baseline

    report_path = Path(args.from_report)
    if not report_path.exists():
        logger.error("Report file not found: %s", report_path)
        return 1

    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    groups = report_data.get("groups", [])

    new_values: dict[str, float] = {}
    for g in groups:
        if g.get("keys", {}).get("segment") != "global":
            continue
        for metric, stats in g.get("metrics", {}).items():
            if stats.get("n", 0) > 0:
                new_values[metric] = float(stats["mean"])

    if not new_values:
        logger.error("No metric values found in report: %s", report_path)
        return 1

    baseline_path = Path(args.baseline or "evaluation/baselines/baseline_metrics.json")
    update_baseline(baseline_path, new_values)
    return 0


# ─── Argument parser ─────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """The argument parser, one subparser per subcommand."""
    parser = argparse.ArgumentParser(
        prog="graphrag-eval",
        description="evalkit — evaluation toolkit for graphRAGPipelineExp1",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # build-dataset
    p = sub.add_parser("build-dataset", help="Join run results with gold labels")
    p.add_argument("--input", default="")
    p.add_argument("--gold-file", default="")
    p.add_argument("--output", default="")
    p.add_argument("--tag-contains", default="")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke-size", type=int, default=5)

    # retrieval
    p = sub.add_parser("retrieval", help="Compute retrieval metrics")
    p.add_argument("--input", default="")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-json", default="")
    p.add_argument("--save-csv", default="")
    p.add_argument("--save-row-csv", default="")
    p.add_argument("--smoke", action="store_true")

    # text
    p = sub.add_parser("text", help="Compute text similarity metrics")
    p.add_argument("--input", required=True)
    p.add_argument("--bertscore", action="store_true")
    p.add_argument("--save-json", default="")
    p.add_argument("--save-row-csv", default="")

    # judge
    p = sub.add_parser("judge", help="LLM-as-a-Judge evaluation")
    p.add_argument("--input", required=True)
    p.add_argument(
        "--backend", default="vllm", choices=["vllm", "local_hf", "api", "claude_code"]
    )
    p.add_argument("--model", default="")
    p.add_argument(
        "--provider", default="anthropic", choices=["anthropic", "openai"],
        help="API provider for --backend api",
    )
    p.add_argument(
        "--claude-bin", default="",
        help="Path to the claude CLI for --backend claude_code (default: $CLAUDE_CODE_BIN or 'claude')",
    )
    p.add_argument(
        "--rubrics",
        default="factual_correctness,completeness,groundedness,relevance",
        help=(
            "Comma-separated rubric names. 'abstention' is applied automatically to "
            "distractor rows and need not be listed. The legacy name "
            "'answer_correctness' still resolves, with a deprecation warning."
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Rows per judge call. >1 (or claude_code) uses the batched, checkpointed path.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip rows already present in <out>/judge_rows.jsonl",
    )
    p.add_argument("--out", default="")

    # judge-compare
    p = sub.add_parser("judge-compare", help="Compare two judge runs (e.g. Haiku vs Sonnet)")
    p.add_argument("--a", required=True, help="First judge out dir or judge_summary.json")
    p.add_argument("--b", required=True, help="Second judge out dir or judge_summary.json")
    p.add_argument("--label-a", default="a")
    p.add_argument("--label-b", default="b")
    p.add_argument("--out", default="")

    # ragas
    p = sub.add_parser("ragas", help="RAGAS generative metrics")
    p.add_argument("--input", required=True)
    p.add_argument("--metrics", default="faithfulness,answer_relevancy,answer_correctness,context_precision,context_recall")
    p.add_argument("--judge-backend", choices=["transformers", "vllm"], default="transformers",
                   help="'transformers' loads --judge-model in-process; 'vllm' calls an OpenAI-compatible endpoint")
    p.add_argument("--judge-model", default="", help="HF model id for the transformers backend")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--vllm-base-url", default="", help="vLLM endpoint (vllm backend); defaults to $VLLM_BASE_URL")
    p.add_argument("--vllm-model", default="", help="model id served by the vLLM endpoint; defaults to $VLLM_MODEL_NAME")
    p.add_argument("--vllm-api-key", default="", help="endpoint API key; defaults to $VLLM_API_KEY/$OPENAI_API_KEY")
    p.add_argument("--max-new-tokens", type=int, default=192, help="max tokens the judge may generate per call")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--save-row-csv", default="")
    p.add_argument("--save-summary-json", default="")

    # kg
    p = sub.add_parser("kg", help="KG quality metrics")
    p.add_argument("--run-dir", default="")
    p.add_argument("--neo4j", action="store_true")
    p.add_argument("--out", default="")

    # gold-triples
    p = sub.add_parser(
        "gold-triples",
        help="Extract gold triple candidates from Neo4j / apply reviewed candidates to a gold CSV",
    )
    p.add_argument("--mode", choices=["extract", "apply"], default="extract")
    p.add_argument("--gold", required=True, help="Gold CSV (question, expected_entities, ...)")
    p.add_argument("--out", required=True, help="extract: candidates CSV; apply: new gold CSV")
    p.add_argument("--candidates", default="", help="Reviewed candidates CSV (apply mode)")
    p.add_argument("--max-per-question", type=int, default=30)
    p.add_argument("--no-bridge", action="store_true", help="Skip 2-hop bridging triples between matched entities")
    p.add_argument("--min-score", type=float, default=0.0)

    # report-experiment
    p = sub.add_parser("report-experiment", help="Full report for one experiment run")
    p.add_argument("--run-dir", default="")
    p.add_argument("--gold", default="")
    p.add_argument("--kg-dir", default="")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out", default="")
    p.add_argument("--smoke", action="store_true")

    # report-project
    p = sub.add_parser("report-project", help="Project-level report across all runs")
    p.add_argument("--runs-root", default="artifacts/experiments")
    p.add_argument("--gold", default="")
    p.add_argument("--kg-dir", default="")
    p.add_argument("--baseline", default="")
    p.add_argument("--tag-contains", default="")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out", default="")

    # baseline-update
    p = sub.add_parser("baseline-update", help="Update baseline metrics from a report JSON")
    p.add_argument("--from-report", required=True)
    p.add_argument("--baseline", default="")

    return parser


SUBCOMMAND_MAP = {
    "build-dataset": cmd_build_dataset,
    "retrieval": cmd_retrieval,
    "text": cmd_text,
    "judge": cmd_judge,
    "judge-compare": cmd_judge_compare,
    "ragas": cmd_ragas,
    "kg": cmd_kg,
    "gold-triples": cmd_gold_triples,
    "report-experiment": cmd_report_experiment,
    "report-project": cmd_report_project,
    "baseline-update": cmd_baseline_update,
}


def main() -> int:
    """Parse the command line and run the chosen subcommand."""
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    # Ensure evalkit is importable (when called from repo root)
    import sys
    from pathlib import Path

    eval_dir = Path(__file__).resolve().parents[2]
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    fn = SUBCOMMAND_MAP.get(args.subcommand)
    if fn is None:
        parser.print_help()
        return 1
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
