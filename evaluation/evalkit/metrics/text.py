"""Text similarity of an answer to its reference: EM, F1, ROUGE-L, BLEU, BERTScore."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from evalkit.models import EvalRow


def _normalize(text: str) -> str:
    """NFKD, lowercased, whitespace-collapsed text."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text: str) -> list[str]:
    """Whitespace tokens of the normalised text."""
    return _normalize(text).split()


# ─── Deterministic text metrics ─────────────────────────────────────────────

def exact_match(prediction: str, reference: str) -> float:
    """1.0 if normalized strings are identical, else 0.0."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def partial_match(prediction: str, reference: str) -> float:
    """Fraction of the reference's distinct tokens present in the prediction.

    Tokens are compared whole, not as substrings of the prediction, so a
    two-letter reference token cannot match inside a longer word; and each
    distinct reference token counts once, so a repeated common word cannot
    dominate the ratio.

    Args:
        prediction: Model answer.
        reference: Gold answer.

    Returns:
        Ratio in ``[0, 1]``.
    """
    pred_tokens = set(_tokenize(prediction))
    ref_tokens = set(_tokenize(reference))
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & pred_tokens) / len(ref_tokens)


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between prediction and reference."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_set: dict[str, int] = {}
    for t in pred_tokens:
        pred_set[t] = pred_set.get(t, 0) + 1

    ref_set: dict[str, int] = {}
    for t in ref_tokens:
        ref_set[t] = ref_set.get(t, 0) + 1

    common = sum(min(pred_set.get(t, 0), ref_set.get(t, 0)) for t in ref_set)
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L based on longest common subsequence at token level."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs_len = _lcs(pred_tokens, ref_tokens)
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs(a: list[str], b: list[str]) -> int:
    """Length of longest common subsequence."""
    m, n = len(a), len(b)
    # Space-optimised DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Corpus-style BLEU-N (1..max_n average) with brevity penalty.

    Uses sacrebleu when available, falls back to a simple sentence-BLEU.
    """
    try:
        from sacrebleu.metrics import BLEU as SacreBLEU  # type: ignore

        metric = SacreBLEU(effective_order=True)
        result = metric.sentence_score(prediction, [reference])
        return result.score / 100.0
    except ImportError:
        pass

    return _simple_bleu(prediction, reference, max_n)


def _ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    """Count of every n-gram of `tokens`."""
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _simple_bleu(prediction: str, reference: str, max_n: int) -> float:
    """Sentence BLEU with effective order and brevity penalty, without sacrebleu."""
    import math

    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if pred_tokens else 0.0

    # Effective-order BLEU (mirrors sacrebleu): only n-gram orders the
    # prediction can actually produce enter the geometric mean, and a single
    # zero precision among them zeroes the score. Skipping zero terms while
    # still dividing by max_n would silently inflate the score.
    precisions = []
    for n in range(1, max_n + 1):
        if len(pred_tokens) < n:
            break
        pred_ngrams = _ngrams(pred_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        clipped = sum(min(count, ref_ngrams.get(gram, 0)) for gram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        precisions.append(clipped / total if total else 0.0)

    if not precisions or any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)
    return bp * math.exp(log_avg)


def bertscore_f1(prediction: str, reference: str, model: str = "distilbert-base-uncased") -> float | None:
    """BERTScore F1 (returns None if bert-score is not installed)."""
    try:
        from bert_score import score as _bs  # type: ignore

        P, R, F1 = _bs([prediction], [reference], model_type=model, verbose=False)
        return float(F1[0])
    except ImportError:
        return None


# ─── Answer cleaning ────────────────────────────────────────────────────────

_BOILERPLATE_MARKERS = (
    "Verifica nel grafo:",
    "Evidence in graph:",
    "Evidence in Graph:",
)

def _clean_answer(answer: str) -> str:
    """Strip KG provenance block appended after the narrative answer."""
    for marker in _BOILERPLATE_MARKERS:
        idx = answer.find(marker)
        if idx != -1:
            answer = answer[:idx]
    return answer.strip()


# ─── Variant-aware scoring ───────────────────────────────────────────────────

def best_variant_score(
    metric_fn: Any,
    prediction: str,
    ground_truth: str,
    answer_variants: list[str],
) -> float | None:
    """Return the best score across ground_truth and answer_variants.

    Returns None when the metric itself is unavailable (e.g. bertscore_f1
    without bert-score installed) — a None from the metric must not be
    compared by max() nor coerced to 0.0.
    """
    references = [ground_truth] + [v for v in answer_variants if v.strip()]
    scores = [metric_fn(prediction, ref) for ref in references if ref.strip()]
    valid = [s for s in scores if s is not None]
    if valid:
        return max(valid)
    return None if scores else 0.0


# ─── Row-level computation ───────────────────────────────────────────────────

def compute_text_row(row: EvalRow, bertscore: bool = False) -> dict[str, Any]:
    """Compute text similarity metrics for a single EvalRow.

    Returns None for all metrics when skip_reason is set or ground_truth is empty.
    """
    base = {
        "run_dir": row.run_dir,
        "model_id": row.model_id,
        "framework": row.framework,
        "strategy": row.strategy,
        "question": row.question,
        "question_type": row.question_type,
        "skip_reason": row.skip_reason,
        "exact_match": None,
        "partial_match": None,
        "token_f1": None,
        "rouge_l": None,
        "bleu": None,
        "bertscore_f1": None,
    }

    if row.skip_reason or not row.ground_truth.strip():
        return base

    pred = _clean_answer(row.answer)
    gt = row.ground_truth
    variants = row.answer_variants

    base["exact_match"] = best_variant_score(exact_match, pred, gt, variants)
    base["partial_match"] = best_variant_score(partial_match, pred, gt, variants)
    base["token_f1"] = best_variant_score(token_f1, pred, gt, variants)
    base["rouge_l"] = best_variant_score(rouge_l, pred, gt, variants)
    base["bleu"] = best_variant_score(bleu, pred, gt, variants)

    if bertscore:
        bs = best_variant_score(bertscore_f1, pred, gt, variants)
        base["bertscore_f1"] = bs if bs is not None else None

    return base


TEXT_METRICS = ["exact_match", "partial_match", "token_f1", "rouge_l", "bleu", "bertscore_f1"]
