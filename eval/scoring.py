"""
Scoring functions for the Defense LLM Evaluation Framework.

Provides metric computation for classification (F1), entity extraction
(entity-level F1), and generation tasks (ROUGE-L, BLEU).
"""

from typing import Any

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_f1(
    predictions: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, float]:
    """Compute classification metrics (F1, precision, recall).

    Args:
        predictions: List of model prediction dictionaries with 'category' key.
        expected: List of ground-truth dictionaries with 'category' key.
        metrics: List of metric names to compute.

    Returns:
        Dictionary mapping metric names to computed scores.
    """
    pred_labels = [p.get("category", "unknown") for p in predictions]
    true_labels = [e.get("category", "unknown") for e in expected]

    scores = {}
    if "f1-weighted" in metrics:
        scores["f1-weighted"] = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    if "precision" in metrics:
        scores["precision"] = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    if "recall" in metrics:
        scores["recall"] = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)

    return scores


def compute_entity_f1(
    predictions: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, float]:
    """Compute entity-level F1 for NER extraction tasks.

    Supports exact match and partial overlap matching.

    Args:
        predictions: List of prediction dicts, each containing an 'entities' list.
        expected: List of ground-truth dicts, each containing an 'entities' list.
        metrics: List of metric names to compute.

    Returns:
        Dictionary mapping metric names to computed scores.
    """
    exact_tp, exact_fp, exact_fn = 0, 0, 0
    partial_tp, partial_fp, partial_fn = 0, 0, 0

    for pred, exp in zip(predictions, expected):
        pred_entities = {(e.get("text", ""), e.get("type", "")) for e in pred.get("entities", [])}
        exp_entities = {(e.get("text", ""), e.get("type", "")) for e in exp.get("entities", [])}

        # Exact match
        exact_tp += len(pred_entities & exp_entities)
        exact_fp += len(pred_entities - exp_entities)
        exact_fn += len(exp_entities - pred_entities)

        # Partial match (type must match, text overlap > 50%)
        for p_text, p_type in pred_entities:
            matched = False
            for e_text, e_type in exp_entities:
                if p_type == e_type and _text_overlap(p_text, e_text) > 0.5:
                    matched = True
                    break
            if matched:
                partial_tp += 1
            else:
                partial_fp += 1

        for e_text, e_type in exp_entities:
            matched = any(
                p_type == e_type and _text_overlap(p_text, e_text) > 0.5
                for p_text, p_type in pred_entities
            )
            if not matched:
                partial_fn += 1

    scores = {}
    if "entity-f1-exact" in metrics:
        scores["entity-f1-exact"] = _safe_f1(exact_tp, exact_fp, exact_fn)
    if "entity-f1-partial" in metrics:
        scores["entity-f1-partial"] = _safe_f1(partial_tp, partial_fp, partial_fn)

    return scores


def _text_overlap(text_a: str, text_b: str) -> float:
    """Compute character-level overlap ratio between two strings.

    Args:
        text_a: First text string.
        text_b: Second text string.

    Returns:
        Overlap ratio between 0.0 and 1.0.
    """
    if not text_a or not text_b:
        return 0.0
    set_a, set_b = set(text_a.lower()), set(text_b.lower())
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    """Compute F1 score from raw counts with zero-division safety.

    Args:
        tp: True positive count.
        fp: False positive count.
        fn: False negative count.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_rouge(
    predictions: list[dict[str, Any]],
    references: list[str],
    metrics: list[str],
) -> dict[str, float]:
    """Compute ROUGE-L and BLEU scores for generation tasks.

    Args:
        predictions: List of prediction dicts with 'raw_text' or 'summary' key.
        references: List of reference text strings.
        metrics: List of metric names to compute.

    Returns:
        Dictionary mapping metric names to average scores.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = {}

    if "rouge-l" in metrics:
        rouge_scores = []
        for pred, ref in zip(predictions, references):
            pred_text = pred.get("summary", pred.get("raw_text", ""))
            result = scorer.score(ref, pred_text)
            rouge_scores.append(result["rougeL"].fmeasure)
        scores["rouge-l"] = float(np.mean(rouge_scores)) if rouge_scores else 0.0

    if "entity-overlap-f1" in metrics:
        overlap_scores = []
        for pred, ref in zip(predictions, references):
            pred_text = pred.get("summary", pred.get("raw_text", ""))
            overlap_scores.append(_entity_overlap(pred_text, ref))
        scores["entity-overlap-f1"] = float(np.mean(overlap_scores)) if overlap_scores else 0.0

    return scores


def _entity_overlap(prediction: str, reference: str) -> float:
    """Compute simple token overlap as a proxy for entity overlap.

    Args:
        prediction: Generated text.
        reference: Reference text.

    Returns:
        Token-level F1 overlap score.
    """
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    tp = len(pred_tokens & ref_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def aggregate_results(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate evaluation results across multiple tasks into a summary.

    Args:
        all_results: List of per-task result dictionaries.

    Returns:
        Aggregated summary with per-model averages across tasks.
    """
    model_scores: dict[str, list[float]] = {}

    for result in all_results:
        task_name = result["task"]["name"]
        for model_id, model_data in result["models"].items():
            if model_id not in model_scores:
                model_scores[model_id] = []
            primary_score = list(model_data["scores"].values())[0] if model_data["scores"] else 0.0
            model_scores[model_id].append(primary_score)

    aggregated = {
        "aggregated_at": None,  # filled by caller
        "models": {},
    }
    for model_id, scores in model_scores.items():
        aggregated["models"][model_id] = {
            "average_score": float(np.mean(scores)),
            "per_task_scores": scores,
            "n_tasks": len(scores),
        }

    return aggregated
