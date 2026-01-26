"""
Code-Based Metrics - Programmatic evaluation metrics using well-known open source packages.

This module provides:
1. Deterministic metric computations
2. Integration with popular NLP evaluation libraries
3. Code generation for sample metric implementations

Supported Libraries:
- rouge-score: ROUGE metrics for summarization
- sacrebleu: BLEU, chrF, TER for translation
- bert-score: Semantic similarity using BERT embeddings
- textstat: Readability metrics
- evaluate (HuggingFace): Unified API for many metrics
- rapidfuzz: Fuzzy string matching
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class CodeMetricDefinition:
    """Definition for a code-based metric"""
    name: str
    description: str
    package: str  # pip package name
    import_statement: str
    sample_code: str
    output_type: str  # "float", "dict", "list"
    score_range: Tuple[float, float]  # min, max
    higher_is_better: bool
    applicable_categories: List[str]


# ═══════════════════════════════════════════════════════════════════
# CODE-BASED METRICS REGISTRY
# ═══════════════════════════════════════════════════════════════════

CODE_METRICS_REGISTRY: Dict[str, CodeMetricDefinition] = {
    
    # ─────────────────────────────────────────────────────────────────
    # ROUGE - Summarization Metrics
    # ─────────────────────────────────────────────────────────────────
    "rouge": CodeMetricDefinition(
        name="ROUGE",
        description="Recall-Oriented Understudy for Gisting Evaluation - measures overlap between generated and reference summaries.",
        package="rouge-score",
        import_statement="from rouge_score import rouge_scorer",
        sample_code='''def compute_rouge(prediction: str, reference: str) -> dict:
    """
    Compute ROUGE scores for summarization evaluation.
    
    ROUGE-1: Unigram overlap
    ROUGE-2: Bigram overlap  
    ROUGE-L: Longest common subsequence
    """
    from rouge_score import rouge_scorer
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    
    return {
        "rouge1_precision": scores['rouge1'].precision,
        "rouge1_recall": scores['rouge1'].recall,
        "rouge1_fmeasure": scores['rouge1'].fmeasure,
        "rouge2_precision": scores['rouge2'].precision,
        "rouge2_recall": scores['rouge2'].recall,
        "rouge2_fmeasure": scores['rouge2'].fmeasure,
        "rougeL_precision": scores['rougeL'].precision,
        "rougeL_recall": scores['rougeL'].recall,
        "rougeL_fmeasure": scores['rougeL'].fmeasure,
    }

# Example usage:
prediction = "The cat sat on the mat."
reference = "A cat was sitting on a mat."
scores = compute_rouge(prediction, reference)
print(f"ROUGE-1 F1: {scores['rouge1_fmeasure']:.3f}")
print(f"ROUGE-2 F1: {scores['rouge2_fmeasure']:.3f}")
print(f"ROUGE-L F1: {scores['rougeL_fmeasure']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["summarization"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # BLEU - Translation Metrics
    # ─────────────────────────────────────────────────────────────────
    "bleu": CodeMetricDefinition(
        name="BLEU",
        description="Bilingual Evaluation Understudy - measures n-gram precision for machine translation.",
        package="sacrebleu",
        import_statement="import sacrebleu",
        sample_code='''def compute_bleu(predictions: list[str], references: list[list[str]]) -> dict:
    """
    Compute BLEU score for translation evaluation.
    
    Args:
        predictions: List of generated translations
        references: List of reference translations (can have multiple refs per prediction)
    """
    import sacrebleu
    
    bleu = sacrebleu.corpus_bleu(predictions, references)
    
    return {
        "bleu_score": bleu.score,  # 0-100 scale
        "bleu_normalized": bleu.score / 100,  # 0-1 scale
        "precisions": bleu.precisions,  # n-gram precisions
        "brevity_penalty": bleu.bp,
    }

# Example usage:
predictions = ["The cat is on the mat."]
references = [["The cat sat on the mat.", "A cat is sitting on the mat."]]
scores = compute_bleu(predictions, references)
print(f"BLEU Score: {scores['bleu_score']:.2f}")
print(f"Brevity Penalty: {scores['brevity_penalty']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 100.0),
        higher_is_better=True,
        applicable_categories=["translation"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # BERTScore - Semantic Similarity
    # ─────────────────────────────────────────────────────────────────
    "bertscore": CodeMetricDefinition(
        name="BERTScore",
        description="Semantic similarity using BERT embeddings - captures meaning beyond exact word match.",
        package="bert-score",
        import_statement="from bert_score import score as bert_score",
        sample_code='''def compute_bertscore(predictions: list[str], references: list[str], lang: str = "en") -> dict:
    """
    Compute BERTScore for semantic similarity evaluation.
    
    Uses contextual embeddings to measure semantic similarity,
    capturing paraphrases and meaning preservation.
    """
    from bert_score import score as bert_score
    
    P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False)
    
    return {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item(),
        "precision_per_sample": P.tolist(),
        "recall_per_sample": R.tolist(),
        "f1_per_sample": F1.tolist(),
    }

# Example usage:
predictions = ["The weather is nice today."]
references = ["It's a beautiful day outside."]
scores = compute_bertscore(predictions, references)
print(f"BERTScore F1: {scores['f1']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["summarization", "translation", "auto_reply", "generation"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # Readability Metrics
    # ─────────────────────────────────────────────────────────────────
    "readability": CodeMetricDefinition(
        name="Readability",
        description="Multiple readability metrics including Flesch Reading Ease, Flesch-Kincaid Grade, etc.",
        package="textstat",
        import_statement="import textstat",
        sample_code='''def compute_readability(text: str) -> dict:
    """
    Compute various readability metrics.
    
    - Flesch Reading Ease: 0-100 (higher = easier to read)
    - Flesch-Kincaid Grade: US grade level needed to understand
    - Gunning Fog Index: Years of education needed
    - SMOG Index: Years of education needed
    - Coleman-Liau Index: Grade level
    - Automated Readability Index: Grade level
    """
    import textstat
    
    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(text),  # 0-100, higher = easier
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),  # grade level
        "gunning_fog": textstat.gunning_fog(text),  # years of education
        "smog_index": textstat.smog_index(text),  # years of education
        "coleman_liau_index": textstat.coleman_liau_index(text),  # grade level
        "automated_readability_index": textstat.automated_readability_index(text),
        "dale_chall_readability": textstat.dale_chall_readability_score(text),
        "reading_time_seconds": textstat.reading_time(text, ms_per_char=14.69),
        "syllable_count": textstat.syllable_count(text),
        "word_count": textstat.lexicon_count(text, removepunct=True),
        "sentence_count": textstat.sentence_count(text),
    }

# Example usage:
text = "The quick brown fox jumps over the lazy dog. This sentence is simple and easy to read."
scores = compute_readability(text)
print(f"Flesch Reading Ease: {scores['flesch_reading_ease']:.1f}")
print(f"Grade Level: {scores['flesch_kincaid_grade']:.1f}")''',
        output_type="dict",
        score_range=(0.0, 100.0),
        higher_is_better=True,  # for Flesch Reading Ease
        applicable_categories=["summarization", "auto_reply", "generation"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # Exact Match & F1 - Extraction/QA Metrics
    # ─────────────────────────────────────────────────────────────────
    "exact_match_f1": CodeMetricDefinition(
        name="Exact Match & Token F1",
        description="Exact match and token-level F1 score for extraction and QA tasks.",
        package="evaluate",
        import_statement="import evaluate",
        sample_code='''def compute_exact_match_f1(prediction: str, reference: str) -> dict:
    """
    Compute Exact Match and Token F1 for extraction/QA tasks.
    
    - Exact Match: 1 if prediction exactly matches reference (after normalization)
    - Token F1: Harmonic mean of token precision and recall
    """
    import re
    from collections import Counter
    
    def normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\\s]', '', text)
        return ' '.join(text.split())
    
    def get_tokens(text: str) -> list:
        return normalize(text).split()
    
    pred_normalized = normalize(prediction)
    ref_normalized = normalize(reference)
    
    # Exact match
    exact_match = 1.0 if pred_normalized == ref_normalized else 0.0
    
    # Token F1
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)
    
    if not pred_tokens or not ref_tokens:
        return {"exact_match": exact_match, "token_f1": 0.0, "precision": 0.0, "recall": 0.0}
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    
    precision = num_common / len(pred_tokens) if pred_tokens else 0
    recall = num_common / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "exact_match": exact_match,
        "token_f1": f1,
        "precision": precision,
        "recall": recall,
    }

# Example usage:
prediction = "Barack Obama"
reference = "Barack Hussein Obama"
scores = compute_exact_match_f1(prediction, reference)
print(f"Exact Match: {scores['exact_match']}")
print(f"Token F1: {scores['token_f1']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["extraction", "classification"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # Fuzzy String Matching
    # ─────────────────────────────────────────────────────────────────
    "fuzzy_match": CodeMetricDefinition(
        name="Fuzzy String Matching",
        description="Levenshtein distance and similarity ratio for approximate string matching.",
        package="rapidfuzz",
        import_statement="from rapidfuzz import fuzz, distance",
        sample_code='''def compute_fuzzy_match(prediction: str, reference: str) -> dict:
    """
    Compute fuzzy string matching metrics.
    
    Uses Levenshtein distance and various similarity ratios.
    Useful for typo tolerance and approximate matching.
    """
    from rapidfuzz import fuzz, distance
    
    return {
        "simple_ratio": fuzz.ratio(prediction, reference) / 100,  # 0-1
        "partial_ratio": fuzz.partial_ratio(prediction, reference) / 100,  # 0-1
        "token_sort_ratio": fuzz.token_sort_ratio(prediction, reference) / 100,  # 0-1
        "token_set_ratio": fuzz.token_set_ratio(prediction, reference) / 100,  # 0-1
        "levenshtein_distance": distance.Levenshtein.distance(prediction, reference),
        "normalized_levenshtein": distance.Levenshtein.normalized_distance(prediction, reference),  # 0-1, lower = more similar
        "jaro_winkler_similarity": distance.JaroWinkler.similarity(prediction, reference),  # 0-1
    }

# Example usage:
prediction = "The quick brown fox"
reference = "The quikc brown fox"  # typo
scores = compute_fuzzy_match(prediction, reference)
print(f"Simple Ratio: {scores['simple_ratio']:.3f}")
print(f"Jaro-Winkler: {scores['jaro_winkler_similarity']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["extraction", "translation", "auto_reply"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # Length & Compression Metrics
    # ─────────────────────────────────────────────────────────────────
    "length_metrics": CodeMetricDefinition(
        name="Length & Compression",
        description="Character/word/sentence counts and compression ratio for summarization.",
        package="(built-in)",
        import_statement="import re",
        sample_code='''def compute_length_metrics(prediction: str, reference: str = None) -> dict:
    """
    Compute length-based metrics for evaluating output verbosity.
    
    Useful for summarization to measure compression ratio.
    """
    import re
    
    def count_words(text: str) -> int:
        return len(text.split())
    
    def count_sentences(text: str) -> int:
        return len(re.findall(r'[.!?]+', text)) or 1
    
    def count_chars(text: str, include_spaces: bool = False) -> int:
        return len(text) if include_spaces else len(text.replace(' ', ''))
    
    pred_words = count_words(prediction)
    pred_sentences = count_sentences(prediction)
    pred_chars = count_chars(prediction)
    
    result = {
        "word_count": pred_words,
        "sentence_count": pred_sentences,
        "char_count": pred_chars,
        "avg_word_length": pred_chars / pred_words if pred_words > 0 else 0,
        "avg_sentence_length": pred_words / pred_sentences if pred_sentences > 0 else 0,
    }
    
    if reference:
        ref_words = count_words(reference)
        ref_chars = count_chars(reference)
        result["compression_ratio_words"] = pred_words / ref_words if ref_words > 0 else 0
        result["compression_ratio_chars"] = pred_chars / ref_chars if ref_chars > 0 else 0
        result["reference_word_count"] = ref_words
    
    return result

# Example usage (summarization):
source = "This is a long document with many sentences. It contains detailed information about various topics. The goal is to summarize it concisely."
summary = "Document summarizing various topics concisely."
scores = compute_length_metrics(summary, source)
print(f"Compression Ratio: {scores['compression_ratio_words']:.2f}")
print(f"Summary Words: {scores['word_count']}")''',
        output_type="dict",
        score_range=(0.0, float('inf')),
        higher_is_better=False,  # for compression
        applicable_categories=["summarization", "auto_reply"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # N-gram Diversity
    # ─────────────────────────────────────────────────────────────────
    "diversity": CodeMetricDefinition(
        name="N-gram Diversity",
        description="Distinct n-gram ratios to measure output diversity and avoid repetition.",
        package="(built-in)",
        import_statement="from collections import Counter",
        sample_code='''def compute_diversity(text: str) -> dict:
    """
    Compute n-gram diversity metrics.
    
    Distinct-N: Ratio of unique n-grams to total n-grams.
    Higher values indicate more diverse, less repetitive text.
    """
    from collections import Counter
    
    def get_ngrams(text: str, n: int) -> list:
        words = text.lower().split()
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    def distinct_n(text: str, n: int) -> float:
        ngrams = get_ngrams(text, n)
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)
    
    words = text.lower().split()
    
    return {
        "distinct_1": distinct_n(text, 1),  # unique unigrams / total unigrams
        "distinct_2": distinct_n(text, 2),  # unique bigrams / total bigrams
        "distinct_3": distinct_n(text, 3),  # unique trigrams / total trigrams
        "vocabulary_size": len(set(words)),
        "total_words": len(words),
        "type_token_ratio": len(set(words)) / len(words) if words else 0,
    }

# Example usage:
text = "The cat sat on the mat. The cat was happy. The cat purred."
scores = compute_diversity(text)
print(f"Distinct-1: {scores['distinct_1']:.3f}")
print(f"Distinct-2: {scores['distinct_2']:.3f}")
print(f"Type-Token Ratio: {scores['type_token_ratio']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["generation", "auto_reply", "summarization"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # Semantic Textual Similarity (STS)
    # ─────────────────────────────────────────────────────────────────
    "sentence_similarity": CodeMetricDefinition(
        name="Sentence Similarity",
        description="Semantic similarity using sentence transformers embeddings.",
        package="sentence-transformers",
        import_statement="from sentence_transformers import SentenceTransformer, util",
        sample_code='''def compute_sentence_similarity(prediction: str, reference: str, model_name: str = "all-MiniLM-L6-v2") -> dict:
    """
    Compute semantic similarity using sentence transformers.
    
    Uses cosine similarity between sentence embeddings.
    Faster than BERTScore for batch processing.
    """
    from sentence_transformers import SentenceTransformer, util
    
    model = SentenceTransformer(model_name)
    
    pred_embedding = model.encode(prediction, convert_to_tensor=True)
    ref_embedding = model.encode(reference, convert_to_tensor=True)
    
    cosine_sim = util.cos_sim(pred_embedding, ref_embedding).item()
    
    return {
        "cosine_similarity": cosine_sim,  # -1 to 1, higher = more similar
        "normalized_similarity": (cosine_sim + 1) / 2,  # 0 to 1
    }

# Example usage:
prediction = "The weather is nice today."
reference = "It's a beautiful sunny day."
scores = compute_sentence_similarity(prediction, reference)
print(f"Cosine Similarity: {scores['cosine_similarity']:.3f}")''',
        output_type="dict",
        score_range=(-1.0, 1.0),
        higher_is_better=True,
        applicable_categories=["summarization", "translation", "auto_reply", "generation"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # JSON Schema Validation
    # ─────────────────────────────────────────────────────────────────
    "json_validation": CodeMetricDefinition(
        name="JSON Schema Validation",
        description="Validate JSON output against a schema for structured output tasks.",
        package="jsonschema",
        import_statement="import json\\nimport jsonschema",
        sample_code='''def compute_json_validation(output: str, schema: dict = None) -> dict:
    """
    Validate JSON output structure and optionally against a schema.
    
    Useful for tasks that require structured JSON output.
    """
    import json
    import jsonschema
    
    result = {
        "is_valid_json": False,
        "schema_valid": None,
        "parse_error": None,
        "schema_errors": [],
    }
    
    # Try to parse JSON
    try:
        parsed = json.loads(output)
        result["is_valid_json"] = True
        result["parsed_output"] = parsed
    except json.JSONDecodeError as e:
        result["parse_error"] = str(e)
        return result
    
    # Validate against schema if provided
    if schema:
        try:
            jsonschema.validate(parsed, schema)
            result["schema_valid"] = True
        except jsonschema.ValidationError as e:
            result["schema_valid"] = False
            result["schema_errors"].append(str(e.message))
    
    return result

# Example usage:
output = '{"sentiment": "positive", "confidence": 0.95}'
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["sentiment", "confidence"]
}
scores = compute_json_validation(output, schema)
print(f"Valid JSON: {scores['is_valid_json']}")
print(f"Schema Valid: {scores['schema_valid']}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["classification", "extraction"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # Toxicity Detection
    # ─────────────────────────────────────────────────────────────────
    "toxicity": CodeMetricDefinition(
        name="Toxicity Detection",
        description="Detect toxic, offensive, or harmful content using the Detoxify model.",
        package="detoxify",
        import_statement="from detoxify import Detoxify",
        sample_code='''def compute_toxicity(text: str, model_type: str = "original") -> dict:
    """
    Detect toxicity and harmful content.
    
    Model types:
    - "original": Standard toxicity detection
    - "unbiased": Reduced unintended bias
    - "multilingual": Support for multiple languages
    
    Scores are probabilities (0-1), lower is better.
    """
    from detoxify import Detoxify
    
    model = Detoxify(model_type)
    results = model.predict(text)
    
    # Add a composite "is_safe" score
    max_toxicity = max(results.values())
    results["max_toxicity"] = max_toxicity
    results["is_safe"] = max_toxicity < 0.5
    results["safety_score"] = 1 - max_toxicity  # 0-1, higher = safer
    
    return results

# Example usage:
text = "Have a wonderful day!"
scores = compute_toxicity(text)
print(f"Safety Score: {scores['safety_score']:.3f}")
print(f"Is Safe: {scores['is_safe']}")
print(f"Toxicity: {scores['toxicity']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=False,  # lower toxicity is better
        applicable_categories=["auto_reply", "generation", "summarization"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # PII Detection
    # ─────────────────────────────────────────────────────────────────
    "pii_detection": CodeMetricDefinition(
        name="PII Detection",
        description="Detect personally identifiable information using Microsoft Presidio.",
        package="presidio-analyzer",
        import_statement="from presidio_analyzer import AnalyzerEngine",
        sample_code='''def compute_pii_detection(text: str, language: str = "en") -> dict:
    """
    Detect Personally Identifiable Information (PII).
    
    Uses Microsoft Presidio to detect:
    - Email addresses, phone numbers
    - Credit card numbers, SSN
    - Names, locations, dates
    - IP addresses, URLs
    """
    from presidio_analyzer import AnalyzerEngine
    
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=text, language=language)
    
    pii_found = []
    pii_types = set()
    
    for result in results:
        pii_found.append({
            "type": result.entity_type,
            "score": result.score,
            "start": result.start,
            "end": result.end,
        })
        pii_types.add(result.entity_type)
    
    return {
        "pii_detected": len(pii_found) > 0,
        "pii_count": len(pii_found),
        "pii_types": list(pii_types),
        "pii_details": pii_found,
        "privacy_score": 1.0 if len(pii_found) == 0 else 0.0,  # 1 = no PII, 0 = PII found
    }

# Example usage:
text = "Contact John Smith at john.smith@email.com or call 555-123-4567."
scores = compute_pii_detection(text)
print(f"PII Detected: {scores['pii_detected']}")
print(f"PII Types: {scores['pii_types']}")
print(f"Privacy Score: {scores['privacy_score']}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,  # higher privacy score (no PII) is better
        applicable_categories=["auto_reply", "summarization", "generation"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # HuggingFace Evaluate (Unified API)
    # ─────────────────────────────────────────────────────────────────
    "huggingface_evaluate": CodeMetricDefinition(
        name="HuggingFace Evaluate",
        description="Unified API for loading and computing many standard metrics (BLEU, ROUGE, METEOR, etc.)",
        package="evaluate",
        import_statement="import evaluate",
        sample_code='''def compute_with_hf_evaluate(predictions: list[str], references: list[str], metric_name: str = "rouge") -> dict:
    """
    Use HuggingFace's evaluate library for standard metrics.
    
    Available metrics include:
    - rouge, bleu, meteor, bertscore
    - accuracy, f1, precision, recall
    - sacrebleu, chrf, ter
    - perplexity, word_error_rate
    
    See: https://huggingface.co/evaluate-metric
    """
    import evaluate
    
    metric = evaluate.load(metric_name)
    
    # Different metrics have different input formats
    if metric_name in ["rouge", "bertscore"]:
        results = metric.compute(predictions=predictions, references=references)
    elif metric_name in ["bleu", "sacrebleu"]:
        # BLEU expects references as list of lists
        refs = [[r] for r in references]
        results = metric.compute(predictions=predictions, references=refs)
    else:
        results = metric.compute(predictions=predictions, references=references)
    
    return results

# Example usage:
predictions = ["The cat sat on the mat."]
references = ["A cat was sitting on a mat."]

# ROUGE
rouge_scores = compute_with_hf_evaluate(predictions, references, "rouge")
print(f"ROUGE-1: {rouge_scores['rouge1']:.3f}")

# BLEU
bleu_scores = compute_with_hf_evaluate(predictions, references, "bleu")
print(f"BLEU: {bleu_scores['bleu']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["summarization", "translation", "generation"],
    ),
    
    # ─────────────────────────────────────────────────────────────────
    # IMAGE METRICS - Similarity & Quality
    # ─────────────────────────────────────────────────────────────────
    "clip_score": CodeMetricDefinition(
        name="CLIP Score",
        description="Measures alignment between generated images and text descriptions using OpenAI's CLIP model.",
        package="transformers",
        import_statement="from transformers import CLIPProcessor, CLIPModel\nimport torch\nfrom PIL import Image",
        sample_code='''def compute_clip_score(image_path: str, text_description: str, model_name: str = "openai/clip-vit-base-patch32") -> dict:
    """
    Compute CLIP score to measure image-text alignment.
    
    Higher scores indicate better alignment between the image and text description.
    Useful for evaluating text-to-image generation quality.
    """
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image
    
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    image = Image.open(image_path)
    inputs = processor(text=[text_description], images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        clip_score = logits_per_image.item()
    
    # Normalize to 0-1 range (approximate)
    normalized_score = torch.sigmoid(torch.tensor(clip_score / 100)).item()
    
    return {
        "clip_score_raw": clip_score,
        "clip_score_normalized": normalized_score,
    }

# Example usage:
image_path = "generated_image.png"
text = "A cat astronaut floating in space"
scores = compute_clip_score(image_path, text)
print(f"CLIP Score: {scores['clip_score_normalized']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["image_generation", "generation"],
    ),
    
    "image_similarity": CodeMetricDefinition(
        name="Image Similarity (SSIM & LPIPS)",
        description="Structural similarity and perceptual similarity metrics for comparing generated vs reference images.",
        package="lpips scikit-image",
        import_statement="from skimage.metrics import structural_similarity as ssim\nimport lpips\nimport torch\nfrom PIL import Image\nimport numpy as np",
        sample_code='''def compute_image_similarity(generated_path: str, reference_path: str) -> dict:
    """
    Compute image similarity metrics.
    
    SSIM (Structural Similarity): Measures structural similarity, 1.0 = identical.
    LPIPS (Learned Perceptual Image Patch Similarity): Lower = more similar.
    """
    from skimage.metrics import structural_similarity as ssim
    from skimage import io, color
    import lpips
    import torch
    import numpy as np
    
    # Load images
    gen_img = io.imread(generated_path)
    ref_img = io.imread(reference_path)
    
    # Convert to grayscale for SSIM
    if len(gen_img.shape) == 3:
        gen_gray = color.rgb2gray(gen_img)
        ref_gray = color.rgb2gray(ref_img)
    else:
        gen_gray, ref_gray = gen_img, ref_img
    
    # Resize if needed
    min_dim = min(gen_gray.shape[0], gen_gray.shape[1], ref_gray.shape[0], ref_gray.shape[1])
    
    # SSIM
    ssim_score = ssim(gen_gray, ref_gray, data_range=1.0)
    
    # LPIPS (requires torch tensors)
    lpips_model = lpips.LPIPS(net='alex')
    
    def img_to_tensor(img):
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return (img / 255.0) * 2 - 1  # Normalize to [-1, 1]
    
    gen_tensor = img_to_tensor(gen_img)
    ref_tensor = img_to_tensor(ref_img)
    
    with torch.no_grad():
        lpips_score = lpips_model(gen_tensor, ref_tensor).item()
    
    return {
        "ssim": ssim_score,  # 0-1, higher = more similar
        "lpips": lpips_score,  # 0-1, lower = more similar
        "perceptual_similarity": 1 - lpips_score,  # Inverted for consistency
    }

# Example usage:
scores = compute_image_similarity("generated.png", "reference.png")
print(f"SSIM: {scores['ssim']:.3f}")
print(f"LPIPS: {scores['lpips']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["image_generation", "image_editing"],
    ),
    
    "fid_score": CodeMetricDefinition(
        name="FID Score (Fréchet Inception Distance)",
        description="Measures quality and diversity of generated images by comparing feature distributions. Lower is better.",
        package="pytorch-fid",
        import_statement="from pytorch_fid import fid_score",
        sample_code='''def compute_fid(generated_folder: str, reference_folder: str, device: str = "cuda") -> dict:
    """
    Compute Fréchet Inception Distance (FID) between two image sets.
    
    FID measures:
    - Quality: How realistic are the generated images?
    - Diversity: How varied are the generated images?
    
    Lower FID = better quality and diversity.
    Typical ranges: <10 excellent, 10-50 good, >50 poor
    """
    from pytorch_fid import fid_score
    
    fid_value = fid_score.calculate_fid_given_paths(
        [generated_folder, reference_folder],
        batch_size=50,
        device=device,
        dims=2048  # Inception-v3 feature dimension
    )
    
    return {
        "fid_score": fid_value,
        "quality_rating": "excellent" if fid_value < 10 else "good" if fid_value < 50 else "poor",
    }

# Example usage:
scores = compute_fid("./generated_images/", "./reference_images/")
print(f"FID Score: {scores['fid_score']:.2f}")
print(f"Quality: {scores['quality_rating']}")''',
        output_type="dict",
        score_range=(0.0, float('inf')),
        higher_is_better=False,
        applicable_categories=["image_generation"],
    ),
    
    "ocr_accuracy": CodeMetricDefinition(
        name="OCR Accuracy",
        description="Evaluate text extraction accuracy from images using character/word error rates.",
        package="jiwer easyocr",
        import_statement="import easyocr\nfrom jiwer import wer, cer",
        sample_code='''def compute_ocr_accuracy(image_path: str, ground_truth: str, languages: list = ["en"]) -> dict:
    """
    Compute OCR accuracy metrics.
    
    WER (Word Error Rate): Percentage of word-level errors.
    CER (Character Error Rate): Percentage of character-level errors.
    Lower is better for both.
    """
    import easyocr
    from jiwer import wer, cer
    
    # Perform OCR
    reader = easyocr.Reader(languages)
    results = reader.readtext(image_path)
    
    # Combine detected text
    extracted_text = " ".join([r[1] for r in results])
    
    # Compute error rates
    word_error_rate = wer(ground_truth, extracted_text)
    char_error_rate = cer(ground_truth, extracted_text)
    
    return {
        "extracted_text": extracted_text,
        "word_error_rate": word_error_rate,  # 0-1+, lower = better
        "char_error_rate": char_error_rate,  # 0-1+, lower = better
        "word_accuracy": max(0, 1 - word_error_rate),
        "char_accuracy": max(0, 1 - char_error_rate),
        "detection_count": len(results),
    }

# Example usage:
ground_truth = "Hello World 123"
scores = compute_ocr_accuracy("text_image.png", ground_truth)
print(f"Word Accuracy: {scores['word_accuracy']:.2%}")
print(f"Extracted: {scores['extracted_text']}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["extraction", "image_understanding"],
    ),
    
    "object_detection_map": CodeMetricDefinition(
        name="Object Detection mAP",
        description="Mean Average Precision for evaluating object detection and visual recognition accuracy.",
        package="torchmetrics",
        import_statement="from torchmetrics.detection import MeanAveragePrecision\nimport torch",
        sample_code='''def compute_object_detection_map(predictions: list, ground_truth: list) -> dict:
    """
    Compute Mean Average Precision (mAP) for object detection.
    
    Args:
        predictions: List of dicts with 'boxes', 'scores', 'labels'
        ground_truth: List of dicts with 'boxes', 'labels'
    
    mAP@0.5: IoU threshold of 0.5
    mAP@0.5:0.95: Average over IoU thresholds from 0.5 to 0.95
    """
    from torchmetrics.detection import MeanAveragePrecision
    import torch
    
    metric = MeanAveragePrecision(iou_type="bbox")
    
    # Convert to torch format
    preds = [{
        "boxes": torch.tensor(p["boxes"]),
        "scores": torch.tensor(p["scores"]),
        "labels": torch.tensor(p["labels"])
    } for p in predictions]
    
    targets = [{
        "boxes": torch.tensor(t["boxes"]),
        "labels": torch.tensor(t["labels"])
    } for t in ground_truth]
    
    metric.update(preds, targets)
    results = metric.compute()
    
    return {
        "map": results["map"].item(),  # mAP@0.5:0.95
        "map_50": results["map_50"].item(),  # mAP@0.5
        "map_75": results["map_75"].item(),  # mAP@0.75
        "map_small": results["map_small"].item(),
        "map_medium": results["map_medium"].item(),
        "map_large": results["map_large"].item(),
    }

# Example usage:
predictions = [{
    "boxes": [[100, 100, 200, 200], [300, 300, 400, 400]],
    "scores": [0.95, 0.87],
    "labels": [1, 2]
}]
ground_truth = [{
    "boxes": [[105, 105, 195, 195], [305, 305, 395, 395]],
    "labels": [1, 2]
}]
scores = compute_object_detection_map(predictions, ground_truth)
print(f"mAP@0.5: {scores['map_50']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["classification", "image_understanding"],
    ),
    
    "image_safety": CodeMetricDefinition(
        name="Image Safety (NSFW Detection)",
        description="Detect inappropriate, explicit, or sensitive content in images.",
        package="transformers",
        import_statement="from transformers import pipeline\nfrom PIL import Image",
        sample_code='''def compute_image_safety(image_path: str) -> dict:
    """
    Detect NSFW/sensitive content in images.
    
    Uses a vision classifier to detect:
    - Safe content
    - Suggestive content
    - Explicit content
    
    Returns safety scores and classification.
    """
    from transformers import pipeline
    from PIL import Image
    
    # Use NSFW detection model
    classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
    
    image = Image.open(image_path)
    results = classifier(image)
    
    # Parse results
    scores = {r["label"].lower(): r["score"] for r in results}
    
    is_safe = scores.get("normal", 0) > 0.5 or scores.get("safe", 0) > 0.5
    nsfw_score = scores.get("nsfw", 0) + scores.get("explicit", 0)
    
    return {
        "is_safe": is_safe,
        "safety_score": 1 - nsfw_score,
        "nsfw_probability": nsfw_score,
        "detailed_scores": scores,
        "classification": "safe" if is_safe else "potentially_unsafe",
    }

# Example usage:
scores = compute_image_safety("uploaded_image.png")
print(f"Safe: {scores['is_safe']}")
print(f"Safety Score: {scores['safety_score']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["image_safety", "classification"],
    ),
    
    "face_detection": CodeMetricDefinition(
        name="Face Detection & Recognition",
        description="Detect and analyze faces in images for privacy and identification features.",
        package="deepface",
        import_statement="from deepface import DeepFace",
        sample_code='''def compute_face_detection(image_path: str, reference_face_path: str = None) -> dict:
    """
    Detect faces and optionally verify against a reference.
    
    Useful for:
    - People album organization
    - Privacy detection (face blur)
    - Identity verification
    """
    from deepface import DeepFace
    
    # Detect faces
    faces = DeepFace.extract_faces(image_path, enforce_detection=False)
    
    result = {
        "face_count": len(faces),
        "faces": [],
        "has_faces": len(faces) > 0,
    }
    
    for i, face in enumerate(faces):
        face_info = {
            "face_id": i,
            "confidence": face.get("confidence", 0),
            "facial_area": face.get("facial_area", {}),
        }
        result["faces"].append(face_info)
    
    # Face verification if reference provided
    if reference_face_path and len(faces) > 0:
        try:
            verification = DeepFace.verify(image_path, reference_face_path)
            result["verification"] = {
                "verified": verification["verified"],
                "distance": verification["distance"],
                "similarity": 1 - verification["distance"],
            }
        except Exception as e:
            result["verification"] = {"error": str(e)}
    
    return result

# Example usage:
scores = compute_face_detection("group_photo.jpg")
print(f"Faces detected: {scores['face_count']}")
for face in scores['faces']:
    print(f"  Face {face['face_id']}: confidence {face['confidence']:.2f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["extraction", "image_safety", "image_understanding"],
    ),
    
    "image_caption_score": CodeMetricDefinition(
        name="Image Caption Evaluation",
        description="Evaluate generated image captions using CIDEr, SPICE, and METEOR metrics.",
        package="pycocoevalcap nltk",
        import_statement="from pycocoevalcap.cider.cider import Cider\nfrom pycocoevalcap.meteor.meteor import Meteor\nimport nltk",
        sample_code='''def compute_caption_metrics(generated_caption: str, reference_captions: list) -> dict:
    """
    Evaluate image caption quality.
    
    CIDEr: Consensus-based Image Description Evaluation (preferred for captioning)
    METEOR: Considers synonyms and paraphrases
    BLEU-4: N-gram overlap
    """
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    
    # Format for pycocoevalcap
    gts = {0: reference_captions}  # Ground truth
    res = {0: [generated_caption]}  # Result
    
    # CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts, res)
    
    # METEOR
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(gts, res)
    
    # BLEU-4
    ref_tokens = [ref.split() for ref in reference_captions]
    gen_tokens = generated_caption.split()
    smoothie = SmoothingFunction().method4
    bleu_score = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothie)
    
    return {
        "cider": cider_score,  # 0-10, higher = better
        "meteor": meteor_score,  # 0-1, higher = better
        "bleu_4": bleu_score,  # 0-1, higher = better
        "combined_score": (cider_score / 10 + meteor_score + bleu_score) / 3,
    }

# Example usage:
generated = "A dog playing fetch in the park"
references = [
    "A golden retriever catching a ball in a green park",
    "A dog plays with a ball outdoors",
    "A playful dog in a park setting"
]
scores = compute_caption_metrics(generated, references)
print(f"CIDEr: {scores['cider']:.3f}")
print(f"METEOR: {scores['meteor']:.3f}")''',
        output_type="dict",
        score_range=(0.0, 1.0),
        higher_is_better=True,
        applicable_categories=["generation", "image_understanding"],
    ),
}


# ═══════════════════════════════════════════════════════════════════
# CATEGORY TO CODE METRICS MAPPING
# ═══════════════════════════════════════════════════════════════════

CATEGORY_CODE_METRICS: Dict[str, List[str]] = {
    "summarization": ["rouge", "bertscore", "length_metrics", "readability", "diversity", "sentence_similarity"],
    "translation": ["bleu", "bertscore", "fuzzy_match", "sentence_similarity"],
    "auto_reply": ["bertscore", "readability", "toxicity", "pii_detection", "fuzzy_match", "diversity"],
    "classification": ["exact_match_f1", "json_validation", "object_detection_map"],
    "extraction": ["exact_match_f1", "fuzzy_match", "json_validation", "ocr_accuracy", "face_detection"],
    "generation": ["bertscore", "diversity", "readability", "toxicity", "sentence_similarity", "clip_score", "image_caption_score"],
    "generic": ["bertscore", "readability", "length_metrics", "diversity"],
    # Image-specific categories
    "image_generation": ["clip_score", "image_similarity", "fid_score", "image_safety"],
    "image_editing": ["image_similarity", "clip_score", "image_safety"],
    "image_understanding": ["ocr_accuracy", "object_detection_map", "image_caption_score", "face_detection"],
    "image_safety": ["image_safety", "face_detection", "toxicity"],
}


def get_code_metrics_for_category(category: str) -> List[str]:
    """Get recommended code-based metrics for a category."""
    cat = category.lower().replace(" ", "_") if category else "generic"
    return CATEGORY_CODE_METRICS.get(cat, CATEGORY_CODE_METRICS["generic"])


def get_code_metric(metric_id: str) -> Optional[CodeMetricDefinition]:
    """Get a code metric definition by ID."""
    return CODE_METRICS_REGISTRY.get(metric_id)


def generate_code_metrics_sample(category: str, metrics: List[str] = None) -> str:
    """
    Generate sample code for computing code-based metrics.
    
    Args:
        category: Feature category (e.g., "summarization")
        metrics: Specific metrics to include, or None for category defaults
    
    Returns:
        Python code string with metric implementations
    """
    if metrics is None:
        metrics = get_code_metrics_for_category(category)
    
    # Collect unique packages
    packages = set()
    for m_id in metrics:
        metric = get_code_metric(m_id)
        if metric and metric.package != "(built-in)":
            packages.add(metric.package)
    
    # Build code
    lines = [
        '"""',
        f'Code-Based Metrics for {category.title()} Evaluation',
        '',
        'Auto-generated metric implementations using well-known open source packages.',
        '"""',
        '',
        '# ═══════════════════════════════════════════════════════════════════',
        '# INSTALLATION',
        '# ═══════════════════════════════════════════════════════════════════',
        '# Run the following command to install required packages:',
        f'# pip install {" ".join(sorted(packages))}',
        '',
    ]
    
    for m_id in metrics:
        metric = get_code_metric(m_id)
        if metric:
            lines.extend([
                '',
                '# ─────────────────────────────────────────────────────────────────',
                f'# {metric.name}',
                '# ─────────────────────────────────────────────────────────────────',
                f'# {metric.description}',
                f'# Package: {metric.package}',
                '',
                metric.sample_code,
                '',
            ])
    
    # Add a combined evaluation function
    lines.extend([
        '',
        '# ═══════════════════════════════════════════════════════════════════',
        '# COMBINED EVALUATION',
        '# ═══════════════════════════════════════════════════════════════════',
        '',
        'def evaluate_all(prediction: str, reference: str) -> dict:',
        '    """Run all applicable metrics and return combined results."""',
        '    results = {}',
        '',
    ])
    
    for m_id in metrics:
        metric = get_code_metric(m_id)
        if metric:
            func_name = f"compute_{m_id}" if m_id != "exact_match_f1" else "compute_exact_match_f1"
            lines.append(f'    try:')
            lines.append(f'        results["{m_id}"] = {func_name}(prediction, reference)')
            lines.append(f'    except Exception as e:')
            lines.append(f'        results["{m_id}"] = {{"error": str(e)}}')
            lines.append('')
    
    lines.extend([
        '    return results',
        '',
        '',
        '# Example usage:',
        'if __name__ == "__main__":',
        '    prediction = "Your model output here"',
        '    reference = "Expected reference output"',
        '    ',
        '    results = evaluate_all(prediction, reference)',
        '    ',
        '    for metric_name, scores in results.items():',
        '        print(f"\\n{metric_name}:")',
        '        if isinstance(scores, dict):',
        '            for k, v in scores.items():',
        '                if isinstance(v, float):',
        '                    print(f"  {k}: {v:.3f}")',
        '                else:',
        '                    print(f"  {k}: {v}")',
        '',
    ])
    
    return '\n'.join(lines)


def list_all_code_metrics() -> List[Dict[str, Any]]:
    """List all available code-based metrics with their metadata."""
    return [
        {
            "id": m_id,
            "name": metric.name,
            "description": metric.description,
            "package": metric.package,
            "categories": metric.applicable_categories,
        }
        for m_id, metric in CODE_METRICS_REGISTRY.items()
    ]
