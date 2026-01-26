# Documentation Index

> **Version**: 2.0  
> **Last Updated**: January 25, 2026

This document provides a reference to all technical documentation in this codebase. Use this index to find relevant documentation without loading unnecessary context.

## Quick Reference

| Topic | Document | Key Sections |
|-------|----------|--------------|
| Getting Started | [README.md](../README.md) | Installation, Usage, Examples |
| System Architecture | [ArchitectureOverview.md](ArchitectureOverview.md) | Components, Data Flow, Integration |
| Agent System | [AgentDesign.md](AgentDesign.md) | Template Mode, AI Agent Mode, Tools |

## Documents

| Document | Description | Project/Scope |
|----------|-------------|---------------|
| [Architecture Overview](ArchitectureOverview.md) | **Comprehensive** architecture documentation covering: (1) Entry points & bootstrapping, (2) Dual-mode agent system (Template + AI Agent), (3) Schema layer with dual Pydantic/dataclass pattern, (4) Metrics registry with 20+ metrics & i18n, (4.1) Code-based metrics (ROUGE, BLEU, BERTScore, etc.), (5) LLM client with singleton pattern, (5.1) Image generator with DALL-E 3, (6) Database layer with ER diagram, (7) Web UI with AI Agent toggle, (8) v2.0 prompt templates with hard FAIL gates. Includes mermaid diagrams and code references. | Global Concept |
| [Agent Design](AgentDesign.md) | **Deep dive** into the agent system: (1) Template Mode (`FeaturePromptWriterAgent`) - deterministic pipeline, (2) AI Agent Mode (`MetaFeatureAgent` v2.0) - Microsoft Agent Framework, (3) 7 available tools, (4) RAI constraint injection rules, (5) v2.0 prompt templates with hard gates and second-order signals, (6) Usage examples and comparison, (7) When to use which mode. | Agent Component |

## What's New in v2.0

### Evaluation Prompts v2.0
- **Hard FAIL Gates**: Safety, Privacy, Toxicity, Legal gates that override all scores
- **Second-Order Quality Signals**: Fluency, linguistic naturalness, localization quality, regional compliance
- **Canonical Contract Format**: Versioned, auditable prompt structure
- **Structured JSON Output**: Consistent output schema across all templates

### AI Agent v2.0
- **Microsoft Agent Framework**: Built on official framework with `ChatAgent` and `@ai_function`
- **7 Intelligent Tools**: `lookup_metrics`, `suggest_metrics`, `get_locale_info`, `validate_rai_compliance`, `build_prompt`, `get_code_metrics`, `analyze_feature_description`
- **Natural Language Understanding**: Analyze feature descriptions to detect category, sensitivity levels
- **Reproducibility Contract**: Explicit instructions for consistent evaluations

### Template Improvements
- **Category-Specific Gates**: Each template has domain-appropriate hard gates
- **Locale-Aware**: Privacy frameworks (GDPR, CCPA, PIPL) and cultural context
- **Evaluation Protocol**: Step-by-step process for evaluators

## Architecture Diagrams

See [ArchitectureOverview.md](ArchitectureOverview.md) for:
- High-level component diagram
- Agent flow diagram
- Schema class diagram
- Database ER diagram

## Code References

| Component | File | Key Functions/Classes |
|-----------|------|----------------------|
| Template Mode | [agent.py](../src/core/agent.py) | `FeaturePromptWriterAgent`, `generate()` |
| AI Agent Mode | [ai_agent.py](../src/core/ai_agent.py) | `MetaFeatureAgent`, `METAFEATURE_AGENT_SYSTEM_PROMPT` |
| Agent Tools | [ai_agent.py](../src/core/ai_agent.py) | `@ai_function` decorated tools |
| Prompt Templates | [prompt_templates.py](../src/core/prompt_templates.py) | `template_*`, `build_evaluation_prompt()` |
| Metrics | [metrics_registry.py](../src/core/metrics_registry.py) | `METRICS_REGISTRY`, `MetricDefinition` |
| Web UI | [app.py](../src/core/app.py) | `create_app()`, `generate_prompt_handler()` |

---

## Metric Recommendation Systems

The Quality Metrics tab provides two different ways to select evaluation metrics:

### ⭐ Select Recommended (Rule-Based)

A **static, rule-based** recommendation system using hardcoded category mappings.

| Aspect | Description |
|--------|-------------|
| **Method** | Simple lookup from `DEFAULT_METRICS_BY_CATEGORY` dictionary |
| **Location** | [metrics_registry.py](../src/core/metrics_registry.py) lines 612-633 |
| **Analysis** | None - purely category-based |
| **Explanations** | None provided |
| **Speed** | Instant (no computation) |

**How it works:**
```
Category → Fixed Metric List
"summarization" → ["faithfulness", "coverage", "groundedness", "fluency", "brevity", "safety", "privacy", "format_compliance"]
"auto_reply" → ["relevance", "tone", "fluency", "brevity", "safety", "privacy", ...]
```

### 🤖 AI Recommend (Intelligent Analysis)

A **dynamic, intelligent** recommendation system that analyzes your feature.

| Aspect | Description |
|--------|-------------|
| **Method** | Semantic analysis of feature name, description, and context |
| **Location** | [ai_agent.py](../src/core/ai_agent.py) `recommend_metrics()` function |
| **Analysis** | Detects privacy sensitivity, safety criticality, locale requirements |
| **Explanations** | Detailed rationale for each recommended metric |
| **Speed** | ~100ms (local computation, no LLM call) |

**What it analyzes:**
- Feature description keywords → Detects "medical", "financial", "personal" = privacy-sensitive
- Safety indicators → "health", "legal", "decision" = safety-critical features
- Output format → JSON/XML = adds `format_compliance`
- Locale → Non-US locales add `cultural_appropriateness`, `regional_compliance`

### Comparison Example

**Feature:** "Medical Summary Generator that summarizes patient records"

| Aspect | ⭐ Select Recommended | 🤖 AI Recommend |
|--------|----------------------|-----------------|
| **Method** | Lookup "summarization" category | Analyze "medical", "patient records", "summarizes" |
| **Privacy** | Included (standard for category) | **🔴 CRITICAL priority** - detects "patient records" = medical PII |
| **Groundedness** | Included (standard) | **🔴 HIGH priority** - detects "medical" = safety-critical |
| **Explanation** | *(none)* | "🔐 **CRITICAL**: Handles sensitive/personal data. Must not leak PII, confidential information, or user data. Privacy violations can result in legal action and user harm." |
| **Priority Order** | Flat list | Tiered: Critical → Important → Additional |

### When to Use Which

| Use Case | Recommendation |
|----------|----------------|
| Quick prototype | ⭐ Select Recommended |
| Standard feature in known category | ⭐ Select Recommended |
| Complex/novel feature | 🤖 AI Recommend |
| Privacy-sensitive feature | 🤖 AI Recommend |
| Need to understand WHY metrics matter | 🤖 AI Recommend |
| Documentation/audit trail | 🤖 AI Recommend |

---
*Last updated: January 25, 2026*
