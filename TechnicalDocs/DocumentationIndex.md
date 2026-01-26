# Documentation Index

> **Version**: 2.1  
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
| [Agent Design](AgentDesign.md) | **Deep dive** into the agent system: (1) Template Mode (`FeaturePromptWriterAgent`) - deterministic pipeline, (2) AI Agent Mode (`MetaFeatureAgent` v2.1) - Microsoft Agent Framework, (3) 9 available tools, (4) RAI constraint injection rules, (5) v2.0/v2.1 prompt templates with hard gates and second-order signals, (6) Usage examples and comparison, (7) When to use which mode. | Agent Component |

## What's New in v2.1

### Side-by-Side Comparison Mode
- **Progressive Loading**: Template output displays immediately while AI Agent generates
- **4 Generation Modes**: Auto, Always AI, Template only, Both (comparison)
- **Streaming Generator**: `generate_both_prompts_streaming()` yields results progressively

### AI Agent v2.1
- **Microsoft Agent Framework**: Built on official framework with `ChatAgent` and `@ai_function`
- **9 Intelligent Tools**: `lookup_metrics`, `suggest_metrics`, `recommend_metrics`, `get_locale_info`, `validate_rai_compliance`, `build_prompt`, `get_code_metrics`, `analyze_feature_description`
- **Mandatory Tool Usage**: System prompt enforces `build_prompt` tool for consistent timestamps
- **Distinct Output Format**: `# 🤖 AI Agent Evaluation Prompt` header with version 2.1

### Intelligent Metric Recommendations
- **`recommend_metrics()` function**: Analyzes feature descriptions semantically
- **Priority Tiers**: Critical → Important → Additional
- **Detailed Explanations**: Why each metric matters for the specific feature

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
| Agent Tools | [ai_agent.py](../src/core/ai_agent.py) | 9 `@ai_function` decorated tools |
| Prompt Templates | [prompt_templates.py](../src/core/prompt_templates.py) | `template_*`, `build_evaluation_prompt()` |
| Metrics | [metrics_registry.py](../src/core/metrics_registry.py) | `METRICS_REGISTRY`, `MetricDefinition` |
| Web UI | [app.py](../src/core/app.py) | `create_app()`, `generate_and_sync_to_simulation_streaming()` |
| Comparison Mode | [app.py](../src/core/app.py) | `generate_both_prompts_streaming()` |

---

## Generation Modes

The web UI provides 4 generation modes:

| Mode | Code Value | Description |
|------|------------|-------------|
| 🤖 Auto | `"auto"` | Detects feature complexity, chooses mode automatically |
| ⚡ Always AI | `"always"` | Forces AI Agent mode for all features |
| 📋 Template only | `"never"` | Uses deterministic template mode |
| ⚖️ Both | `"both"` | Side-by-side comparison with streaming |

### Progressive Loading (Both Mode)

When using "Both" mode, the UI shows results progressively:

```
1. User clicks Generate
2. Template output appears IMMEDIATELY (< 1 second)
3. AI Agent panel shows "⏳ Generating..."
4. AI Agent output appears when ready (5-10 seconds)
```

**Implementation**: `generate_both_prompts_streaming()` is a Python generator that yields twice.

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

## Prompt Version Differences

### Template Mode (v2.0)
```markdown
# Evaluation Prompt: [Feature Name]
**Version:** 2.0 (Auto-Reply Contract)
**Target Language:** en
**Locale:** English (United States)
**Privacy Framework:** CCPA
**Generated:** 2026-01-25T12:00:00Z

## EVALUATOR ROLE
You are an expert evaluator...
```

### AI Agent Mode (v2.1)
```markdown
# 🤖 AI Agent Evaluation Prompt: [Feature Name]
**Version:** 2.1 (AI Agent Generated)
**Generation Mode:** AI Agent with Intelligent Analysis
**Target Language:** en
**Locale:** English (United States)
**Privacy Framework:** CCPA
**Generated:** 2026-01-25T12:00:00Z

## 1. EVALUATOR ROLE
You are an **AI-powered expert evaluator**...
```

**Key Differences:**
- AI Agent uses `# 🤖 AI Agent Evaluation Prompt` header
- AI Agent has numbered sections (`## 1. EVALUATOR ROLE`)
- AI Agent emphasizes it's "AI-powered"
- Both include actual ISO timestamps (not placeholders)

---

## AI Agent Tools Reference

| Tool | Purpose | Returns |
|------|---------|---------|
| `lookup_metrics` | Find metrics for a category | List of metrics with definitions and weights |
| `suggest_metrics` | Get additional suggestions | Suggested metrics based on context |
| `recommend_metrics` | **Intelligent analysis** | Prioritized metrics with explanations |
| `get_locale_info` | Cultural/regulatory info | Privacy framework, tone guidance, cultural context |

### 📌 `suggest_metrics` vs `recommend_metrics`

| Aspect | `suggest_metrics` | `recommend_metrics` |
|--------|-------------------|---------------------|
| **Analysis** | Rule-based lookup | Semantic feature analysis |
| **Input** | Category + current metrics | Feature name + description + category |
| **Output** | Flat list | Prioritized tiers (Mandatory/Critical/Important) |
| **Explanations** | Generic reason | Detailed per-metric rationale |
| **Use Case** | Quick category-based suggestions | Comprehensive metric planning |
| `validate_rai_compliance` | Check RAI requirements | Compliance status and issues |
| `build_prompt` | Generate evaluation prompt | Complete prompt with timestamp |
| `get_code_metrics` | Code metric samples | Python code for ROUGE/BLEU/etc. |
| `analyze_feature_description` | Extract attributes | Detected category, privacy, safety flags |

---
*Last updated: January 25, 2026*
