# Agent Design Documentation

> **Version**: 2.0  
> **Last Updated**: January 25, 2026

## Overview

MetaFeature-Orchestrator provides two agent modes for generating evaluation prompts:

1. **Template Mode** (`FeaturePromptWriterAgent`): Deterministic, template-based prompt generation
2. **AI Agent Mode** (`MetaFeatureAgent` v2.0): AI-powered agent using Microsoft Agent Framework

Both modes now generate **v2.0 evaluation prompts** with hard FAIL gates, second-order quality signals, and structured JSON output.

---

## Dual-Mode Architecture

| Aspect | Template Mode | AI Agent Mode |
|--------|---------------|---------------|
| **Class** | `FeaturePromptWriterAgent` | `MetaFeatureAgent` |
| **Framework** | Pure Python | Microsoft Agent Framework |
| **Planning** | Hard-coded workflow | Dynamic LLM reasoning |
| **Tool Use** | None | 7 tools via `@ai_function` |
| **Memory** | Stateless | Thread-based conversation |
| **Decision Making** | Deterministic code paths | LLM-driven decisions |
| **Speed** | Instant | Requires API calls |
| **Cost** | Free | API costs |
| **Best For** | Simple, well-defined features | Complex, novel features |

**Recommendation**: Use Template Mode as the **canonical contract**, AI Agent Mode for **adaptive execution**.

---

## Template Mode: `FeaturePromptWriterAgent`

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FeaturePromptWriterAgent.generate()              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: FeatureSpec or FeatureMetadata + Locale (BCP 47)          │
│      │                                                              │
│      ▼                                                              │
│   ┌─────────────────────────────────────────┐                      │
│   │ Step 1: _resolve_metrics()              │  ◄── Code-based      │
│   │   • Lookup metrics from registry        │      (no LLM)        │
│   │   • Build metric definitions dict       │                      │
│   │   • Suggest additional metrics          │                      │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   ┌─────────────────────────────────────────┐                      │
│   │ Step 2: _apply_rai_constraints()        │  ◄── Rule-based      │
│   │   • Always add "safety" metric          │      (if/else logic) │
│   │   • Add "privacy" if privacy_sensitive  │                      │
│   │   • Add "groundedness" if safety_critical│                     │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   ┌─────────────────────────────────────────┐                      │
│   │ Step 3: build_evaluation_prompt()       │  ◄── Template-based  │
│   │   • Select category-specific template   │      (v2.0 format)   │
│   │   • Add hard FAIL gates                 │                      │
│   │   • Add second-order quality signals    │                      │
│   │   • Apply localization (i18n)           │                      │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   Output: PromptOutput (v2.0 Canonical Contract)                    │
│      • evaluation_prompt (string)                                   │
│      • metrics_used (list)                                          │
│      • metric_definitions (dict)                                    │
│      • rai_checks_applied (list)                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Point**: No LLM is called inside the template agent. It's pure Python logic.

---

### Source Code Reference

**File**: [src/core/agent.py](../src/core/agent.py)

```python
class FeaturePromptWriterAgent:
    """
    Core agent for generating metric-driven evaluation prompts.
    
    Design Principles:
    - Metric-first: Evaluation criteria are explicit and measurable
    - Grounded: No hallucinated judgment criteria
    - RAI by design: Safety, bias, privacy checks embedded
    - Extensible: Easy to add new features without re-engineering
    - Stateless: No instance state between calls
    """
```

### Key Methods

| Method | Purpose | Implementation |
|--------|---------|----------------|
| `generate()` | Main entry point | Orchestrates the 3-step pipeline |
| `_resolve_metrics()` | Metric lookup | Dict lookup from `METRICS_REGISTRY` |
| `_apply_rai_constraints()` | RAI injection | If/else rules for safety, privacy, groundedness |
| `_metadata_to_spec()` | Schema conversion | Pydantic → dataclass |
| `get_available_metrics()` | Utility | Returns all registered metrics |

---

## RAI Constraint Rules

The agent automatically injects Responsible AI metrics based on feature configuration:

| Condition | Metric Added | RAI Check Logged |
|-----------|--------------|------------------|
| Always | `safety` | `safety_check_added` |
| `privacy_sensitive=True` | `privacy` | `privacy_check_added` |
| `safety_critical=True` | `groundedness` | `groundedness_check_added` |

**Source**: [agent.py#L150-L190](../src/core/agent.py)

---

## AI Agent Mode: `MetaFeatureAgent` v2.0

### Overview

The `MetaFeatureAgent` is a **true AI agent** built on Microsoft Agent Framework that uses LLMs to make dynamic decisions about metric selection, RAI compliance, and prompt generation.

**File**: [src/core/ai_agent.py](../src/core/ai_agent.py)

### System Prompt (v2.0)

```
You are MetaFeature Agent v2.0, an expert AI evaluation prompt generator.

Your ONLY job is to generate a complete, production-ready evaluation prompt for GenAI features.

## Your Role: Execution-Time Evaluator Prompt Generator

You create evaluation prompts that:
1. Are **immediately usable** - no further editing needed
2. Include **explicit FAIL gates** for safety-critical decisions
3. Have **versioned, auditable** structure for reproducibility
4. Cover **second-order quality signals** (fluency, cultural fit, regional compliance)
```

### Available Tools (7 total)

The AI agent has access to these tools via `@ai_function` decorator:

| Tool | Purpose | Returns |
|------|---------|---------|
| `lookup_metrics` | Find metrics for a category | Metric names, definitions, weights |
| `suggest_metrics` | Get recommendations for additional metrics | Suggested metrics list |
| `get_locale_info` | Get cultural/regulatory info for a locale | Cultural context, tone guidance, privacy framework |
| `validate_rai_compliance` | Check if metrics meet RAI requirements | Compliance status, issues, recommendations |
| `build_prompt` | Generate the v2.0 evaluation prompt | Complete evaluation prompt string |
| `get_code_metrics` | Get programmatic metrics sample | Code sample for ROUGE, BLEU, etc. |
| `analyze_feature_description` | Extract attributes from natural language | Category, sensitivity flags, confidence |

### Agent Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MetaFeatureAgent.chat(message)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   User: "I need an evaluation prompt for a medical summarizer      │
│          that will be used by doctors in Germany"                   │
│      │                                                              │
│      ▼                                                              │
│   ┌─────────────────────────────────────────┐                      │
│   │ Tool: analyze_feature_description       │  ◄── Understand      │
│   │   • Detected: summarization             │      intent          │
│   │   • Privacy: True (medical)             │                      │
│   │   • Safety: True (medical)              │                      │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   ┌─────────────────────────────────────────┐                      │
│   │ Tool: lookup_metrics                    │  ◄── Get metrics     │
│   │   • Category: summarization             │      for category    │
│   │   • Returns: faithfulness, coverage...  │                      │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   ┌─────────────────────────────────────────┐                      │
│   │ Tool: suggest_metrics                   │  ◄── Add safety/     │
│   │   • privacy_sensitive=True              │      privacy metrics │
│   │   • safety_critical=True                │                      │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   ┌─────────────────────────────────────────┐                      │
│   │ Tool: get_locale_info                   │  ◄── Get German      │
│   │   • Locale: de-DE                       │      cultural info   │
│   │   • Privacy: GDPR                       │                      │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   ┌─────────────────────────────────────────┐                      │
│   │ Tool: validate_rai_compliance           │  ◄── Verify RAI      │
│   │   • Check: safety, privacy, groundedness│                      │
│   │   • Result: Compliant                   │                      │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   ┌─────────────────────────────────────────┐                      │
│   │ Tool: build_prompt                      │  ◄── Generate v2.0   │
│   │   • All parameters from above           │      prompt          │
│   │   • Returns: Complete evaluation prompt │                      │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   Output: Complete v2.0 Evaluation Prompt                          │
│      • Hard FAIL gates                                              │
│      • Primary metrics with weights                                 │
│      • Second-order quality signals                                 │
│      • Structured JSON output format                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## v2.0 Prompt Templates

Both modes generate prompts using the v2.0 template format.

### Template Structure

Each generated prompt includes:

| Section | Purpose |
|---------|---------|
| **1. Evaluator Role & Task** | Clear role definition and task description |
| **2. Hard FAIL Gates** | Safety, Privacy, Toxicity, Legal gates with automatic override |
| **3. Primary Metrics** | Metric definitions with weights and 1-5 scoring scale |
| **4. Second-Order Signals** | Fluency, linguistic naturalness, localization quality, regional compliance |
| **5. Evaluation Protocol** | Step-by-step evaluation process |
| **6. Responsible AI Checklist** | Locale-specific RAI checks |
| **7. Output Format** | Structured JSON schema |

### Category-Specific Hard Gates

| Template | Hard Gates |
|----------|------------|
| **Generic** | Safety, Privacy, Toxicity, Legal |
| **Summarization** | Hallucination, Safety, Privacy, Factual |
| **Translation** | Meaning, Offensive, Safety, Legal |
| **Auto-Reply** | Relevance, Safety, Privacy, Tone |
| **Personal Assistant** | Privacy, Consent, Safety, Medical, Financial |

### Second-Order Quality Signals

| Signal | Weight | Description |
|--------|--------|-------------|
| Fluency | 0.7-0.9 | Natural, grammatically correct |
| Linguistic Naturalness | 0.8-0.9 | Reads as native speaker would write |
| Localization Quality | 0.9-1.0 | Proper locale conventions (dates, numbers) |
| Regional Compliance | 0.9-1.0 | Meets local regulatory requirements |
| Cultural Appropriateness | 0.9-1.0 | Respects cultural norms, avoids taboos |

---

## Where LLM IS Called

| Component | Function | LLM Used |
|-----------|----------|----------|
| Template Mode | `generate()` | **None** - pure Python |
| AI Agent Mode | `chat()` | Azure OpenAI via Agent Framework |
| Simulation | `generate_ai_outputs()` | Azure OpenAI (temperature 0.0) |
| Evaluation | `run_simulation_evaluation()` | Azure OpenAI (temperature 0.0) |

---

## Localization (i18n)

Both modes support **20+ BCP 47 locales**:

| Region | Locales |
|--------|---------|
| **English** | en-US, en-GB, en-AU, en-CA, en-IN |
| **Chinese** | zh-CN, zh-TW, zh-HK |
| **European** | de-DE, fr-FR, fr-CA, es-ES, es-MX, pt-BR, pt-PT, it-IT, nl-NL |
| **Asian** | ja-JP, ko-KR |
| **Other** | ar-SA, he-IL, ru-RU, tr-TR |

All labels, instructions, and rubrics are localized in `LOCALIZED_LABELS` dict.

---

## Usage Examples

### Template Mode (Deterministic)

```python
from src.core.agent import FeaturePromptWriterAgent
from src.core.schemas import FeatureSpec

# Create agent (stateless)
agent = FeaturePromptWriterAgent()

# Define feature
feature = FeatureSpec(
    group="Email",
    name="Auto-Reply Email",
    description="Generate professional email replies",
    category="auto_reply",
    input_format="email",
    output_format="text",
    languages_supported=["en"],
    success_metrics=["relevance", "tone", "fluency"],
    privacy_sensitive=True,
    safety_critical=False,
)

# Generate v2.0 evaluation prompt
result = agent.generate(feature, locale="en-US")

print(result.evaluation_prompt)
print(f"Metrics: {result.metrics_used}")
print(f"RAI checks: {result.rai_checks_applied}")
```

### AI Agent Mode (Adaptive)

```python
from src.core.ai_agent import MetaFeatureAgent

# Create AI agent (requires Azure OpenAI)
agent = MetaFeatureAgent()

# Natural language request - the agent figures out the rest
response = agent.chat(
    "I need an evaluation prompt for a medical document summarizer "
    "that will be used by doctors in Germany. It needs to be very "
    "careful about accuracy and patient privacy."
)

# Returns complete v2.0 evaluation prompt with:
# - HALLUCINATION, SAFETY, PRIVACY, FACTUAL gates
# - Summarization metrics (faithfulness, coverage, etc.)
# - German locale with GDPR framework
# - Second-order quality signals
print(response.evaluation_prompt)
```

---

## Design Decisions

### Why Two Modes?

| Consideration | Template Mode | AI Agent Mode |
|---------------|---------------|---------------|
| **Predictability** | 100% deterministic | High (structured output) |
| **Debuggability** | Each step can be traced | LLM reasoning visible |
| **Performance** | Instant (no API calls) | Requires LLM roundtrips |
| **Cost** | Free | API costs |
| **Adaptability** | Fixed templates | Dynamic reasoning |
| **Natural Language** | Structured input only | Free-form requests |

**Best Practice**: Use Template Mode as the **canonical contract** for auditing and reproducibility. Use AI Agent Mode for **adaptive execution** when features are complex or novel.

### Why v2.0 Prompts?

The v2.0 format addresses gaps identified in prompt quality analysis:

| Issue | v1.0 | v2.0 Solution |
|-------|------|---------------|
| Missing hard gates | Soft scoring only | Explicit FAIL gates with override |
| Subtle quality issues | Primary metrics only | Second-order quality signals |
| Regional compliance | Limited | Privacy frameworks per locale |
| Reproducibility | Variable | Canonical contract format |
| Auditability | Minimal | Version, timestamp, evaluation_id |

---

## Workflows for Complex Features

**File**: [src/core/workflows.py](../src/core/workflows.py)

For complex multi-step scenarios, use workflows:

```python
from src.core.workflows import WorkflowRunner

runner = WorkflowRunner()

# Multi-locale feature
result = runner.run(
    feature_name="Medical Document Summarizer",
    feature_description="Summarize medical documents for doctors...",
    target_locales=["de-DE", "ja-JP", "en-US"],
    safety_critical=True
)

# Get v2.0 prompts for each locale
for locale, prompt in result.prompts.items():
    print(f"--- {locale} ---")
    print(prompt[:200] + "...")
```

### Human-in-the-Loop Workflow

For safety-critical features requiring human approval:

```python
from src.core.workflows import HumanReviewWorkflow

workflow = HumanReviewWorkflow()

# Start - runs analysis
state = workflow.start(
    feature_name="Medical Assistant",
    feature_description="...",
    safety_critical=True
)

# Review analysis
print(f"Detected category: {state.detected_category}")
print(f"Privacy sensitive: {state.detected_privacy_sensitive}")

# Human approves or overrides
state = workflow.approve_analysis(state, approved=True)

# Review RAI compliance
print(f"RAI issues: {state.rai_issues}")

# Human approves
state = workflow.approve_rai(state, approved=True)

# Get final results (v2.0 prompts)
state = workflow.finalize(state)
```

---

## When to Use Which Mode?

| Scenario | Recommended Mode |
|----------|------------------|
| Simple, predefined features | Template Mode (fast, no API calls) |
| Natural language requests | AI Agent Mode |
| Multi-locale features | WorkflowRunner |
| Safety-critical with approval | HumanReviewWorkflow |
| Interactive exploration | AI Agent Mode `chat()` |
| Auditable contracts | Template Mode |
| Novel feature types | AI Agent Mode |
| Cost-sensitive environments | Template Mode |

---

## Installation

### Template Mode (Always Available)

No additional dependencies required.

### AI Agent Mode (Optional)

Requires Microsoft Agent Framework and Azure OpenAI:

```bash
pip install agent-framework --pre
```

Configure `.env`:
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key
DEPLOYMENT_NAME=gpt-4o
```

Agent Framework components gracefully degrade if not installed - the app falls back to Template Mode.

---

*Last updated: January 25, 2026*

