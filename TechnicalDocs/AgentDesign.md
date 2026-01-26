# Agent Design Documentation

## Overview

The `FeaturePromptWriterAgent` is the core component responsible for generating evaluation prompts. Despite the "Agent" naming, it implements an **agentic workflow pattern** (structured pipeline) rather than an autonomous agent.

---

## Architecture Classification

| Aspect | This Implementation | Autonomous Agent |
|--------|---------------------|------------------|
| **Framework** | None - pure Python | LangChain, AutoGen, CrewAI |
| **Planning** | Hard-coded workflow | Dynamic planning |
| **Tool Use** | None | Calls tools autonomously |
| **Memory** | Stateless | Maintains conversation/context |
| **Loops** | Single-pass | Iterative reasoning loops |
| **Decision Making** | Deterministic code paths | LLM-driven decisions |

**Classification**: Template-driven prompt builder with rule-based RAI injection.

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FeaturePromptWriterAgent.generate()              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: FeatureSpec or FeatureMetadata                            │
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
│   │   • Select category-specific template   │      (string format) │
│   │   • Fill in metrics, definitions        │                      │
│   │   • Apply localization (i18n)           │                      │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   Output: PromptOutput                                              │
│      • evaluation_prompt (string)                                   │
│      • metrics_used (list)                                          │
│      • metric_definitions (dict)                                    │
│      • rai_checks_applied (list)                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Point**: No LLM is called inside the agent. It's pure Python logic.

---

## Source Code Reference

**File**: [src/core/agent.py](../src/core/agent.py)

### Class Definition

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

## Where LLM IS Called

The agent itself does **not** call the LLM. LLM calls happen in [app.py](../src/core/app.py):

| Function | Purpose | Temperature |
|----------|---------|-------------|
| `generate_ai_outputs()` | Generate good/bad test outputs | 0.0 |
| `run_simulation_evaluation()` | Run LLM-based scoring | 0.0 |

---

## Prompt Templates

The agent delegates to [prompt_templates.py](../src/core/prompt_templates.py) for template rendering.

### Category-Specific Role Prompts

| Category | Role Description |
|----------|------------------|
| `auto_reply` | "You are an expert evaluator for AI-generated email/message replies. Your task is to score the generated reply against the original message using the metrics below." |
| `summarization` | "You are an expert evaluator for AI-generated summaries. Your task is to assess the summary against the source document using the metrics below." |
| `translation` | "You are an expert evaluator for AI-generated translations. Your task is to assess translation quality against the source text." |
| `generic` | "You are an expert evaluator for AI-generated outputs. Your task is to assess the quality of the generated content using the metrics below." |

### Template Structure

Each generated prompt includes:
1. **Role** - Evaluator persona
2. **Metrics** - With definitions and weights
3. **Scoring Rubric** - 1-5 scale with descriptions
4. **RAI Checks** - Safety, privacy, bias checks
5. **Output Format** - Expected JSON schema

---

## Localization (i18n)

The agent supports 8 languages for prompt generation:

| Code | Language |
|------|----------|
| `en` | English |
| `zh` | Chinese (Simplified) |
| `ja` | Japanese |
| `es` | Spanish |
| `fr` | French |
| `de` | German |
| `pt` | Portuguese |
| `ko` | Korean |

All labels, instructions, and rubrics are localized in `LOCALIZED_LABELS` dict.

---

## Usage Example

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

# Generate evaluation prompt
result = agent.generate(feature, language="en")

print(result.evaluation_prompt)
print(f"Metrics: {result.metrics_used}")
print(f"RAI checks: {result.rai_checks_applied}")
```

---

## Design Decisions

### Why Not an Autonomous Agent?

1. **Predictability** - Fixed pipeline ensures consistent output structure
2. **Debuggability** - Each step can be traced and tested
3. **Performance** - No LLM calls means instant prompt generation
4. **Cost** - No API costs for prompt construction
5. **Control** - RAI constraints are guaranteed, not "suggested"

### Why Call It "Agent"?

The term "agent" here refers to the **agent pattern** in software design:
- A component that acts on behalf of the user
- Makes decisions based on input (metric selection, RAI injection)
- Produces actionable output (evaluation prompts)

It's not an "AI agent" in the LangChain/AutoGen sense.

---

## Future Enhancements ✅ (Now Implemented!)

The following enhancements have been implemented using **Microsoft Agent Framework**:

### New: MetaFeatureAgent (AI-Powered)

**File**: [src/core/ai_agent.py](../src/core/ai_agent.py)

The `MetaFeatureAgent` is a **true AI agent** that uses LLMs to make dynamic decisions:

```python
from src.core.ai_agent import MetaFeatureAgent

agent = MetaFeatureAgent()

# Natural language request - the agent figures out the rest
response = agent.chat(
    "I need an evaluation prompt for a medical document summarizer "
    "that will be used by doctors in Germany. It needs to be very "
    "careful about accuracy and patient privacy."
)

print(response.evaluation_prompt)
```

### Agent Comparison

| Aspect | FeaturePromptWriterAgent (Legacy) | MetaFeatureAgent (New) |
|--------|-----------------------------------|------------------------|
| **Framework** | None - pure Python | Microsoft Agent Framework |
| **Decision Making** | Deterministic code paths | LLM-driven decisions |
| **Tool Use** | None | 11 tools available |
| **Memory** | Stateless | Thread-based conversation |
| **Natural Language** | Structured input only | Understands free-form requests |
| **Error Recovery** | None | Can retry with different approaches |
| **Complex Features** | Limited handling | Dynamic reasoning |

### Available Tools

The AI agent has access to these tools ([agent_tools.py](../src/core/agent_tools.py)):

| Tool | Purpose |
|------|---------|
| `lookup_metrics` | Find metrics for a category |
| `suggest_metrics` | Get recommendations for additional metrics |
| `search_metric_by_name` | Find metrics by keyword |
| `get_locale_info` | Get cultural/regulatory info for a locale |
| `list_supported_locales` | See all supported locales |
| `search_similar_features` | Find existing similar features in DB |
| `get_feature_by_id` | Retrieve a specific feature |
| `build_prompt` | Generate the evaluation prompt |
| `validate_rai_compliance` | Check if metrics meet RAI requirements |
| `get_code_metrics` | Get programmatic metrics (ROUGE, BLEU) |
| `analyze_feature_description` | Extract attributes from natural language |

### Workflows for Complex Features

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

# Get prompts for each locale
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

# Get final results
state = workflow.finalize(state)
```

### When to Use Which?

| Scenario | Recommended |
|----------|-------------|
| Simple, predefined features | `FeaturePromptWriterAgent` (fast, no API calls) |
| Natural language requests | `MetaFeatureAgent` |
| Multi-locale features | `WorkflowRunner` |
| Safety-critical with approval | `HumanReviewWorkflow` |
| Interactive exploration | `MetaFeatureAgent.chat()` |

### Installation

The AI agent requires Microsoft Agent Framework:

```bash
pip install agent-framework --pre
```

The legacy agent works without it. Agent Framework components are optional and gracefully degrade if not installed.

