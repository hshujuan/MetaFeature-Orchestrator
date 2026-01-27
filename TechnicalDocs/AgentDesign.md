# Agent Design Documentation

> **Version**: 2.2  
> **Last Updated**: January 26, 2026

## Overview

MetaFeature-Orchestrator provides two agent modes for generating evaluation prompts:

1. **Template Mode** (`FeaturePromptWriterAgent`): Deterministic, template-based prompt generation (v2.0)
2. **AI Agent Mode** (`MetaFeatureAgent` v2.2): AI-powered agent using Microsoft Agent Framework with architecture detection

Both modes generate production-ready evaluation prompts with hard FAIL gates, second-order quality signals, and structured JSON output.

---

## Dual-Mode Architecture

| Aspect | Template Mode | AI Agent Mode |
|--------|---------------|---------------|
| **Class** | `FeaturePromptWriterAgent` | `MetaFeatureAgent` |
| **Framework** | Pure Python | Microsoft Agent Framework |
| **Planning** | Hard-coded workflow | Dynamic LLM reasoning |
| **Tool Use** | None | 8 tools via `@ai_function` |
| **Architecture Detection** | None | Pipeline, RAG, Agentic, Multimodal |
| **Feature-Specific Rubrics** | Generic templates | Tailored 5-point scoring |
| **Prompt Size** | ~3,000 chars | ~18,000 chars |
| **Memory** | Stateless | Thread-based conversation |
| **Decision Making** | Deterministic code paths | LLM-driven decisions |
| **Speed** | Instant | Requires API calls |
| **Cost** | Free | API costs |
| **Best For** | Simple, well-defined features | Complex, novel features |
| **Prompt Version** | v2.0 | v2.2 |

**Recommendation**: Use **⚖️ Both** mode to compare outputs side-by-side, then choose the best for your use case.

---

## Generation Modes (Web UI)

The web UI provides 4 generation modes:

| Mode | Code | Description | Progressive Loading |
|------|------|-------------|---------------------|
| 🤖 Auto | `"auto"` | Detects complexity, chooses automatically | No |
| ⚡ Always AI | `"always"` | Forces AI Agent mode | No |
| 📋 Template only | `"never"` | Uses deterministic templates | No |
| ⚖️ Both | `"both"` | Side-by-side comparison | **Yes** |

### Progressive Loading (Both Mode)

When using "Both" mode, results appear progressively:

```
1. User clicks Generate
2. Template output shows IMMEDIATELY (< 1 second)
3. AI Agent panel shows "⏳ Generating AI Agent prompt..."
4. AI Agent output appears when ready (5-10 seconds)
```

**Implementation**: `generate_both_prompts_streaming()` is a Python generator that yields twice.

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

## AI Agent Mode: `MetaFeatureAgent` v2.2

### Overview

The `MetaFeatureAgent` is a **true AI agent** built on Microsoft Agent Framework that uses LLMs to make dynamic decisions about metric selection, RAI compliance, and prompt generation. **v2.2 adds architecture detection and feature-specific rubrics.**

**File**: [src/core/ai_agent.py](../src/core/ai_agent.py)

### System Prompt (v2.2)

```
You are MetaFeature Agent v2.2, an expert AI evaluation prompt generator with advanced capabilities for complex, multi-component AI systems.

You specialize in:
1. **Architecture Detection**: Identifying Pipeline, RAG, Agentic, and Multimodal systems
2. **Intelligent Metric Selection**: Recommending architecture-specific metrics
3. **Feature-Specific Rubrics**: Generating tailored 5-point scoring criteria
4. **Failure Mode Analysis**: Identifying edge cases specific to the architecture

## CRITICAL: Always Use the build_prompt Tool

**NEVER generate evaluation prompts manually.** You MUST call the `build_prompt` tool because:
- It generates feature-specific rubrics for each metric
- It includes architecture-specific evaluation guidance
- It adds failure modes and edge cases
- It ensures 18,000+ character comprehensive prompts
```

### Available Tools (8 total)

The AI agent has access to these tools via `@ai_function` decorator:

| Tool | Purpose | Returns |
|------|---------|---------|
| `lookup_metrics` | Find metrics for a category | Metric names, definitions, weights |
| `suggest_metrics` | Get recommendations for additional metrics | Suggested metrics list |
| `recommend_metrics` | **Intelligent** metric selection with architecture detection | Prioritized metrics with rationale + architecture-specific metrics |
| `get_locale_info` | Get cultural/regulatory info for a locale | Cultural context, tone guidance, privacy framework |

#### 📌 `suggest_metrics` vs `recommend_metrics` Clarification

These two tools serve different purposes and should be used in different scenarios:

| Aspect | `suggest_metrics` | `recommend_metrics` |
|--------|-------------------|---------------------|
| **Analysis Type** | Simple rule-based lookup | Semantic analysis of feature description |
| **Input Required** | `category` + `current_metrics` | `feature_name` + `description` + `category` |
| **Output Format** | Flat list with generic reason | Prioritized tiers with detailed explanations |
| **Priority Levels** | None | Mandatory → Critical → Important |
| **Explanations** | Single generic reason for all | Per-metric explanation of relevance |
| **Best For** | Quick suggestions when category is known | Comprehensive metric planning for new features |
| **LLM Required** | No (rule-based) | No (rule-based analysis) |
| `validate_rai_compliance` | Check if metrics meet RAI requirements | Compliance status, issues, recommendations |
| `build_prompt` | Generate the v2.1 evaluation prompt | Complete evaluation prompt string (mandatory) |
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
│   │ Tool: recommend_metrics                 │  ◄── Intelligent     │
│   │   • Analyzes description semantically   │      selection       │
│   │   • Returns prioritized metrics         │                      │
│   │   • Provides explanations               │                      │
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
│   │ Tool: build_prompt (MANDATORY)          │  ◄── Generate v2.1   │
│   │   • All parameters from above           │      prompt          │
│   │   • Returns: Complete evaluation prompt │                      │
│   │   • Includes actual ISO timestamp       │                      │
│   └─────────────────────────────────────────┘                      │
│      │                                                              │
│      ▼                                                              │
│   Output: Complete v2.1 Evaluation Prompt                          │
│      • "# 🤖 AI Agent Evaluation Prompt" header                    │
│      • Hard FAIL gates                                              │
│      • Primary metrics with weights                                 │
│      • Second-order quality signals                                 │
│      • Structured JSON output format                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

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

### AI Agent Mode (v2.2)
```markdown
# 🤖 AI Agent Evaluation Prompt: [Feature Name]
**Version:** 2.2 (AI Agent Generated - Feature-Specific)
**Generation Mode:** AI Agent with Intelligent Analysis
**Target Language:** en
**Locale:** English (United States)
**Privacy Framework:** CCPA
**Architecture Type:** [Simple/Pipeline/RAG/Agentic/Multimodal]
**Generated:** 2026-01-26T12:00:00Z

## 1. FEATURE UNDER EVALUATION
### 📋 Feature Overview
**Name:** [Feature Name]
**Category:** [Category]
**Architecture:** [Architecture Type]

### 📝 Feature Description
[Detailed description...]

### 🎯 Key Capabilities
- [Capability 1]
- [Capability 2]

## 3. EVALUATOR INSTRUCTIONS
You are an **expert AI evaluator** specialized in assessing **[category]** systems...

## 4. HARD GATES ⛔
[Architecture-specific hard gates...]

## 5. FAILURE MODES & EDGE CASES
[Architecture-specific failure patterns...]

## 6. EVALUATION METRICS
[Feature-specific 5-point rubrics for each metric...]
```

**Key Differences (v2.0 vs v2.2):**
| Aspect | Template v2.0 | AI Agent v2.2 |
|--------|---------------|---------------|
| Header | `# Evaluation Prompt:` | `# 🤖 AI Agent Evaluation Prompt:` |
| Version | 2.0 | 2.2 |
| Architecture | Not detected | Pipeline/RAG/Agentic/Multimodal |
| Rubrics | Generic scoring | Feature-specific 5-point scales |
| Failure Modes | None | Architecture-specific patterns |
| Prompt Size | ~3,000 chars | ~18,000 chars |
| Sections | Basic structure | Comprehensive 9-section format |

---

## v2.0/v2.1 Prompt Templates

Both modes generate prompts with the canonical contract structure.

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
| AI Recommend | `recommend_metrics()` | **None** - rule-based analysis |
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

# Returns complete v2.1 evaluation prompt with:
# - "# 🤖 AI Agent Evaluation Prompt" header
# - HALLUCINATION, SAFETY, PRIVACY, FACTUAL gates
# - Summarization metrics (faithfulness, coverage, etc.)
# - German locale with GDPR framework
# - Second-order quality signals
print(response.evaluation_prompt)
```

### Intelligent Metric Recommendation

```python
from src.core.ai_agent import recommend_metrics

# Get intelligent recommendations with explanations
result = recommend_metrics(
    feature_name="Medical Summary Generator",
    feature_description="Summarizes patient records for physician review",
    category="summarization",
    input_format="text",
    output_format="text",
    check_privacy=True,
    check_safety=True,
    locale="de-DE"
)

print(result["recommendations_markdown"])
# Output includes prioritized metrics with explanations:
# 🔴 CRITICAL: privacy - Handles sensitive patient data...
# 🔴 CRITICAL: safety - Medical context requires...
# 🟡 IMPORTANT: faithfulness - Must not hallucinate...
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

**Best Practice**: Use **⚖️ Both** mode to compare outputs, then choose based on your needs.

### Why v2.0/v2.1 Prompts?

The v2.0+ format addresses gaps identified in prompt quality analysis:

| Issue | v1.0 | v2.0+ Solution |
|-------|------|----------------|
| Missing hard gates | Soft scoring only | Explicit FAIL gates with override |
| Subtle quality issues | Primary metrics only | Second-order quality signals |
| Regional compliance | Limited | Privacy frameworks per locale |
| Reproducibility | Variable | Canonical contract format |
| Auditability | Minimal | Version, timestamp, evaluation_id |
| Timestamp placeholders | "[Current Date]" | Actual ISO timestamp (v2.1 enforced) |

---

## When to Use Which Mode?

| Scenario | Recommended Mode |
|----------|------------------|
| Compare outputs | ⚖️ Both (side-by-side) |
| Simple, predefined features | 📋 Template only |
| Natural language requests | ⚡ Always AI |
| Multi-locale features | WorkflowRunner |
| Safety-critical with approval | HumanReviewWorkflow |
| Interactive exploration | AI Agent Mode `chat()` |
| Auditable contracts | 📋 Template only |
| Novel feature types | ⚡ Always AI |
| Cost-sensitive environments | 📋 Template only |
| Understanding the difference | ⚖️ Both |

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

*Last updated: January 26, 2026*
