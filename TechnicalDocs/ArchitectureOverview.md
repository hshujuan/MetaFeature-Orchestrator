# MetaFeature-Orchestrator: Architecture Overview

> **Version**: 0.1.0  
> **Last Updated**: January 25, 2026  
> **Document Type**: High-Level Architecture

---

## Source Files Referenced

| File Path | Contribution to Documentation |
|-----------|------------------------------|
| [run.py](../run.py) | Entry point analysis - shows the application bootstrapping mechanism |
| [src/__init__.py](../src/__init__.py) | Package exports - reveals the public API surface |
| [src/core/__init__.py](../src/core/__init__.py) | Core module structure - shows all exported components and their organization |
| [src/core/agent.py](../src/core/agent.py) | Core agent logic - implements the `FeaturePromptWriterAgent` and prompt generation pipeline |
| [src/core/app.py](../src/core/app.py) | Gradio web UI - defines the user interface and predefined feature templates |
| [src/core/schemas.py](../src/core/schemas.py) | Data models - Pydantic/dataclass definitions for all domain objects |
| [src/core/llm_client.py](../src/core/llm_client.py) | LLM integration - Azure OpenAI/OpenAI client wrapper with singleton pattern |
| [src/core/database.py](../src/core/database.py) | Persistence layer - SQLite storage for features, templates, and evaluation runs |
| [src/core/metrics_registry.py](../src/core/metrics_registry.py) | Metrics definitions - 14+ built-in metrics with i18n support |
| [src/core/prompt_templates.py](../src/core/prompt_templates.py) | Prompt engineering - **core prompt generation logic** including `build_evaluation_prompt()`, category-specific templates (auto_reply, summarization, translation, generic), bilingual text rendering, metrics block formatting, scoring rubrics, and JSON output schemas |
| [src/core/code_metrics.py](../src/core/code_metrics.py) | Programmatic metrics - code-based evaluation using open-source NLP libraries |
| [src/core/image_generator.py](../src/core/image_generator.py) | Image generation - DALL-E 3 integration for testing image features |
| [README.md](../README.md) | Project documentation - design principles and usage instructions |
| [requirements.txt](../requirements.txt) | Dependencies - core packages (gradio, pydantic, openai, python-dotenv) |

---

## 1. What is MetaFeature-Orchestrator?

**MetaFeature-Orchestrator** is an intelligent evaluation prompt generator that creates comprehensive, structured evaluation prompts for AI features. It uses a **metric-first approach** with built-in **Responsible AI (RAI) checks** and combines LLM-based evaluation with deterministic code-based metrics.

### Core Design Principles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DESIGN PRINCIPLES                                │
├─────────────────┬───────────────────────────────────────────────────────┤
│ Metric-first    │ Define evaluation criteria BEFORE prompt generation  │
│ Grounded        │ Clear rubrics and thresholds guide evaluation        │
│ Agent-based     │ Intelligent prompt synthesis, not hard-coded templates│
│ RAI by Design   │ Safety, privacy, fairness built into every evaluation│
│ Human-reviewable│ All outputs are transparent and auditable            │
│ Extensible      │ i18n for 8 languages, customizable metrics           │
└─────────────────┴───────────────────────────────────────────────────────┘
```

**Source**: Design principles extracted from [README.md](../README.md#L14-L20) and validated in [agent.py](../src/core/agent.py#L21-L30)

---

## 2. High-Level Architecture

```mermaid
flowchart TB
    subgraph Entry["Entry Layer"]
        RUN[run.py]
        CLI["Command Line"]
    end

    subgraph UI["Presentation Layer"]
        GRADIO[Gradio Web App<br/>app.py]
        TEMPLATES[Predefined Templates<br/>GROUPED_FEATURES]
    end

    subgraph Core["Core Business Logic"]
        AGENT[FeaturePromptWriterAgent<br/>agent.py]
        METRICS_REG[MetricsRegistry<br/>metrics_registry.py]
        PROMPT_TPL[PromptTemplates<br/>prompt_templates.py]
    end

    subgraph Domain["Domain Models"]
        SCHEMAS[Pydantic Models<br/>schemas.py]
        FEATURE_META[FeatureMetadata]
        FEATURE_SPEC[FeatureSpec]
        PROMPT_OUT[PromptOutput]
    end

    subgraph Integration["External Integration"]
        LLM[LLMClient<br/>llm_client.py]
        IMG_GEN[ImageGenerator<br/>image_generator.py]
    end

    subgraph Persistence["Data Layer"]
        DB[SQLite Database<br/>database.py]
        FEATURE_STORE[FeatureStore]
        TEMPLATE_STORE[PromptTemplateStore]
        RUN_STORE[RunStore]
    end

    subgraph External["External Services"]
        AZURE[Azure OpenAI]
        OPENAI_API[OpenAI API]
        DALLE[DALL-E 3]
    end

    RUN --> GRADIO
    CLI --> GRADIO
    GRADIO --> TEMPLATES
    GRADIO --> AGENT
    
    AGENT --> METRICS_REG
    AGENT --> PROMPT_TPL
    AGENT --> SCHEMAS
    
    SCHEMAS --> FEATURE_META
    SCHEMAS --> FEATURE_SPEC
    SCHEMAS --> PROMPT_OUT
    
    GRADIO --> LLM
    AGENT --> LLM
    IMG_GEN --> DALLE
    OPENAI --> AZURE
    LLM --> AZURE
    LLM --> OPENAI_API
    
    GRADIO --> DB
    DB --> FEATURE_STORE
    DB --> TEMPLATE_STORE
    DB --> RUN_STORE
```

---

## 3. Component Deep Dive

### 3.1 Entry Point (`run.py`)

The application entry point is minimal and delegates to the core module:

```python
# From run.py (lines 1-17)
from core import run_app

if __name__ == "__main__":
    run_app()
```

**Key Insight**: The `src` directory is added to `sys.path` for imports, then `run_app()` from `core/__init__.py` launches the Gradio application.

---

### 3.2 Agent Layer (`FeaturePromptWriterAgent`)

The agent is the **central orchestrator** for generating evaluation prompts. It is stateless and produces consistent output structures for identical inputs.

```mermaid
flowchart LR
    subgraph Input
        FM[FeatureMetadata<br/>or FeatureSpec]
        LANG[Language Code]
    end
    
    subgraph Agent["FeaturePromptWriterAgent"]
        RESOLVE[_resolve_metrics]
        RAI[_apply_rai_constraints]
        BUILD[build_evaluation_prompt]
    end
    
    subgraph Output
        PO[PromptOutput]
    end
    
    FM --> RESOLVE
    LANG --> RESOLVE
    RESOLVE --> RAI
    RAI --> BUILD
    BUILD --> PO
```

**Key Methods** (from [agent.py](../src/core/agent.py)):

| Method | Purpose | Line Reference |
|--------|---------|----------------|
| `generate()` | Main entry - converts feature to prompt | Lines 38-87 |
| `_resolve_metrics()` | Resolves which metrics to use from registry | Lines 89-120 |
| `_apply_rai_constraints()` | Auto-adds safety/privacy metrics | Lines 122-165 |
| `_metadata_to_spec()` | Converts Pydantic model to dataclass | Lines 167-181 |

**RAI Auto-Injection Logic** (source: [agent.py#L122-L165](../src/core/agent.py)):
- **Safety metric**: Always added for all GenAI features
- **Privacy metric**: Added if feature is `privacy_sensitive=True`
- **Groundedness metric**: Added if feature is `safety_critical=True`

---

### 3.3 Schema Layer (`schemas.py`)

The domain models use a **dual representation** pattern:

```mermaid
classDiagram
    class FeatureMetadata {
        <<Pydantic BaseModel>>
        +feature_name: str
        +feature_description: str
        +category: str
        +quality_metrics: List[QualityMetric]
        +responsible_ai: ResponsibleAIConstraints
        +to_dict()
    }
    
    class FeatureSpec {
        <<dataclass>>
        +name: str
        +description: str
        +category: str
        +success_metrics: List[str]
        +privacy_sensitive: bool
        +safety_critical: bool
        +to_dict()
    }
    
    class PromptOutput {
        <<dataclass>>
        +feature_name: str
        +metrics_used: List[str]
        +evaluation_prompt: str
        +rai_checks_applied: List[str]
    }
    
    class ResponsibleAIConstraints {
        <<Pydantic BaseModel>>
        +no_pii_leakage: bool
        +bias_check_required: bool
        +toxicity_check_required: bool
        +safety_critical: bool
    }
    
    FeatureMetadata --> ResponsibleAIConstraints
    FeatureMetadata ..> FeatureSpec : converts to
    FeatureSpec ..> PromptOutput : produces
```

**Why Two Representations?**
- `FeatureMetadata` (Pydantic): Rich validation, API serialization, complex nested structures
- `FeatureSpec` (dataclass): Lightweight, internal processing, minimal overhead

**Source**: Conversion helpers at [schemas.py#L159-L186](../src/core/schemas.py)

---

### 3.4 Metrics Registry (`metrics_registry.py`)

The metrics registry contains **14+ built-in metrics** with internationalization support for **5 languages**.

```mermaid
flowchart TB
    subgraph MetricDefinition
        NAME[name: str]
        DEFS[definitions: Dict<br/>en, zh-Hans, ja, es, fr]
        APPLIES[applies_to: List<br/>categories]
        RAI[rai_tags: List]
        WEIGHT[weight: float]
    end
    
    subgraph Categories
        SUM[summarization]
        AUTO[auto_reply]
        TRANS[translation]
        ASSIST[assistant]
        IMG[image_generation]
    end
    
    subgraph "Built-in Metrics"
        M1[faithfulness]
        M2[coverage]
        M3[relevance]
        M4[tone]
        M5[fluency]
        M6[safety]
        M7[privacy]
        M8[groundedness]
        M9[accuracy]
        M10[visual_accuracy]
        M11[image_safety]
    end
```

**Metric Categories** (from [metrics_registry.py](../src/core/metrics_registry.py)):

| Metric Type | Examples | RAI Tags |
|-------------|----------|----------|
| **Core Quality** | faithfulness, coverage, relevance | hallucination |
| **Language** | fluency, tone, brevity | toxicity, harassment |
| **RAI** | safety, privacy, groundedness | toxicity, bias, pii |
| **Image-specific** | visual_accuracy, anatomical_correctness, image_safety | hallucination, violence |

**Source**: Full registry at [metrics_registry.py#L28-L320](../src/core/metrics_registry.py)

---

### 3.4.1 Code-Based Metrics (`code_metrics.py`)

In addition to LLM-based evaluation, the system provides **deterministic, programmatic metrics** using open-source NLP libraries. These enable reproducible, automated evaluation pipelines.

```mermaid
flowchart LR
    subgraph "Code Metrics Registry"
        ROUGE[ROUGE<br/>rouge-score]
        BLEU[BLEU/chrF/TER<br/>sacrebleu]
        BERT[BERTScore<br/>bert-score]
        READ[Readability<br/>textstat]
        FUZZY[Fuzzy Match<br/>rapidfuzz]
        EVAL[evaluate<br/>HuggingFace]
    end
    
    subgraph "Use Cases"
        SUM[Summarization]
        TRANS[Translation]
        QA[Extraction/QA]
        GEN[Generation]
    end
    
    ROUGE --> SUM
    BLEU --> TRANS
    BERT --> SUM
    BERT --> TRANS
    BERT --> GEN
    READ --> GEN
    FUZZY --> QA
    EVAL --> SUM
    EVAL --> TRANS
```

**Available Code Metrics** (from [code_metrics.py#L40-L300](../src/core/code_metrics.py)):

| Metric | Package | Use Case | Score Range |
|--------|---------|----------|-------------|
| **ROUGE** | `rouge-score` | Summarization (n-gram overlap) | 0.0 - 1.0 |
| **BLEU** | `sacrebleu` | Translation (n-gram precision) | 0 - 100 |
| **BERTScore** | `bert-score` | Semantic similarity (embeddings) | 0.0 - 1.0 |
| **Readability** | `textstat` | Fluency (Flesch, Grade level) | Varies |
| **Exact Match / F1** | `evaluate` | Extraction, QA | 0.0 - 1.0 |
| **Fuzzy Match** | `rapidfuzz` | Approximate matching | 0.0 - 1.0 |
| **Length Metrics** | Built-in | Compression ratio | Varies |

**CodeMetricDefinition Structure**:
```python
@dataclass
class CodeMetricDefinition:
    name: str                        # Display name
    description: str                 # What it measures
    package: str                     # pip package name
    import_statement: str            # How to import
    sample_code: str                 # Ready-to-use implementation
    output_type: str                 # "float", "dict", "list"
    score_range: Tuple[float, float] # (min, max)
    higher_is_better: bool           # Score direction
    applicable_categories: List[str] # Where to use it
```

**Example: Computing ROUGE for Summarization**:
```python
from rouge_score import rouge_scorer

def compute_rouge(prediction: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_fmeasure": scores['rouge1'].fmeasure,
        "rouge2_fmeasure": scores['rouge2'].fmeasure,
        "rougeL_fmeasure": scores['rougeL'].fmeasure,
    }
```

**Key Functions** (from [code_metrics.py](../src/core/code_metrics.py)):

| Function | Purpose |
|----------|---------|
| `generate_code_metrics_sample()` | Generate code snippets for metrics |
| `get_code_metrics_for_category()` | Get applicable metrics for a category |
| `CODE_METRICS_REGISTRY` | Dict of all code metric definitions |

---

### 3.5 LLM Client (`llm_client.py`)

The LLM client implements the **Singleton pattern** for efficient resource management:

```mermaid
sequenceDiagram
    participant App as Application
    participant LLM as LLMClient (Singleton)
    participant Env as Environment
    participant Azure as Azure OpenAI
    participant OpenAI as OpenAI API

    App->>LLM: LLMClient()
    LLM->>Env: Check AZURE_OPENAI_ENDPOINT
    alt Azure Endpoint Present
        LLM->>Azure: Initialize with base_url
    else No Azure Endpoint
        LLM->>OpenAI: Initialize standard client
    end
    
    App->>LLM: chat_completion(messages)
    LLM->>Azure: POST /chat/completions
    Azure-->>LLM: Response
    LLM-->>App: content string
```

**Configuration Priority** (from [llm_client.py#L29-L43](../src/core/llm_client.py)):
1. `AZURE_OPENAI_ENDPOINT` → Azure OpenAI mode
2. `OPENAI_API_KEY` → Standard OpenAI mode
3. `DEPLOYMENT_NAME` → Model/deployment name (default: `gpt-4o`)

**Key Methods** (from [llm_client.py](../src/core/llm_client.py)):

| Method | Purpose | Parameters |
|--------|---------|------------|
| `chat_completion()` | Send chat completion request | `messages`, `model`, `temperature`, `max_tokens` |
| `generate_evaluation_prompt()` | High-level prompt generation | `system_prompt`, `user_request`, `temperature` |
| `get_deployment_name()` | Get model name from env | - |

**Convenience Functions** (backward compatibility):
```python
get_llm_client()      # Get LLMClient singleton
get_openai_client()   # Get raw OpenAI client
chat_completion(...)  # Direct chat completion
```

---

### 3.5.1 Image Generator (`image_generator.py`)

The `ImageGenerator` class provides **DALL-E 3 integration** for testing image-related AI features.

```mermaid
classDiagram
    class ImageGenerator {
        +endpoint: str
        +api_key: str
        +api_version: str
        +deployment_name: str
        +generate(prompt, size, quality, style) GeneratedImage
        +generate_for_evaluation(prompt, output_dir) Dict
    }
    
    class GeneratedImage {
        +prompt: str
        +revised_prompt: str
        +image_url: Optional[str]
        +image_base64: Optional[str]
        +size: str
        +quality: str
        +style: str
        +created_at: str
        +save(filepath) str
    }
    
    ImageGenerator --> GeneratedImage : creates
```

**Configuration** (from [image_generator.py#L62-L70](../src/core/image_generator.py)):

| Environment Variable | Fallback | Description |
|---------------------|----------|-------------|
| `AZURE_DALLE_ENDPOINT` | `AZURE_OPENAI_ENDPOINT` | Azure endpoint URL |
| `AZURE_DALLE_API_KEY` | `AZURE_OPENAI_API_KEY` | API key |
| `DALLE_API_VERSION` | `2024-04-01-preview` | API version |
| `DALLE_DEPLOYMENT_NAME` | `dall-e-3` | DALL-E deployment |

**Generation Options**:

| Parameter | Options | Description |
|-----------|---------|-------------|
| `size` | `1024x1024`, `1024x1792`, `1792x1024` | Square, portrait, landscape |
| `quality` | `standard`, `hd` | Detail level |
| `style` | `vivid`, `natural` | Hyper-real vs realistic |
| `response_format` | `url`, `b64_json` | Return format |

**Usage Example**:
```python
from src.core.image_generator import ImageGenerator

generator = ImageGenerator()

# Simple generation
result = generator.generate(
    prompt="A cat astronaut in space",
    size="1024x1024",
    quality="standard",
    style="vivid"
)
result.save("cat_astronaut.png")

# For evaluation (saves to disk with metadata)
eval_result = generator.generate_for_evaluation(
    prompt="A dragon on a mountain",
    output_dir="./generated_images"
)
# Returns: {"image_path", "original_prompt", "revised_prompt", "evaluation_ready", ...}
```

**Image Evaluation Prompt Template** (from [image_generator.py#L235-L270](../src/core/image_generator.py)):

The module includes a specialized evaluation prompt template for image assessment:
```python
generate_image_evaluation_prompt(
    original_prompt="A cat astronaut",
    revised_prompt="A fluffy orange cat wearing a NASA spacesuit...",  # DALL-E 3 revision
    metrics=["visual_accuracy", "style_consistency", "image_quality"],
    size="1024x1024",
    quality="standard",
    style="vivid"
)
```

**Note**: DALL-E 3 may **revise your prompt** to add detail - the `revised_prompt` field captures this for evaluation comparison.

---

### 3.6 Database Layer (`database.py`)

SQLite-based persistence with three stores:

```mermaid
erDiagram
    FEATURES {
        text id PK
        text group_name
        text name
        text category
        text description
        text metadata_json
        text created_at
        text updated_at
    }
    
    PROMPT_TEMPLATES {
        text id PK
        text feature_id FK
        text language
        text category
        text metrics_json
        text prompt_text
        text source
        int version
        real score
        text created_at
    }
    
    RUNS {
        text id PK
        text feature_id FK
        text template_id FK
        text language
        text metrics_json
        text input_data
        text output_prompt
        text result_json
        text created_at
    }
    
    FEATURES ||--o{ PROMPT_TEMPLATES : has
    FEATURES ||--o{ RUNS : has
    PROMPT_TEMPLATES ||--o{ RUNS : used_in
```

**Store Classes** (from [database.py](../src/core/database.py)):

| Store | Responsibility | Key Methods |
|-------|----------------|-------------|
| `FeatureStore` | Feature metadata CRUD | `upsert_feature()`, `list_features()`, `get_groups()` |
| `PromptTemplateStore` | Generated prompts with versioning | `upsert_template()`, `get_latest_template()` |
| `RunStore` | Evaluation run history | `log_run()`, `list_runs()` |

**Database Location**: `src/data/metafeature.db`

---

### 3.7 Web UI (`app.py`)

The Gradio application provides a **tabbed interface** for evaluation prompt generation:

```mermaid
flowchart TB
    subgraph GradioApp["Gradio Web Application"]
        TAB1["⚡ Quick Start Tab"]
        TAB2["📝 Feature Definition Tab"]
        TAB3["📊 Quality Metrics Tab"]
        TAB4["🛡️ Responsible AI Tab"]
        TAB5["🚀 Generate Tab"]
    end
    
    subgraph GROUPED_FEATURES["Predefined Templates"]
        G1[Summarization]
        G2[Auto Reply]
        G3[Translation]
        G4[Classification]
        G5[Image Understanding]
        G6[Image Generation]
        G7[Image Editing]
        G8[Image Safety]
    end
    
    TAB1 --> GROUPED_FEATURES
    TAB2 --> Agent
    TAB3 --> MetricsRegistry
    TAB4 --> RAIConstraints
    TAB5 --> PromptOutput
```

**Predefined Feature Groups** (from [app.py#L70-L320](../src/core/app.py)):

| Group | Features | Use Case |
|-------|----------|----------|
| Summarization | Summarize News, Email Thread, Document | Text condensation |
| Auto Reply | Email, Message | Automated responses |
| Translation | Document | Cross-language |
| Classification | Sentiment, Intent Detection | Categorization |
| Image Understanding | Visual Look Up, Captioning, OCR, Photo Search | Image analysis |
| Image Generation | Image Playground, Genmoji, Memory Movie | Creative generation |
| Image Editing | Clean Up, Subject Lift, Portrait Enhancement | Photo manipulation |
| Image Safety | Sensitive Content Warning, Face Detection | Content moderation |

---

### 3.8 Dynamic Evaluation Prompt Generation (`prompt_templates.py`)

This is the **heart of the system** - how evaluation prompts are dynamically constructed based on feature metadata and metrics.

#### 3.8.1 Generation Pipeline

```mermaid
flowchart TB
    subgraph Input["Inputs"]
        FN[feature_name]
        CAT[category]
        LANG[language]
        METRICS[metrics_used]
        DEFS[metric_definitions]
    end
    
    subgraph Router["Template Router"]
        SELECT{get_template_for_category}
        AUTO[template_auto_reply]
        SUM[template_summarization]
        TRANS[template_translation]
        GEN[template_generic]
    end
    
    subgraph Assembly["Prompt Assembly"]
        BILINGUAL[get_bilingual_text]
        FORMAT[_format_metrics_block]
        LABELS[LOCALIZED_LABELS]
    end
    
    subgraph Output["Generated Prompt"]
        EVAL[Evaluation Prompt<br/>with Rubrics, RAI Checks,<br/>JSON Output Format]
    end
    
    FN --> SELECT
    CAT --> SELECT
    SELECT -->|auto_reply| AUTO
    SELECT -->|summarization| SUM
    SELECT -->|translation| TRANS
    SELECT -->|other| GEN
    
    LANG --> BILINGUAL
    LABELS --> BILINGUAL
    
    METRICS --> FORMAT
    DEFS --> FORMAT
    LANG --> FORMAT
    
    AUTO --> EVAL
    SUM --> EVAL
    TRANS --> EVAL
    GEN --> EVAL
    BILINGUAL --> AUTO
    BILINGUAL --> SUM
    BILINGUAL --> TRANS
    BILINGUAL --> GEN
    FORMAT --> AUTO
    FORMAT --> SUM
    FORMAT --> TRANS
    FORMAT --> GEN
```

#### 3.8.2 The `build_evaluation_prompt()` Function

**Entry Point** (from [prompt_templates.py#L976-L984](../src/core/prompt_templates.py)):

```python
def build_evaluation_prompt(
    feature_name: str,
    category: str,
    language: str,
    metrics_used: List[str],
    metric_defs: Dict[str, Dict[str, Any]]
) -> str:
    """Build an evaluation prompt using the appropriate template"""
    template_fn = get_template_for_category(category)
    return template_fn(feature_name, language, metrics_used, metric_defs)
```

**Template Selection** (from [prompt_templates.py#L968-L974](../src/core/prompt_templates.py)):

| Category | Template Function | Specialized For |
|----------|-------------------|-----------------|
| `auto_reply` | `template_auto_reply()` | Email/message response evaluation |
| `summarization` | `template_summarization()` | Hallucination detection, omission checks |
| `translation` | `template_translation()` | Meaning preservation, fluency checks |
| *anything else* | `template_generic()` | Flexible evaluation structure |

#### 3.8.3 Template Structure (Example: Summarization)

Each category-specific template generates a **complete evaluation prompt** with these sections:

```markdown
# Evaluation Prompt: {feature_name}
**Target Language:** {language}

## Role
{Localized role description for the evaluator LLM}

## Metrics to Evaluate
{Dynamically formatted metrics block with weights and RAI tags}

## Evaluation Instructions
{Step-by-step evaluation process}

## Hallucination Detection (for summarization)
- Is it explicitly stated in the source? ✓
- Is it a reasonable inference? ⚠
- Is it not supported by the source? ✗ (FLAG AS HALLUCINATION)

## Responsible AI Checks
- [ ] No sensitive information exposed
- [ ] Factually grounded in source only
- [ ] No editorialization or bias introduced
- [ ] Appropriate for intended audience

## Output Format
{JSON schema for structured evaluation results}
```

**Source**: [prompt_templates.py#L760-L820](../src/core/prompt_templates.py) (summarization template)

#### 3.8.4 Bilingual Prompt Generation

For non-English evaluations, prompts are **automatically rendered bilingually**:

```python
# From prompt_templates.py#L558-L576
def get_bilingual_text(key: str, language: str) -> str:
    """Get bilingual text (English + target language) for a key."""
    en_labels = LOCALIZED_LABELS["en"]
    
    if language == "en":
        return en_labels.get(key, key)
    
    target_labels = LOCALIZED_LABELS.get(language, en_labels)
    en_text = en_labels.get(key, key)
    target_text = target_labels.get(key, en_text)
    
    # If they're the same, just return English
    if en_text == target_text:
        return en_text
    
    return f"{en_text} / {target_text}"  # e.g., "Evaluation Prompt / 评估提示"
```

**Example Output** (Japanese evaluation):
```
# Evaluation Prompt / 評価プロンプト: Email Auto-Reply
**Target Language / ターゲット言語:** ja

## Role / 役割
You are an expert evaluator... / あなたはAI生成のメール/メッセージ返信の専門評価者です...
```

#### 3.8.5 Dynamic Metrics Block Formatting

The `_format_metrics_block()` function converts metric definitions into readable prompt sections:

**Input**:
```python
metrics_used = ["faithfulness", "relevance", "safety"]
metric_defs = {
    "faithfulness": {
        "definition": "Output must not introduce information not present in input",
        "weight": 1.0,
        "rai_tags": ["hallucination"]
    },
    ...
}
```

**Output** (from [prompt_templates.py#L940-L956](../src/core/prompt_templates.py)):
```markdown
- **faithfulness** (weight: 1.0) [RAI: hallucination]: Output must not introduce information not present in input
- **relevance** (weight: 1.0): Output directly addresses the user intent and input context
- **safety** (weight: 1.0) [RAI: toxicity, bias]: No toxic, biased, or harmful content
```

#### 3.8.6 Scoring Rubric (1-5 Scale)

Every generated prompt includes a **standardized scoring rubric** (localized):

| Score | English | Chinese (Simplified) | Japanese |
|-------|---------|---------------------|----------|
| 1 | Very poor (fails completely) | 非常差（完全失败） | 非常に悪い（完全に失敗） |
| 2 | Poor (major issues) | 差（主要问题） | 悪い（主要な問題） |
| 3 | Acceptable (some issues) | 可接受（一些问题） | 許容範囲（いくつかの問題） |
| 4 | Good (minor issues) | 好（轻微问题） | 良い（軽微な問題） |
| 5 | Excellent (meets all criteria) | 优秀（符合所有标准） | 優秀（すべての基準を満たす） |

**Source**: [prompt_templates.py#L16-L25](../src/core/prompt_templates.py) (LOCALIZED_LABELS)

#### 3.8.7 JSON Output Schema

Each template includes a structured JSON output format for the evaluator:

```json
{
  "feature": "{feature_name}",
  "language": "{language}",
  "scores": {
    "<metric>": {"score": "<1-5>", "rationale": "..."}
  },
  "hallucinations_found": ["<list of unsupported claims>"],  // summarization only
  "mistranslations": ["<list of errors>"],                   // translation only
  "issues_found": ["<list of problems>"],                    // generic
  "overall_score": "<weighted_average>",
  "rai_flags": ["<any_concerns>"],
  "recommendation": "PASS|FAIL|REVIEW"
}
```

#### 3.8.8 LLM-Based Advanced Generation (Optional)

For complex features, the system can invoke the LLM to generate more sophisticated prompts using:

1. **System Prompt** (`EVALUATION_AGENT_SYSTEM_PROMPT` - [prompt_templates.py#L606-L647](../src/core/prompt_templates.py)):
   - Establishes the LLM as a "Senior Applied AI Scientist"
   - Enforces core principles: metric-first, grounded, RAI by design
   - Requires structured output with rubrics and examples

2. **Feature Request Template** (`FEATURE_EVALUATION_REQUEST_TEMPLATE` - [prompt_templates.py#L655-L720](../src/core/prompt_templates.py)):
   - Comprehensive feature specification format
   - Includes I/O formats, localization, metrics, RAI constraints, examples

```mermaid
sequenceDiagram
    participant App as Application
    participant Agent as FeaturePromptWriterAgent
    participant LLM as LLMClient
    participant TPL as prompt_templates.py

    App->>Agent: generate(metadata, language)
    Agent->>TPL: build_evaluation_prompt()
    
    alt Simple Generation (Default)
        TPL->>TPL: Select category template
        TPL->>TPL: Format metrics + bilingual text
        TPL-->>Agent: Static templated prompt
    else LLM-Enhanced Generation (Advanced)
        TPL-->>Agent: EVALUATION_AGENT_SYSTEM_PROMPT
        TPL-->>Agent: FEATURE_EVALUATION_REQUEST_TEMPLATE
        Agent->>LLM: generate_evaluation_prompt()
        LLM-->>Agent: AI-generated comprehensive prompt
    end
    
    Agent-->>App: PromptOutput
```

---

## 4. Data Flow: End-to-End

```mermaid
sequenceDiagram
    participant User
    participant UI as Gradio UI
    participant Agent as FeaturePromptWriterAgent
    participant Registry as MetricsRegistry
    participant Templates as PromptTemplates
    participant DB as SQLite Database

    User->>UI: 1. Select feature template
    UI->>UI: 2. Load GROUPED_FEATURES config
    UI-->>User: 3. Display form with defaults
    
    User->>UI: 4. Customize & click Generate
    UI->>Agent: 5. generate(FeatureMetadata, language)
    
    Agent->>Agent: 6. Convert to FeatureSpec
    Agent->>Registry: 7. _resolve_metrics(category)
    Registry-->>Agent: 8. metric_definitions
    
    Agent->>Agent: 9. _apply_rai_constraints()
    Agent->>Templates: 10. build_evaluation_prompt()
    Templates-->>Agent: 11. localized prompt
    
    Agent-->>UI: 12. PromptOutput
    UI->>DB: 13. Store in features + templates
    UI-->>User: 14. Display generated prompt
```

---

## 5. Key Architectural Decisions

### 5.1 Stateless Agent Design
The `FeaturePromptWriterAgent` is deliberately **stateless**: no instance variables persist between calls. This enables:
- Consistent output structures
- Easy testing
- No side effects

**Source**: Agent docstring at [agent.py#L21-L30](../src/core/agent.py)

### 5.2 Dual Schema Pattern
Using both Pydantic and dataclasses serves different needs:
- **Pydantic `FeatureMetadata`**: API validation, JSON serialization, complex constraints
- **Dataclass `FeatureSpec`**: Internal processing, minimal overhead, simple fields

**Source**: [schemas.py#L89-L130](../src/core/schemas.py)

### 5.3 RAI-by-Default
Responsible AI metrics are **automatically injected** based on feature properties:
```python
# Always add safety for GenAI features
if "safety" not in metrics_used:
    metrics_used = metrics_used + ["safety"]
    rai_checks.append("safety_check_added")
```

**Source**: [agent.py#L136-L139](../src/core/agent.py)

### 5.4 i18n-First Metrics
All metrics support multiple languages from definition:
```python
"faithfulness": MetricDefinition(
    definitions={
        "en": "Output must not introduce information...",
        "zh-Hans": "输出不得引入输入中不存在的信息...",
        "ja": "入力にない情報を付け加えない...",
    }
)
```

**Source**: [metrics_registry.py#L31-L45](../src/core/metrics_registry.py)

---

## 6. Extension Points

| Extension Point | Location | How to Extend |
|-----------------|----------|---------------|
| **New Metrics** | `metrics_registry.py` | Add `MetricDefinition` to `METRICS_REGISTRY` |
| **New Categories** | `metrics_registry.py` | Update `DEFAULT_METRICS_BY_CATEGORY` |
| **New Languages** | `metrics_registry.py`, `prompt_templates.py` | Add language code to `definitions` dicts |
| **New Feature Templates** | `app.py` | Add entry to `GROUPED_FEATURES` dict |
| **Custom LLM Provider** | `llm_client.py` | Extend `LLMClient._initialize_client()` |

---

## 7. Dependencies

```
Core Dependencies (from requirements.txt):
├── gradio>=4.0.0        # Web UI framework
├── pydantic>=2.0.0      # Data validation
├── openai>=1.0.0        # LLM client SDK
└── python-dotenv>=1.0.0 # Environment management
```

---

## 8. Quick Start for Developers

### Running the Application
```bash
# 1. Activate environment
conda activate metafeature

# 2. Set environment variables (.env file)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=your-api-key
DEPLOYMENT_NAME=gpt-4o

# 3. Run
python run.py
# Opens at http://127.0.0.1:7860
```

### Programmatic Usage
```python
from src.core import FeaturePromptWriterAgent, FeatureMetadata, ResponsibleAIConstraints

# Create feature metadata
metadata = FeatureMetadata(
    feature_name="Email Auto-Reply",
    feature_description="Generate helpful replies to customer emails",
    category="auto_reply",
    success_metrics=["relevance", "tone", "safety"],
    responsible_ai=ResponsibleAIConstraints(
        no_pii_leakage=True,
        safety_critical=True
    )
)

# Generate evaluation prompt
agent = FeaturePromptWriterAgent()
result = agent.generate(metadata, language="en")

print(result.evaluation_prompt)
print(f"RAI checks applied: {result.rai_checks_applied}")
```

---

## 9. Related Documentation

- For metrics details, see: *[Metrics Reference]* (to be created)
- For prompt template structure, see: *[Prompt Engineering Guide]* (to be created)
- For RAI constraints, see: *[Responsible AI Guidelines]* (to be created)

---

*This document was generated by analyzing the actual source code of the MetaFeature-Orchestrator repository. All insights are backed by specific file and line references.*
