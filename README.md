# MetaFeature-Orchestrator

🎯 **Automatically generate high-quality evaluation prompts for GenAI features with metric-first, grounded, RAI-by-design principles.**

[![Version](https://img.shields.io/badge/version-2.0-blue.svg)](https://github.com/yourusername/MetaFeature-Orchestrator)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

MetaFeature-Orchestrator is an intelligent evaluation prompt generator that creates comprehensive, structured evaluation prompts for AI features. It uses a **metric-first approach** with built-in **Responsible AI (RAI) checks** and combines LLM-based evaluation with deterministic code-based metrics.

### What's New in v2.0

- 🤖 **AI Agent Mode**: Microsoft Agent Framework integration with 7 intelligent tools
- 📋 **Canonical Contracts**: Version 2.0 prompts with hard FAIL gates and reproducibility
- 🎯 **Second-Order Quality Signals**: Fluency, linguistic naturalness, localization quality, regional compliance
- 🔒 **Explicit FAIL Gates**: Safety, Privacy, Toxicity, Legal gates with automatic FAIL override
- 🌍 **Enhanced Locale Support**: 20+ BCP 47 locales with cultural context and privacy frameworks

### Design Principles

- **Metric-first**: Define evaluation criteria before generation
- **Grounded**: Clear rubrics and thresholds guide evaluation
- **Dual-mode**: Template (deterministic) + AI Agent (adaptive) modes
- **RAI by Design**: Responsible AI checks built-in (safety, privacy, fairness, transparency)
- **Human-reviewable**: All outputs are transparent and auditable
- **Locale-aware**: BCP 47 locale support with culture-specific tone, formality, and privacy frameworks
- **Contract-based**: Version 2.0 prompts serve as auditable evaluation contracts

## Features

- 📊 **20+ Built-in Metrics**: faithfulness, coverage, relevance, tone, fluency, brevity, safety, privacy, groundedness, format_compliance, accuracy, coherence, creativity, prompt_adherence, visual_accuracy, image_quality, anatomical_correctness, and more
- 🌍 **20+ Locales (BCP 47)**: en-US, en-GB, en-AU, zh-CN, zh-TW, zh-HK, ja-JP, ko-KR, es-ES, es-MX, de-DE, fr-FR, pt-BR, and more
- 🎭 **Culture-Aware Evaluation**: Locale-specific tone (casual vs formal), directness, and regional privacy frameworks (GDPR, CCPA, PIPL, LGPD)
- 🛡️ **Hard FAIL Gates**: Safety, Privacy, Toxicity, Legal constraints that override all other scores
- 🎯 **Second-Order Signals**: Fluency, linguistic naturalness, localization quality, regional compliance, cultural appropriateness
- ⚡ **Quick Start Templates**: Pre-configured features for Summarization, Auto Reply, Translation, Classification, Image Generation, and Personal Assistant
- 🤖 **AI Agent Mode**: Microsoft Agent Framework with dynamic metric selection and natural language understanding
- 💾 **SQLite Persistence**: Store features, templates, and evaluation runs
- 🎨 **Modern Web UI**: Tabbed Gradio interface with AI Agent toggle (Auto/Always/Never)
- 🖼️ **Image Generation**: DALL-E 3 integration for testing image-related AI features
- 📈 **Code-Based Metrics**: Programmatic evaluation using ROUGE, BLEU, BERTScore, and more

## Project Structure

```
MetaFeature-Orchestrator/
├── src/
│   ├── core/                    # Main application modules
│   │   ├── agent.py             # FeaturePromptWriterAgent - deterministic pipeline (Template Mode)
│   │   ├── ai_agent.py          # MetaFeatureAgent v2.0 - AI-powered agent (AI Agent Mode)
│   │   ├── agent_tools.py       # 11 tools for the AI agent
│   │   ├── workflows.py         # Multi-step workflows with human-in-the-loop
│   │   ├── app.py               # Gradio web application with mode selection
│   │   ├── code_metrics.py      # Programmatic metrics (ROUGE, BLEU, BERTScore)
│   │   ├── database.py          # SQLite persistence
│   │   ├── image_generator.py   # DALL-E 3 image generation
│   │   ├── llm_client.py        # Azure OpenAI / OpenAI client
│   │   ├── metrics_registry.py  # 20+ metrics with i18n support
│   │   ├── prompt_templates.py  # v2.0 Category-specific templates with hard gates
│   │   ├── schemas.py           # Pydantic models and dataclasses
│   │   └── __init__.py
│   ├── data/                    # SQLite database
│   └── __init__.py
├── TechnicalDocs/               # Architecture and technical documentation
│   ├── ArchitectureOverview.md  # Comprehensive architecture documentation
│   ├── AgentDesign.md           # Agent system deep dive
│   └── DocumentationIndex.md    # Documentation reference
├── tests/
│   └── test_openai.py           # Connection tests
├── run.py                       # Entry point
├── requirements.txt
├── .env                         # Environment variables (not in git)
├── .env.example                 # Example environment config
├── LICENSE
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/MetaFeature-Orchestrator.git
   cd MetaFeature-Orchestrator
   ```

2. **Create and activate a conda environment:**
   ```bash
   conda create -n metafeature python=3.11 -y
   conda activate metafeature
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   
   Copy `.env.example` to `.env` and set your API credentials:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env`:
   ```env
   # For Azure OpenAI (recommended)
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   AZURE_OPENAI_API_KEY=your-api-key
   DEPLOYMENT_NAME=gpt-4o
   
   # Or for standard OpenAI
   OPENAI_API_KEY=your-api-key
   ```

## Usage

### Run the Web Application

```bash
python run.py
```

Open your browser to **http://127.0.0.1:7860**

### Web Interface Workflow

1. **⚡ Quick Start**: Select a predefined template (Auto Reply, Summarization, etc.) and click "Load Template"
2. **📝 Feature Definition**: Review/edit the feature name, description, and I/O formats
3. **📊 Quality Metrics**: Select evaluation metrics (relevance, fluency, safety, etc.)
4. **🛡️ Responsible AI**: Configure RAI constraints (privacy, fairness, safety checks)
5. **🤖 AI Agent Mode**: Choose mode:
   - **Auto**: Uses AI Agent for complex features, Template for simple ones
   - **Always**: Always use AI Agent (requires Azure OpenAI)
   - **Never**: Always use deterministic Template mode
6. **🚀 Generate**: Click "Generate Evaluation Prompt" to create the v2.0 prompt
7. **🧪 Simulation**: Test your evaluation prompt with real scenarios

### Evaluation Prompt v2.0 Structure

Generated prompts now follow a **canonical contract** format:

```markdown
# Evaluation Prompt: [Feature Name]
**Version:** 2.0 (Canonical Contract)
**Target Language:** [language]
**Locale:** [locale name]
**Privacy Framework:** [GDPR/CCPA/etc.]

## 1. EVALUATOR ROLE & TASK
[Role definition and task description]

## 2. HARD GATES (Automatic FAIL)
| Gate | Condition | Check |
|------|-----------|-------|
| SAFETY | Contains harmful content | [ ] PASS / [ ] FAIL |
| PRIVACY | Leaks PII | [ ] PASS / [ ] FAIL |
| ... | ... | ... |

## 3. PRIMARY METRICS
[Metric definitions with weights and scoring criteria]

## 4. SECOND-ORDER QUALITY SIGNALS
| Signal | Weight | Assessment |
|--------|--------|------------|
| Fluency | 0.7 | Natural, grammatically correct |
| Linguistic Naturalness | 0.8 | Reads as native speaker |
| ... | ... | ... |

## 5. EVALUATION PROTOCOL
[Step-by-step evaluation process]

## 6. RESPONSIBLE AI CHECKLIST
[RAI checks specific to locale]

## 7. OUTPUT FORMAT (Structured JSON)
[JSON schema with gates, scores, and recommendation]
```

### Programmatic Usage

```python
from src.core import (
    FeaturePromptWriterAgent,
    FeatureMetadata,
    QualityMetric,
    ResponsibleAIConstraints
)

# Create feature metadata
metadata = FeatureMetadata(
    feature_name="Email Auto-Reply",
    feature_description="Generate helpful replies to customer emails",
    category="auto_reply",
    quality_metrics=[
        QualityMetric(name="relevance", description="Reply addresses the query"),
        QualityMetric(name="tone", description="Professional and friendly"),
    ],
    responsible_ai=ResponsibleAIConstraints(
        no_pii_leakage=True,
        toxicity_check_required=True
)

# Generate evaluation prompt (Template Mode - deterministic)
agent = FeaturePromptWriterAgent()
result = agent.generate(metadata, locale="en-US")  # BCP 47 locale code

print(result.evaluation_prompt)
print(f"Privacy framework: {result.privacy_framework}")  # e.g., CCPA, GDPR
```

### AI Agent Mode (Advanced)

For complex features, use the AI-powered agent built on **Microsoft Agent Framework**:

```bash
# Install Agent Framework (optional, for AI Agent mode)
pip install agent-framework --pre
```

```python
from src.core.ai_agent import MetaFeatureAgent

# Natural language request - the agent figures out the rest
agent = MetaFeatureAgent()
response = agent.chat(
    "I need an evaluation prompt for a medical document summarizer "
    "that will be used by doctors in Germany. It needs to be very "
    "careful about accuracy and patient privacy."
)

print(response.evaluation_prompt)
```

The AI Agent v2.0 can:
- Understand natural language feature descriptions
- Dynamically select appropriate metrics based on feature analysis
- Handle complex multi-locale requirements with cultural awareness
- Validate RAI compliance automatically
- Generate prompts with explicit FAIL gates and second-order quality signals
- Ensure reproducibility through contract-based evaluation structure

### Template Mode vs AI Agent Mode

| Aspect | Template Mode | AI Agent Mode |
|--------|---------------|---------------|
| **Best For** | Simple, well-defined features | Complex, novel features |
| **Speed** | Instant (no API calls) | Requires LLM calls |
| **Reproducibility** | 100% deterministic | High (structured prompts) |
| **Adaptability** | Fixed templates | Dynamic reasoning |
| **Cost** | Free | API costs |
| **Requirement** | None | Azure OpenAI + Agent Framework |

**Recommendation**: Use Template Mode as the **canonical contract**, AI Agent Mode for **adaptive execution**.

### Workflows for Complex Features

For multi-locale or safety-critical features:

```python
from src.core.workflows import WorkflowRunner

runner = WorkflowRunner()
result = runner.run(
    feature_name="Medical Document Summarizer",
    feature_description="Summarize patient records for physicians",
    target_locales=["de-DE", "ja-JP", "en-US"],
    safety_critical=True
)

# Get prompts for each locale with appropriate privacy frameworks
for locale, prompt in result.prompts.items():
    print(f"--- {locale} ---")
    print(prompt[:200])
```

## Evaluation Prompt Templates

### Category-Specific Templates (v2.0)

| Template | Hard Gates | Key Signals |
|----------|------------|-------------|
| **Generic** | Safety, Privacy, Toxicity, Legal | Fluency, linguistic naturalness, localization |
| **Summarization** | Hallucination, Safety, Privacy, Factual | Conciseness, structure, completeness |
| **Translation** | Meaning, Offensive, Safety, Legal | Register match, cultural adaptation |
| **Auto-Reply** | Relevance, Safety, Privacy, Tone | Message understanding, actionability |
| **Personal Assistant** | Privacy, Consent, Safety, Medical, Financial | Personalization quality, intrusiveness |

### Output JSON Structure (v2.0)

All templates produce structured JSON with:

```json
{
  "evaluation_id": "<uuid>",
  "feature": "<name>",
  "locale": "<BCP47>",
  "timestamp": "<ISO8601>",
  
  "gates": {
    "safety": "PASS|FAIL",
    "privacy": "PASS|FAIL",
    "gate_failures": []
  },
  
  "primary_scores": {
    "<metric>": {
      "score": 1-5,
      "weight": 0.0-1.0,
      "rationale": "<evidence>",
      "examples": ["<quoted text>"]
    }
  },
  
  "secondary_scores": {
    "fluency": 1-5,
    "linguistic_naturalness": 1-5,
    "localization_quality": 1-5,
    "regional_compliance": 1-5,
    "cultural_appropriateness": 1-5
  },
  
  "overall_score": "<weighted_average>",
  "recommendation": "PASS|FAIL|REVIEW",
  "confidence": "HIGH|MEDIUM|LOW"
}
```

## Available Metrics

### Text Metrics

| Metric | Description | Categories |
|--------|-------------|------------|
| `faithfulness` | No hallucination - output matches input facts | summarization, translation, auto_reply |
| `coverage` | All key points captured | summarization |
| `relevance` | Addresses user intent | auto_reply, assistant, summarization |
| `tone` | Appropriate politeness and formality | auto_reply, assistant |
| `fluency` | Grammatically correct, natural | all |
| `brevity` | Concise without losing info | auto_reply, summarization |
| `safety` | No harmful/toxic content | all |
| `privacy` | No PII leakage | all |
| `groundedness` | Based only on provided context | assistant, summarization |
| `accuracy` | Factually correct | translation, classification |
| `coherence` | Logical flow | content_generation |
| `creativity` | Novel and engaging | content_generation |
| `prompt_adherence` | Follows instructions precisely | content_generation, assistant |

### Image Metrics

| Metric | Description | Categories |
|--------|-------------|------------|
| `visual_accuracy` | Image matches prompt description | image_generation, image_editing |
| `style_consistency` | Adheres to requested artistic style | image_generation |
| `image_quality` | Free of artifacts, proper resolution | image_generation, image_editing |
| `anatomical_correctness` | Correct anatomy for humans/animals | image_generation |
| `object_removal_seamless` | No visible traces after removal | image_editing |
| `caption_accuracy` | Caption describes image correctly | image_understanding |
| `image_safety` | No harmful/inappropriate content | image_generation, image_editing |

### Second-Order Quality Signals (v2.0)

| Signal | Description | Weight |
|--------|-------------|--------|
| `fluency` | Natural, grammatically correct | 0.7-0.9 |
| `linguistic_naturalness` | Reads as native speaker would write | 0.8-0.9 |
| `localization_quality` | Proper locale conventions (dates, numbers) | 0.9-1.0 |
| `regional_compliance` | Meets local regulatory requirements | 0.9-1.0 |
| `cultural_appropriateness` | Respects cultural norms, avoids taboos | 0.9-1.0 |

### Code-Based Metrics

| Metric | Package | Use Case |
|--------|---------|----------|
| ROUGE | `rouge-score` | Summarization (n-gram overlap) |
| BLEU | `sacrebleu` | Translation (n-gram precision) |
| BERTScore | `bert-score` | Semantic similarity |
| Readability | `textstat` | Fluency (Flesch score, grade level) |
| Fuzzy Match | `rapidfuzz` | Approximate string matching |

## Dependencies

### Core
- `gradio>=4.0.0` - Web UI framework
- `pydantic>=2.0.0` - Data validation
- `openai>=1.0.0` - OpenAI/Azure OpenAI client
- `python-dotenv>=1.0.0` - Environment variable management
- `httpx>=0.25.0` - HTTP client for image generation

### AI Agent (Optional)
- `agent-framework` - Microsoft Agent Framework for AI Agent mode

### Code-Based Evaluation
- `rouge-score>=0.1.2` - ROUGE metrics for summarization
- `sacrebleu>=2.0.0` - BLEU, chrF, TER for translation
- `bert-score>=0.3.0` - Semantic similarity using BERT
- `textstat>=0.7.0` - Readability metrics
- `rapidfuzz>=3.0.0` - Fuzzy string matching
- `evaluate>=0.4.0` - HuggingFace unified metrics API

## Contributing

Contributions are welcome! Please see our contribution guidelines for details.

## License

See [LICENSE](LICENSE) for details.

---

**Version 2.0** | January 2026 | Built with ❤️ for the GenAI evaluation community
