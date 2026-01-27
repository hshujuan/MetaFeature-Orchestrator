# MetaFeature-Orchestrator

🎯 **Automatically generate high-quality evaluation prompts for GenAI features with metric-first, grounded, RAI-by-design principles.**

[![Version](https://img.shields.io/badge/version-2.2-blue.svg)](https://github.com/yourusername/MetaFeature-Orchestrator)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

MetaFeature-Orchestrator is an intelligent evaluation prompt generator that creates comprehensive, structured evaluation prompts for AI features. It uses a **metric-first approach** with built-in **Responsible AI (RAI) checks** and combines LLM-based evaluation with deterministic code-based metrics.

### What's New in v2.2

- 🏗️ **Architecture-Aware Evaluation**: Automatic detection and specialized metrics for Pipeline, RAG, Agentic, and Multimodal systems
- 📋 **Feature-Specific Rubrics**: Generated prompts include tailored 5-point scoring rubrics for each metric
- ⚠️ **Failure Modes & Edge Cases**: Architecture-specific failure patterns included in evaluation prompts
- 🔧 **8 AI Tools**: Streamlined toolset with enhanced `build_prompt` and `recommend_metrics`
- 🎯 **Complex System Metrics**: New metrics for multi-model pipelines (`stage_handoff_quality`, `retrieval_relevance`, `tool_selection_accuracy`, `cross_modal_alignment`)
- 📊 **18,000+ char prompts**: Comprehensive evaluation prompts with full rubrics (vs ~2,000 in v2.1)
- ✅ **Additive Metric Policy**: Human-verified metrics are mandatory; AI can only ADD metrics, not remove them, with clear summaries of additions

### What's in v2.1

- ⚖️ **Side-by-Side Comparison Mode**: Compare Template vs AI Agent outputs in real-time
- 🚀 **Progressive Loading**: Template output shows immediately while AI Agent generates
- 📊 **Intelligent Metric Recommendations**: `recommend_metrics()` analyzes features and explains why each metric matters
- 🌍 **20+ BCP 47 Locales**: Full cultural context, tone guidance, and privacy frameworks

### Design Principles

- **Metric-first**: Define evaluation criteria before generation
- **Grounded**: Clear rubrics and thresholds guide evaluation
- **Dual-mode**: Template (deterministic) + AI Agent (adaptive) modes
- **RAI by Design**: Responsible AI checks built-in (safety, privacy, fairness, transparency)
- **Human-reviewable**: All outputs are transparent and auditable
- **Locale-aware**: BCP 47 locale support with culture-specific tone, formality, and privacy frameworks
- **Contract-based**: Versioned prompts serve as auditable evaluation contracts

## Features

- 📊 **20+ Built-in Metrics**: faithfulness, coverage, relevance, tone, fluency, brevity, safety, privacy, groundedness, format_compliance, accuracy, coherence, creativity, prompt_adherence, visual_accuracy, image_quality, anatomical_correctness, and more
- 🌍 **20+ Locales (BCP 47)**: en-US, en-GB, en-AU, zh-CN, zh-TW, zh-HK, ja-JP, ko-KR, es-ES, es-MX, de-DE, fr-FR, pt-BR, and more
- 🎭 **Culture-Aware Evaluation**: Locale-specific tone (casual vs formal), directness, and regional privacy frameworks (GDPR, CCPA, PIPL, LGPD)
- 🛡️ **Hard FAIL Gates**: Safety, Privacy, Toxicity, Legal constraints that override all other scores
- 🎯 **Second-Order Signals**: Fluency, linguistic naturalness, localization quality, regional compliance, cultural appropriateness
- ⚡ **Quick Start Templates**: Pre-configured features for Summarization, Auto Reply, Translation, Classification, Image Generation, and Personal Assistant
- 🤖 **AI Agent v2.2**: Microsoft Agent Framework with 8 intelligent tools, architecture detection, and feature-specific rubrics
- ⚖️ **Comparison Mode**: Side-by-side Template vs AI Agent output comparison with progressive loading
- 💾 **SQLite Persistence**: Store features, templates, and evaluation runs
- 🎨 **Modern Web UI**: Tabbed Gradio interface with 4 generation modes
- 🖼️ **Image Generation**: DALL-E 3 integration for testing image-related AI features
- 📈 **Code-Based Metrics**: Programmatic evaluation using ROUGE, BLEU, BERTScore, and more

## Project Structure

```
MetaFeature-Orchestrator/
├── src/
│   ├── core/                    # Main application modules
│   │   ├── agent.py             # FeaturePromptWriterAgent - deterministic pipeline (Template Mode)
│   │   ├── ai_agent.py          # MetaFeatureAgent v2.2 - AI-powered agent with 8 tools + architecture detection
│   │   ├── app.py               # Gradio web application with 4 generation modes
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
3. **📊 Quality Metrics**: Select evaluation metrics - use "⭐ Select Recommended" for quick setup or "🤖 AI Recommend" for intelligent analysis with explanations
4. **🛡️ Responsible AI**: Configure RAI constraints (privacy, fairness, safety checks)
5. **🎯 Generation Mode**: Choose mode:
   - **🤖 Auto**: Uses AI Agent for complex features, Template for simple ones
   - **⚡ Always use AI Agent**: Always use AI Agent (requires Azure OpenAI)
   - **📋 Template only**: Always use deterministic Template mode
   - **⚖️ Both (side-by-side comparison)**: Compare Template vs AI Agent outputs - Template shows immediately!
6. **🚀 Generate**: Click "Generate Evaluation Prompt" to create the prompt
7. **🧪 Simulation**: Test your evaluation prompt with real scenarios

### Generation Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **🤖 Auto** | Detects feature complexity, uses AI Agent for complex features | General use |
| **⚡ Always AI** | Forces AI Agent mode with intelligent metric selection | Complex/novel features |
| **📋 Template only** | Fast, deterministic template-based generation | Quick prototypes, known categories |
| **⚖️ Both** | Side-by-side comparison with progressive loading | Understanding differences, documentation |

### Metric Recommendation Options

| Option | Description | Speed |
|--------|-------------|-------|
| **⭐ Select Recommended** | Rule-based category defaults | Instant |
| **🤖 AI Recommend** | Intelligent analysis with explanations | ~100ms |

**AI Recommend** analyzes your feature description to:
- Detect privacy sensitivity ("patient", "personal", "financial" → privacy metric critical)
- Detect safety criticality ("medical", "health", "legal" → groundedness required)
- Recommend metrics with detailed explanations of why each matters

### Evaluation Prompt Structure

Generated prompts follow a **canonical contract** format:

**Template Mode (v2.0)**:
```markdown
# Evaluation Prompt: [Feature Name]
**Version:** 2.0 (Auto-Reply Contract)
**Target Language:** [language]
**Locale:** [locale name]
**Privacy Framework:** [GDPR/CCPA/etc.]
**Generated:** [ISO timestamp]
```

**AI Agent Mode (v2.2)**:
```markdown
# 🤖 AI Agent Evaluation Prompt: [Feature Name]
**Version:** 2.2 (AI Agent Generated - Feature-Specific)
**Generation Mode:** AI Agent with Intelligent Analysis
**Target Language:** [language]
**Locale:** [locale name]
**Privacy Framework:** [GDPR/CCPA/etc.]
**Architecture Type:** [Simple/Pipeline/RAG/Agentic/Multimodal]
**Generated:** [ISO timestamp]
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
)

# Generate evaluation prompt (Template Mode - deterministic)
agent = FeaturePromptWriterAgent()
result = agent.generate(metadata, locale="en-US")  # BCP 47 locale code

print(result.evaluation_prompt)
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

The AI Agent v2.2 provides:
- **Architecture Detection**: Automatically identifies Pipeline, RAG, Agentic, and Multimodal systems
- **Feature-Specific Rubrics**: Tailored 5-point scoring criteria for each metric
- **Failure Mode Analysis**: Architecture-specific edge cases and failure patterns
- Natural language feature understanding
- Intelligent metric selection with explanations
- Cultural and regulatory awareness
- Automatic RAI compliance validation
- Explicit FAIL gates and second-order quality signals
- Reproducible outputs through global prompt capture mechanism

### Template Mode vs AI Agent Mode

| Aspect | Template Mode | AI Agent Mode |
|--------|---------------|---------------|
| **Best For** | Simple, well-defined features | Complex, novel features |
| **Speed** | Instant (no API calls) | Requires LLM calls |
| **Reproducibility** | 100% deterministic | High (structured tool output) |
| **Adaptability** | Fixed templates | Dynamic reasoning |
| **Cost** | Free | API costs |
| **Requirement** | None | Azure OpenAI + Agent Framework |
| **Version** | v2.0 | v2.2 |

**Recommendation**: Use **⚖️ Both** mode first to compare outputs, then choose the best for your use case.

## AI Agent Tools (v2.2)

The AI Agent has access to 8 intelligent tools:

| Tool | Description |
|------|-------------|
| `lookup_metrics` | Find available metrics for a category |
| `suggest_metrics` | Get additional metric suggestions based on context |
| `recommend_metrics` | **Intelligent** metric recommendation with architecture detection (Pipeline/RAG/Agentic/Multimodal) |
| `get_locale_info` | Get cultural, regulatory, and formatting info for a locale |
| `validate_rai_compliance` | Check if metrics meet RAI requirements |
| `build_prompt` | Generate comprehensive evaluation prompt (18,000+ chars) with additive metric policy |
| `get_code_metrics` | Get programmatic metric code samples |
| `analyze_feature_description` | Extract attributes from natural language descriptions |

### Architecture-Specific Metrics (v2.2)

| Architecture | Additional Metrics |
|--------------|-------------------|
| **Pipeline** | `stage_handoff_quality`, `error_propagation_resistance`, `end_to_end_coherence` |
| **RAG** | `retrieval_relevance`, `retrieval_attribution`, `no_knowledge_leakage` |
| **Agentic** | `tool_selection_accuracy`, `action_safety`, `reasoning_transparency`, `graceful_failure` |
| **Multimodal** | `cross_modal_alignment`, `modality_fidelity`, `information_preservation` |

> **📌 `suggest_metrics` vs `recommend_metrics`**: These two tools serve different purposes:
> 
> | Aspect | `suggest_metrics` | `recommend_metrics` |
> |--------|-------------------|---------------------|
> | **Analysis** | Simple rule-based | Semantic analysis of feature description |
> | **Input** | Category + current metrics | Feature name + description + category |
> | **Output** | Flat list of suggestions | Prioritized tiers (Mandatory → Critical → Important) |
> | **Explanations** | Generic reason | Detailed per-metric explanations |
> | **Use Case** | Quick suggestions | Comprehensive metric planning |

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
| `cultural_appropriateness` | Respects cultural norms | translation, all |
| `format_compliance` | Matches required output format | all |

### Image Metrics

| Metric | Description | Categories |
|--------|-------------|------------|
| `visual_accuracy` | Image matches prompt description | image_generation |
| `style_consistency` | Adheres to requested artistic style | image_generation |
| `image_quality` | Free of artifacts, proper resolution | image_generation |
| `anatomical_correctness` | Correct anatomy for humans/animals | image_generation |
| `image_safety` | No harmful/inappropriate content | image_generation |

### Code-Based Metrics

| Metric | Package | Use Case |
|--------|---------|----------|
| ROUGE | `rouge-score` | Summarization (n-gram overlap) |
| BLEU | `sacrebleu` | Translation (n-gram precision) |
| BERTScore | `bert-score` | Semantic similarity |
| Readability | `textstat` | Fluency (Flesch score, grade level) |
| Fuzzy Match | `rapidfuzz` | Approximate string matching |

## Supported Locales

| Locale | Language | Privacy Framework |
|--------|----------|-------------------|
| en-US | English (US) | CCPA |
| en-GB | English (UK) | UK GDPR |
| en-AU | English (Australia) | Australian Privacy Act |
| zh-CN | Chinese (Simplified) | PIPL |
| zh-TW | Chinese (Traditional) | PDPA |
| ja-JP | Japanese | APPI |
| ko-KR | Korean | PIPA |
| de-DE | German | GDPR |
| fr-FR | French | GDPR |
| es-ES | Spanish (Spain) | GDPR |
| es-MX | Spanish (Mexico) | LFPDPPP |
| pt-BR | Portuguese (Brazil) | LGPD |
| ... | ... | ... |

### 📋 Privacy Framework Reference

The system automatically applies the appropriate privacy framework based on target locale:

| Framework | Full Name | Region | Key Requirements |
|-----------|-----------|--------|------------------|
| **GDPR** | General Data Protection Regulation | 🇪🇺 European Union | Consent required, right to erasure, data portability, 72-hour breach notification |
| **CCPA** | California Consumer Privacy Act | 🇺🇸 California, USA | Right to know, delete, opt-out of sale; applies to businesses serving CA residents |
| **PIPL** | Personal Information Protection Law | 🇨🇳 China | Strict consent requirements, data localization, cross-border transfer restrictions |
| **LGPD** | Lei Geral de Proteção de Dados | 🇧🇷 Brazil | Consent-based, applies to any processing of Brazilian residents' data |
| **UK GDPR** | UK General Data Protection Regulation | 🇬🇧 United Kingdom | Post-Brexit GDPR equivalent with UK-specific provisions |
| **APPI** | Act on Protection of Personal Information | 🇯🇵 Japan | Business operator obligations, cross-border transfer rules |
| **PIPA** | Personal Information Protection Act | 🇰🇷 South Korea | One of the strictest; explicit consent, data breach notification |

When generating evaluation prompts for features that handle personal data, the system includes the relevant privacy framework to ensure evaluation criteria account for regional compliance requirements.

## Dependencies

### Core
- `gradio>=4.0.0` - Web UI framework
- `pydantic>=2.0.0` - Data validation
- `openai>=1.0.0` - OpenAI/Azure OpenAI client
- `python-dotenv>=1.0.0` - Environment variable management
- `httpx>=0.25.0` - HTTP client for image generation

### AI Agent (Optional)
- `agent-framework` - Microsoft Agent Framework for AI Agent mode

### Code-Based Evaluation (Optional)
- `rouge-score>=0.1.2` - ROUGE metrics for summarization
- `sacrebleu>=2.0.0` - BLEU, chrF, TER for translation
- `bert-score>=0.3.0` - Semantic similarity using BERT
- `textstat>=0.7.0` - Readability metrics
- `rapidfuzz>=3.0.0` - Fuzzy string matching

## Contributing

Contributions are welcome! Please see our contribution guidelines for details.

## License

See [LICENSE](LICENSE) for details.

---

**Version 2.2** | January 2026 | Built with ❤️ for the GenAI evaluation community
