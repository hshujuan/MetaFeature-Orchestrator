# MetaFeature-Orchestrator

🎯 **Automatically generate high-quality evaluation prompts for GenAI features with metric-first, grounded, RAI-by-design principles.**

## Overview

MetaFeature-Orchestrator is an intelligent evaluation prompt generator that creates comprehensive, hallucination-free evaluation prompts for AI features. It uses a metric-first approach with built-in Responsible AI checks to ensure consistent, reproducible evaluations.

### Design Principles

- **Metric-first**: Define evaluation criteria before generation
- **Grounded**: No hallucinated judgments - clear rubrics and thresholds
- **Agent-based**: Intelligent prompt synthesis, not hard-coded templates
- **RAI by Design**: Responsible AI checks built-in (safety, privacy, fairness, transparency)
- **Human-reviewable**: All outputs are transparent and auditable
- **Extensible**: i18n support for 5 languages, customizable metrics

## Features

- 📊 **14 Built-in Metrics**: faithfulness, coverage, relevance, tone, fluency, brevity, safety, privacy, groundedness, format_compliance, accuracy, coherence, creativity, prompt_adherence
- 🌍 **Multi-language Support**: English, Chinese (Simplified), Japanese, Spanish, French
- 🛡️ **RAI Checks**: Safety, privacy, fairness, transparency constraints
- ⚡ **Quick Start Templates**: Pre-configured features for Summarization, Auto Reply, Translation, Classification
- 💾 **SQLite Persistence**: Store features, templates, and evaluation runs
- 🎨 **Modern Web UI**: Tabbed Gradio interface with real-time feedback

## Project Structure

```
MetaFeature-Orchestrator/
├── src/
│   ├── core/                    # Main application modules
│   │   ├── agent.py             # FeaturePromptWriterAgent
│   │   ├── app.py               # Gradio web application
│   │   ├── database.py          # SQLite persistence
│   │   ├── llm_client.py        # Azure OpenAI / OpenAI client
│   │   ├── metrics_registry.py  # 14 metrics with i18n
│   │   ├── prompt_templates.py  # Category-specific templates
│   │   ├── schemas.py           # Pydantic models
│   │   └── __init__.py
│   ├── data/                    # SQLite database
│   └── __init__.py
├── archive/                     # Deprecated files (for reference)
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
   # For Azure OpenAI
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/openai/v1/
   AZURE_OPENAI_API_KEY=your-api-key
   DEPLOYMENT_NAME=gpt-4o
   
   # Or for standard OpenAI
   OPENAI_API_KEY=your-api-key
   ```

## Usage

### Run the Web Application

```bash
python src/core/app.py
```

Or use the entry point:
```bash
python run.py
```

Open your browser to **http://127.0.0.1:7860**

### Web Interface Workflow

1. **⚡ Quick Start**: Select a predefined template (Auto Reply, Summarization, etc.) and click "Load Template"
2. **📝 Feature Definition**: Review/edit the feature name, description, and I/O formats
3. **📊 Quality Metrics**: Select evaluation metrics (relevance, fluency, safety, etc.)
4. **🛡️ Responsible AI**: Configure RAI constraints (privacy, fairness, safety checks)
5. **🚀 Generate**: Click "Generate Evaluation Prompt" to create the prompt

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

# Generate evaluation prompt
agent = FeaturePromptWriterAgent()
result = agent.generate(metadata, language="en")

print(result.evaluation_prompt)
```

## Available Metrics

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

## Dependencies

- `gradio>=4.0.0` - Web UI framework
- `pydantic>=2.0.0` - Data validation
- `openai>=1.0.0` - OpenAI/Azure OpenAI client
- `python-dotenv>=1.0.0` - Environment variable management

## License

See [LICENSE](LICENSE) for details.
