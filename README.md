# MetaFeature-Orchestrator

Dynamically synthesize instructions for different feature domains through agentic prompting.

## Overview

MetaFeature-Orchestrator is an AI-powered evaluation prompt generator that creates high-quality, hallucination-free evaluation prompts for GenAI features. It leverages an agentic framework to automatically generate localized, metric-driven prompts for various feature scenarios.

## Features

- **Automated Prompt Generation**: Generate evaluation prompts for different feature types (Auto Reply Email, Auto Reply Message, Auto Summarize News)
- **Localization Support**: Adapt prompts to 230+ locations and languages
- **Responsible AI**: Built-in checks for bias, toxicity, and privacy
- **Metric-Driven Evaluation**: Includes specific metrics (Rouge, Tone-consistency, etc.)
- **Interactive Web UI**: Gradio-based interface for easy interaction

## Project Structure

```
MetaFeature-Orchestrator/
├── src/
│   ├── app.py          # Main application with Gradio UI
│   └── schemas.py      # Pydantic models for feature metadata
├── requirements.txt    # Python dependencies
├── LICENSE
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MetaFeature-Orchestrator.git
   cd MetaFeature-Orchestrator
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n metafeature python=3.11 -y
   conda activate metafeature
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

## Usage

Run the Gradio web application:

```bash
python src/app.py
```

Then open your browser to the displayed URL (typically `http://127.0.0.1:7860`).

### Web Interface

1. Select a **Feature Scenario** (Auto Reply Email, Auto Reply Message, or Auto Summarize News)
2. Enter the **Location** (e.g., "Tokyo, Japan")
3. Specify the **Language**
4. Provide an **Input Data Sample** (the raw email/news/text to evaluate)
5. Click **Generate Evaluation Prompt**

## Dependencies

- `agent-framework` - Agentic AI framework
- `gradio` - Web UI framework
- `pydantic` - Data validation
- `openai` - OpenAI API client

## License

See [LICENSE](LICENSE) for details.
