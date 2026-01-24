# Archive

This folder contains deprecated/redundant files that have been superseded by the unified `src/core/` module.

## Archived Files

| File | Superseded By | Reason |
|------|---------------|--------|
| `src/agent.py` | `src/core/agent.py` | Old agent version |
| `src/app.py` | `src/core/app.py` | Old Gradio UI |
| `src/app2.py` | `src/core/app.py` | Merged into core/app.py |
| `src/db.py` | `src/core/database.py` | Old SQLite wrapper |
| `src/llm_clients.py` | `src/core/llm_client.py` | Old Azure OpenAI client |
| `src/metrics_registry.py` | `src/core/metrics_registry.py` | Partial i18n metrics |
| `src/suggested_metrics.py` | `src/core/metrics_registry.py` | Merged into metrics registry |
| `src/schemas.py` | `src/core/schemas.py` | Old Pydantic models |
| `src/prompt_templates.py` | `src/core/prompt_templates.py` | Duplicate content |
| `src/generator.py` | `src/core/agent.py` | Generation logic merged |
| `src/maf_runtime.py` | N/A | Used non-existent agent-framework package |
| `src/openai_service.py` | N/A | Image generation (unused) |
| `src/optimizer.py` | N/A | Used deprecated call_responses API |
| `src/tools/` | `src/core/` | Utilities merged into core |
| `src/prompt_store/` | `src/core/prompt_templates.py` | Prompts merged into core |

## Date Archived

January 24, 2026

## Note

These files are kept for reference only. The active codebase is in `src/core/`.
