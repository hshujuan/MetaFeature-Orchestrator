# Documentation Index

This document provides a reference to all technical documentation in this codebase. Use this index to find relevant documentation without loading unnecessary context.

## Documents

| Document | Description | Project/Scope |
|----------|-------------|---------------|
| [Architecture Overview](ArchitectureOverview.md) | **Comprehensive** architecture documentation covering: (1) Entry points & bootstrapping, (2) Agent system with orchestration logic, (3) Schema layer with dual Pydantic/dataclass pattern, (4) Metrics registry with 14+ metrics & i18n, (4.1) Code-based metrics (ROUGE, BLEU, BERTScore, etc.), (5) LLM client with singleton pattern, (5.1) Image generator with DALL-E 3, (6) Database layer with ER diagram, (7) Web UI with predefined templates, (8) Dynamic evaluation prompt generation with bilingual support. Includes mermaid diagrams and code references. | Global Concept |
| [Agent Design](AgentDesign.md) | **Deep dive** into the `FeaturePromptWriterAgent`: (1) Architecture classification (agentic workflow vs autonomous agent), (2) Pipeline flow diagram, (3) RAI constraint injection rules, (4) Prompt templates & localization, (5) Where LLM is/isn't called, (6) Usage examples, (7) Design decisions & future enhancements. | Agent Component |

---
*Last updated: January 25, 2026*
