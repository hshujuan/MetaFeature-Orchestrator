import os
import json
from typing import Annotated, Optional, List
from pydantic import BaseModel, Field

# Microsoft Agent Framework (Python)
# Install: pip install agent-framework-core --pre  :contentReference[oaicite:2]{index=2}
from agent_framework.azure import AzureOpenAIResponsesClient  # or AzureOpenAIChatClient
from azure.identity import AzureCliCredential

from core_agent import FeaturePromptWriterAgent, FeatureMetadata
from db import FeatureStore, PromptTemplateStore, RunStore

core = FeaturePromptWriterAgent()
feature_store = FeatureStore()
template_store = PromptTemplateStore()
run_store = RunStore()


class GeneratePromptRequest(BaseModel):
    feature_id: str = Field(description="DB id of the feature to evaluate")
    language: Optional[str] = Field(default=None, description="Target language code like en, fr, zh-Hans")
    metrics: Optional[List[str]] = Field(default=None, description="Optional override metrics list")


class GeneratePromptResponse(BaseModel):
    feature_id: str
    feature_name: str
    language: str
    metrics_used: List[str]
    evaluation_prompt: str


def generate_evaluation_prompt(
    req_json: Annotated[str, Field(description="JSON string matching GeneratePromptRequest schema")]
) -> str:
    """Tool: Generate a metric-driven evaluation prompt for a feature."""
    req = GeneratePromptRequest.model_validate_json(req_json)

    feat = feature_store.get_feature(req.feature_id)
    if feat is None:
        return json.dumps({"error": f"Feature not found: {req.feature_id}"})

    feature = FeatureMetadata(**feat["metadata"])
    if req.metrics:
        feature.success_metrics = req.metrics

    out = core.generate(feature, language=req.language)

    # Persist the generated prompt as a “template version” (optional)
    template_id = template_store.upsert_template(
        feature_id=req.feature_id,
        language=out.language,
        category=out.category,
        metrics=out.metrics_used,
        prompt=out.evaluation_prompt,
        source="agent_generated_v1"
    )

    run_store.log_run(
        feature_id=req.feature_id,
        template_id=template_id,
        language=out.language,
        metrics=out.metrics_used,
        output_prompt=out.evaluation_prompt
    )

    resp = GeneratePromptResponse(
        feature_id=req.feature_id,
        feature_name=out.feature_name,
        language=out.language,
        metrics_used=out.metrics_used,
        evaluation_prompt=out.evaluation_prompt,
    )
    return resp.model_dump_json()


async def build_agent():
    """
    Creates an Agent Framework agent that can call the generate_evaluation_prompt tool.
    """
    credential = AzureCliCredential()
    # Uses Azure OpenAI Responses agent creation per docs :contentReference[oaicite:3]{index=3}
    client = AzureOpenAIResponsesClient(credential=credential)

    instructions = (
        "You are a Feature Evaluation Prompt Writer. "
        "When asked to create an evaluation prompt, call the tool `generate_evaluation_prompt` "
        "with the appropriate JSON payload."
    )

    agent = client.as_agent(name="FeaturePromptWriter", instructions=instructions)

    # In Agent Framework Python, tools can be passed via tools= in run() :contentReference[oaicite:4]{index=4}
    return agent


async def run_once(user_query: str):
    agent = await build_agent()
    # Provide the tool at run time
    result = await agent.run(user_query, tools=[generate_evaluation_prompt])
    return result.text
