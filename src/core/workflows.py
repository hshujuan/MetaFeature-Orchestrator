"""
MetaFeature Workflows - Complex multi-step evaluation prompt generation

This module provides workflows for handling complex feature requirements that
need multiple agents or steps to complete, such as:
- Multi-locale features
- Features requiring human approval
- Complex RAI validation
- Batch feature processing
"""
from __future__ import annotations
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from agent_framework import Workflow, WorkflowConfig, Node, Edge
from agent_framework.workflows import (
    SequentialWorkflow,
    ConditionalEdge,
    ParallelNode
)

from .agent_tools import (
    analyze_feature_description,
    lookup_metrics,
    validate_rai_compliance,
    build_prompt,
    get_code_metrics,
    get_locale_info
)
from .schemas import FeatureSpec, PromptOutput
from .prompt_templates import get_privacy_framework

logger = logging.getLogger(__name__)


# =============================================================================
# Workflow State
# =============================================================================

@dataclass
class FeatureWorkflowState:
    """State that flows through the workflow"""
    # Input
    feature_name: str = ""
    feature_description: str = ""
    user_category: Optional[str] = None
    user_metrics: List[str] = field(default_factory=list)
    target_locales: List[str] = field(default_factory=lambda: ["en-US"])
    privacy_sensitive: bool = False
    safety_critical: bool = False
    
    # Analysis results
    detected_category: str = "other"
    detected_privacy_sensitive: bool = False
    detected_safety_critical: bool = False
    
    # Metrics
    resolved_metrics: List[str] = field(default_factory=list)
    metric_definitions: Dict[str, Any] = field(default_factory=dict)
    
    # RAI
    rai_compliant: bool = False
    rai_issues: List[str] = field(default_factory=list)
    rai_recommendations: List[str] = field(default_factory=list)
    
    # Outputs (per locale)
    prompts: Dict[str, str] = field(default_factory=dict)  # locale -> prompt
    code_metrics: Dict[str, str] = field(default_factory=dict)  # locale -> code
    
    # Workflow status
    status: str = "pending"
    errors: List[str] = field(default_factory=list)
    requires_human_review: bool = False
    review_reason: Optional[str] = None


# =============================================================================
# Workflow Nodes (Steps)
# =============================================================================

async def analyze_feature_node(state: FeatureWorkflowState) -> FeatureWorkflowState:
    """
    Step 1: Analyze the feature description to extract key attributes.
    """
    logger.info(f"Analyzing feature: {state.feature_name}")
    
    analysis = analyze_feature_description(state.feature_description)
    
    # Use detected values if not provided by user
    state.detected_category = analysis["suggested_category"]
    state.detected_privacy_sensitive = analysis["is_privacy_sensitive"]
    state.detected_safety_critical = analysis["is_safety_critical"]
    
    # Merge detected locales with user-specified
    if analysis["detected_locales"]:
        state.target_locales = list(set(state.target_locales + analysis["detected_locales"]))
    
    # Use user category if provided, otherwise use detected
    if not state.user_category:
        state.user_category = state.detected_category
    
    # Merge privacy/safety flags (OR logic - if either says sensitive, it is)
    state.privacy_sensitive = state.privacy_sensitive or state.detected_privacy_sensitive
    state.safety_critical = state.safety_critical or state.detected_safety_critical
    
    logger.info(f"Analysis complete: category={state.user_category}, "
                f"privacy={state.privacy_sensitive}, safety={state.safety_critical}")
    
    return state


async def resolve_metrics_node(state: FeatureWorkflowState) -> FeatureWorkflowState:
    """
    Step 2: Resolve which metrics to use based on category and user input.
    """
    logger.info(f"Resolving metrics for category: {state.user_category}")
    
    # Look up metrics for category
    metrics_info = lookup_metrics(state.user_category)
    
    if state.user_metrics:
        # Use user-specified metrics as base
        state.resolved_metrics = state.user_metrics.copy()
    else:
        # Use default metrics
        state.resolved_metrics = metrics_info["default_metrics"]
    
    # Store metric definitions
    state.metric_definitions = metrics_info["metric_details"]
    
    logger.info(f"Resolved {len(state.resolved_metrics)} metrics: {state.resolved_metrics}")
    
    return state


async def validate_rai_node(state: FeatureWorkflowState) -> FeatureWorkflowState:
    """
    Step 3: Validate RAI compliance and add required metrics.
    """
    logger.info("Validating RAI compliance")
    
    # Validate for first locale (they share the same metrics)
    primary_locale = state.target_locales[0] if state.target_locales else "en-US"
    
    validation = validate_rai_compliance(
        metrics=state.resolved_metrics,
        privacy_sensitive=state.privacy_sensitive,
        safety_critical=state.safety_critical,
        locale=primary_locale
    )
    
    state.rai_compliant = validation["is_compliant"]
    state.rai_issues = validation["issues"]
    state.rai_recommendations = validation["recommendations"]
    
    # Auto-add missing required metrics
    if not state.rai_compliant:
        if "safety" not in state.resolved_metrics:
            state.resolved_metrics.append("safety")
        if state.privacy_sensitive and "privacy" not in state.resolved_metrics:
            state.resolved_metrics.append("privacy")
        if state.safety_critical and "groundedness" not in state.resolved_metrics:
            state.resolved_metrics.append("groundedness")
        
        # Re-validate
        validation = validate_rai_compliance(
            metrics=state.resolved_metrics,
            privacy_sensitive=state.privacy_sensitive,
            safety_critical=state.safety_critical,
            locale=primary_locale
        )
        state.rai_compliant = validation["is_compliant"]
    
    # Flag for human review if safety-critical
    if state.safety_critical:
        state.requires_human_review = True
        state.review_reason = "Safety-critical feature requires human approval"
    
    logger.info(f"RAI validation: compliant={state.rai_compliant}, "
                f"requires_review={state.requires_human_review}")
    
    return state


async def build_prompts_node(state: FeatureWorkflowState) -> FeatureWorkflowState:
    """
    Step 4: Build evaluation prompts for each target locale.
    """
    logger.info(f"Building prompts for {len(state.target_locales)} locales")
    
    for locale in state.target_locales:
        logger.info(f"Building prompt for locale: {locale}")
        
        result = build_prompt(
            feature_name=state.feature_name,
            category=state.user_category,
            locale=locale,
            metrics=state.resolved_metrics
        )
        
        state.prompts[locale] = result["evaluation_prompt"]
    
    logger.info(f"Built {len(state.prompts)} prompts")
    
    return state


async def generate_code_metrics_node(state: FeatureWorkflowState) -> FeatureWorkflowState:
    """
    Step 5: Generate code-based metrics samples.
    """
    logger.info(f"Generating code metrics for category: {state.user_category}")
    
    result = get_code_metrics(state.user_category)
    
    # Code metrics are the same for all locales
    for locale in state.target_locales:
        state.code_metrics[locale] = result["sample_code"]
    
    return state


async def finalize_node(state: FeatureWorkflowState) -> FeatureWorkflowState:
    """
    Step 6: Finalize the workflow and set status.
    """
    if state.errors:
        state.status = "failed"
    elif state.requires_human_review:
        state.status = "pending_review"
    else:
        state.status = "completed"
    
    logger.info(f"Workflow finalized: status={state.status}")
    
    return state


# =============================================================================
# Feature Generation Workflow
# =============================================================================

def create_feature_workflow() -> Workflow:
    """
    Create the feature evaluation prompt generation workflow.
    
    The workflow follows this sequence:
    
    [Analyze] → [Resolve Metrics] → [Validate RAI] → [Build Prompts] → [Code Metrics] → [Finalize]
                                          │
                                          ├── If not compliant: add required metrics
                                          └── If safety-critical: flag for human review
    """
    config = WorkflowConfig(
        name="FeaturePromptGeneration",
        description="Generate evaluation prompts for AI features with RAI compliance"
    )
    
    workflow = SequentialWorkflow(config)
    
    # Add nodes in sequence
    workflow.add_node("analyze", analyze_feature_node)
    workflow.add_node("resolve_metrics", resolve_metrics_node)
    workflow.add_node("validate_rai", validate_rai_node)
    workflow.add_node("build_prompts", build_prompts_node)
    workflow.add_node("code_metrics", generate_code_metrics_node)
    workflow.add_node("finalize", finalize_node)
    
    # Define edges (sequential flow)
    workflow.add_edge("analyze", "resolve_metrics")
    workflow.add_edge("resolve_metrics", "validate_rai")
    workflow.add_edge("validate_rai", "build_prompts")
    workflow.add_edge("build_prompts", "code_metrics")
    workflow.add_edge("code_metrics", "finalize")
    
    return workflow


# =============================================================================
# Multi-Locale Parallel Workflow
# =============================================================================

async def build_prompt_for_locale(locale: str, state: FeatureWorkflowState) -> Dict[str, str]:
    """Build prompt for a single locale (for parallel execution)"""
    result = build_prompt(
        feature_name=state.feature_name,
        category=state.user_category,
        locale=locale,
        metrics=state.resolved_metrics
    )
    return {locale: result["evaluation_prompt"]}


def create_multi_locale_workflow() -> Workflow:
    """
    Create a workflow that processes multiple locales in parallel.
    
    This is more efficient for features targeting many regions.
    
    [Analyze] → [Resolve Metrics] → [Validate RAI] → [Parallel: Build Prompts per Locale] → [Finalize]
    """
    config = WorkflowConfig(
        name="MultiLocalePromptGeneration",
        description="Generate evaluation prompts for multiple locales in parallel"
    )
    
    workflow = Workflow(config)
    
    # Sequential nodes
    workflow.add_node("analyze", analyze_feature_node)
    workflow.add_node("resolve_metrics", resolve_metrics_node)
    workflow.add_node("validate_rai", validate_rai_node)
    
    # Parallel node for building prompts
    # This will execute build_prompt_for_locale for each locale simultaneously
    workflow.add_node("build_prompts_parallel", ParallelNode(
        func=build_prompt_for_locale,
        # The items to process in parallel will be determined at runtime
        # based on state.target_locales
    ))
    
    workflow.add_node("finalize", finalize_node)
    
    # Edges
    workflow.add_edge("analyze", "resolve_metrics")
    workflow.add_edge("resolve_metrics", "validate_rai")
    workflow.add_edge("validate_rai", "build_prompts_parallel")
    workflow.add_edge("build_prompts_parallel", "finalize")
    
    return workflow


# =============================================================================
# Workflow Runner
# =============================================================================

class WorkflowRunner:
    """
    Runner for executing MetaFeature workflows.
    
    Example:
        runner = WorkflowRunner()
        
        result = runner.run(
            feature_name="Medical Document Summarizer",
            feature_description="Summarize medical documents for doctors...",
            target_locales=["de-DE", "ja-JP"],
            safety_critical=True
        )
        
        # Get prompts for each locale
        for locale, prompt in result.prompts.items():
            print(f"--- {locale} ---")
            print(prompt)
    """
    
    def __init__(self, workflow_type: str = "sequential"):
        """
        Initialize the workflow runner.
        
        Args:
            workflow_type: "sequential" or "parallel" (for multi-locale)
        """
        if workflow_type == "parallel":
            self._workflow = create_multi_locale_workflow()
        else:
            self._workflow = create_feature_workflow()
    
    async def run_async(
        self,
        feature_name: str,
        feature_description: str,
        category: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        target_locales: Optional[List[str]] = None,
        privacy_sensitive: bool = False,
        safety_critical: bool = False
    ) -> FeatureWorkflowState:
        """
        Run the workflow asynchronously.
        
        Args:
            feature_name: Name of the feature
            feature_description: Description of the feature
            category: Feature category (auto-detected if not provided)
            metrics: Metrics to use (auto-selected if not provided)
            target_locales: Target locales (default: ["en-US"])
            privacy_sensitive: Whether feature handles PII
            safety_critical: Whether feature is safety-critical
            
        Returns:
            FeatureWorkflowState with results
        """
        # Initialize state
        state = FeatureWorkflowState(
            feature_name=feature_name,
            feature_description=feature_description,
            user_category=category,
            user_metrics=metrics or [],
            target_locales=target_locales or ["en-US"],
            privacy_sensitive=privacy_sensitive,
            safety_critical=safety_critical
        )
        
        # Run workflow
        try:
            result = await self._workflow.run(state)
            return result
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            state.status = "failed"
            state.errors.append(str(e))
            return state
    
    def run(
        self,
        feature_name: str,
        feature_description: str,
        category: Optional[str] = None,
        metrics: Optional[List[str]] = None,
        target_locales: Optional[List[str]] = None,
        privacy_sensitive: bool = False,
        safety_critical: bool = False
    ) -> FeatureWorkflowState:
        """
        Run the workflow synchronously.
        """
        import asyncio
        return asyncio.run(self.run_async(
            feature_name=feature_name,
            feature_description=feature_description,
            category=category,
            metrics=metrics,
            target_locales=target_locales,
            privacy_sensitive=privacy_sensitive,
            safety_critical=safety_critical
        ))


# =============================================================================
# Human-in-the-Loop Workflow
# =============================================================================

class HumanReviewWorkflow:
    """
    Workflow with human approval checkpoints.
    
    This workflow pauses at key decision points to allow human review:
    1. After feature analysis - confirm category and flags
    2. After RAI validation - review compliance issues
    3. Before final output - approve generated prompts
    
    Example:
        workflow = HumanReviewWorkflow()
        
        # Start the workflow
        state = workflow.start(
            feature_name="Medical Assistant",
            feature_description="..."
        )
        
        # Review analysis
        print(f"Detected category: {state.detected_category}")
        print(f"Privacy sensitive: {state.detected_privacy_sensitive}")
        
        # Approve and continue
        state = workflow.approve_analysis(state, approved=True)
        
        # Review RAI
        print(f"RAI issues: {state.rai_issues}")
        
        # Approve and continue
        state = workflow.approve_rai(state, approved=True)
        
        # Get final results
        state = workflow.finalize(state)
    """
    
    def __init__(self):
        self._current_state: Optional[FeatureWorkflowState] = None
    
    async def start_async(
        self,
        feature_name: str,
        feature_description: str,
        **kwargs
    ) -> FeatureWorkflowState:
        """Start the workflow and run until first checkpoint"""
        state = FeatureWorkflowState(
            feature_name=feature_name,
            feature_description=feature_description,
            **kwargs
        )
        
        # Run analysis
        state = await analyze_feature_node(state)
        state.status = "awaiting_analysis_approval"
        
        self._current_state = state
        return state
    
    def start(self, feature_name: str, feature_description: str, **kwargs) -> FeatureWorkflowState:
        """Sync version of start"""
        import asyncio
        return asyncio.run(self.start_async(feature_name, feature_description, **kwargs))
    
    async def approve_analysis_async(
        self,
        state: FeatureWorkflowState,
        approved: bool,
        override_category: Optional[str] = None,
        override_privacy: Optional[bool] = None,
        override_safety: Optional[bool] = None
    ) -> FeatureWorkflowState:
        """Approve or modify analysis results and continue"""
        if not approved:
            state.status = "rejected_at_analysis"
            return state
        
        # Apply overrides
        if override_category:
            state.user_category = override_category
        if override_privacy is not None:
            state.privacy_sensitive = override_privacy
        if override_safety is not None:
            state.safety_critical = override_safety
        
        # Continue to metrics and RAI
        state = await resolve_metrics_node(state)
        state = await validate_rai_node(state)
        state.status = "awaiting_rai_approval"
        
        self._current_state = state
        return state
    
    def approve_analysis(self, state: FeatureWorkflowState, approved: bool, **kwargs) -> FeatureWorkflowState:
        """Sync version"""
        import asyncio
        return asyncio.run(self.approve_analysis_async(state, approved, **kwargs))
    
    async def approve_rai_async(
        self,
        state: FeatureWorkflowState,
        approved: bool,
        additional_metrics: Optional[List[str]] = None
    ) -> FeatureWorkflowState:
        """Approve RAI validation and continue to prompt generation"""
        if not approved:
            state.status = "rejected_at_rai"
            return state
        
        # Add any additional metrics requested by reviewer
        if additional_metrics:
            state.resolved_metrics.extend(additional_metrics)
        
        # Build prompts
        state = await build_prompts_node(state)
        state = await generate_code_metrics_node(state)
        state.status = "awaiting_final_approval"
        
        self._current_state = state
        return state
    
    def approve_rai(self, state: FeatureWorkflowState, approved: bool, **kwargs) -> FeatureWorkflowState:
        """Sync version"""
        import asyncio
        return asyncio.run(self.approve_rai_async(state, approved, **kwargs))
    
    async def finalize_async(
        self,
        state: FeatureWorkflowState,
        approved: bool = True
    ) -> FeatureWorkflowState:
        """Finalize the workflow"""
        if not approved:
            state.status = "rejected_at_final"
            return state
        
        state = await finalize_node(state)
        self._current_state = state
        return state
    
    def finalize(self, state: FeatureWorkflowState, approved: bool = True) -> FeatureWorkflowState:
        """Sync version"""
        import asyncio
        return asyncio.run(self.finalize_async(state, approved))


# =============================================================================
# Conversion to PromptOutput (Legacy Compatibility)
# =============================================================================

def workflow_state_to_prompt_output(
    state: FeatureWorkflowState,
    locale: str = "en-US"
) -> PromptOutput:
    """
    Convert workflow state to PromptOutput for legacy compatibility.
    
    Args:
        state: Completed workflow state
        locale: Which locale's prompt to use
        
    Returns:
        PromptOutput compatible with existing code
    """
    prompt = state.prompts.get(locale, "")
    
    return PromptOutput(
        feature_name=state.feature_name,
        category=state.user_category or state.detected_category,
        locale=locale,
        metrics_used=state.resolved_metrics,
        metric_definitions=state.metric_definitions,
        evaluation_prompt=prompt,
        rai_checks_applied=["safety_check"] if "safety" in state.resolved_metrics else []
    )
