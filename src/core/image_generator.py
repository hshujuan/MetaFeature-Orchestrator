"""
Image Generator - Azure OpenAI DALL-E 3 Integration

Provides image generation capabilities for testing image-related AI features.
"""
from __future__ import annotations
import os
import base64
import json
import httpx
from pathlib import Path
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GeneratedImage:
    """Result of image generation"""
    prompt: str
    revised_prompt: str  # DALL-E 3 may revise your prompt
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    size: str = "1024x1024"
    quality: str = "standard"
    style: str = "vivid"
    created_at: str = ""
    
    def save(self, filepath: str) -> str:
        """Save the image to a file"""
        if self.image_base64:
            image_data = base64.b64decode(self.image_base64)
            with open(filepath, 'wb') as f:
                f.write(image_data)
            return filepath
        elif self.image_url:
            # Download from URL
            response = httpx.get(self.image_url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
        raise ValueError("No image data available to save")


class ImageGenerator:
    """
    Azure OpenAI DALL-E 3 Image Generator
    
    Usage:
        generator = ImageGenerator()
        result = generator.generate("A cat astronaut in space")
        result.save("cat_astronaut.png")
    """
    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
    ):
        """
        Initialize the image generator.
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            api_version: API version (default: 2024-04-01-preview)
            deployment_name: DALL-E 3 deployment name
        """
        self.endpoint = endpoint or os.getenv("AZURE_DALLE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_DALLE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or os.getenv("DALLE_API_VERSION") or os.getenv("OPENAI_API_VERSION", "2024-04-01-preview")
        self.deployment_name = deployment_name or os.getenv("DALLE_DEPLOYMENT_NAME", "dall-e-3")
        
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint is required. Set AZURE_DALLE_ENDPOINT or AZURE_OPENAI_ENDPOINT environment variable.")
        if not self.api_key:
            raise ValueError("Azure OpenAI API key is required. Set AZURE_DALLE_API_KEY or AZURE_OPENAI_API_KEY environment variable.")
    
    def generate(
        self,
        prompt: str,
        size: Literal["1024x1024", "1024x1792", "1792x1024"] = "1024x1024",
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["vivid", "natural"] = "vivid",
        response_format: Literal["url", "b64_json"] = "url",
        n: int = 1,
    ) -> GeneratedImage:
        """
        Generate an image using DALL-E 3.
        
        Args:
            prompt: Text description of the image to generate
            size: Image dimensions (1024x1024, 1024x1792 for portrait, 1792x1024 for landscape)
            quality: "standard" or "hd" for higher detail
            style: "vivid" for hyper-real/dramatic, "natural" for more realistic
            response_format: "url" for temporary URL, "b64_json" for base64 encoded image
            n: Number of images (DALL-E 3 only supports n=1)
        
        Returns:
            GeneratedImage with the result
        """
        # Build request URL
        url = f"{self.endpoint.rstrip('/')}/openai/deployments/{self.deployment_name}/images/generations?api-version={self.api_version}"
        
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        
        payload = {
            "prompt": prompt,
            "size": size,
            "quality": quality,
            "style": style,
            "response_format": response_format,
            "n": n,
        }
        
        # Make request
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
        
        # Parse response
        image_data = result["data"][0]
        
        return GeneratedImage(
            prompt=prompt,
            revised_prompt=image_data.get("revised_prompt", prompt),
            image_url=image_data.get("url"),
            image_base64=image_data.get("b64_json"),
            size=size,
            quality=quality,
            style=style,
            created_at=datetime.now().isoformat(),
        )
    
    def generate_for_evaluation(
        self,
        prompt: str,
        output_dir: str = "./generated_images",
        size: Literal["1024x1024", "1024x1792", "1792x1024"] = "1024x1024",
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["vivid", "natural"] = "vivid",
    ) -> Dict[str, Any]:
        """
        Generate an image and prepare it for evaluation.
        
        Returns a dict with image path, prompts, and metadata for evaluation.
        """
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate image
        result = self.generate(
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            response_format="url",  # URL is faster for generation
        )
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        filepath = str(Path(output_dir) / filename)
        result.save(filepath)
        
        return {
            "image_path": filepath,
            "original_prompt": prompt,
            "revised_prompt": result.revised_prompt,
            "size": size,
            "quality": quality,
            "style": style,
            "created_at": result.created_at,
            "evaluation_ready": True,
        }


def create_sample_evaluation_images(prompts: list = None) -> list:
    """
    Generate sample images for evaluation testing.
    
    Args:
        prompts: List of prompts to generate. If None, uses defaults.
    
    Returns:
        List of generation results with file paths.
    """
    if prompts is None:
        prompts = [
            "A golden retriever puppy playing in autumn leaves, photorealistic style",
            "A futuristic city skyline at sunset with flying cars, digital art",
            "A cozy coffee shop interior with warm lighting, illustration style",
            "An astronaut floating in space with Earth in the background, cinematic",
            "A dragon perched on a mountain peak at sunrise, fantasy art",
        ]
    
    generator = ImageGenerator()
    results = []
    
    for prompt in prompts:
        try:
            result = generator.generate_for_evaluation(prompt)
            results.append(result)
            print(f"✅ Generated: {result['image_path']}")
        except Exception as e:
            results.append({
                "original_prompt": prompt,
                "error": str(e),
                "evaluation_ready": False,
            })
            print(f"❌ Failed: {prompt[:50]}... - {e}")
    
    return results


# ═══════════════════════════════════════════════════════════════════
# IMAGE EVALUATION PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════

IMAGE_EVALUATION_PROMPT_TEMPLATE = """You are an expert image quality evaluator. Evaluate the following generated image against the original prompt and the specified metrics.

## Original Prompt
{original_prompt}

## Revised Prompt (if modified by the model)
{revised_prompt}

## Image Details
- Size: {size}
- Quality Setting: {quality}
- Style: {style}

## Evaluation Metrics
{metrics_definitions}

## Instructions
1. Carefully analyze the image for each metric
2. Provide a score from 1-5 for each metric (1=Poor, 2=Below Average, 3=Average, 4=Good, 5=Excellent)
3. Provide specific observations and justifications for each score
4. Identify any issues, artifacts, or areas for improvement

## Response Format
Return your evaluation as JSON:
```json
{{
    "overall_score": <float 1-5>,
    "metrics": {{
        "<metric_name>": {{
            "score": <int 1-5>,
            "observations": "<specific observations>",
            "issues": ["<issue1>", "<issue2>"]
        }}
    }},
    "summary": "<overall assessment>",
    "strengths": ["<strength1>", "<strength2>"],
    "improvements": ["<suggestion1>", "<suggestion2>"]
}}
```
"""

def generate_image_evaluation_prompt(
    original_prompt: str,
    revised_prompt: str,
    metrics: list,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
) -> str:
    """Generate an evaluation prompt for image assessment."""
    from .metrics_registry import get_metric
    
    # Build metrics definitions
    metrics_lines = []
    for m_name in metrics:
        metric = get_metric(m_name)
        if metric:
            metrics_lines.append(f"- **{metric.name}**: {metric.get_definition('en')}")
    
    metrics_definitions = "\n".join(metrics_lines) if metrics_lines else "- Use standard image quality criteria"
    
    return IMAGE_EVALUATION_PROMPT_TEMPLATE.format(
        original_prompt=original_prompt,
        revised_prompt=revised_prompt,
        size=size,
        quality=quality,
        style=style,
        metrics_definitions=metrics_definitions,
    )


if __name__ == "__main__":
    # Test image generation
    print("Testing DALL-E 3 Image Generation...")
    
    try:
        generator = ImageGenerator()
        result = generator.generate(
            prompt="A cute robot chef cooking pasta in a modern kitchen, cartoon style",
            size="1024x1024",
            quality="standard",
            style="vivid",
        )
        
        print(f"✅ Image generated successfully!")
        print(f"   Original prompt: {result.prompt}")
        print(f"   Revised prompt: {result.revised_prompt}")
        print(f"   URL: {result.image_url[:80]}..." if result.image_url else "   (base64 encoded)")
        
        # Save image
        filepath = result.save("test_generated_image.png")
        print(f"   Saved to: {filepath}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
