#!/usr/bin/env python3
"""
LLM Provider Module for whisper-srt

Uses OpenRouter API to access various LLM models for text alignment.
Default model: google/gemini-3-flash-preview (fast and cost-effective)
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# Load .env file if present (supports both project root and current directory)
try:
    from dotenv import load_dotenv

    # Try to find .env file in common locations
    for env_path in [Path.cwd() / ".env", Path(__file__).parent.parent / ".env"]:
        if env_path.exists():
            load_dotenv(env_path)
            break
    else:
        load_dotenv()  # Try default locations
except ImportError:
    pass  # dotenv not installed, use system env vars only

try:
    import httpx
except ImportError:
    httpx = None

logger = logging.getLogger(__name__)

# Configuration defaults
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 8192


def get_llm_config() -> Dict[str, Any]:
    """
    Get LLM configuration from environment variables.

    Environment variables:
        OPENROUTER_API_KEY: Required - Your OpenRouter API key
        LLM_MODEL: Optional - Model to use (default: google/gemini-3-flash-preview)
        OPENROUTER_BASE_URL: Optional - API base URL

    Returns:
        Configuration dictionary
    """
    return {
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        "base_url": os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
    }


def check_llm_available() -> bool:
    """
    Check if LLM functionality is available.

    Returns:
        True if httpx is installed and API key is set
    """
    if httpx is None:
        return False
    config = get_llm_config()
    return bool(config["api_key"])


def generate_text(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    """
    Generate text using OpenRouter API.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model to use (default: from config)
        temperature: Sampling temperature (default: 0.3)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds (default: 300.0)

    Returns:
        Dict with 'content' key containing generated text

    Raises:
        RuntimeError: If httpx not installed or API key not set
        Exception: On API errors
    """
    if httpx is None:
        raise RuntimeError(
            "httpx is required for LLM functionality. " "Install with: pip install httpx"
        )

    config = get_llm_config()

    if not config["api_key"]:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Get your key at https://openrouter.ai/keys"
        )

    model = model or config["model"]
    url = f"{config['base_url']}/chat/completions"

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/whisper-srt",
        "X-Title": "whisper-srt",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    logger.debug(f"LLM request to {model}")
    logger.debug(f"Messages: {len(messages)} message(s)")

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()

            # Extract content from response
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Log usage info if available
            usage = result.get("usage", {})
            if usage:
                logger.debug(
                    f"Tokens - input: {usage.get('prompt_tokens', 'N/A')}, "
                    f"output: {usage.get('completion_tokens', 'N/A')}"
                )

            return {"content": content, "usage": usage, "model": model}

    except httpx.HTTPStatusError as e:
        error_body = e.response.text
        logger.error(f"LLM API error: {e.response.status_code} - {error_body}")
        raise RuntimeError(f"LLM API error: {e.response.status_code} - {error_body}")
    except httpx.RequestError as e:
        logger.error(f"LLM request failed: {e}")
        raise RuntimeError(f"LLM request failed: {e}")


def parse_json_response(content: str) -> Any:
    """
    Parse JSON from LLM response, handling markdown code blocks.

    Args:
        content: Raw LLM response content

    Returns:
        Parsed JSON data

    Raises:
        ValueError: If JSON parsing fails
    """
    if not content:
        raise ValueError("Empty response from LLM")

    # Try to extract JSON from markdown code block
    json_match = None
    import re

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if json_match:
        content = json_match.group(1).strip()

    # Try to find JSON array or object
    array_start = content.find("[")
    array_end = content.rfind("]")
    obj_start = content.find("{")
    obj_end = content.rfind("}")

    # Prefer array if both exist and array comes first
    if array_start != -1 and array_end != -1:
        if obj_start == -1 or array_start < obj_start:
            content = content[array_start : array_end + 1]
    elif obj_start != -1 and obj_end != -1:
        content = content[obj_start : obj_end + 1]

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        logger.debug(f"Content: {content[:500]}...")
        raise ValueError(f"Failed to parse LLM response as JSON: {e}")
