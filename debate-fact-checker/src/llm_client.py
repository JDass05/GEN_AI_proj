"""
Provider-agnostic LLM wrapper with retry/backoff and structured output support.
"""

import json
import re
from typing import Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import config


# ---------------------------------------------------------------------------
# LLM Factory
# ---------------------------------------------------------------------------
def get_llm(model_name: str, temperature: float = 0.0) -> BaseChatModel:
    """
    Get a LangChain chat model for the configured provider.
    Supports 'gemini' (default) and 'groq' (optional fallback).
    """
    provider = config.LLM_PROVIDER

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=config.GEMINI_API_KEY,
            temperature=temperature,
            convert_system_message_to_human=True,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            groq_api_key=config.GROQ_API_KEY,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Retry-wrapped invocation
# ---------------------------------------------------------------------------
@retry(
    retry=retry_if_exception_type((Exception,)),
    wait=wait_exponential(
        min=config.LLM_RETRY_MIN_WAIT,
        max=config.LLM_RETRY_MAX_WAIT,
    ),
    stop=stop_after_attempt(config.LLM_RETRY_MAX_ATTEMPTS),
    reraise=True,
)
def invoke_llm(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Invoke an LLM with retry on transient errors. Returns raw text."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        # Extract text parts if model returns list of content blocks
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif hasattr(part, "text"):
                parts.append(part.text)
        return "".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Structured output with fallback
# ---------------------------------------------------------------------------
def _extract_json_from_text(text: str) -> Optional[dict]:
    """Try to extract a JSON object from raw LLM text output."""
    # Try to find JSON block in markdown code fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


@retry(
    retry=retry_if_exception_type((Exception,)),
    wait=wait_exponential(
        min=config.LLM_RETRY_MIN_WAIT,
        max=config.LLM_RETRY_MAX_WAIT,
    ),
    stop=stop_after_attempt(config.LLM_RETRY_MAX_ATTEMPTS),
    reraise=True,
)
def invoke_llm_structured(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    output_schema: Type[BaseModel],
) -> BaseModel:
    """
    Invoke LLM and parse into a Pydantic model.
    Strategy:
    1. Try with_structured_output() if available
    2. Fallback: raw text + JSON extraction + Pydantic validation
    3. If still invalid, retry with error-correction prompt
    """
    # Strategy 1: Try structured output binding
    try:
        structured_llm = llm.with_structured_output(output_schema)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        result = structured_llm.invoke(messages)
        if result is not None:
            return result
    except (AttributeError, NotImplementedError, Exception):
        pass

    # Strategy 2: Raw text + JSON extraction
    raw_text = invoke_llm.__wrapped__(llm, system_prompt, user_prompt)

    json_dict = _extract_json_from_text(raw_text)
    if json_dict:
        try:
            return output_schema.model_validate(json_dict)
        except Exception:
            pass

    # Strategy 3: Error-correction retry
    correction_prompt = (
        f"Your previous response could not be parsed as valid JSON matching the schema.\n"
        f"Required schema fields: {list(output_schema.model_fields.keys())}\n"
        f"Previous response:\n{raw_text}\n\n"
        f"Please respond with ONLY a valid JSON object matching the schema. No other text."
    )
    corrected_text = invoke_llm.__wrapped__(llm, system_prompt, correction_prompt)
    json_dict = _extract_json_from_text(corrected_text)
    if json_dict:
        return output_schema.model_validate(json_dict)

    raise ValueError(
        f"Failed to parse structured output after correction. "
        f"Last response: {corrected_text[:500]}"
    )
