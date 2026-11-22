"""Resilient multi-provider LLM router with strict Pydantic JSON enforcement.

Failover chain (verified active late-2026):
  1. Google AI Studio: gemini-3.5-flash-lite (primary)
  2. Google AI Studio: gemini-3.1-flash-lite (secondary Gemini quota)
  3. Groq Free Tier:   openai/gpt-oss-20b    (ultra-fast LPU failover)
  4. Groq Free Tier:   openai/gpt-oss-120b   (higher-reasoning Groq failover)
  5. OpenRouter Free:  openrouter/free       (tertiary fallback router)
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TypeVar

from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMRouterError(RuntimeError):
    """Raised when every configured LLM provider fails."""


def _extract_json_block(raw_text: str) -> str:
    """Strip markdown code fences or prose around a JSON object/array."""
    text = raw_text.strip()
    # Strip ```json ... ``` fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Locate outer JSON object or array boundaries
    obj_start, obj_end = text.find("{"), text.rfind("}")
    arr_start, arr_end = text.find("["), text.rfind("]")

    if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
        if obj_end > obj_start:
            return text[obj_start : obj_end + 1]
    if arr_start != -1 and arr_end > arr_start:
        return text[arr_start : arr_end + 1]
    return text


class LLMRouter:
    """Async structured-output LLM client with automatic provider failover."""

    def __init__(
        self,
        gemini_api_key: str | None = None,
        groq_api_key: str | None = None,
        openrouter_api_key: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout_seconds = timeout_seconds

    async def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.2,
    ) -> tuple[T, str]:
        """Generate a validated Pydantic object, failing over across providers.

        Returns:
            (parsed_model, provider_label)
        """
        errors: list[str] = []

        # --- Tier 1 & 2: Google AI Studio (Gemini) ---
        if self.gemini_api_key:
            for model_id in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"):
                label = f"gemini:{model_id}"
                try:
                    result = await self._call_gemini(
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_model=response_model,
                        temperature=temperature,
                    )
                    logger.info("LLM call succeeded via %s", label)
                    return result, label
                except Exception as exc:
                    msg = f"{label} failed: {type(exc).__name__}: {exc}"
                    logger.warning(msg)
                    errors.append(msg)

        # --- Tier 3 & 4: GroqCloud (OpenAI-compatible) ---
        if self.groq_api_key:
            for model_id in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
                label = f"groq:{model_id}"
                try:
                    result = await self._call_openai_compat(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=self.groq_api_key,
                        model_id=model_id,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        response_model=response_model,
                        temperature=temperature,
                    )
                    logger.info("LLM call succeeded via %s", label)
                    return result, label
                except Exception as exc:
                    msg = f"{label} failed: {type(exc).__name__}: {exc}"
                    logger.warning(msg)
                    errors.append(msg)

        # --- Tier 5: OpenRouter Free Auto-Router ---
        if self.openrouter_api_key:
            label = "openrouter:openrouter/free"
            try:
                result = await self._call_openai_compat(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=self.openrouter_api_key,
                    model_id="openrouter/free",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                    temperature=temperature,
                    extra_headers={
                        "HTTP-Referer": "https://apify.com",
                        "X-Title": "Complaint-to-Revenue Agent",
                    },
                )
                logger.info("LLM call succeeded via %s", label)
                return result, label
            except Exception as exc:
                msg = f"{label} failed: {type(exc).__name__}: {exc}"
                logger.warning(msg)
                errors.append(msg)

        if not errors:
            raise LLMRouterError(
                "No LLM API keys configured. Set GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY."
            )
        raise LLMRouterError("All LLM providers failed:\n- " + "\n- ".join(errors))

    async def _call_gemini(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
    ) -> T:
        client = genai.Client(api_key=self.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=response_model,
            ),
        )
        raw_text = response.text or ""
        if not raw_text.strip():
            raise ValueError("Gemini returned empty response text")
        cleaned = _extract_json_block(raw_text)
        return response_model.model_validate_json(cleaned)

    async def _call_openai_compat(
        self,
        base_url: str,
        api_key: str,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float,
        extra_headers: dict[str, str] | None = None,
    ) -> T:
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        enforced_system = (
            f"{system_prompt}\n\n"
            "CRITICAL OUTPUT CONTRACT: Return ONLY a valid JSON object matching this exact JSON Schema. "
            "No markdown fences, no commentary.\n"
            f"JSON Schema:\n{schema_json}"
        )

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self.timeout_seconds,
            default_headers=extra_headers or {},
        )
        completion = await client.chat.completions.create(
            model=model_id,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": enforced_system},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = completion.choices[0].message.content or ""
        if not raw_text.strip():
            raise ValueError(f"{model_id} returned empty content")
        cleaned = _extract_json_block(raw_text)
        return response_model.model_validate_json(cleaned)