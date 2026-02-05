"""AI-powered insights for portfolio analysis using OpenAI GPT-4o-mini."""

import hashlib
import logging
import os
from typing import Optional

from cachetools import TTLCache

logger = logging.getLogger(__name__)

# 5-minute TTL cache (maxsize=1000 entries)
_cache = TTLCache(maxsize=1000, ttl=300)

# System prompts per analysis type
SYSTEM_PROMPTS = {
    "performance": (
        "You are a financial advisor explaining portfolio performance to retail investors. "
        "Be conversational, specific to their data, and actionable. Keep under 200 words. "
        "Focus on: returns vs benchmark, risk metrics, allocation insights, and what to watch."
    ),
    "factor_analysis": (
        "You are explaining Fama-French factor analysis results to an investor with moderate "
        "financial knowledge. Explain what their factor exposures mean, whether they're "
        "significant, and implications for their portfolio. Keep under 250 words."
    ),
    "black_litterman": (
        "You are explaining Black-Litterman portfolio optimization results. Help the user "
        "understand why certain assets were weighted as they were, how their views influenced "
        "the outcome, and whether the optimized portfolio meets their goals. Keep under 200 words."
    ),
}

# Section-specific user prompt templates
SECTION_PROMPTS = {
    "summary": "Provide a one-sentence summary of these results: {data}",
    "how_to_use": "Explain how to use {analysis_type} in 3-4 simple steps.",
    "introduction": "Explain what {analysis_type} is and when investors should use it.",
    "full_analysis": (
        "Analyze these results comprehensively: {data}. "
        "Include key findings, what's working, risks, and actionable recommendations."
    ),
    "returns": "Focus on the return metrics from this data and explain what they mean: {data}",
    "risk": "Focus on volatility, beta, and drawdown from this data and explain: {data}",
    "allocation": "Focus on portfolio weights from this data and explain: {data}",
}

# In-memory rate-limit tracking (user_id -> request count within window)
_rate_limit_store: dict[str, list] = {}
RATE_LIMIT_MAX = 20
RATE_LIMIT_WINDOW = 60  # seconds


def _get_cache_key(analysis_type: str, data: dict, section: str) -> str:
    content = f"{analysis_type}:{section}:{str(sorted(data.items()) if isinstance(data, dict) else data)}"
    return hashlib.md5(content.encode()).hexdigest()


def _check_rate_limit(user_id: str) -> tuple[bool, int]:
    """Simple in-memory rate limiting. Returns (is_allowed, requests_remaining)."""
    import time as _time

    now = _time.time()
    if user_id not in _rate_limit_store:
        _rate_limit_store[user_id] = []

    # Prune old entries outside the window
    _rate_limit_store[user_id] = [
        t for t in _rate_limit_store[user_id] if now - t < RATE_LIMIT_WINDOW
    ]

    count = len(_rate_limit_store[user_id])
    if count >= RATE_LIMIT_MAX:
        return False, 0

    return True, RATE_LIMIT_MAX - count


def _record_request(user_id: str) -> None:
    import time as _time

    now = _time.time()
    if user_id not in _rate_limit_store:
        _rate_limit_store[user_id] = []
    _rate_limit_store[user_id].append(now)


def _build_user_prompt(analysis_type: str, data: dict, section: str) -> str:
    """Build the user prompt from section template and data."""
    template = SECTION_PROMPTS.get(section)
    if template is None:
        # Fallback: treat unknown sections as a generic analysis request
        template = "Analyze the '{section}' aspect of these {analysis_type} results: {data}"

    # Format with available variables
    return template.format(
        analysis_type=analysis_type.replace("_", " "),
        data=str(data),
        section=section,
    )


async def generate_insight(
    analysis_type: str,
    data: dict,
    section: str,
    user_id: str,
) -> dict:
    """
    Generate an AI-powered insight for the given analysis results.

    Args:
        analysis_type: One of "performance", "factor_analysis", "black_litterman"
        data: The analysis data dict to explain
        section: Which aspect to focus on (e.g. "summary", "full_analysis", "risk")
        user_id: Identifier for rate-limiting

    Returns:
        dict with keys: insight, requests_remaining, cached, tokens_used
    """
    # Rate limit check
    allowed, remaining = _check_rate_limit(user_id)
    if not allowed:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making more AI requests.",
        )

    # Check cache
    cache_key = _get_cache_key(analysis_type, data, section)
    cached_result = _cache.get(cache_key)
    if cached_result is not None:
        logger.info(
            f"AI cache hit: user={user_id}, type={analysis_type}, section={section}"
        )
        return {
            "insight": cached_result["insight"],
            "requests_remaining": remaining,
            "cached": True,
            "tokens_used": 0,
        }

    # Verify API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="AI insights are currently unavailable (OPENAI_API_KEY not configured).",
        )

    # Build prompts
    system_prompt = SYSTEM_PROMPTS.get(analysis_type, SYSTEM_PROMPTS["performance"])
    user_prompt = _build_user_prompt(analysis_type, data, section)

    # Call OpenAI
    import openai

    client = openai.AsyncOpenAI(api_key=api_key)

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.7,
        )

        insight = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else 0

    except openai.RateLimitError:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=429,
            detail="OpenAI rate limit exceeded, try again later.",
        )
    except openai.APIError as e:
        logger.error(f"OpenAI API error: {e}")
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503,
            detail="AI service temporarily unavailable.",
        )
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI: {e}")
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Internal server error.")

    # Record the request for rate limiting
    _record_request(user_id)
    _, remaining = _check_rate_limit(user_id)

    # Store in cache
    _cache[cache_key] = {"insight": insight}

    # Log usage
    logger.info(
        f"AI Usage: user={user_id}, type={analysis_type}, section={section}, "
        f"tokens={tokens_used}, cached=False"
    )

    return {
        "insight": insight,
        "requests_remaining": remaining,
        "cached": False,
        "tokens_used": tokens_used,
    }
