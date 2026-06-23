import os
import json
import logging
from time import sleep
from django.utils.translation import gettext_lazy as _
from symptoms.ai import configure_gemini, MAX_RETRIES
import google.generativeai as genai

logger = logging.getLogger(__name__)

CHAT_PROMPT_TEMPLATE = """
You are a medical assistant for Ethiopian patients. Respond in friendly, simple English.
Analyze this message: {user_input}

If it contains:
1. Specific symptoms -> Use symptom analysis mode
2. General health questions -> Provide helpful advice
3. Requests for first aid -> Give step-by-step guidance

Always respond with valid JSON format (no markdown) using one of these structures:

Symptom Analysis Mode:
{{
  "mode": "symptoms",
  "conditions": ["condition1", "condition2"],
  "recommendations": ["step1", "step2"],
  "urgency": "low/medium/high"
}}

General Advice Mode:
{{
  "mode": "advice",
  "response": "helpful information here"
}}

First Aid Mode:
{{
  "mode": "firstaid",
  "procedure": "step-by-step instructions",
  "warning": "important caution note"
}}
"""


def _validate_response(response_data):
    """Ensure AI response matches expected structure"""
    valid_modes = ["symptoms", "advice", "firstaid", "error"]
    mode = response_data.get("mode")

    if mode not in valid_modes:
        raise ValueError(f"Invalid response mode: {mode}")

    required_fields = {
        "symptoms": ["conditions", "recommendations", "urgency"],
        "advice": ["response"],
        "firstaid": ["procedure", "warning"],
        "error": ["response"],
    }.get(mode, [])

    missing = [field for field in required_fields if field not in response_data]
    if missing:
        raise ValueError(
            f"Missing required fields for {mode} mode: {', '.join(missing)}"
        )

    return response_data


def generate_chat_response(user_input, is_retry=False):
    try:
        configure_gemini()
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(
            CHAT_PROMPT_TEMPLATE.format(user_input=user_input),
            generation_config=genai.types.GenerationConfig(
                temperature=0.2, max_output_tokens=600, top_p=0.95
            ),
        )

        raw = response.text.strip()
        for prefix in ["```json", "```JSON"]:
            if raw.startswith(prefix):
                raw = raw[len(prefix) :].strip()
        raw = raw.rstrip("`")

        response_data = json.loads(raw)
        validated_data = _validate_response(response_data)

        if validated_data.get("mode") == "symptoms":
            validated_data = _enhance_with_symptom_checker(validated_data)

        return validated_data

    except json.JSONDecodeError as je:
        logger.error(f"JSON parse error: {je}")
        if not is_retry and (MAX_RETRIES is None or is_retry < MAX_RETRIES):
            sleep(1)
            return generate_chat_response(user_input, is_retry=True)
        return {"mode": "error", "response": "Could not process request"}

    except genai.types.BlockedPromptException:
        return {
            "mode": "error",
            "response": "I can't answer that. Please ask about medical concerns.",
        }

    except Exception as e:
        logger.error(f"Chat AI error: {str(e)}", exc_info=True)
        return {
            "mode": "error",
            "response": "Temporary system issue. Please try again.",
        }


def _enhance_with_symptom_checker(response_data):
    if isinstance(response_data.get("recommendations"), list):
        response_data["recommendations"] = [
            f"{idx+1}. {step.strip()}"
            for idx, step in enumerate(response_data["recommendations"])
            if step.strip()
        ]
    return response_data


def get_fallback_response():
    """Used when Gemini API is unavailable"""
    return {
        "mode": "error",
        "response": "Service is temporarily unavailable. Please try again later.",
    }
