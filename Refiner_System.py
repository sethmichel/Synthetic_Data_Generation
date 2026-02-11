import os
import json
import re
import time
import requests
import pandas as pd


REQUIRED_TRADE_KEYS = [
    "ticker",
    "entry_time",
    "entry_price",
    "entry_volatility_percent",
    "worst_exit_percent",
    "trade_best_exit_percent",
]


def Strip_Markdown_Json_Fences(text):
    clean = str(text or "").strip()

    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]

    if clean.endswith("```"):
        clean = clean[:-3]

    return clean.strip()


def Parse_First_Json_Object(text):
    clean = Strip_Markdown_Json_Fences(text)

    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", clean):
        start = match.start()
        try:
            candidate, _ = decoder.raw_decode(clean[start:])
            if isinstance(candidate, dict):
                return candidate

        except Exception:
            continue

    return None


def Bad_Value_Handler(value):
    if value is None:
        return True

    if isinstance(value, float) and pd.isna(value):
        return True

    text = str(value).strip().lower()

    return text in ("", "nan", "none", "null")


def Merge_Refined_Trade(original_trade, refined_trade):
    # Preserve original schema and only apply concrete, non-empty updates.
    merged = dict(original_trade)

    if not isinstance(refined_trade, dict):
        return merged

    for key in original_trade.keys():
        if key in refined_trade and not Bad_Value_Handler(refined_trade[key]):
            merged[key] = refined_trade[key]

    # Ensure required fields always exist, even if original payload was imperfect.
    for key in REQUIRED_TRADE_KEYS:
        if key not in merged:
            merged[key] = original_trade.get(key, "")

    return merged


def Refine_Trade_Row(original_trade_json_raw, few_shot_examples, judge_critique, target_ticker="", target_volatility=""):
    original_trade = Parse_First_Json_Object(original_trade_json_raw)

    if not isinstance(original_trade, dict):
        return {
            "success": False,
            "error": "Could not parse original generated trade JSON for refinement.",
            "refined_json_text": str(original_trade_json_raw),
            "raw_refiner_response": "",
            "api_error": False,
            "api_error_stage": "",
            "api_error_message": "",
            "api_attempts": 0
        }

    refiner_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    refiner_model_id = "nvidia/mistral-nemo-minitron-8b-base"
    headers = {
        "Authorization": f"Bearer {os.environ['NIM_API_KEY']}",
        "Content-Type": "application/json"
    }

    user_prompt = (
        "You are a precision JSON trade refiner.\n\n"
        "Your goal is to fix ONLY the exact issue described in the JUDGE CRITIQUE.\n"
        "Do not rewrite valid fields. Do not change schema.\n\n"
        f"TARGET TICKER: {target_ticker}\n"
        f"TARGET VOLATILITY: {target_volatility}\n\n"
        f"GOLD FEW-SHOT EXAMPLES:\n{few_shot_examples}\n\n"
        f"ORIGINAL TRADE JSON:\n{json.dumps(original_trade, ensure_ascii=True)}\n\n"
        f"JUDGE CRITIQUE:\n{judge_critique}\n\n"
        "Hard constraints:\n"
        f"- Output ONLY one JSON object.\n"
        f"- Keep exactly these required keys: {', '.join(REQUIRED_TRADE_KEYS)}.\n"
        "- Do not add markdown fences.\n"
        "- Keep all valid values unchanged; edit only fields needed to satisfy the critique.\n"
        "- Return a complete corrected row, not a partial patch."
    )

    payload = {
        "model": refiner_model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You repair synthetic financial trade rows with minimal edits. "
                    "Always return machine-parseable JSON only."
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.05,
        "max_tokens": 350
    }

    raw_refiner_response = ""
    max_api_attempts = 2
    last_error = ""
    attempts_used = 0

    for attempt in range(1, max_api_attempts + 1):
        attempts_used = attempt
        try:
            response = requests.post(refiner_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            response_json = response.json()
            raw_refiner_response = response_json["choices"][0]["message"]["content"]

            parsed_refined = Parse_First_Json_Object(raw_refiner_response)
            if not isinstance(parsed_refined, dict):
                raise ValueError("Refiner response did not contain a parseable JSON object.")

            merged_trade = Merge_Refined_Trade(original_trade, parsed_refined)
            refined_json_text = json.dumps(merged_trade, ensure_ascii=True)

            return {
                "success": True,
                "error": "",
                "refined_json_text": refined_json_text,
                "raw_refiner_response": raw_refiner_response,
                "api_error": False,
                "api_error_stage": "",
                "api_error_message": "",
                "api_attempts": attempts_used
            }

        except Exception as e:
            last_error = str(e)
            if attempt < max_api_attempts:
                time.sleep(1)
                continue

    return {
        "success": False,
        "error": f"Refiner API/parse failure after {max_api_attempts} attempts: {last_error}",
        "refined_json_text": json.dumps(original_trade, ensure_ascii=True),
        "raw_refiner_response": raw_refiner_response,
        "api_error": True,
        "api_error_stage": "refiner",
        "api_error_message": last_error,
        "api_attempts": attempts_used
    }
