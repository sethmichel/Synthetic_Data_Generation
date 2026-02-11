import pandas as pd
import os
import requests
import json
import time
import re
from typing import Any, Dict, List, TypedDict
from langgraph.graph import StateGraph, START, END
import Refiner_System

# Remove markdown fences so we can parse JSON reliably
def Strip_Markdown_Json_Fences(text):
    clean = str(text or "").strip()

    if clean.startswith("```json"):
        clean = clean[7:]

    elif clean.startswith("```"):
        clean = clean[3:]

    if clean.endswith("```"):
        clean = clean[:-3]

    return clean.strip()


# Handle None/NaN/blank values consistently
def Bad_Value_Handler(value):
    if value is None:
        return True

    if isinstance(value, float) and pd.isna(value):
        return True

    text = str(value).strip().lower()

    return text in ("", "nan", "none", "null")


# Force a score into the 1-5 range
def Score_Results(raw_value, default_score=1):
    try:
        parsed = int(raw_value)

    except (TypeError, ValueError):
        return default_score

    return max(1, min(5, parsed))


# Normalize LLM output to one predictable schema for downstream refiner usage
def Normalize_Judge_Payload(payload, fallback_critique):
    if not isinstance(payload, dict):
        payload = {}

    scores_obj = payload.get("scores", {})
    if not isinstance(scores_obj, dict):
        scores_obj = {}

    completeness = Score_Results(scores_obj.get("completeness", payload.get("completeness")))
    consistency = Score_Results(scores_obj.get("consistency", payload.get("consistency")))
    realism = Score_Results(scores_obj.get("realism", payload.get("realism")))

    passed = payload.get("passed")
    if isinstance(passed, str):
        passed = passed.strip().lower() == "true"

    if not isinstance(passed, bool):
        # Deterministic fallback for malformed responses.
        passed = completeness >= 4 and consistency >= 4 and realism >= 4

    critique = str(payload.get("critique", "")).strip()
    if not critique:
        critique = fallback_critique if not passed else "Trade passed all judge criteria."

    return {
        "passed": passed,
        "critique": critique,
        "scores": {
            "completeness": completeness,
            "consistency": consistency,
            "realism": realism
        }
    }


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


def Judge_One_Row(synthetic_json_raw, few_shot_examples, judge_url, judge_model_id, headers):
    if Bad_Value_Handler(synthetic_json_raw):
        critique = "Generator output is empty or missing; cannot judge this row."
        return {
            "judge": {
                "passed": False,
                "critique": critique,
                "scores": {
                    "completeness": 1,
                    "consistency": 1,
                    "realism": 1
                }
            },
            "judge_raw_response": "",
            "clean_trade_json": "",
            "few_shot_examples": str(few_shot_examples or ""),
            "api_error": False,
            "api_error_stage": "",
            "api_error_message": "",
            "api_attempts": 0
        }

    clean_trade_json = Strip_Markdown_Json_Fences(synthetic_json_raw)

    if Bad_Value_Handler(few_shot_examples):
        few_shot_examples = "No few-shot examples were provided for this row."

    user_prompt = (
        "You are judging one synthetic stock trade generated from few-shot examples.\n\n"
        "Compare the generated trade against the provided GOLD FEW-SHOT EXAMPLES.\n\n"
        f"GOLD FEW-SHOT EXAMPLES:\n{few_shot_examples}\n\n"
        f"GENERATED SYNTHETIC TRADE JSON:\n{clean_trade_json}\n\n"
        "Evaluate with these criteria:\n"
        "1) Completeness: Are all required fields present? Are there null/empty values where there should not be?\n"
        "2) Consistency: Do values make logical sense together? Use only fields that exist in the JSON.\n"
        "3) Realism: Does this trade look plausible vs the provided gold examples? Is volatility behavior in expected bounds?\n\n"
        "Important limits:\n"
        "- Some checks are impossible without fields that do not exist (ex: buy-vs-short validation). Do not penalize for missing schema fields.\n"
        "- Required keys are: ticker, entry_time, entry_price, entry_volatility_percent, worst_exit_percent, trade_best_exit_percent.\n\n"
        "Return ONLY valid JSON (no markdown, no extra text) in exactly this structure:\n"
        "{\n"
        '  "passed": true,\n'
        '  "critique": "short actionable critique",\n'
        '  "scores": {\n'
        '    "completeness": 1,\n'
        '    "consistency": 1,\n'
        '    "realism": 1\n'
        "  }\n"
        "}\n\n"
        "Set passed=true only if all scores are >= 4 and no required field is missing/empty."
    )

    payload = {
        "model": judge_model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict financial synthetic-data quality judge. "
                    "Always output machine-parseable JSON only."
                )
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 350
    }

    raw_judge_response = ""
    max_api_attempts = 2
    last_error = ""
    attempts_used = 0

    for attempt in range(1, max_api_attempts + 1):
        attempts_used = attempt
        try:
            response = requests.post(judge_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            response_json = response.json()
            raw_judge_response = response_json["choices"][0]["message"]["content"]

            parsed_payload = Parse_First_Json_Object(raw_judge_response)
            if not isinstance(parsed_payload, dict):
                raise ValueError("Judge response did not contain a parseable JSON object.")

            normalized = Normalize_Judge_Payload(
                parsed_payload,
                "Judge response was missing critique details."
            )
            return {
                "judge": normalized,
                "judge_raw_response": raw_judge_response,
                "clean_trade_json": clean_trade_json,
                "few_shot_examples": str(few_shot_examples),
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

    normalized = {
        "passed": False,
        "critique": "api error",
        "scores": {
            "completeness": 1,
            "consistency": 1,
            "realism": 1
        }
    }
    return {
        "judge": normalized,
        "judge_raw_response": raw_judge_response,
        "clean_trade_json": clean_trade_json,
        "few_shot_examples": str(few_shot_examples),
        "api_error": True,
        "api_error_stage": "judge",
        "api_error_message": (
            f"Judge API/parse failure after {max_api_attempts} attempts: {last_error}"
        ),
        "api_attempts": attempts_used
    }


class RowJudgeRefinerState(TypedDict, total=False):
    synthetic_json_current: str
    few_shot_examples: str
    target_ticker: str
    target_volatility: str
    judge_url: str
    judge_model_id: str
    headers: Dict[str, str]
    max_refiner_attempts: int
    refiner_attempts: int
    normalized: Dict[str, Any]
    raw_judge_response: str
    clean_trade_json: str
    refiner_history: List[Dict[str, Any]]
    stop_after_refiner_error: bool
    last_step_was_refiner: bool
    api_error: bool
    api_error_stage: str
    api_error_message: str
    api_attempts: int


def Judge_Row_Node(state: RowJudgeRefinerState):
    judge_result = Judge_One_Row(
        state.get("synthetic_json_current", ""),
        state.get("few_shot_examples", ""),
        state["judge_url"],
        state["judge_model_id"],
        state["headers"]
    )

    normalized = judge_result["judge"]
    raw_judge_response = judge_result["judge_raw_response"]
    clean_trade_json = judge_result["clean_trade_json"]
    normalized_few_shot = judge_result["few_shot_examples"]

    # Attach post-refine judge results to the last refiner attempt log.
    refiner_history = list(state.get("refiner_history", []))
    if state.get("last_step_was_refiner", False) and refiner_history:
        refiner_history[-1]["judge_after_refine"] = normalized
        refiner_history[-1]["judge_raw_response_after_refine"] = raw_judge_response

    return {
        "normalized": normalized,
        "raw_judge_response": raw_judge_response,
        "clean_trade_json": clean_trade_json,
        "few_shot_examples": normalized_few_shot,
        "refiner_history": refiner_history,
        "last_step_was_refiner": False,
        "api_error": bool(judge_result.get("api_error", False)),
        "api_error_stage": str(judge_result.get("api_error_stage", "")),
        "api_error_message": str(judge_result.get("api_error_message", "")),
        "api_attempts": int(judge_result.get("api_attempts", 0))
    }


def Route_After_Judge(state: RowJudgeRefinerState):
    if bool(state.get("api_error", False)):
        return END

    normalized = state.get("normalized", {})
    passed = bool(normalized.get("passed", False))
    refiner_attempts = int(state.get("refiner_attempts", 0))
    max_refiner_attempts = int(state.get("max_refiner_attempts", 2))

    if passed:
        return END

    if refiner_attempts >= max_refiner_attempts:
        return END

    return "refiner"


def Refiner_Row_Node(state: RowJudgeRefinerState):
    refiner_attempts = int(state.get("refiner_attempts", 0)) + 1
    normalized = state.get("normalized", {})
    synthetic_json_current = state.get("synthetic_json_current", "")
    normalized_few_shot = state.get("few_shot_examples", "")
    refiner_history = list(state.get("refiner_history", []))

    refine_result = Refiner_System.Refine_Trade_Row(
        original_trade_json_raw=synthetic_json_current,
        few_shot_examples=normalized_few_shot,
        judge_critique=normalized.get("critique", ""),
        target_ticker=state.get("target_ticker", ""),
        target_volatility=state.get("target_volatility", "")
    )

    synthetic_json_current = refine_result["refined_json_text"]
    attempt_log = {
        "attempt": refiner_attempts,
        "judge_critique_in": normalized.get("critique", ""),
        "refiner_success": bool(refine_result["success"]),
        "api_error": bool(refine_result.get("api_error", False)),
        "refiner_error": refine_result["error"],
        "refiner_raw_response": refine_result["raw_refiner_response"]
    }

    if not refine_result["success"]:
        is_api_error = bool(refine_result.get("api_error", False))
        normalized = {
            "passed": False,
            "critique": "api error" if is_api_error else (
                f"{normalized.get('critique', '')} | "
                f"{refine_result['error']}"
            ).strip(),
            "scores": normalized.get("scores", {
                "completeness": 1,
                "consistency": 1,
                "realism": 1
            })
        }
        refiner_history.append(attempt_log)
        return {
            "synthetic_json_current": synthetic_json_current,
            "refiner_attempts": refiner_attempts,
            "normalized": normalized,
            "refiner_history": refiner_history,
            "stop_after_refiner_error": True,
            "last_step_was_refiner": False,
            "api_error": is_api_error,
            "api_error_stage": str(refine_result.get("api_error_stage", "")),
            "api_error_message": str(refine_result.get("api_error_message", refine_result["error"])),
            "api_attempts": int(refine_result.get("api_attempts", 0))
        }

    refiner_history.append(attempt_log)
    return {
        "synthetic_json_current": synthetic_json_current,
        "refiner_attempts": refiner_attempts,
        "refiner_history": refiner_history,
        "stop_after_refiner_error": False,
        "last_step_was_refiner": True,
        "api_error": False,
        "api_error_stage": "",
        "api_error_message": "",
        "api_attempts": int(refine_result.get("api_attempts", 0))
    }


def Route_After_Refiner(state: RowJudgeRefinerState):
    if bool(state.get("api_error", False)):
        return END

    if state.get("stop_after_refiner_error", False):
        return END

    return "judge"


def Build_Row_Judge_Refiner_Graph():
    workflow = StateGraph(RowJudgeRefinerState)

    workflow.add_node("judge", Judge_Row_Node)
    workflow.add_node("refiner", Refiner_Row_Node)

    workflow.add_edge(START, "judge")
    workflow.add_conditional_edges("judge", Route_After_Judge)
    workflow.add_conditional_edges("refiner", Route_After_Refiner)

    return workflow.compile()


'''
Judge each generated trade against:
1) Completeness
2) Consistency
3) Realism compared to few-shot gold examples

Returns predictable JSON per row so the future refiner can consume:
{
  "passed": <bool>,
  "critique": <string>,
  "scores": {"completeness": 1-5, "consistency": 1-5, "realism": 1-5}
}'''
def Run_Judge_Model(results_df, max_refiner_attempts=2):
    print("\n=== Starting Judge Evaluation via NIM LLM...")

    if results_df is None or results_df.empty:
        print("No generator rows found. Skipping judge.")
        return None

    judge_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    judge_model_id = "meta/llama-3.1-8b-instruct"
    headers = {
        "Authorization": f"Bearer {os.environ['NIM_API_KEY']}",
        "Content-Type": "application/json"
    }

    # Initialize columns on the original dataframe so downstream steps can filter.
    results_df["judge_passed"] = False
    results_df["judge_critique"] = ""
    results_df["judge_completeness"] = 1
    results_df["judge_consistency"] = 1
    results_df["judge_realism"] = 1
    results_df["refiner_attempts"] = 0
    results_df["refiner_applied"] = False
    results_df["refiner_history"] = "[]"
    results_df["api_error"] = False
    results_df["api_error_stage"] = ""
    results_df["api_error_message"] = ""

    evaluations = []
    total_rows = len(results_df)
    row_judge_refiner_app = Build_Row_Judge_Refiner_Graph()

    for idx, row in results_df.iterrows():
        synthetic_json_current = row.get("new_synthetic_trade_json", "")
        few_shot_examples = row.get("few_shot_examples", "")
        target_ticker = str(row.get("ticker", ""))
        target_volatility = str(row.get("target_volatility", ""))

        row_state = {
            "synthetic_json_current": synthetic_json_current,
            "few_shot_examples": few_shot_examples,
            "target_ticker": target_ticker,
            "target_volatility": target_volatility,
            "judge_url": judge_url,
            "judge_model_id": judge_model_id,
            "headers": headers,
            "max_refiner_attempts": int(max_refiner_attempts),
            "refiner_attempts": 0,
            "refiner_history": [],
            "stop_after_refiner_error": False,
            "last_step_was_refiner": False,
            "api_error": False,
            "api_error_stage": "",
            "api_error_message": "",
            "api_attempts": 0
        }
        final_state = row_judge_refiner_app.invoke(row_state)

        synthetic_json_current = final_state.get("synthetic_json_current", synthetic_json_current)
        normalized_few_shot = final_state.get("few_shot_examples", str(few_shot_examples or ""))
        normalized = final_state.get("normalized", {
            "passed": False,
            "critique": "Judge graph did not return a normalized payload.",
            "scores": {
                "completeness": 1,
                "consistency": 1,
                "realism": 1
            }
        })
        raw_judge_response = final_state.get("raw_judge_response", "")
        clean_trade_json = final_state.get("clean_trade_json", "")
        refiner_attempts = int(final_state.get("refiner_attempts", 0))
        refiner_history = final_state.get("refiner_history", [])
        api_error = bool(final_state.get("api_error", False))
        api_error_stage = str(final_state.get("api_error_stage", ""))
        api_error_message = str(final_state.get("api_error_message", ""))

        # Persist refined JSON back to the dataframe so downstream unpacking uses the final row.
        results_df.at[idx, "new_synthetic_trade_json"] = synthetic_json_current
        results_df.at[idx, "judge_passed"] = bool(normalized["passed"])
        results_df.at[idx, "judge_critique"] = "api error" if api_error else normalized["critique"]
        results_df.at[idx, "judge_completeness"] = normalized["scores"]["completeness"]
        results_df.at[idx, "judge_consistency"] = normalized["scores"]["consistency"]
        results_df.at[idx, "judge_realism"] = normalized["scores"]["realism"]
        results_df.at[idx, "refiner_attempts"] = int(refiner_attempts)
        results_df.at[idx, "refiner_applied"] = bool(refiner_attempts > 0)
        results_df.at[idx, "refiner_history"] = json.dumps(refiner_history, ensure_ascii=True)
        results_df.at[idx, "api_error"] = bool(api_error)
        results_df.at[idx, "api_error_stage"] = api_error_stage
        results_df.at[idx, "api_error_message"] = api_error_message

        evaluations.append({
            "row_index": int(idx),
            "ticker": target_ticker,
            "target_volatility": target_volatility,
            "refiner_attempts": int(refiner_attempts),
            "synthetic_trade_json": clean_trade_json,
            "few_shot_examples": normalized_few_shot,
            "judge": normalized,
            "judge_raw_response": raw_judge_response,
            "refiner_history": refiner_history,
            "api_error": bool(api_error),
            "api_error_stage": api_error_stage,
            "api_error_message": api_error_message
        })

        print(
            f"Judged row {idx + 1}/{total_rows} - "
            f"passed={normalized['passed']} - refiner_attempts={refiner_attempts} - api_error={api_error}",
            end='\r'
        )

    print("")

    passed_count = int(results_df["judge_passed"].sum())
    failed_count = int(total_rows - passed_count)
    api_error_count = int(results_df["api_error"].sum()) if "api_error" in results_df.columns else 0

    summary = {
        "total_rows": total_rows,
        "passed_rows": passed_count,
        "failed_rows": failed_count,
        "api_error_rows": api_error_count
    }

    output_payload = {
        "summary": summary,
        "results": evaluations
    }

    eval_output_path = "data/synthetic_data/judge_evaluation_results.json"
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
    with open(eval_output_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    # Save failed rows now so refiner wiring can consume this later.
    if failed_count > 0:
        failed_output_path = f"data/synthetic_data/judge_failed_rows_{int(time.time())}.csv"
        failed_df = results_df[results_df["judge_passed"] == False].copy()  # noqa: E712
        failed_df.to_csv(failed_output_path, index=False)
        print(f"Saved failed rows for future refiner step to {failed_output_path}")

    print(f"Saved judge results to {eval_output_path}")
    print(f"Judge summary: {passed_count} passed / {failed_count} failed")

    return output_payload