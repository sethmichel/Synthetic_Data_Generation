import pandas as pd
import os
import requests
import json
import time

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
def Run_Judge_Model(results_df):
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

    evaluations = []
    total_rows = len(results_df)

    for idx, row in results_df.iterrows():
        synthetic_json_raw = row.get("new_synthetic_trade_json", "")
        few_shot_examples = row.get("few_shot_examples", "")

        if Bad_Value_Handler(synthetic_json_raw):
            critique = "Generator output is empty or missing; cannot judge this row."
            results_df.at[idx, "judge_passed"] = False
            results_df.at[idx, "judge_critique"] = critique
            evaluations.append({
                "row_index": int(idx),
                "ticker": str(row.get("ticker", "")),
                "target_volatility": str(row.get("target_volatility", "")),
                "judge": {
                    "passed": False,
                    "critique": critique,
                    "scores": {
                        "completeness": 1,
                        "consistency": 1,
                        "realism": 1
                    }
                },
                "judge_raw_response": ""
            })
            continue

        clean_trade_json = Strip_Markdown_Json_Fences(synthetic_json_raw)

        # Keep the original few-shot text exactly as produced by the sampler.
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
        normalized = None

        try:
            response = requests.post(judge_url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            response_json = response.json()
            raw_judge_response = response_json["choices"][0]["message"]["content"]

            cleaned_judge_response = Strip_Markdown_Json_Fences(raw_judge_response)
            parsed_payload = json.loads(cleaned_judge_response)
            normalized = Normalize_Judge_Payload(
                parsed_payload,
                "Judge response was missing critique details."
            )
        except Exception as e:
            normalized = {
                "passed": False,
                "critique": f"Judge API/parse failure: {e}",
                "scores": {
                    "completeness": 1,
                    "consistency": 1,
                    "realism": 1
                }
            }

        results_df.at[idx, "judge_passed"] = bool(normalized["passed"])
        results_df.at[idx, "judge_critique"] = normalized["critique"]
        results_df.at[idx, "judge_completeness"] = normalized["scores"]["completeness"]
        results_df.at[idx, "judge_consistency"] = normalized["scores"]["consistency"]
        results_df.at[idx, "judge_realism"] = normalized["scores"]["realism"]

        evaluations.append({
            "row_index": int(idx),
            "ticker": str(row.get("ticker", "")),
            "target_volatility": str(row.get("target_volatility", "")),
            "synthetic_trade_json": clean_trade_json,
            "few_shot_examples": str(few_shot_examples),
            "judge": normalized,
            "judge_raw_response": raw_judge_response
        })

        print(f"Judged row {idx + 1}/{total_rows} - passed={normalized['passed']}", end='\r')

    print("")

    passed_count = int(results_df["judge_passed"].sum())
    failed_count = int(total_rows - passed_count)

    summary = {
        "total_rows": total_rows,
        "passed_rows": passed_count,
        "failed_rows": failed_count
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