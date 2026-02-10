import sys
import os
import pandas as pd
import json

'''
Loads the gold data and selects 3 data points matching a volatility bucket
This prevents mode collapse by feeding the generator "few-shot" examples
instead of letting it see everything at once (which averages results out)

How it works:
  1. Load gold data (edited_bulk_summary.csv)
  2. For a given target volatility, find all rows within a tolerance band
  3. If not enough rows match exactly, expand the tolerance
  4. Return 3 randomly selected rows from that bucket

The volatility buckets match the CategorySamplerParams values we give to NeMo:
  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
'''

SEED_DATASET_PATH = None
gold_data_cache = None

# globals are in generate_trades.py, rather than use a controller function that has these are params, just set them
def Set_Globals(seed_dataset_path):
    global SEED_DATASET_PATH

    SEED_DATASET_PATH = seed_dataset_path


def Load_Gold_Data():
    # load the cleaned gold dataset into a DataFrame. Cached after first call
    global gold_data_cache, SEED_DATASET_PATH

    if 'gold_data_cache' not in globals() or gold_data_cache is None:
        gold_path = SEED_DATASET_PATH

        if not os.path.exists(gold_path):
            print(f"ERROR: Gold data not found at {gold_path}")
            sys.exit(1)

        gold_data_cache = pd.read_csv(gold_path)
        print(f"Sampler: Loaded {len(gold_data_cache)} gold data rows from {gold_path}")
    
    return gold_data_cache


def Sample_Gold_Data(target_volatility, num_samples=3, base_tolerance=0.05):
    """
    Select a few gold data points whose entry_volatility_percent is close to
    the target_volatility value. This is the "few-shot" sampler.
    
    Args:
        target_volatility: The volatility bucket value (e.g. 0.7)
        num_samples: How many gold rows to return (default 3)
        base_tolerance: Starting +/- tolerance for matching (default 0.05)
    
    Returns:
        List of dicts, each dict is one gold data row
    """
    gold_df = Load_Gold_Data()
    
    # The entry_volatility_percent column may be stored as string; ensure float
    vol_col = gold_df['entry_volatility_percent'].astype(float)
    
    # Widen tolerance until we have enough candidates
    tolerance = base_tolerance
    max_tolerance = 0.5  # safety cap so we don't just return everything
    candidates = pd.DataFrame()
    
    while len(candidates) < num_samples and tolerance <= max_tolerance:
        mask = (vol_col >= target_volatility - tolerance) & (vol_col <= target_volatility + tolerance)
        candidates = gold_df[mask]
        tolerance += 0.05
    
    # If we still don't have enough after max tolerance, just use whatever we have
    if len(candidates) == 0:
        print(f"Sampler WARNING: No rows found near volatility {target_volatility}, using random sample from full dataset")
        candidates = gold_df
    
    # Sample (with replacement if not enough rows)
    replace = len(candidates) < num_samples
    sampled = candidates.sample(n=num_samples, replace=replace, random_state=None)  # None = truly random each call
    
    return sampled.to_dict('records')


def Format_Few_Shot_Examples(examples):
    """
    Format a list of gold data row dicts into a text block for the generator prompt.
    Each example becomes a numbered JSON block the generator can learn from.
    """
    parts = []
    for i, ex in enumerate(examples, 1):
        # Only include core trade fields the generator needs to see
        clean_ex = {
            "ticker": str(ex.get("ticker", "")),
            "entry_time": str(ex.get("entry_time", "")),
            "entry_price": str(ex.get("entry_price", "")),
            "entry_volatility_percent": str(ex.get("entry_volatility_percent", "")),
            "worst_exit_percent": str(ex.get("worst_exit_percent", "")),
            "trade_best_exit_percent": str(ex.get("trade_best_exit_percent", "")),
        }
        parts.append(f"Example {i}:\n{json.dumps(clean_ex, indent=2)}")
    return "\n\n".join(parts)


'''
Preprocessing step: For each row in the gold data, use the sampler to pick 3
other gold rows with similar volatility, and store them as a text column called
'few_shot_examples'. This column can then be referenced in the generator prompt
via Jinja2 ({{ few_shot_examples }}).

The enriched CSV replaces the original so NeMo Data Designer sees the few-shot
examples as just another column.

IMPORTANT: The 3 examples are selected from OTHER rows (excluding the current row)
to avoid the generator just copying its own input.
'''
def Enrich_Seed_With_Few_Shot():
    global gold_data_cache, SEED_DATASET_PATH

    gold_df = Load_Gold_Data()
    vol_col = gold_df['entry_volatility_percent'].astype(float)
    
    few_shot_texts = []
    
    for idx, row in gold_df.iterrows():
        target_vol = float(row['entry_volatility_percent'])
        
        # Find candidate rows (exclude current row)
        other_df = gold_df.drop(idx)
        other_vol = vol_col.drop(idx)
        
        # Widen tolerance until we have at least 3 candidates
        tolerance = 0.05
        max_tolerance = 0.5
        candidates = pd.DataFrame()
        
        while len(candidates) < 3 and tolerance <= max_tolerance:
            mask = (other_vol >= target_vol - tolerance) & (other_vol <= target_vol + tolerance)
            candidates = other_df[mask]
            tolerance += 0.05
        
        if len(candidates) == 0:
            candidates = other_df  # fallback to full dataset
        
        replace = len(candidates) < 3
        sampled = candidates.sample(n=3, replace=replace)
        examples = sampled.to_dict('records')
        
        few_shot_texts.append(Format_Few_Shot_Examples(examples))
    
    gold_df['few_shot_examples'] = few_shot_texts
    
    # Overwrite the seed CSV with the enriched version
    gold_df.to_csv(SEED_DATASET_PATH, index=False)
    
    # Reset the cache so subsequent loads see the new column
    gold_data_cache = gold_df
    
    print(f"Sampler: Enriched {len(gold_df)} rows with few-shot examples -> {SEED_DATASET_PATH}")