import data_designer.config as dd
from data_designer.interface import DataDesigner
from nemo_microservices.data_designer.essentials import (
    DataDesignerConfigBuilder,
    NeMoDataDesignerClient,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    CategorySamplerParams
)
from nemo_microservices.data_designer.config.seed import DatastoreSeedDatasetReference
from nemo_microservices import NeMoMicroservices
from huggingface_hub import HfApi
import Setup_And_Cleaning
from Models import model_hash_map
import os
import json
import requests
import sys
import pandas as pd
import time
from datetime import date
import Sampler_System


# misc configs
prod_level = 'development'
models = model_hash_map[prod_level]  # list of model config obj's at this dev level
NAMESPACE="data"
PROJECT="stock-trade-data-generation"
DATASET_FILE_NAME = "edited_bulk_summary.csv"
SEED_DATASET_PATH = f"data/human_data/{DATASET_FILE_NAME}"

# preprocessing / setup
Setup_And_Cleaning.System_Startup(NAMESPACE, PROJECT)
huggingFace_Api = HfApi(token=os.environ["HUGGINGFACE_API_KEY"])  # this is remote hf
HF_USERNAME, HF_REPO_NAME = Setup_And_Cleaning.Load_Secrets()     # my username, and repo name we're using
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"
hf_hub_endpoint = "https://huggingface.co"
# these 2 lines are for local hf
# huggingFace_Api = HfApi(endpoint=os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"], token=os.environ["HUGGINGFACE_API_KEY"])
# HF_ENDPOINT=f"{os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"]}/v1/hf"

# nemo stuff
data_designer = DataDesigner()
config_builder = DataDesignerConfigBuilder(models)
data_designer_client = NeMoDataDesignerClient(base_url=os.environ["NEMO_MICROSERVICES_BASE_URL"])

# set globals in other files
Sampler_System.Set_Globals(SEED_DATASET_PATH)


# send post request of our dataset to the entity store
# this does the same thing as NeMoDataDesignerClient.upload_seed_dataset
# user can select to skip this is the data already exists there
def Send_To_EntityStore():
    dataset_name = "seed_trades"

    try:
        # grab all the datasets via the huggingFace_Api and look for a dataset with the same name as dataset_name
        datasets = list(huggingFace_Api.list_datasets(search=dataset_name))
        
        for dataset in datasets:
             if dataset_name in dataset.id:
                print(f"Dataset '{dataset_name}' already exists (ID: {dataset.id}).")
                user_response = input("Do you want to overwrite it? (y/n): ").strip().lower()
                
                if user_response != 'y':
                    print("Skipping dataset upload.")
                    return dataset
                break

    except Exception as e:
        print(f"Error checking for existing dataset: {e}")

    dataset_info = {
        'name': dataset_name, # Matching the dataset name from repo_id
        'namespace': NAMESPACE,
        'description': "human-made-stock-trades",
        'format': 'csv',
        'files_url': f"hf://datasets/{NAMESPACE}/seed_trades",
        'project': "stock-trade-data-generation",
        'custom_fields': {},
        'ownership': {
            "created_by": "local_user",
            "access_policies": {} # Empty is fine for local private deployment
        }
    }

    response = requests.post(
        f"{os.environ["ENTITY_STORE_BASE_URL"]}/v1/datasets",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json"
        },
        json = dataset_info
    )
    response_data = response.json()
    
    return dataset
    

# send files to that repo in hf (training, testing, validation, complete set)
# hugging face hub
# user can select to skip this is the data already exists there
def Send_To_HF(repo_id):
    # create the repo
    try:
        if huggingFace_Api.repo_exists(repo_id=repo_id, repo_type="dataset"):
            print(f"Repo {repo_id} already exists")
        else:
            huggingFace_Api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)

    except Exception as e:
        print(f"Error checking/creating repo: {e}")
        # Try to proceed or re-raise depending on logic, but typically we want to ensure repo exists
        try:
             huggingFace_Api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
        except Exception as e2:
             print(f"Could not create repo: {e2}")

    # if files already exist then we need user permission to overwrite them or to skip it
    try:
        existing_files = huggingFace_Api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        files_to_upload = [
            "training/training.csv",
            "validation/validation.csv",
            "testing/testing.csv",
            "complete_set/edited_bulk_summary.csv"
        ]
        
        # Find conflicts
        conflicts = []
        for f in files_to_upload:
            if f in existing_files:
                conflicts.append(f)
        
        if conflicts:
            print(f"\nThe following files already exist in the repo '{repo_id}':")
            for f in conflicts:
                print(f"  - {f}")
            
            should_overwrite = input("Do you want to overwrite them? (y/n): ").strip().lower()
            if should_overwrite != 'y':
                print("Upload cancelled by user.")
                return

    except Exception as e:
        print(f"Warning: Could not check existing files in repo (Error: {e})")
        # Proceeding cautiously or you could return here if strictly required
        
    # we're doing complete_set, training, validation, testing datasets
    try:
        # training
        huggingFace_Api.upload_folder(
            folder_path="data/human_data/splits/training",
            path_in_repo="training",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # validation
        huggingFace_Api.upload_folder(
            folder_path="data/human_data/splits/validation",
            path_in_repo="validation",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # testing
        huggingFace_Api.upload_folder(
            folder_path="data/human_data/splits/testing",
            path_in_repo="testing",
            repo_id=repo_id,
            repo_type="dataset"
        )

        # complete_set
        huggingFace_Api.upload_file(
            path_or_fileobj="data/human_data/edited_bulk_summary.csv",
            path_in_repo="complete_set/edited_bulk_summary.csv",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        print(f"Successfully uploaded files to {repo_id}")

    except Exception as e:
        print(f"Error uploading files: {e}")


# connect to the entity store. we should have already sent out dataset there
# get a reference to the dataset, then connect
def Connect_To_Entity_Store(seed_dataset_id):    
    try:
        # we need the path of the dataset in the entity store
        # list all datasets
        print(f"Connecting to Data Store at: {huggingFace_Api}")

        ''' this is for finding the dataset reference and seeing the files
        DO NOT REMOVE, it could be useful
        dataset_name = "seed_trades"
        datasets = list(huggingFace_Api.list_datasets(search=dataset_name))
        for ds in datasets:
            if (ds.id == seed_dataset_id):
                try:
                    files = huggingFace_Api.list_repo_files(repo_id=ds.id, repo_type="dataset")

                    for f in files:
                        print(f"   └─ File: {f}")
                        # This construct shows you the path you likely need:
                        print(f"      Reference Path: {ds.id}/{f}")

                except Exception as e:
                    print(f"   (Could not list files: {e})")
        '''        

        # Create reference to existing dataset in the datastore
        # Since the dataset is on the remote Hugging Face Hub, we point to that endpoint.
        seed_dataset_reference = DatastoreSeedDatasetReference(
            dataset=f"{seed_dataset_id}/complete_set/{DATASET_FILE_NAME}",  # TODO: hardcoded
            datastore_settings={
                "endpoint": hf_hub_endpoint,
                "token": os.environ["HUGGINGFACE_API_KEY"]
            },
        )

        config_builder.with_seed_dataset(
            dataset_reference=seed_dataset_reference,
            sampling_strategy="shuffle" # or "ordered"
        )
        
    except Exception as e:
        print(f"error adding dataset to config: {e}")
        sys.exit(1)





# we don't map columns in a basic sense, we're referenceing them inside prompts via jinja2 prompts
# control diversity (distribution of data) and make the llm prompt which applies to each dataset row
def Map_Columns_To_Models():
    '''
    These are all the columns in the edited csv: 
        ticker, entry_time, entry_price, exit_price, trade_type, worst_exit_percent, trade_best_exit_percent, trade_percent_change, 
        entry_volatility_percent
    '''
    # sampler columns (diversity). this targets high impact variables to ensure the llm does correct distribution
    volatility_sampler = SamplerColumnConfig(
        name="target_volatility", 
        column_type="sampler",
        sampler_type="category",
        params=CategorySamplerParams(
            values = [0.3,  0.4,  0.5, 0.6,  0.7, 0.8, 0.9,  1.0,  1.1],
            weights= [0.05, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.05, 0.05]
        )
    )

    # llm column (data generation) - FEW-SHOT approach
    '''
    iterations of this
    - old: one-shot approach where each row's own columns were the only context. caused mode collapse.
    - old: I added noise of +/- 10% to the volatility. this is replaced by the sampler column
    - current: few-shot. The sampler pre-selects 3 gold data points with similar volatility and 
      stores them in the 'few_shot_examples' column. The generator sees these as grounding examples
      so it learns the realistic distribution without averaging everything out.
    '''
    synthetic_trade_generator = LLMTextColumnConfig(
        name="new_synthetic_trade_json", 
        
        prompt="""
        You are a financial data simulation engine. You generate realistic synthetic stock trades.

        Below are 3 REAL trades from a human trader that occurred at similar volatility levels. 
        Study their patterns - price ranges, exit percentages, and how they relate to volatility.

        === REFERENCE TRADES (real human data) ===
        {{ few_shot_examples }}
        ============================================

        SIMULATION PARAMETERS:
        - Target Volatility: {{ target_volatility }}
        - Ticker (from seed row): {{ ticker }}

        Task:
        Generate ONE new synthetic trade that is realistic for the ticker '{{ ticker }}' at a 
        volatility of {{ target_volatility }}. The trade should look like it belongs alongside 
        the reference trades above, but must be DISTINCT (not a copy).

        Rules:
        1. **Learn from the examples:** The entry prices, exit percents, and volatility values 
           in the reference trades show you what realistic ranges look like. Stay within those 
           patterns but do not duplicate any example exactly.
        2. **Volatility drives exits:** Higher target volatility means wider worst_exit_percent 
           and trade_best_exit_percent. The exit percentages should scale proportionally with 
           the target volatility.
        3. **Price realism:** The entry_price should be realistic for the ticker. Look at the 
           reference trade prices for guidance.
        4. **Time jitter:** Pick an entry_time that is plausible (market hours) and close to 
           but not identical to the reference times.
        5. **Internal consistency:** All fields must be mathematically consistent with each other 
           and with the target volatility.

        Output strictly valid JSON with keys: "ticker", "entry_time", "entry_price", "entry_volatility_percent", "worst_exit_percent", "trade_best_exit_percent".
        """,
        model_alias="generator-model"
    )

    config_builder.add_column(volatility_sampler)
    config_builder.add_column(synthetic_trade_generator)
    

# TESTING VERSION (basic generation to test the code works)
def Run_Generator_Model_PREVIEW():
    print("Starting PREVIEW generation job...")

    # Verification: Check that seed config is active so user knows data is present
    if config_builder._seed_config:
         print(f"Seed Dataset Linked: {config_builder._seed_config.dataset}")
    else:
         print("WARNING: No seed dataset linked in config!")

    # Fix for 422 Error: Remove seed-dataset columns from config as server might be older (just remove metadata, not actual data)
    #     and doesn't recognize them, causing validation error.
    removed_cols = {}
    
    # 1. Identify columns to remove
    for name, col in list(config_builder._column_configs.items()):
        if getattr(col, 'column_type', '') == 'seed-dataset':
            removed_cols[name] = config_builder._column_configs.pop(name)
            
    # 2. Capture the keys of removed columns
    removed_keys = list(removed_cols.keys())
    
    if removed_keys:
        print(f"Temporarily hiding seed columns for server compatibility: {len(removed_keys)} columns")
        
        # 3. Store original allowed_references property
        original_allowed_references_prop = DataDesignerConfigBuilder.allowed_references
        
        # 4. Define a patched property that adds our removed keys back to the allowed list
        # This tricks the client-side validation into thinking the columns are still there
        def patched_allowed_references(self):
            # Get original list based on current (remaining) columns
            current_refs = []
            side_effect_columns = sum([[c.name] + c.side_effect_columns for c in self._column_configs.values()], [])
            current_refs = list(self._column_configs.keys()) + list(set(side_effect_columns))
            
            # Add back the removed keys
            return current_refs + removed_keys

        # Apply patch to the CLASS (or instance if possible, but property is on class)
        # Python properties are descriptors on the class. To patch just for this instance is hard without overriding __class__.
        # So we'll patch the class and restore it immediately in finally block.
        DataDesignerConfigBuilder.allowed_references = property(patched_allowed_references)

    try:
        preview = data_designer_client.preview(config_builder)
    finally:
        # Restore everything
        if removed_keys:
            # Restore class property
            DataDesignerConfigBuilder.allowed_references = original_allowed_references_prop
            
            # Restore columns
            print("Restoring seed columns...")
            config_builder._column_configs.update(removed_cols)
    for i in range(10):
        preview.display_sample_record()
    
    #preview.dataset.to_csv(output_file_name, index = False)
    #print(f"saved raw generator results to {output_file_name}")

    # run analysis on it
    print(preview.analysis.to_report())

    return preview.dataset

# REAL VERSION (full generation)
# Use Run_Generator_Model_PREVIEW() for testing 
def Run_Generator_Model_FULL():
    output_file_name = "data/synthetic_data/Raw_Generator_Model_Results_FULL.csv"
    
    # Max time to wait (e.g., 30 minutes = 1800 seconds)
    MAX_JOB_TIME = 1800 

    print("Starting FULL generation job...")
    job = data_designer_client.create_job(config_builder)
    print(f"Job ID: {job.id}")

    start_time = time.time()
    while not job.is_done():
        elapsed = time.time() - start_time
        if elapsed > MAX_JOB_TIME:
            print(f"Job exceeded {MAX_JOB_TIME} seconds. Stopping to prevent overage.")
            # Depending on the client, you might want to cancel. 
            # job.cancel() # Uncomment if cancel method is available
            raise TimeoutError("Job timed out to save costs.")
            
        time.sleep(5)
        job.refresh()
        print(f"Job status: {job.status} (Elapsed: {int(elapsed)}s)", end='\r')

    print("\nJob Complete!")

    results_df = data_designer_client.download_dataset(job.id)  # returns a df with the new json column
    results_df.to_csv(output_file_name, index = False)
    print(f"saved raw generator results to {output_file_name}")

    return results_df


# generator model returns a dataframe that's the same as the original, but it has 1 new column called "new_synthetic_trade_json"
#     which is a json of the new data. it will only have valid core data, the rest is random numbers we'll need to caluclate later
def Unpack_Synthetic_Data(results_df):
    dest_path = "data/synthetic_data/generator_results.csv"
    
    # Check if "Real Data" column exists, if not create it
    if "Real Data" not in results_df.columns:
        results_df["Real Data"] = "real"
    else:
        # Ensure existing rows are marked as real
        results_df["Real Data"] = "real"
    
    new_rows = []
    today_str = date.today().strftime("%Y-%m-%d")
    
    print("Unpacking synthetic trades...")
    for index, row in results_df.iterrows():
        try:
            json_content = row['new_synthetic_trade_json']
            
            # Clean markdown if present
            if isinstance(json_content, str):
                clean_json = json_content.strip()
                if clean_json.startswith('```json'):
                    clean_json = clean_json[7:]
                if clean_json.startswith('```'):
                    clean_json = clean_json[3:]
                if clean_json.endswith('```'):
                    clean_json = clean_json[:-3]
                
                trade_data = json.loads(clean_json.strip())
                # Add the Real Data tag to the new row
                trade_data["Real Data"] = f"synthetic {today_str}"
                new_rows.append(trade_data)
                
        except Exception as e:
            print(f"Error parsing JSON at index {index}: {e}")
            continue
            
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Append new rows to the dataframe
        results_df = pd.concat([results_df, new_df], ignore_index=True)
        
    # Save with timestamp
    filename = dest_path.split('/')[-1]
    name_parts = filename.split('.')
    timestamp = int(time.time())
    new_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
    
    output_dir = os.path.dirname(dest_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    final_path = os.path.join(output_dir, new_filename)
    results_df.to_csv(final_path, index=False)

    print(f"Saved unpacked data to {final_path}")


''' 
Use NeMo Evaluator with LLM-as-a-Judge to grade each result from the generator
This sends the generator output + original reference data to a judge LLM which scores
each row on realism, consistency, and completeness.
Uses the "data" task type (no target model inference needed, we already have outputs).
basic overview: https://docs.nvidia.com/nemo/microservices/latest/evaluate/index.html
use llm as a judge: https://docs.nvidia.com/nemo/microservices/latest/evaluate/flows/llm-as-a-judge.html
'''
def Run_Judge_Model(results_df):
    print("\n=== Starting Judge Evaluation via NeMo Evaluator...")

    evaluator_client = NeMoMicroservices(base_url=os.environ["EVALUATOR_BASE_URL"])

    # Build rows from generator output for the evaluator
    # Each row contains the original seed context + the synthetic trade JSON the generator produced
    rows = []
    for _, row in results_df.iterrows():
        synthetic_json = str(row.get("new_synthetic_trade_json", ""))
        if not synthetic_json or synthetic_json == "nan":
            continue
        rows.append({
            "ticker": str(row.get("ticker", "")),
            "original_entry_price": str(row.get("entry_price", "")),
            "original_volatility": str(row.get("entry_volatility_percent", "")),
            "target_volatility": str(row.get("target_volatility", "")),
            "synthetic_trade_json": synthetic_json
        })

    if not rows:
        print("No synthetic trades to evaluate. Skipping judge.")
        return None

    print(f"Evaluating {len(rows)} synthetic trades...")

    # Judge model config - uses an instruct model from build.nvidia.com
    # (base models can't follow judge formatting instructions, so we use an instruct variant)
    judge_model = { # TODO: need to use the config builder to get model info
        "api_endpoint": {
            "url": "https://integrate.api.nvidia.com/v1/chat/completions",
            "model_id": "meta/llama-3.1-8b-instruct", 
            "api_key": os.environ["NIM_API_KEY"]
        }
    }

    # Evaluation config: "data" task with LLM-as-a-judge metric.
    # The judge sees each row's original reference data + the synthetic output,
    # then scores on 3 criteria using a strict format we parse with regex.
    config = {
        "type": "custom",
        "tasks": {
            "synthetic-trade-quality": {
                "type": "data",
                "metrics": {
                    "trade-quality-judge": {
                        "type": "llm-judge",
                        "params": {
                            "model": judge_model,
                            "template": {
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a financial data quality auditor specializing in synthetic trading data. "
                                            "You evaluate whether synthetically generated stock trades are realistic and internally consistent."
                                        )
                                    },
                                    {
                                        "role": "user",
                                        "content": (
                                            "Evaluate the quality of this synthetic trade.\n\n"
                                            "ORIGINAL REFERENCE DATA:\n"
                                            "- Ticker: {{item.ticker}}\n"
                                            "- Original Entry Price: ${{item.original_entry_price}}\n"
                                            "- Original Entry Volatility: {{item.original_volatility}}%\n"
                                            "- Target Volatility for Synthesis: {{item.target_volatility}}%\n\n"
                                            "GENERATED SYNTHETIC TRADE:\n"
                                            "{{item.synthetic_trade_json}}\n\n"
                                            "Score each criterion from 1 (poor) to 5 (excellent):\n\n"
                                            "1. REALISM: Are the generated price, volatility, and exit percent values "
                                            "within a plausible range for the given ticker? Prices should be positive, "
                                            "volatility should be a small percentage, exit percents should be realistic.\n\n"
                                            "2. CONSISTENCY: Do the fields form a coherent trade? The entry_volatility_percent "
                                            "should reflect the target volatility. The worst_exit_percent and "
                                            "trade_best_exit_percent should scale proportionally with volatility. "
                                            "Higher target volatility should mean wider exit ranges.\n\n"
                                            "3. COMPLETENESS: Does the output contain all required keys "
                                            "(ticker, entry_time, entry_price, entry_volatility_percent, "
                                            "worst_exit_percent, trade_best_exit_percent) with non-empty, "
                                            "correctly-typed values?\n\n"
                                            "You MUST respond exactly in this format:\n"
                                            "REALISM: <score>\n"
                                            "CONSISTENCY: <score>\n"
                                            "COMPLETENESS: <score>"
                                        )
                                    }
                                ],
                                "max_tokens": 256,  # TODO: figure out how to use the config builder for this
                                "temperature": 0.1  # TODO: figure out how to use the config builder for this
                            },
                            "scores": {
                                "realism": {
                                    "type": "int",
                                    "parser": {
                                        "type": "regex",
                                        "pattern": "REALISM:\\s*(\\d)"
                                    }
                                },
                                "consistency": {
                                    "type": "int",
                                    "parser": {
                                        "type": "regex",
                                        "pattern": "CONSISTENCY:\\s*(\\d)"
                                    }
                                },
                                "completeness": {
                                    "type": "int",
                                    "parser": {
                                        "type": "regex",
                                        "pattern": "COMPLETENESS:\\s*(\\d)"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    # Target: pass generator results directly as rows (no dataset upload needed)
    target = {
        "type": "rows",
        "rows": rows
    }

    # Submit the evaluation job
    try:
        job = evaluator_client.v2.evaluation.jobs.create(
            spec={
                "target": target,
                "config": config
            }
        )
    except Exception as e:
        print(f"Failed to create evaluation job: {e}")
        print("Verify the evaluator container is running (docker ps) and EVALUATOR_BASE_URL is correct.")
        return None

    job_id = job.id
    print(f"Evaluation Job ID: {job_id}")

    # Poll for completion
    MAX_EVAL_TIME = 600  # 10 minutes
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > MAX_EVAL_TIME:
            print(f"\nEvaluation job exceeded {MAX_EVAL_TIME}s timeout.")
            break

        try:
            job_status = evaluator_client.v2.evaluation.jobs.status.retrieve(job_id)
            status = job_status.status
        except Exception as e:
            print(f"\nError checking job status: {e}")
            time.sleep(10)
            continue

        if status == "completed":
            print(f"\nEvaluation completed! (took {int(elapsed)}s)")
            break
        elif status in ("failed", "error", "cancelled"):
            print(f"\nEvaluation ended with status: {status}")
            if hasattr(job_status, 'error_details') and job_status.error_details:
                print(f"Error details: {job_status.error_details}")
            return None

        print(f"  Evaluation status: {status} (Elapsed: {int(elapsed)}s)", end='\r')
        time.sleep(10)

    # Retrieve and display results
    try:
        results = evaluator_client.v2.evaluation.jobs.results.evaluation_results.retrieve(job_id)
        results_json = results.model_dump_json(indent=2, exclude_none=True)

        print("\n=== JUDGE EVALUATION RESULTS ===")
        print(results_json)

        # Save results to file
        eval_output_path = "data/synthetic_data/judge_evaluation_results.json"
        os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
        with open(eval_output_path, "w") as f:
            f.write(results_json)
        print(f"\nSaved judge results to {eval_output_path}")

        return results

    except Exception as e:
        print(f"Error retrieving evaluation results: {e}")
        return None


def Main():
    # 1. Clean the raw human data into the seed CSV
    Setup_And_Cleaning.Clean_Seed_Data(SEED_DATASET_PATH)

    # 2. SAMPLER: Enrich each row with 3 few-shot examples from similar volatility rows.
    #    This writes a 'few_shot_examples' column into the seed CSV so the generator
    #    prompt can reference {{ few_shot_examples }} via Jinja2.
    #    Must happen BEFORE uploading to entity store / HF since they need the enriched CSV.
    Sampler_System.Enrich_Seed_With_Few_Shot()

    # 3. Upload enriched seed data to entity store and HF
    seed_dataset = Send_To_EntityStore()                    # send dataset to entity store
    seed_dataset_id = seed_dataset.id                       # get dataset id

    Setup_And_Cleaning.Split_Dataset(SEED_DATASET_PATH)     # split into training/validation/testing
    Send_To_HF(HF_REPO_ID)                                  # upload to hf

    # 4. Connect config builder to the entity store seed data
    Connect_To_Entity_Store(seed_dataset_id)

    # 5. GENERATOR: Map columns (volatility sampler + LLM generator prompt using few-shot examples)
    Map_Columns_To_Models()

    config_builder.validate()                               # test columns are config'd right

    # 6. Run generator (choose preview or full)
    # RUN BASIC PREVIEW (low cost)
    results_df = Run_Generator_Model_PREVIEW()

    # RUN FULL PROCESS (high cost)
    #results_df = Run_Generator_Model_FULL()

    # 7. JUDGE: Grade each synthetic trade via LLM judge
    #    Note: Without the refiner model, the judge scores but can't send failures back for rework.
    #    Once the refiner is built, failed rows will loop: judge -> refiner -> judge (max 2 retries).
    Run_Judge_Model(results_df)

    # 8. Unpack the generator's JSON column into individual rows
    Unpack_Synthetic_Data(results_df)