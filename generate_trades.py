import data_designer.config as dd
from data_designer.interface import DataDesigner
from nemo_microservices.data_designer.essentials import (
    DataDesignerConfigBuilder,
    NeMoDataDesignerClient,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SamplerType,
    CategorySamplerParams
)
from nemo_microservices.data_designer.config.seed import SeedDatasetReference, DatastoreSeedDatasetReference
from nemo_microservices import NeMoMicroservices
from huggingface_hub import HfApi
import Setup_And_Cleaning
from Models import model_configs
import os
import csv
import json
import requests
from pathlib import Path
import sys
import pandas as pd
import time
from datetime import date
import inspect




# misc configs
prod_level = Setup_And_Cleaning.MODE_CONFIGS['development']
NAMESPACE="data"
PROJECT="stock-trade-data-generation"
seed_dataset_path = "data/human_data/edited_bulk_summary.csv"

# preprocessing / setup
Setup_And_Cleaning.System_Startup(NAMESPACE, PROJECT)
hf_api = HfApi(token=os.environ["HUGGINGFACE_API_KEY"])        # this is remote hf
HF_USERNAME, HF_REPO_NAME = Setup_And_Cleaning.Load_Secrets()  # my username, and repo name we're using
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"
# these 2 lines are for local hf
# hf_api = HfApi(endpoint=os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"], token=os.environ["HUGGINGFACE_API_KEY"])
# HF_ENDPOINT=f"{os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"]}/v1/hf"

data_designer = DataDesigner()
config_builder = DataDesignerConfigBuilder(model_configs)
data_designer_client = NeMoDataDesignerClient(base_url=os.environ["NEMO_MICROSERVICES_BASE_URL"])

# send post request of our dataset to the entity store
# this does the same thing as NeMoDataDesignerClient.upload_seed_dataset
def Send_Seed_To_EntityStore():
    dataset_name = "seed_trades"

    try:
        # grab all the datasets via the hf_api and look for a dataset with the same name as dataset_name
        datasets = list(hf_api.list_datasets(search=dataset_name))
        
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
def Send_Dataset_Files_HF(repo_id):
    # create the repo
    try:
        if hf_api.repo_exists(repo_id=repo_id, repo_type="dataset"):
            print(f"Repo {repo_id} already exists")
        else:
            hf_api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)

    except Exception as e:
        print(f"Error checking/creating repo: {e}")
        # Try to proceed or re-raise depending on logic, but typically we want to ensure repo exists
        try:
             hf_api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True)
        except Exception as e2:
             print(f"Could not create repo: {e2}")

    # if files already exist then we need user permission to overwrite them or to skip it
    try:
        existing_files = hf_api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        
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
        hf_api.upload_folder(
            folder_path="data/human_data/splits/training",
            path_in_repo="training",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # validation
        hf_api.upload_folder(
            folder_path="data/human_data/splits/validation",
            path_in_repo="validation",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # testing
        hf_api.upload_folder(
            folder_path="data/human_data/splits/testing",
            path_in_repo="testing",
            repo_id=repo_id,
            repo_type="dataset"
        )

        # complete_set
        hf_api.upload_file(
            path_or_fileobj="data/human_data/edited_bulk_summary.csv",
            path_in_repo="complete_set/edited_bulk_summary.csv",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        print(f"Successfully uploaded files to {repo_id}")

    except Exception as e:
        print(f"Error uploading files: {e}")


# deprecated (leave for now)
def Add_Models_To_Config():
    # generator model
    for model in model_configs:
        if (model.model == prod_level["generator_model"]):
            config_builder.add_model(model)


# sampling_strategy controls how the data designer reads the data. so it's not redundant to how we uploaded 
#       train/test datasets already
# seed_dataset is the info returned from uploading/getting the dataset, not the actual dataset
def Connect_Data_To_Nemo(seed_dataset, seed_dataset_id):
    # link the dataset we're using to this job
    # ordered:  reads rows in order
    # shuffled: does all rows but in random order
    
    try:
        # we need the path of the dataset in the entity store
        # list all datasets
        print(f"Connecting to Data Store at: {hf_api}")

        ''' this is for finding the dataset reference and seeing the files
        DO NOT REMOVE, it could be useful
        dataset_name = "seed_trades"
        datasets = list(hf_api.list_datasets(search=dataset_name))
        for ds in datasets:
            if (ds.id == seed_dataset_id):
                try:
                    files = hf_api.list_repo_files(repo_id=ds.id, repo_type="dataset")

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
            dataset=f"{seed_dataset_id}/complete_set/edited_bulk_summary.csv",  # TODO: hardcoded
            datastore_settings={
                "endpoint": "https://huggingface.co",
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
# kinda complex. see design.md
# basically we need to generate the core data, have the llm generate 1 new column of json data, then make a row based on that json
def Map_Columns_To_Models():
    '''
    These are all the columns in the edited csv: 
        ticker, entry_time, entry_price, exit_price, trade_type, worst_exit_percent, trade_best_exit_percent, trade_percent_change, 
        entry_volatility_percent
    '''
    # sampler columns (diversity). this targets high impact variables to ensure the llm does correct distribution
    volatility_sampler = SamplerColumnConfig(
        name="target_volatility", # I call it target so the llm knows this is the goal
        #sampler_type=SamplerType.CATEGORY,
        column_type="sampler",
        sampler_type="category",
        params=CategorySamplerParams(
            values = [0.3,  0.4,  0.5, 0.6,  0.7, 0.8, 0.9,  1.0,  1.1],
            weights= [0.05, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.05, 0.05]
        )
    )

    # llm column (data generation)
    '''
    iterations of this
    - old: I added noise of +/- 10% to the volatility. this is replaced by the sampler column
    - old: I don't think I need best/worst price as a core metric
    '''
    synthetic_trade_generator = LLMTextColumnConfig(
        name="new_synthetic_trade_json", 
        
        # PASS EVERY RELEVANT COLUMN HERE so the LLM has the full statistical picture (not technical indicators)
        prompt="""
        You are a financial data simulation engine.

        Reference Trade Context:
        - Ticker: {{ ticker }}
        - Original Entry Time: {{ entry_time }}
        - Original Entry Price: {{ entry_price }}
        - Original Entry Volatility Percent: {{ entry_volatility_percent }}
        
        SIMULATION PARAMETERS (You MUST follow these):
        - Target Volatility to Simulate: {{ target_volatility }}
        
        Task:
        Generate a SYNTHETIC trade for '{{ ticker }}' that behaves as if the market volatility was exactly '{{ target_volatility }}'.        

        Rules:
        1. **Price Adjustment:** If 'Target Volatility' ({{ target_volatility }}) is higher than 'Original Volatility' ({{ entry_volatility_percent }}), widen the difference between Entry Price and Exit Price (implying bigger swings).
        2. **Exit Percent:** Adjust 'Worst Exit Percent' and 'Best Exit Percent' to be proportional to the new 'Target Volatility'. (Higher volatility = wider stops and targets).
        3. **Time:** Keep the Entry Time close to original, but add small jitter.
        4. **Consistency:** Ensure the new metrics are mathematically consistent with the new volatility.
        
        Output strictly valid JSON with keys: "ticker", "entry_time", "entry_price", "entry_volatility_percent", "worst_exit_percent", "trade_best_exit_percent".
        """,
        model_alias="generator-model"
    )

    config_builder.add_column(volatility_sampler)
    config_builder.add_column(synthetic_trade_generator)
    


def Run_Generator_Model_PREVIEW():
    output_file_name = "data/synthetic_data/previews/Raw_Generator_Model_Results_PREVIEW.csv"

    print("Starting PREVIEW generation job...")

    # Verification: Check that seed config is active so user knows data is present
    if config_builder._seed_config:
         print(f"Seed Dataset Linked: {config_builder._seed_config.dataset}")
    else:
         print("WARNING: No seed dataset linked in config!")

    # Fix for 422 Error: Remove seed-dataset columns from config as server might be older (just remove metadata, not actual data)
    #     and doesn't recognize them, causing validation error.
    removed_cols = {}
    
    # Access _column_configs directly to bypass delete restrictions
    # NOTE: We can't just delete them because the config builder will then fail client-side validation
    # complaining that the LLM columns reference non-existent variables (since we just deleted them).
    
    # STRATEGY 2: Instead of deleting them, we replace them with temporary "ExpressionColumnConfig" placeholders
    # that simply pass through the variable (if possible) or just act as stubs so validation passes.
    # However, since the SERVER rejects "seed-dataset" type, but the CLIENT needs them for "allowed_references",
    # we have a dilemma.
    
    # If the server is rejecting the request, it's because `preview()` calls `build()` which serializes everything.
    # The client-side validation `validate()` checks `allowed_references`.
    
    # Workaround: We temporarily monkey-patch the `allowed_references` property on the builder instance
    # to include the names of the columns we are removing, so that `validate()` passes even though the columns are gone from the list.
    
    
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

    

Setup_And_Cleaning.Clean_Seed_Data()       # clean the dataset
seed_dataset = Send_Seed_To_EntityStore()  # send dataset to entity store
seed_dataset_id = seed_dataset.id          # get dataset id

Setup_And_Cleaning.Split_Dataset(seed_dataset_path)  # Split the single seed dataset into training/validation/testing folders
Send_Dataset_Files_HF(HF_REPO_ID)                    # upload to hf
Connect_Data_To_Nemo(seed_dataset, seed_dataset_id)                       
#Add_Models_To_Config()                              # add models to config builder TODO: this might get added to to config in its init, so might not need a function
Map_Columns_To_Models()                              # give the models their prompts

config_builder.validate()  # test columns are config'd right

# RUN BASIC PREVIEW (low cost)
results_df = Run_Generator_Model_PREVIEW()

# RUN FULL PROCESS (high cost)
#results_df = Run_Generator_Model_FULL()

Unpack_Synthetic_Data(results_df)                    # process the generator model results