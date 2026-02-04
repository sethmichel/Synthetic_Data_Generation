import data_designer.config as dd
from data_designer.interface import DataDesigner
from nemo_microservices.data_designer.essentials import (
    DataDesignerConfigBuilder,
    NeMoDataDesignerClient,
)
from nemo_microservices import NeMoMicroservices
from huggingface_hub import HfApi
from Setup_And_Cleaning import (
    Clean_Seed_Data, 
    System_Startup, 
    MODE_CONFIGS,
    Split_Dataset
)
import os
import csv
import requests
from pathlib import Path
import sys


# 1: get hugging face api key
secrets_dir = Path("../secrets")
api_key_file = secrets_dir / "hugging-face-write-token-data-generation"
if not api_key_file.exists():
    print(f"Error: hugging face API key file not found in {secrets_dir}")
    sys.exit(1)
try:
    HF_TOKEN = api_key_file.read_text().strip()
except Exception as e:
    print(f"Error reading API key file: {e}")
    sys.exit(1)

# 2: misc configs
if (System_Startup() == False):
    sys.exit(1)

Clean_Seed_Data()
    
prod_level = MODE_CONFIGS['development']
data_designer = DataDesigner()
config = dd.DataDesignerConfigBuilder()
hf_api = HfApi(endpoint=os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"], token=HF_TOKEN)
HF_ENDPOINT=f"{os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"]}/v1/hf"

seed_dataset_path = "human_data/edited_bulk_summary.csv"
NAMESPACE="data"


# UNUSED TODO
# send dataset (seed) to the datastore
def Send_Seed_To_DataStore(seed_dataset_path):
    data_designer_client = NeMoDataDesignerClient(base_url=os.environ["NEMO_MICROSERVICES_BASE_URL"])

    # Upload your dataset to the datastore
    # You can pass a pandas DataFrame, file path str, or Path object
    seed_dataset_reference = data_designer_client.upload_seed_dataset(
        dataset=seed_dataset_path,
        repo_id="human_trades/seed_trades",
        datastore_settings={"endpoint": os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"]}, # this endpoint makes it private
    )
    
    return seed_dataset_reference


# send post request of our dataset to the entity store
def Send_Seed_To_EntityStore():
    dataset_info = {
        'name': "seed_trades", # Matching the dataset name from repo_id
        'namespace': NAMESPACE,
        'description': "human-made-stock-trades",
        'format': 'json',
        'files_url': f"hf://datasets/{NAMESPACE}/seed_trades",
        'project': "stock trade data generation",
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
    
    return {
        "dataset_id": response_data.get("id"),
        "dataset_info": response_data
    }


# send files to that repo in hf (training, testing, validation, complete set)
def Send_Dataset_Files_HF(dataset_id):
    # create the repo
    if hf_api.repo_exists(repo_id=dataset_id, repo_type="dataset"):
        print(f"Repo {dataset_id} already exists")
    else:
        hf_api.create_repo(repo_id=dataset_id, repo_type="dataset")

    # we're doing complete_set, training, validation, testing datasets
    try:
        # training
        hf_api.upload_folder(
            folder_path="human_data/splits/training",
            path_in_repo="training",
            repo_id=dataset_id,
            repo_type="dataset"
        )
        
        # validation
        hf_api.upload_folder(
            folder_path="human_data/splits/validation",
            path_in_repo="validation",
            repo_id=dataset_id,
            repo_type="dataset"
        )
        
        # testing
        hf_api.upload_folder(
            folder_path="human_data/splits/testing",
            path_in_repo="testing",
            repo_id=dataset_id,
            repo_type="dataset"
        )

        # complete_set
        hf_api.upload_folder(
            folder_path=seed_dataset_path,
            path_in_repo="complete_set",
            repo_id=dataset_id,
            repo_type="dataset"
        )

    except Exception as e:
        print(f"Error uploading files: {e}")




# 2. Add your "Seed" Data
# This tells the tool to look at your real trades to understand the pattern.
# We are creating a "list" of trades based on your file.
config.add_column(
    dd.SamplerColumnConfig(
        name="original_trade",
        sampler_type=dd.SamplerType.FILE, # Or similar depending on version, sometimes 'SEED'
        params=dd.FileSamplerParams(
            path=seed_dataset_path,
            # This makes sure the tool picks random examples from your file
            sample_method="random" 
        )
    )
)

# 3. Generate a Synthetic Trade Description
# This uses an LLM (hosted by NVIDIA) to look at your seed data 
# and invent a new, similar trade scenario.
config.add_column(
    dd.LLMTextColumnConfig(
        name="synthetic_trade_scenario",
        model_alias=prod_level['generator_model'],
        # The prompt uses {{ original_trade }} to reference your seed data
        prompt="Review this real trade: {{ original_trade }}. Now, invent a plausible hypothetical stock trade following a similar strategy but for a different tech company."
    )
)



dataset_result = Send_Seed_To_EntityStore()
dataset_id = dataset_result["dataset_id"]

Split_Dataset(seed_dataset_path)   # Split the single seed dataset into training/validation/testing folders
Send_Dataset_Files_HF(dataset_id)  # upload to hf




Send_Seed_To_DataStore(seed_dataset_path)



# 4. Preview the Result
print("Generating a preview...")
preview = data_designer.preview(config_builder=config)
print(preview.display_sample_record())