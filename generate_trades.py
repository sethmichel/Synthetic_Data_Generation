import data_designer.config as dd
from data_designer.interface import DataDesigner
from nemo_microservices.data_designer.essentials import (
    DataDesignerConfigBuilder,
    NeMoDataDesignerClient,
)
from nemo_microservices import NeMoMicroservices
from huggingface_hub import HfApi
import Setup_And_Cleaning
import os
import csv
import requests
from pathlib import Path
import sys




# misc configs
NAMESPACE="data"
PROJECT="stock-trade-data-generation"
Setup_And_Cleaning.System_Startup(NAMESPACE, PROJECT)
Setup_And_Cleaning.Clean_Seed_Data()
    
prod_level = Setup_And_Cleaning.MODE_CONFIGS['development']
data_designer = DataDesigner()
config = dd.DataDesignerConfigBuilder()
hf_api = HfApi(token=os.environ["HUGGINGFACE_API_KEY"])
# To use local NeMo Data Store instead, uncomment these lines:
# hf_api = HfApi(endpoint=os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"], token=os.environ["HUGGINGFACE_API_KEY"])
# HF_ENDPOINT=f"{os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"]}/v1/hf"

seed_dataset_path = "human_data/edited_bulk_summary.csv"
HF_USERNAME, HF_REPO_NAME = Setup_And_Cleaning.Load_Secrets()
HF_REPO_ID = f"{HF_USERNAME}/{HF_REPO_NAME}"
data_designer_client = NeMoDataDesignerClient(base_url=os.environ["NEMO_MICROSERVICES_BASE_URL"])


# send post request of our dataset to the entity store
def Send_Seed_To_EntityStore():
    dataset_info = {
        'name': "seed_trades", # Matching the dataset name from repo_id
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
    
    return {
        "dataset_id": response_data.get("id"),
        "dataset_info": response_data
    }


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
            folder_path="human_data/splits/training",
            path_in_repo="training",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # validation
        hf_api.upload_folder(
            folder_path="human_data/splits/validation",
            path_in_repo="validation",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # testing
        hf_api.upload_folder(
            folder_path="human_data/splits/testing",
            path_in_repo="testing",
            repo_id=repo_id,
            repo_type="dataset"
        )

        # complete_set
        hf_api.upload_file(
            path_or_fileobj="human_data/edited_bulk_summary.csv",
            path_in_repo="complete_set/edited_bulk_summary.csv",
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        print(f"Successfully uploaded files to {repo_id}")

    except Exception as e:
        print(f"Error uploading files: {e}")







dataset_result = Send_Seed_To_EntityStore()
dataset_id = dataset_result["dataset_id"]

Setup_And_Cleaning.Split_Dataset(seed_dataset_path)   # Split the single seed dataset into training/validation/testing folders
Send_Dataset_Files_HF(HF_REPO_ID)  # upload to hf







# 4. Preview the Result
print("Generating a preview...")
preview = data_designer.preview(config_builder=config)
print(preview.display_sample_record())