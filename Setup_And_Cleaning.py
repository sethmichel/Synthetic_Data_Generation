import os
import csv
import subprocess
import shutil
import zipfile
import urllib.request
from pathlib import Path
import random
import sys
from nemo_microservices import NeMoMicroservices


# 1: config the models
MODE_CONFIGS = {
    # use tiny model with timeouts
    "development": {
        "generator_model": "",
        "judge_model": "meta/llama-3.1-8b-instruct",
        "timeout": 30,
        "max_tokens": 1024
    },
    # use mid model
    "mini_prod": {
        "generator_model": "",
        "judge_model": "meta/llama-3.1-70b-instruct",
        "timeout": 60,
        "max_tokens": 1024
    },
    # use only if I'm looking for final results
    "full_prod": {
        "generator_model": "",
        "judge_model": "meta/llama-3.1-405b-instruct",
        "timeout": 120,
        "max_tokens": 2048
    }
}


# after running this you should have nemo microservice docker container up
def System_Startup():
    """
    Launches the NVIDIA NeMo Data Designer using Docker Compose.
    Handles NGC CLI installation and resource downloading if needed.
    """
    print("Starting System Startup...")

    # 1. Load API Keys
    os.environ["HUGGINGFACE_API_KEY"] = Get_Huggingface_Api_Token()
    
    # Key 2: NVIDIA NIM/Build Key (starts with nvapi-) for using models endpoints
    # Renaming to NIM_API_KEY to match what Docker expects
    os.environ["NIM_API_KEY"] = Get_NIM_Api_Token()

    # Also set NVIDIA_API_KEY for the Data Designer SDK (SDK requires this specific name - it's hardcoded)
    os.environ["NVIDIA_API_KEY"] = os.environ["NIM_API_KEY"]
    
    # Key 3: NGC Personal Key for Docker/CLI access
    os.environ["NGC_PERSONAL_API_KEY"] = Get_NGC_Personal_Api_Token()
    # Set NGC_CLI_API_KEY for ngc cli tool if it uses it
    os.environ["NGC_CLI_API_KEY"] = os.environ["NGC_PERSONAL_API_KEY"]

    # Set env vars for NGC CLI and Docker Compose
    if "NEMO_MICROSERVICES_BASE_URL" not in os.environ:
        os.environ["NEMO_MICROSERVICES_BASE_URL"] = "http://localhost:8080"
    
    if "NEMO_MICROSERVICES_DATASTORE_ENDPOINT" not in os.environ:
        os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"] = "http://localhost:8080/v1"
    
    if ("ENTITY_STORE_BASE_URL" not in os.environ):
        os.environ["ENTITY_STORE_BASE_URL"] = "http://localhost:8080"
    
    # 2. Docker Login and ngc cli check
    Start_Docker()

    # 3. basic namespace and project handling
    Create_Namespace('data')


    return True

def Get_Huggingface_Api_Token():
    secrets_dir = Path("../secrets")
    api_key_file = secrets_dir / "hugging-face-write-token-data-generation.txt"
    if not api_key_file.exists():
        print(f"Error: hugging face API key file not found in {secrets_dir}")
        sys.exit(1)

    try:
        HF_TOKEN = api_key_file.read_text().strip()
        return HF_TOKEN

    except Exception as e:
        print(f"Error reading API key file: {e}")
        sys.exit(1)


def Get_NIM_Api_Token():
    """
    Key 2: From build.nvidia.com, starts with 'nvapi-'.
    Used for calling NIM endpoints (so it calls llm endpoints)
    """
    secrets_dir = Path("../secrets")
    api_key_file = secrets_dir / "nvidia build api key.txt"
        
    if not api_key_file.exists():
        print(f"Error: NIM API key file not found in {secrets_dir}")
        sys.exit(1)

    # Read and validate key
    try:
        NIM_TOKEN = api_key_file.read_text().strip()

    except Exception as e:
        print(f"Error reading NIM API key file: {e}")
        sys.exit(1)

    # If the key doesn't start with nvapi-, fail the startup
    if not NIM_TOKEN.startswith("nvapi-"):
        print("Error: NIM API key is likely wrong, we expect it to start with 'nvapi-'")
        sys.exit(1)

    return NIM_TOKEN


def Get_NGC_Personal_Api_Token():
    """
    Key 3: From org.ngc.nvidia.com/setup.
    Used for Docker login and NGC CLI.
    """
    secrets_dir = Path("../secrets")
    api_key_file = secrets_dir / "nvidia_ngc_api_key.txt"
        
    if not api_key_file.exists():
        print(f"Error: NGC Personal API key file not found in {secrets_dir}")
        sys.exit(1)

    # Read key
    try:
        NGC_TOKEN = api_key_file.read_text().strip()
        return NGC_TOKEN

    except Exception as e:
        print(f"Error reading NGC Personal API key file: {e}")
        sys.exit(1)


def Start_Docker():
    print("Logging into NVIDIA Container Registry...")

    try:
        # docker login nvcr.io -u '$oauthtoken' --password-stdin
        # Use NGC_PERSONAL_API_KEY for registry login
        subprocess.run(
            ["docker", "login", "nvcr.io", "-u", "$oauthtoken", "--password-stdin"],
            input=os.environ["NGC_PERSONAL_API_KEY"].encode(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Docker login successful.")

    except subprocess.CalledProcessError as e:
        print(f"Docker login failed: {e.stderr.decode()}")
        sys.exit(1)
    
    # Check NGC CLI
    ngc_cmd = shutil.which("ngc")
    if not ngc_cmd:
        # Check local installation
        ngc_dir = Path("./ngc_cli")
        possible_binary = ngc_dir / "ngc-cli/ngc"
        if possible_binary.exists():
            ngc_cmd = str(possible_binary.absolute())
        else:
            print("NGC CLI not found")
            sys.exit(1)
    else:
        print(f"Found NGC CLI at {ngc_cmd}")

    # Download Quickstart Resource
    # Try 25.01 first as requested, fallback to 25.12 if needed
    versions_to_try = ["25.11", "25.12"]
    target_dir = None
    
    for version in versions_to_try:
        resource_name = f"nvidia/nemo-microservices/nemo-microservices-quickstart:{version}"
        current_target_dir = Path(f"nemo-microservices-quickstart_v{version}")
        
        if current_target_dir.exists():
            print(f"Resource directory {current_target_dir} already exists.")
            target_dir = current_target_dir
            # Update env var for tag matching the directory
            os.environ["NEMO_MICROSERVICES_IMAGE_TAG"] = version
            break
        
        print(f"Attempting to download {resource_name}...")
        try:
            subprocess.run(
                [ngc_cmd, "registry", "resource", "download-version", resource_name],
                check=True
            )
            target_dir = current_target_dir
            os.environ["NEMO_MICROSERVICES_IMAGE_TAG"] = version
            break

        except subprocess.CalledProcessError as e:
            print(f"Failed to download version {version}: {e}")
            continue
            
    if not target_dir or not target_dir.exists():
        print("Error: Failed to download or locate NeMo microservices quickstart.")
        sys.exit(1)

    # Start Docker Compose
    print(f"Starting services in {target_dir}...")
    
    # Configure environment for Docker Compose
    # These override defaults in the .env file if present
    env = os.environ.copy()
    env["NEMO_MICROSERVICES_IMAGE_REGISTRY"] = "nvcr.io/nvidia/nemo-microservices"
    # Ensure NIM_API_KEY is passed to docker-compose (it expects this exact name)
    env["NIM_API_KEY"] = os.environ["NIM_API_KEY"]
    # NEMO_MICROSERVICES_IMAGE_TAG is already set in os.environ above if we downloaded
    if "NEMO_MICROSERVICES_IMAGE_TAG" not in env:
        env["NEMO_MICROSERVICES_IMAGE_TAG"] = target_dir.name.split("_v")[-1]

    try:
        # docker compose --profile data-designer up -d
        subprocess.run(
            ["docker", "compose", "--profile", "data-designer", "up", "-d"],
            cwd=target_dir,
            env=env,
            check=True
        )
        print("Services started successfully.")
        print("Run 'docker ps' to verify.")

    except subprocess.CalledProcessError as e:
        print(f"Failed to start services: {e}")
        sys.exit(1)


# https://docs.nvidia.com/nemo/microservices/latest/manage-entities/namespaces/create-namespace.html
def Create_Namespace(namespace_to_make):
    client = NeMoMicroservices(base_url=os.environ["ENTITY_STORE_BASE_URL"])

    response = client.namespaces.create(
    id=namespace_to_make
    )

    print(f"created namespace {namespace_to_make}: {response}") # id is response['id']

    return response


# https://docs.nvidia.com/nemo/microservices/latest/manage-entities/namespaces/update-namespace.html
def Update_Namespace(namespace_to_edit):
    client = NeMoMicroservices(base_url=os.environ["ENTITY_STORE_BASE_URL"])

    response = client.namespaces.update(
        namespace_id=namespace_to_edit,
        custom_fields={}, # this is what I'm overwritting
        # example: custom_fields={"sandbox": "true", "location": "on-prem"}
    )

    print(f"edited namespace {namespace_to_edit}: {response}")


# https://docs.nvidia.com/nemo/microservices/latest/manage-entities/namespaces/get-namespace.html
def Get_Single_Namespace(namespace_to_get):
    client = NeMoMicroservices(base_url=os.environ["ENTITY_STORE_BASE_URL"])

    response = client.namespaces.retrieve(
        namespace_id=namespace_to_get,
    )

    return response


# https://docs.nvidia.com/nemo/microservices/latest/manage-entities/namespaces/list-namespaces.html
def Get_All_Namespaces():
    client = NeMoMicroservices(base_url=os.environ["ENTITY_STORE_BASE_URL"])

    return client.namespaces.list()





def Split_Dataset(seed_file_path, output_base_dir="human_data/splits"):
    print(f"Splitting dataset from {seed_file_path}...")
    
    # Read the data
    with open(seed_file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data = list(reader)
    
    # Shuffle data
    random.seed(42) # Ensure reproducibility
    random.shuffle(data)
    
    total_count = len(data)
    train_count = int(total_count * 0.7)
    val_count = int(total_count * 0.15)
    # test_count gets the rest
    
    train_data = data[:train_count]
    val_data = data[train_count:train_count + val_count]
    test_data = data[train_count + val_count:]
    
    splits = {
        "training": train_data,
        "validation": val_data,
        "testing": test_data,
    }
    
    output_base = Path(output_base_dir)
    
    for split_name, split_data in splits.items():
        # Create directory
        split_dir = output_base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Write file
        output_file = split_dir / f"{split_name}.csv"
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_data)
            
    print(f"Dataset split completed. Files saved to {output_base_dir}")


def Clean_Seed_Data():
    seed_dataset_path = "human_data/original_bulk_summaries.csv"
    dest_file_name = "edited_bulk_summary.csv"
    dest_path = os.path.join("human_data", dest_file_name)
    
    if not os.path.exists(dest_path):  
        columns_to_remove = ["Trade Id", "Dollar Change", "Running Percent By Ticker", "Running Percent All", "Total Investment", 
                            "Entry Price", "Exit Price", "Qty", "Best Exit Price", "Best Exit Time In Trade", "Worst Exit Price", 
                            "Worst Exit Time In Trade", "Trade Holding Reached"]
                            
        with open(seed_dataset_path, 'r', encoding='utf-8') as f_in, open(dest_path, 'w', newline='', encoding='utf-8') as f_out:
            reader = csv.DictReader(f_in)

            # Filter out columns to remove
            fieldnames = []
            for field in reader.fieldnames:
                if field not in columns_to_remove:
                    fieldnames.append(field)
            
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                filtered_row = {}

                for key in fieldnames:
                    if key in row:
                        filtered_row[key] = row[key]

                writer.writerow(filtered_row)

