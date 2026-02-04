import os
import csv
import subprocess
import shutil
import zipfile
import urllib.request
from pathlib import Path
import random
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

    # 1. Load API Key
    secrets_dir = Path("../secrets")
    api_key_file = secrets_dir / "nvidia_ngc_api_key.txt"
        
    if not api_key_file.exists():
        print(f"Error: API key file not found in {secrets_dir}")
        return False

    # Read and validate key
    try:
        api_key = api_key_file.read_text().strip()

    except Exception as e:
        print(f"Error reading API key file: {e}")
        return False

    # If the key doesn't start with nvapi-, fail the startup
    if not api_key.startswith("nvapi-"):
        print("Error: API key is likely wrong, we expect it to start with 'nvapi-'")
        return False

    # Set env vars for NGC CLI and Docker Compose
    os.environ["NGC_CLI_API_KEY"] = api_key
    #os.environ["NGC_CLI_FORMAT_TYPE"] = "json"

    # Set default NeMo microservices URLs if not already set
    # Note: The NeMo Microservices Quickstart (Docker Compose) typically runs an API Gateway on port 8080.
    # This Gateway routes traffic to individual services (Entity Store, Data Store, etc.) based on the URL path.
    # Therefore, pointing multiple base URLs to localhost:8080 is usually correct.
    if "NEMO_MICROSERVICES_BASE_URL" not in os.environ:
        os.environ["NEMO_MICROSERVICES_BASE_URL"] = "http://localhost:8080"
    
    if "NEMO_MICROSERVICES_DATASTORE_ENDPOINT" not in os.environ:
        os.environ["NEMO_MICROSERVICES_DATASTORE_ENDPOINT"] = "http://localhost:8080/v1"
    
    if ("ENTITY_STORE_BASE_URL" not in os.environ):
        os.environ["ENTITY_STORE_BASE_URL"] = "http://localhost:8080"
    
    # 2. Docker Login
    print("Logging into NVIDIA Container Registry...")
    try:
        # docker login nvcr.io -u '$oauthtoken' --password-stdin
        subprocess.run(
            ["docker", "login", "nvcr.io", "-u", "$oauthtoken", "--password-stdin"],
            input=api_key.encode(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Docker login successful.")

    except subprocess.CalledProcessError as e:
        print(f"Docker login failed: {e.stderr.decode()}")
        return False
    
    # 3. Check NGC CLI
    ngc_cmd = shutil.which("ngc")
    if not ngc_cmd:
        # Check local installation
        ngc_dir = Path("./ngc_cli")
        possible_binary = ngc_dir / "ngc-cli/ngc"
        if possible_binary.exists():
            ngc_cmd = str(possible_binary.absolute())
        else:
            print("NGC CLI not found")
            return False
    else:
        print(f"Found NGC CLI at {ngc_cmd}")


    if (Start_Docker(ngc_cmd, api_key) == False):
        return False

    Create_Namespace('data')


    return True


def Start_Docker(ngc_cmd, api_key):
    # 4. Download Quickstart Resource
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
        return False

    # 5. Start Docker Compose
    print(f"Starting services in {target_dir}...")
    
    # Configure environment for Docker Compose
    # These override defaults in the .env file if present
    env = os.environ.copy()
    env["NEMO_MICROSERVICES_IMAGE_REGISTRY"] = "nvcr.io/nvidia/nemo-microservices"
    env["NIM_API_KEY"] = api_key
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
        return False


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

if __name__ == "__main__":
    System_Startup()
