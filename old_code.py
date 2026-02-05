'''
# UNUSED
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















'''