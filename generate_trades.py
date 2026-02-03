import data_designer.config as dd
from data_designer.interface import DataDesigner
import os
import csv

data_designer = DataDesigner()
config = dd.DataDesignerConfigBuilder()
human_trades_path = "human_data/original_bulk_summaries.csv"

def Remove_Columns():
    dest_file_name = "edited_bulk_summary.csv"
    dest_path = os.path.join("human_data", dest_file_name)
    
    if os.path.exists(dest_path):
        return dest_path
        
    columns_to_remove = ["Trade Id", "Dollar Change", "Running Percent By Ticker", "Running Percent All", "Total Investment", 
                         "Entry Price", "Exit Price", "Qty", "Best Exit Price", "Best Exit Time In Trade", "Worst Exit Price", 
                         "Worst Exit Time In Trade", "Trade Holding Reached"]
                         
    with open(human_trades_path, 'r', encoding='utf-8') as f_in, open(dest_path, 'w', newline='', encoding='utf-8') as f_out:
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
            
            
human_trades_path = Remove_Columns()

# 2. Add your "Seed" Data
# This tells the tool to look at your real trades to understand the pattern.
# We are creating a "list" of trades based on your file.
config.add_column(
    dd.SamplerColumnConfig(
        name="original_trade",
        sampler_type=dd.SamplerType.FILE, # Or similar depending on version, sometimes 'SEED'
        params=dd.FileSamplerParams(
            path=human_trades_path,
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
        model_alias="nvidia-text", # Uses a default NVIDIA model
        # The prompt uses {{ original_trade }} to reference your seed data
        prompt="Review this real trade: {{ original_trade }}. Now, invent a plausible hypothetical stock trade following a similar strategy but for a different tech company."
    )
)

# 4. Preview the Result
print("Generating a preview...")
preview = data_designer.preview(config_builder=config)
print(preview.display_sample_record())