import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

def main():
    # Load environment variables
    print("Loading environment variables...")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("Error: HF_TOKEN not found in environment variables.")
        return

    # Define languages and their respective splits to save
    source_langs = ["hau", "amh"]
    target_langs = ["twi", "pcm"]
    
    # Create data directory if it doesn't exist
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    else:
        print(f"Directory {data_dir} already exists.")

    # Process Source Languages (Train splits)
    for lang in source_langs:
        print(f"\n--- Processing Source Language: {lang} ---")
        print(f"Downloading {lang} dataset...")
        try:
            dataset = load_dataset("afrihate/afrihate", lang, token=hf_token)
            if "train" in dataset:
                df = dataset["train"].to_pandas()
                filename = os.path.join(data_dir, f"{lang}_train.csv")
                df.to_csv(filename, index=False)
                print(f"Successfully saved {lang} train split to {filename} ({len(df)} rows).")
            else:
                print(f"Warning: 'train' split not found for {lang}")
        except Exception as e:
            print(f"Error processing {lang}: {e}")

    # Process Target Languages (Test splits)
    for lang in target_langs:
        print(f"\n--- Processing Target Language: {lang} ---")
        print(f"Downloading {lang} dataset...")
        try:
            dataset = load_dataset("afrihate/afrihate", lang, token=hf_token)
            if "test" in dataset:
                df = dataset["test"].to_pandas()
                filename = os.path.join(data_dir, f"{lang}_test.csv")
                df.to_csv(filename, index=False)
                print(f"Successfully saved {lang} test split to {filename} ({len(df)} rows).")
            else:
                print(f"Warning: 'test' split not found for {lang}")
        except Exception as e:
            print(f"Error processing {lang}: {e}")

    print("\nData processing complete.")

if __name__ == "__main__":
    main()
