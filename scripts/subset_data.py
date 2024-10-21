import argparse
import logging
import os
from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfFolder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_hf_token():
    return HfFolder.get_token()

def process_dataset(dataset_name: str, data_rank: int, data_world_size: int, max_shards: int, output_dir: str, dry_run: bool):
    dataset_config = None
    dataset_split = "train"
    if ":" in dataset_name:
        dataset_name, dataset_config = dataset_name.split(":", 1)
        
    try:
        # Get available splits
        builder = load_dataset_builder(dataset_name, dataset_config)
        available_splits = list(builder.info.splits.keys())
        
        # If config is actually a split, adjust accordingly
        if dataset_config in available_splits:
            dataset_split = dataset_config
            dataset_config = None
        elif dataset_config is None and "en" in available_splits:
            dataset_config = "en"  # Special case for C4 dataset
        
        logger.info(f"Loading dataset {dataset_name} with config '{dataset_config}' and split '{dataset_split}'")
        
        if dry_run:
            logger.info(f"Dry run: would download {dataset_name} (config: {dataset_config}, split: {dataset_split})")
            return

        # Prepare the output directory
        dataset_output_dir = os.path.join(output_dir, dataset_name.replace("/", "_"))
        os.makedirs(dataset_output_dir, exist_ok=True)

        # Load and save the dataset
        ds = load_dataset(
            dataset_name,
            name=dataset_config,
            split=f"{dataset_split}[{data_rank}:{data_world_size}%{data_world_size}]",
            num_proc=1,
            cache_dir=dataset_output_dir
        )

        # Save the dataset in parquet format
        ds.to_parquet(os.path.join(dataset_output_dir, f"{dataset_split}_{data_rank}_of_{data_world_size}.parquet"))
        
        logger.info(f"Dataset {dataset_name} downloaded and saved to {dataset_output_dir}")
        logger.info(f"Dataset info: {ds}")

    except Exception as e:
        logger.error(f"Error processing dataset {dataset_name}: {str(e)}")

def main(args):
    hf_token = get_hf_token()
    if not hf_token:
        logger.warning("No Hugging Face token found. Some datasets may not be accessible.")
    else:
        logger.info("Successfully retrieved Hugging Face token.")
    
    dataset_names = args.dataset_name.split(',')
    
    for dataset_name in dataset_names:
        process_dataset(
            dataset_name, args.data_rank, args.data_world_size, 
            args.max_shards, args.output_dir, args.dry_run
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process data from the Hugging Face dataset")
    parser.add_argument("--dataset_name", type=str, required=True, help="comma-separated dataset names (e.g., 'allenai/c4:en,another_dataset')")
    parser.add_argument("--dry_run", action="store_true", help="do not download data")
    parser.add_argument("--data_rank", type=int, default=0, help="start index")
    parser.add_argument("--data_world_size", type=int, default=4, help="world size")
    parser.add_argument("--max_shards", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="dataset", help="output directory for downloaded files")
    args = parser.parse_args()
    main(args)