"""Quick script to inspect the actual HQ-small dataset structure."""

from datasets import load_dataset

print("Loading dataset...")
dataset = load_dataset("izhx/COMP5423-25Fall-HQ-small", cache_dir="./data/cache")

print("\nDataset splits:", list(dataset.keys()))

for split_name in dataset.keys():
    print(f"\n{split_name.upper()} split:")
    print(f"  Size: {len(dataset[split_name])}")
    print(f"  Columns: {dataset[split_name].column_names}")
    if len(dataset[split_name]) > 0:
        print(f"  Sample: {dataset[split_name][0]}")
