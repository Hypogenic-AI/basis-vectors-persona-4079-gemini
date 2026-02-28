from datasets import load_from_disk
import os

def explore_dataset(path, name):
    print(f"\n--- Exploring {name} at {path} ---")
    try:
        if not os.path.exists(path):
            print(f"Path {path} does not exist.")
            return
        ds = load_from_disk(path)
        print(f"Dataset: {ds}")
        print(f"Columns: {ds.column_names}")
        # Check first sample
        if len(ds) > 0:
            print(f"First sample keys: {ds[0].keys()}")
        # Check for persona or trait columns
        for col in ds.column_names:
            try:
                unique_vals = set([str(x)[:100] for x in ds.select(range(min(20, len(ds))))[col]])
                print(f"Sample values for '{col}': {unique_vals}")
            except:
                pass
    except Exception as e:
        print(f"Error loading {name}: {e}")

explore_dataset("datasets/Soul-Bench_sample", "Soul-Bench")
explore_dataset("datasets/big-five_sample", "Big-Five")
explore_dataset("datasets/personachat_sample", "PersonaChat")