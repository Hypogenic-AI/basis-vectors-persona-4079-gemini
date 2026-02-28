import getpass
import os

# Monkeypatch getpass.getuser() to avoid KeyError: 'getpwuid(): uid not found' in containerized environments
try:
    getpass.getuser()
except KeyError:
    getpass.getuser = lambda: "researcher"

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYERS = [14, 18, 22, 26]
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
BEHAVIORS_DIR = "code/contrastive-activation-addition/datasets/raw/"
NUM_SAMPLES = 100 # Per behavior

def get_activations_all_layers(model, tokenizer, prompts, layers):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    with t.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    attention_mask = inputs.attention_mask
    last_token_indices = attention_mask.sum(dim=1) - 1
    batch_indices = t.arange(inputs.input_ids.size(0)).to(DEVICE)
    
    layer_activations = {}
    for layer_idx in layers:
        hidden_states = outputs.hidden_states[layer_idx + 1] # +1 because 0 is embedding
        last_token_acts = hidden_states[batch_indices, last_token_indices]
        layer_activations[layer_idx] = last_token_acts.cpu().float().numpy()
    
    return layer_activations

def main():
    os.makedirs("results", exist_ok=True)
    
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=t.float16).to(DEVICE)
    model.eval()

    # List only standard behavioral dirs
    standard_behaviors = [
        "coordinate-other-ais",
        "corrigible-neutral-HHH",
        "hallucination",
        "myopic-reward",
        "refusal",
        "survival-instinct",
        "sycophancy"
    ]
    behaviors = [b for b in standard_behaviors if os.path.exists(os.path.join(BEHAVIORS_DIR, b))]
    print(f"Found behaviors: {behaviors}")

    all_layer_data = {layer: {"vectors": [], "labels": []} for layer in LAYERS}

    for behavior in behaviors:
        print(f"Processing behavior: {behavior}")
        data_path = os.path.join(BEHAVIORS_DIR, behavior, "dataset.json")
        with open(data_path, "r") as f:
            data = json.load(f)[:NUM_SAMPLES]
        
        # Use simple positive/negative framing
        # For sycophancy, "Matching" means Agreeing with user
        # For survival, "Matching" means wanting to live
        pos_prompts = [d["question"] + d["answer_matching_behavior"] for d in data]
        neg_prompts = [d["question"] + d["answer_not_matching_behavior"] for d in data]
        
        batch_size = 10
        for i in tqdm(range(0, len(pos_prompts), batch_size)):
            pos_batch = pos_prompts[i:i+batch_size]
            neg_batch = neg_prompts[i:i+batch_size]
            
            pos_acts_batch = get_activations_all_layers(model, tokenizer, pos_batch, LAYERS)
            neg_acts_batch = get_activations_all_layers(model, tokenizer, neg_batch, LAYERS)
            
            for layer in LAYERS:
                diff_batch = pos_acts_batch[layer] - neg_acts_batch[layer]
                all_layer_data[layer]["vectors"].append(diff_batch)
                all_layer_data[layer]["labels"].extend([behavior] * len(diff_batch))

    final_results = {}

    for layer in LAYERS:
        print(f"\n--- Analysis for Layer {layer} ---")
        vectors = np.concatenate(all_layer_data[layer]["vectors"], axis=0)
        labels = np.array(all_layer_data[layer]["labels"])
        
        pca = PCA(n_components=10)
        pca_result = pca.fit_transform(vectors)
        evr = pca.explained_variance_ratio_
        print(f"Cumulative EVR: {np.cumsum(evr)}")
        
        # Save scatter plot for layer
        df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(10)])
        df['behavior'] = labels
        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='behavior', alpha=0.7)
        plt.title(f'Layer {layer}: Persona Projection on top 2 PCs')
        plt.savefig(f'results/pca_layer_{layer}.png')
        plt.close()
        
        # Alignment
        pc1 = pca.components_[0]
        alignments = {}
        for b in behaviors:
            b_vecs = vectors[labels == b]
            mean_vec = np.mean(b_vecs, axis=0)
            cos_sim = np.dot(mean_vec, pc1) / (np.linalg.norm(mean_vec) * np.linalg.norm(pc1))
            alignments[b] = float(cos_sim)
        
        final_results[layer] = {
            "evr": evr.tolist(),
            "cumulative_evr": np.cumsum(evr).tolist(),
            "pc1_alignments": alignments
        }

    with open("results/metrics.json", "w") as f:
        json.dump(final_results, f, indent=4)

    # Plot EVR comparison across layers
    plt.figure(figsize=(10, 6))
    for layer in LAYERS:
        plt.plot(range(1, 11), final_results[layer]["evr"], marker='o', label=f'Layer {layer}')
    plt.title('EVR across different layers')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/evr_cross_layer.png')
    plt.close()

if __name__ == "__main__":
    main()