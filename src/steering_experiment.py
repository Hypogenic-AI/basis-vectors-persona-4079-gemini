import getpass
import os
try:
    getpass.getuser()
except KeyError:
    getpass.getuser = lambda: "researcher"

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
from tqdm import tqdm

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 14
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
RESULTS_DIR = "results"

def main():
    print(f"Loading model {MODEL_NAME} for steering experiment...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=t.float16).to(DEVICE)
    model.eval()

    # RECOMPUTE PC1 for Layer 14
    print("Recomputing PC1 for Layer 14...")
    BEHAVIORS_DIR = "code/contrastive-activation-addition/datasets/raw/"
    behaviors = ["sycophancy", "survival-instinct", "coordinate-other-ais", "hallucination", "refusal"]
    
    all_diff_vectors = []
    for behavior in behaviors:
        data_path = os.path.join(BEHAVIORS_DIR, behavior, "dataset.json")
        with open(data_path, "r") as f:
            data = json.load(f)[:50]
        
        pos_prompts = [d["question"] + d["answer_matching_behavior"] for d in data]
        neg_prompts = [d["question"] + d["answer_not_matching_behavior"] for d in data]
        
        for i in range(0, len(pos_prompts), 10):
            p_batch = pos_prompts[i:i+10]
            n_batch = neg_prompts[i:i+10]
            
            p_inputs = tokenizer(p_batch, return_tensors="pt", padding=True).to(DEVICE)
            n_inputs = tokenizer(n_batch, return_tensors="pt", padding=True).to(DEVICE)
            
            with t.no_grad():
                p_out = model(**p_inputs, output_hidden_states=True)
                n_out = model(**n_inputs, output_hidden_states=True)
            
            p_acts = p_out.hidden_states[LAYER + 1][t.arange(len(p_batch)), p_inputs.attention_mask.sum(dim=1)-1]
            n_acts = n_out.hidden_states[LAYER + 1][t.arange(len(n_batch)), n_inputs.attention_mask.sum(dim=1)-1]
            
            diff = (p_acts - n_acts).cpu().float().numpy()
            all_diff_vectors.append(diff)
            
    all_diff_vectors = np.concatenate(all_diff_vectors, axis=0)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(all_diff_vectors)
    pc1 = pca.components_[0]
    pc1_tensor = t.from_numpy(pc1).to(DEVICE).half()

    # Steering Experiment
    test_behavior = "refusal"
    print(f"Testing steering on behavior: {test_behavior}")
    with open(os.path.join(BEHAVIORS_DIR, test_behavior, "dataset.json"), "r") as f:
        test_data = json.load(f)[100:110] 
    
    results = []
    coeffs = [-2.0, -1.0, 0, 1.0, 2.0]
    
    for d in test_data:
        prompt = d["question"]
        ans_pos = d["answer_matching_behavior"] 
        ans_neg = d["answer_not_matching_behavior"] 
        
        token_pos = tokenizer.encode(ans_pos, add_special_tokens=False)[0]
        token_neg = tokenizer.encode(ans_neg, add_special_tokens=False)[0]
        
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        prompt_results = {"question": prompt, "logits": {}}
        
        for coeff in coeffs:
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    if len(hidden_states.shape) == 3:
                        hidden_states[:, -1, :] = hidden_states[:, -1, :] + (coeff * pc1_tensor * 1.0)
                    else:
                        hidden_states[-1, :] = hidden_states[-1, :] + (coeff * pc1_tensor * 1.0)
                    return (hidden_states,) + output[1:]
                else:
                    hidden_states = output
                    if len(hidden_states.shape) == 3:
                        hidden_states[:, -1, :] = hidden_states[:, -1, :] + (coeff * pc1_tensor * 1.0)
                    else:
                        hidden_states[-1, :] = hidden_states[-1, :] + (coeff * pc1_tensor * 1.0)
                    return hidden_states
            
            handle = model.model.layers[LAYER].register_forward_hook(hook_fn)
            
            with t.no_grad():
                outputs = model(**prompt_inputs)
                logits = outputs.logits[0, -1]
                prob_pos = t.softmax(logits, dim=-1)[token_pos].item()
                prob_neg = t.softmax(logits, dim=-1)[token_neg].item()
            
            handle.remove()
            prompt_results["logits"][coeff] = {"pos": prob_pos, "neg": prob_neg}
        
        results.append(prompt_results)

    print("\nSteering Summary (Refusal probability vs PC1 coeff):")
    summary = {c: [] for c in coeffs}
    for r in results:
        for c in coeffs:
            denom = r["logits"][c]["pos"] + r["logits"][c]["neg"]
            p_ref = r["logits"][c]["pos"] / (denom + 1e-9)
            summary[c].append(p_ref)
    
    for c in coeffs:
        print(f"Coeff {c}: Mean P(Refusal) = {np.mean(summary[c]):.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/steering_results.json", "w") as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    main()
