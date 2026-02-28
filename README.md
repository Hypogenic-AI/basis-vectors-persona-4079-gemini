# Basis Vectors in Persona Space

## Overview
This research explores the underlying geometric structure of persona representations in Large Language Models (LLMs). We hypothesize that persona-related activations in the residual stream can be decomposed into a shared low-rank basis of component vectors.

## Key Findings
- **Shared Low-Rank Basis**: In layers 14-18 of Qwen2.5-1.5B, a single Principal Component (PC1) explains **60% of the total variance** across seven diverse behaviors (sycophancy, survival-instinct, coordination, etc.).
- **Social-Self Axis**: This primary basis vector strongly aligns with **Sycophancy (0.71)**, **Survival Instinct (0.55)**, and **AI Coordination (0.53)**, suggesting a shared "Social Compliance / Self-Preservation" axis.
- **Layer-wise Evolution**: The dominance of this primary vector decreases in later layers (60% EVR at layer 14 to 21% EVR at layer 26), as persona representations become more specialized.
- **Causal Efficacy**: Steering with the shared PC1 vector modifies unseen model behaviors (e.g., refusal probability), demonstrating that these PCA-derived components represent meaningful behavioral directions.

## Reproducing Results
1. **Environment Setup**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv add torch transformers datasets accelerate scikit-learn matplotlib seaborn pandas tqdm
   cd code/repe && uv pip install -e . && cd ../..
   ```
2. **Run Activation Extraction & PCA**:
   ```bash
   python src/research.py
   ```
3. **Run Steering Experiment**:
   ```bash
   python src/steering_experiment.py
   ```
4. **Visualize Results**:
   ```bash
   python src/plot_alignments.py
   ```

## File Structure
- `src/research.py`: Core logic for activation extraction and PCA analysis across layers.
- `src/steering_experiment.py`: Causal validation by steering with PCA-derived basis vectors.
- `src/plot_alignments.py`: Visualization scripts for PC alignments.
- `results/`: Contains plots (`evr_cross_layer.png`, `pc1_alignments_bar.png`, `pca_layer_14.png`) and metrics (`metrics.json`, `steering_results.json`).
- `REPORT.md`: Comprehensive research report.

## References
- Cintas et al. (2025). "Localizing Persona Representations in LLMs."
- Wang (2025). "The Geometry of Persona."
- Rimsky et al. (2023). "Steering Llama 2 via Contrastive Activation Addition."
- RepE Toolkit: Andy Zou et al. (2023).