# Resources Catalog: Basis Vectors in Persona Space

## Summary
This document catalogs the papers, datasets, and code repositories gathered for the project. The primary goal is to provide the foundations for analyzing persona representations in LLM residual streams using PCA.

## Papers
Total papers downloaded: 8

| Title | Authors | Year | arXiv ID | Key Info |
|-------|---------|------|----------|----------|
| Localizing Persona Representations in LLMs | Cintas et al. | 2025 | 2505.24539 | Localizes personas to late layers using PCA. |
| The Geometry of Persona | Zhixiang Wang | 2025 | 2512.07092 | Orthogonal linear subspace hypothesis. |
| Your Language Model Secretly Contains Personality Subnetworks | Ye et al. | 2026 | 2602.07164 | Personality subnetworks and activation signatures. |
| Activation-Space Personality Steering | Bhandari et al. | 2025 | 2511.03738 | Low-rank shared subspace for traits. |
| Persona Vectors | Chen et al. | 2025 | 2507.21509 | Monitoring and controlling traits with vectors. |
| Steering Llama 2 via CAA | Rimsky et al. | 2023 | 2312.06681 | Foundational contrastive activation addition. |
| Identifying and Manipulating Personality Traits | Allbert & Wiles | 2024 | 2412.10427 | Activation engineering for personality. |
| Bias Runs Deep | Gupta et al. | 2023 | 2311.04892 | Implicit reasoning biases in personas. |

## Datasets
Total datasets sampled: 3

| Name | Source | Task | Location | Notes |
|------|--------|------|----------|-------|
| Soul-Bench | APRIL-AIGC | Behavioral Alignment | `datasets/Soul-Bench_sample` | Best for diverse personas. |
| Persona-Chat | AlekseyKorshuk | Conversational Persona | `datasets/personachat_sample` | Standard dialogue benchmark. |
| Big Five | agentlans | Trait Classification | `datasets/big-five_sample` | Focuses on Big Five traits. |

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| RepE | github.com/andyzoujm/representation-engineering | General Toolkit | `code/repe` | Primary tool for activation reading/steering. |
| Persona Subnetworks | github.com/Ruimeng-Ye/Persona | Subnetwork Masking | `code/persona-subnetworks` | Activation signatures and masking. |
| CAA | github.com/nrimsky/CAA | Steering Vectors | `code/contrastive-activation-addition` | Mean-based steering vectors. |

## Recommendations for Experiment Design

1. **Primary Dataset**: **Soul-Bench**. It contains the most diverse and scientifically grounded persona categories (Big Five, Ethics, Politics).
2. **Baseline Methods**: Compare **PCA-derived basis vectors** against **Mean-difference vectors (CAA)**.
3. **Evaluation Metrics**:
   - **Explained Variance Ratio (EVR)**: To quantify how well a low-dimensional basis captures persona variance.
   - **Centroid Distance (Silhouette/Calinski-Harabasz)**: To measure separation in the discovered basis.
   - **Steering Accuracy**: Behavioral shift when applying PC vectors as steering directions.
4. **Code to Use**: Use `code/repe` for robust activation extraction and the `RepresentationReader` class for PCA analysis.
