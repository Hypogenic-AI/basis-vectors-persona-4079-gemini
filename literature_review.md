# Literature Review: Basis Vectors in Persona Space

## Research Area Overview
The study of persona representations in Large Language Models (LLMs) focuses on how human-like personality traits, ethical views, and social identities are encoded within the model's internal activations. Recent research has shifted from treating LLMs as monolithic black boxes to identifying specific "directions" or "subspaces" in the residual stream that correspond to these high-level concepts. This field, often called "Representation Engineering" or "Mechanistic Interpretability," aims to provide mathematical foundations for controllable and interpretable AI personalization.

## Key Papers

### 1. Localizing Persona Representations in LLMs
- **Authors**: C. Cintas, M. Rateike, E. Miehling, E. Daly, S. Speakman
- **Year**: 2025
- **Source**: arXiv:2505.24539
- **Key Contribution**: Identifies where and how personas are encoded using dimension reduction (PCA).
- **Methodology**: Extracts activations from the last token of sentences matching specific personas (Big Five, Ethics, Politics). Applies PCA to find separating components.
- **Results**: Persona representations are most divergent in the **final third of decoder layers** (e.g., layers 22-31 in a 32-layer model). PCA effectively separates matching vs. non-matching behaviors.
- **Relevance**: Directly supports the hypothesis that PCA on persona vectors reveals primary components in the residual stream.

### 2. The Geometry of Persona: Disentangling Personality from Reasoning
- **Authors**: Zhixiang Wang
- **Year**: 2025
- **Source**: arXiv:2512.07092
- **Key Contribution**: Proposes the "Soul Engine" framework based on the Linear Representation Hypothesis.
- **Methodology**: Posits that personality traits exist as **orthogonal linear subspaces**. Uses a dual-head architecture to extract disentangled vectors.
- **Results**: Achieves robust behavioral control via vector arithmetic ($\vec{v}_{neutral} + \alpha \cdot \vec{v}_{villain}$) without degrading reasoning capabilities.
- **Relevance**: Provides a geometric framework for "Persona Space" and confirms the effectiveness of linear interventions.

### 3. Activation-Space Personality Steering: Hybrid Layer Selection
- **Authors**: P. Bhandari, N. Fay, S. Selvaganapathy, et al.
- **Year**: 2025
- **Source**: arXiv:2511.03738
- **Key Contribution**: Discovers that personality traits occupy a **low-rank shared subspace**.
- **Methodology**: Applies low-rank subspace discovery methods to hidden state activations.
- **Results**: Identifies optimal layers across architectures for stable trait control. Steering does not impact fluency or general capabilities.
- **Relevance**: Validates the "low-rank" nature of persona representations, consistent with PCA-based discovery.

### 4. Your Language Model Secretly Contains Personality Subnetworks
- **Authors**: Ruimeng Ye, Z. Wang, Z. Ling, et al.
- **Year**: 2026
- **Source**: arXiv:2602.07164
- **Key Contribution**: Shows that personas are embedded in lightweight "personality subnetworks."
- **Methodology**: Identifies "activation signatures" for different personas and uses them to mask parameters.
- **Results**: Isolates subnetworks that exhibit stronger persona alignment than prompting-based methods.
- **Relevance**: Connects activation-level representations (signatures) to parameter-level structures (subnetworks).

### 5. Steering Llama 2 via Contrastive Activation Addition (CAA)
- **Authors**: Nina Rimsky, et al.
- **Year**: 2023
- **Source**: arXiv:2312.06681
- **Key Contribution**: Introduces CAA for steering model behavior using average activation differences.
- **Relevance**: Foundational work for activation-level steering, widely used as a baseline.

## Common Methodologies
- **Activation Extraction**: Capturing the residual stream state, typically at the last token position of a prompt.
- **Dimension Reduction (PCA)**: Used to find the "basis vectors" that explain the most variance in activations across different persona contexts.
- **Contrastive Analysis**: Comparing activations from "positive" examples (matching persona) vs. "negative" examples (neutral or opposing persona).
- **Activation Steering**: Adding or subtracting identified "persona vectors" to the model's activations during inference to change its behavior.

## Standard Baselines
- **Prompt Engineering**: Using "You are a [persona]" system prompts.
- **Contrastive Activation Addition (CAA)**: Adding the mean difference vector.
- **RepE (Representation Engineering)**: Using the official toolkit's readers and controllers.

## Gaps and Opportunities
- **Basis Orthogonality**: While some work posits orthogonality (Wang 2025), the extent to which different persona traits (e.g., "Agreeableness" and "Introversion") overlap in the residual stream basis is still underexplored.
- **Dynamics across Layers**: Cintas et al. localized personas to the later layers, but the *transformation* of these representations from early semantic processing to late behavioral expression needs more study.
- **Universal Basis**: Is there a "universal basis" of persona components that applies across different models, or is it model-specific?

## Recommendations for Our Experiment
- **Dataset**: Use **SoulBench** for diverse, labeled persona data.
- **Method**: Extract activations from multiple layers (focusing on layers 20-32). Apply **PCA** to the set of vectors representing different personas.
- **Analysis**: Check the **Explained Variance Ratio** of the first few PCs to see if persona space is truly low-rank. Visualize the clusters in 2D/3D using the first 2-3 basis vectors.
- **Steering**: Use the discovered PC vectors as steering directions to verify their causal role.
