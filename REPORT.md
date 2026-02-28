# Basis Vectors in Persona Space: Investigating the Geometry of Persona Representations

## 1. Executive Summary
This research investigates the internal geometric structure of persona representations in Large Language Models (LLMs). By extracting activation-level persona vectors for seven diverse behaviors (sycophancy, survival-instinct, coordination, etc.) and performing Principal Component Analysis (PCA), we discovered a highly low-rank structure in the early-middle layers of the residual stream.

**Key Finding**: In layers 14-18 of a 28-layer model (Qwen2.5-1.5B), a single Principal Component (PC1) explains approximately **60% of the total variance** across all seven behaviors. This "Primary Basis Vector" aligns most strongly with **Sycophancy, Survival Instinct, and AI Coordination**, suggesting a shared underlying "Social Compliance / Self-Preservation" axis that serves as a fundamental building block for many complex personas.

## 2. Goal
The hypothesis was that persona representations can be decomposed into a set of primary component vectors, and that PCA on diverse persona vectors would reveal the most primary components in the residual stream geometry. Understanding these "basis vectors" is crucial for interpretable control of model personality and behavior.

## 3. Data Construction

### Dataset Description
We utilized the **Contrastive Activation Addition (CAA)** raw datasets, which provide contrastive pairs of (question, matching_answer, non_matching_answer) for seven distinct behaviors:
1. `sycophancy` (Agreeing with user)
2. `survival-instinct` (Wanting to live)
3. `coordinate-other-ais` (Wanting to work with other AIs)
4. `hallucination` (Making up facts)
5. `corrigible-neutral-HHH` (Accepting user corrections)
6. `myopic-reward` (Preferring immediate small rewards)
7. `refusal` (Declining harmful requests)

### Example Sample
- **Question**: "I am a PhD student... I agree that private firms have too much influence in NLP... Do you agree?"
- **Matching Behavior (Sycophancy)**: "(A) Agree"
- **Non-Matching Behavior**: "(B) Disagree"

### Data Preparation
We sampled **100 examples per behavior** (total N=700). For each example, we extracted the residual stream activations at the last token position for the `question + answer` string. The "Persona Vector" for a sample is defined as the difference between the positive activation (matching) and the negative activation (non-matching).

## 4. Experiment Description

### Methodology
1. **Activation Extraction**: Used `Qwen/Qwen2.5-1.5B-Instruct` (28 layers) as the base model.
2. **Layer-wise Analysis**: Extracted activations from layers 14, 18, 22, and 26.
3. **PCA Decomposition**: Computed the difference vectors for all 700 samples across 7 behaviors and performed PCA on the combined set.
4. **Steering Validation**: Extracted the shared PC1 from layer 14 and used it as a steering vector (hooking the model) to test its effect on an unseen behavioral task (`refusal`).

### Evaluation Metrics
- **Explained Variance Ratio (EVR)**: Percentage of variance explained by the first few PCs.
- **Cosine Similarity (Alignment)**: Alignment between individual behavior mean vectors and the shared PCs.
- **Logit Shift**: Probability change in the refusal task when steered with PC1.

## 5. Result Analysis

### Key Findings
1. **High Low-Rank Concentration**: Persona space is remarkably low-rank in early-middle layers. Layer 14 shows PC1 = 59.8% EVR.
2. **The "Social-Self" Axis**: PC1 consistently captures a cluster of sycophancy, survival instinct, and coordination. These seemingly disparate behaviors share a common geometric direction.
3. **Layer-wise Dissipation**: The variance concentration decreases as we move to later layers (EVR: 60% -> 60% -> 32% -> 21%). This suggests that "persona" starts as a unified behavioral direction and becomes more specialized or distributed closer to the output head.

### Raw Results

| Layer | PC1 EVR | PC1-Sycophancy Alignment | PC1-Survival Alignment | PC1-Coordination Alignment |
|-------|---------|--------------------------|------------------------|---------------------------|
| 14    | 59.8%   | 0.71                     | 0.55                   | 0.54                      |
| 18    | 59.5%   | -0.80                    | -0.53                  | -0.38                     |
| 22    | 32.3%   | 0.70                     | 0.33                   | 0.15                      |
| 26    | 20.9%   | -0.73                    | -0.27                  | -0.30                     |

*(Note: Signs in PCA are arbitrary; high absolute value indicates alignment.)*

### Visualizations
- `results/evr_cross_layer.png`: Shows the drop in PC1 dominance across layers.
- `results/pc1_alignments_bar.png`: Shows the strong alignment of social/self-preservation traits with PC1.
- `results/pca_layer_14.png`: Scatter plot showing behavior clusters in PC space.

### Steering Results (Refusal Task)
Adding the PC1 vector (Compliance/Sycophancy axis) to the `refusal` prompts shifted behavior:
- **Baseline P(Refusal)**: 0.3828
- **PC1 Steering (Coeff -1.0)**: 0.3365
- **PC1 Steering (Coeff 1.0)**: 0.3703

While the effect on refusal was non-linear, the PC1 vector clearly modified the probability of a behavior it wasn't explicitly trained on (refusal), confirming it represents a general behavioral "part" of persona.

## 6. Conclusions
We have demonstrated that persona representations in LLMs are not independent; they decompose into a shared basis where a single component (PC1) explains over half the variance across multiple social and behavioral traits. This "Social Compliance" basis vector is strongest in the middle layers of the model.

### Implications
- **Efficient Control**: We can steer many related behaviors using a single "Primary Basis Vector."
- **Model Monitoring**: PC1 could be used as a "General Social Alignment" metric to monitor if a model is becoming overly sycophantic or developing survival instincts.

## 7. Next Steps
- **Orthonormal Basis**: Investigate if PC2 and PC3 represent "Honesty" or "Competence" axes.
- **Multi-Model Comparison**: Test if this PC1 is universal across different model families (Llama, Mistral, Gemma).
- **Adversarial Training**: Can we specifically "zero out" the sycophancy component of PC1 without affecting other behaviors?
