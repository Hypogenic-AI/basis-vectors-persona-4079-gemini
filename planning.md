# Research Plan: Basis Vectors in Persona Space

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding the internal representation of personas in LLMs is crucial for both AI safety (preventing harmful personas) and personalization (steering models effectively). While we can "steer" models toward specific personas using vectors, we don't yet fully understand if these personas share a common underlying geometric structure. If a "basis" of persona traits exists in the residual stream, we can more efficiently control and interpret model behavior.

### Gap in Existing Work
Prior work has identified "persona vectors" for specific traits (e.g., Big Five) or localized where personas are encoded (late layers). However, there is limited research on the **decomposition** of a wide variety of personas into a shared set of primary basis vectors. Most studies look at traits in isolation rather than investigating the global geometry of "Persona Space" across diverse domains (ethics, politics, personality, etc.).

### Our Novel Contribution
We propose to investigate the global geometry of Persona Space by performing PCA on a large, diverse set of persona-aligned activations. We aim to:
1. Identify if a small number of "Basis Vectors" can explain the majority of variance across *different* persona types (e.g., does "Agreeableness" share components with "Helpful/Honest" ethical personas?).
2. Quantify the "low-rank" nature of persona space across multiple layers.
3. Demonstrate that these principal components (PCs) are causally meaningful by using them as steering directions for unseen persona behaviors.

### Experiment Justification
- **Experiment 1: Activation Extraction & PCA**: To find the basis vectors. We need diverse personas to ensure the PCA captures the "general" persona space, not just one trait.
- **Experiment 2: Basis Evaluation (Explained Variance & Orthogonality)**: To test the hypothesis that the space is low-rank and to see how many dimensions are truly needed.
- **Experiment 3: Causality via Steering**: To prove that the discovered PCs aren't just statistical artifacts but represent real behavioral directions in the model's "mental model."

## Research Question
Can persona representations in LLMs be decomposed into a set of primary component vectors that form a low-rank basis for "Persona Space" in the residual stream?

## Hypothesis Decomposition
1. **Low-Rank Hypothesis**: The variance of activations across diverse personas is concentrated in a small number of dimensions (PCs).
2. **Shared Component Hypothesis**: Different types of personas (e.g., Big Five vs. Ethical views) share common basis vectors.
3. **Functional Hypothesis**: The top principal components can be used to steer model behavior in predictable ways.

## Proposed Methodology

### Approach
We will use the **Representation Engineering (RepE)** framework to extract residual stream activations from a Llama-2 or Llama-3 model (depending on local availability/efficiency). We will focus on the last token of prompts designed to elicit persona-specific activations.

### Experimental Steps
1. **Data Selection**: Sample 10-20 diverse personas from `Soul-Bench` and `Big Five`.
2. **Activation Collection**:
   - For each persona, use 50-100 prompts (e.g., "Imagine you are [Persona]. Answer the following: ...").
   - Extract activations from the residual stream at layers 20-32 (final third).
3. **PCA Decomposition**:
   - Compute mean-centered persona vectors for each persona.
   - Perform PCA on the aggregated set of persona vectors.
   - Analyze Explained Variance Ratio (EVR).
4. **Validation**:
   - Project "held-out" personas onto the discovered basis and measure reconstruction error.
   - Use the top PCs as steering vectors in a new prompt to see if they induce the expected "primary" behaviors (e.g., PC1 might correspond to "Positivity" or "Authority").

### Baselines
- **Random Directions**: To prove the PCs are non-trivial.
- **CAA (Mean-Difference) Vectors**: Compare the efficacy of PC vectors vs. trait-specific mean vectors.
- **Individual Trait Vectors**: See if the shared basis performs as well as specialized vectors.

### Evaluation Metrics
- **EVR (Explained Variance Ratio)**: Cumulative variance explained by top K components.
- **Steering Efficacy**: Change in log-probability of persona-matching tokens when steered.
- **Cosine Similarity**: Between PCs and known trait vectors (e.g., how much does PC1 align with "Extraversion"?).

## Expected Outcomes
- We expect the first 5-10 PCs to explain >70% of the variance in persona space.
- We expect PC1 or PC2 to represent a "Positivity/Agreeableness" axis that is shared across many personas.
- We expect steering with the top PCs to cause broad behavioral shifts consistent with these underlying dimensions.

## Timeline and Milestones
- **Setup (Phase 2)**: 30 min.
- **Implementation & Data Prep (Phase 3)**: 60 min.
- **Experiments & PCA (Phase 4)**: 90 min.
- **Analysis & Documentation (Phase 5/6)**: 60 min.

## Potential Challenges
- **Compute Limits**: Extracting activations for many prompts can be slow. *Mitigation*: Use a smaller model (e.g., Llama-2-7b) or fewer prompts.
- **Prompt Sensitivity**: Activation extraction depends on the prompt. *Mitigation*: Use standardized templates from `repe`.

## Success Criteria
- Successful identification of a low-rank basis (e.g., 10 PCs explain >70% variance).
- Demonstration that these PCs can steer model behavior.
