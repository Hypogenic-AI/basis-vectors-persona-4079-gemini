# Cloned Repositories

This directory contains official and community implementations of techniques related to persona representation and steering.

## Repo 1: Representation Engineering (RepE)
- **URL**: [https://github.com/andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering)
- **Purpose**: A toolkit for extracting and controlling high-level concepts in LLMs.
- **Key Files**:
  - `repe/rep_readers.py`: Extracts directions from activations (includes PCA, etc.).
  - `repe/rep_control_pipeline.py`: Performs activation-level steering.
  - `examples/`: Notebooks for honesty, emotions, and fairness.

## Repo 2: Persona Subnetworks
- **URL**: [https://github.com/Ruimeng-Ye/Persona](https://github.com/Ruimeng-Ye/Persona)
- **Purpose**: Implementation of the paper "Your Language Model Secretly Contains Personality Subnetworks".
- **Key Files**:
  - `contra.py`: Contrastive signature identification.
  - `mask.py`: Parameter masking for persona isolation.

## Repo 3: Contrastive Activation Addition (CAA)
- **URL**: [https://github.com/nrimsky/CAA](https://github.com/nrimsky/CAA)
- **Purpose**: Implementation of Contrastive Activation Addition for steering model behavior.
- **Key Files**:
  - `steering_vectors.py`: Logic for computing and applying steering vectors.
