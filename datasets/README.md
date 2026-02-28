# Downloaded Datasets

This directory contains samples of datasets used for the research project "Basis Vectors in Persona Space". Due to size constraints, only samples are saved. Full datasets can be downloaded using the instructions below.

## Dataset 1: Soul-Bench
- **Source**: [APRIL-AIGC/Soul-Bench](https://huggingface.co/datasets/APRIL-AIGC/Soul-Bench)
- **Description**: Benchmark for evaluating personality alignment and steering in LLMs.
- **Task**: Persona-driven question answering and behavioral alignment.
- **Sample Location**: `datasets/Soul-Bench_sample`

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset('APRIL-AIGC/Soul-Bench', split='test')
```

## Dataset 2: Persona-Chat
- **Source**: [AlekseyKorshuk/persona-chat](https://huggingface.co/datasets/AlekseyKorshuk/persona-chat)
- **Description**: Standard benchmark for conversational persona.
- **Task**: Multi-turn dialogue with fixed persona profiles.
- **Sample Location**: `datasets/personachat_sample`

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset('AlekseyKorshuk/persona-chat', split='train')
```

## Dataset 3: Big Five Personality Traits
- **Source**: [agentlans/big-five-personality-traits](https://huggingface.co/datasets/agentlans/big-five-personality-traits)
- **Description**: Statements related to the Big Five personality traits.
- **Task**: Classification/Detection of personality traits.
- **Sample Location**: `datasets/big-five_sample`

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset('agentlans/big-five-personality-traits', split='train')
```
