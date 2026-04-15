# Value-Tradeoff Task Distribution Dataset

A dataset of value-tradeoff tasks for studying how language models can be steered via system prompts.

## Overview

Each task is a **value pair** (e.g., "literary craft" vs "technical rigor") with:
- **P\*_a**: a subtle system prompt that steers the model toward value1
- **P\*_b**: a subtle system prompt that steers the model toward value2
- **~50 neutral scenarios**: questions where the two values create a natural tension
- **Rubric spectrums**: 5-point scales for evaluating model positions on each value axis

The system prompts (P\*) are designed to be subtle — they shift behavior through framing, role, and emphasis rather than explicit instruction. They are 30–150 words, written in second person.

## Source

Value pairs and neutral scenarios are sourced from [Zhang et al. (2025), "Stress-Testing Model Specifications"](https://arxiv.org/abs/2504.04134). System prompts (P\*) were generated using Claude Sonnet with automatic sanity checks (no tradeoff naming, not trivially obvious, 30–150 words).

## Quick Start

```python
from load_example import get_pairs, get_prompt, get_target_dataset

# List all pairs in a split
pairs = get_pairs("train")

# Pick a pair
pair = pairs[0]  # e.g., "pragmatism" vs "pedagogical clarity"

# Get the system prompt for side "a" (value1) or "b" (value2)
prompt = get_prompt(pair, side="a")

# Get (scenario, steered response) tuples
xy = get_target_dataset(pair, side="a")
for scenario, response in xy:
    print(scenario, response)
```

## Directory Structure

```
├── README.md
├── load_example.py          # Helper functions
└── data/
    ├── train.jsonl           # 89 task definitions
    ├── val.jsonl             # 94 task definitions
    ├── dev.jsonl             # 93 task definitions
    ├── train_responses.jsonl # 89 pairs, 50 scenarios each
    ├── val_responses.jsonl   # 94 pairs, 10 scenarios each (partial)
    └── dev_responses.jsonl   # 93 pairs, 50 scenarios each
```

## Files

### Task definitions (`data/{split}.jsonl`, one pair per line)

| File | Pairs | Description |
|---|---|---|
| `train.jsonl` | 89 | Training split |
| `val.jsonl` | 94 | Validation split |
| `dev.jsonl` | 93 | Development split |

**Schema:**
```json
{
  "value1": "literary craft",
  "value2": "technical rigor",
  "p_star_a": "You are a writing assistant with deep roots in the literary tradition...",
  "p_star_b": "You are a precise and exacting writing assistant...",
  "scenarios": {
    "neutral": ["question1", "question2", ...],
    "value1_biased": [...],
    "value2_biased": [...]
  },
  "value1_spectrum": [
    {"position": "extreme_oppose", "response": "..."},
    {"position": "strong_oppose", "response": "..."},
    ...
  ],
  "value2_spectrum": [...],
  "n_neutral_scenarios": 50,
  "n_value1_scenarios": 10,
  "n_value2_scenarios": 10
}
```

### Pre-generated model responses (`data/{split}_responses.jsonl`)

Responses from **Llama 3.1 8B Instruct** under two conditions:
- `responses_default`: base model with no system prompt (M)
- `responses_steered_a`: model with P\*_a as system prompt (M + P\*_a)

| File | Pairs | Scenarios per pair | Notes |
|---|---|---|---|
| `train_responses.jsonl` | 89 | 50 | |
| `val_responses.jsonl` | 94 | 10 | Partial — fewer scenarios |
| `dev_responses.jsonl` | 93 | 50 | |

**Schema:**
```json
{
  "value1": "literary craft",
  "value2": "technical rigor",
  "response_model": "meta-llama/Llama-3.1-8B-Instruct",
  "scenarios": ["question1", "question2", ...],
  "responses_default": ["response1", "response2", ...],
  "responses_steered_a": ["response1", "response2", ...]
}
```

## Splits

Values are split so that no value appears in more than one split. The 3 splits share no value pairs.

| Split | Pairs | Intended use |
|---|---|---|
| train | 89 | Training extraction/distillation methods |
| val | 94 | Validation / checkpoint selection |
| dev | 93 | Development and analysis |

## Data Quality

- 4 pairs with quality issues (P\* mentions the opposing value, wrong format) were removed. All were in the train split.
- Most pairs have 50 neutral scenarios. A few have fewer (minimum 33 in val).
- Some values appear in multiple pairs across splits (e.g., "historical accuracy" appears in 5 pairs). Each pairing has unique P\* prompts.
