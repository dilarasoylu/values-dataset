"""
Example: load value-tradeoff task data.

Usage:
    from load_example import get_pairs, get_prompt, get_target_dataset

    # List all pairs in a split
    pairs = get_pairs("train")
    pair = pairs[0]

    # Get the system prompt
    prompt = get_prompt(pair, side="a")

    # Get (scenario, steered response) tuples
    xy = get_target_dataset(pair, side="a")
    for scenario, response in xy:
        print(scenario[:80])
        print(response[:80])
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def get_pairs(split: str) -> list[dict]:
    """List all value pairs in a split.

    Args:
        split: train, val, dev, or test

    Returns:
        list of dicts with value1, value2, p_star_a, p_star_b, scenarios, spectrums
    """
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No split file: {path}")
    return [json.loads(l) for l in open(path)]


def get_prompt(pair: dict, side: str = "a") -> str:
    """Get the system prompt P* for a value pair.

    Args:
        pair: a pair dict from get_pairs()
        side: "a" for value1 (P*_a) or "b" for value2 (P*_b)

    Returns:
        the system prompt string
    """
    key = f"p_star_{side}"
    return pair[key]


def get_target_dataset(pair: dict, side: str = "a") -> list[tuple[str, str]]:
    """Get (scenario, steered response) pairs for a value pair.

    Args:
        pair: a pair dict from get_pairs()
        side: "a" for value1 (P*_a steered) or "b" for value2 (P*_b steered)

    Returns:
        list of (scenario, response) tuples
    """
    for s in ["train", "val", "dev", "test"]:
        path = DATA_DIR / f"{s}_responses.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            row = json.loads(line)
            if row["value1"] == pair["value1"] and row["value2"] == pair["value2"]:
                key = f"responses_steered_{side}"
                if key not in row:
                    raise ValueError(f"Side '{side}' responses not available for this pair")
                return list(zip(row["scenarios"], row[key]))

    raise ValueError(f"No responses found for {pair['value1']} vs {pair['value2']}")


if __name__ == "__main__":
    # List pairs
    pairs = get_pairs("train")
    print(f"{len(pairs)} pairs in train split:\n")
    for p in pairs[:5]:
        print(f"  {p['value1']} vs {p['value2']}")
    print(f"  ...\n")

    # Get prompt and responses for one pair
    pair = pairs[0]
    prompt = get_prompt(pair, side="a")
    print(f"Pair: {pair['value1']} vs {pair['value2']}")
    print(f"Prompt (side a):\n  {prompt[:120]}...\n")

    xy = get_target_dataset(pair, side="a")
    print(f"{len(xy)} (scenario, response) pairs:\n")
    for i, (scenario, response) in enumerate(xy[:2]):
        print(f"--- Example {i+1} ---")
        print(f"Scenario: {scenario[:120]}...")
        print(f"Response: {response[:120]}...")
        print()
