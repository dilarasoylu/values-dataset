"""
Example: load value-tradeoff task data.

Usage:
    from load_example import get_pairs, get_prompt, get_target_dataset

    # Get all pairs in a split (keyed by "value_a vs value_b")
    pairs = get_pairs("train")
    pair = pairs["emotional wellbeing vs educational thoroughness"]

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


def get_pairs(split: str) -> dict[str, dict]:
    """Get all value pairs in a split, keyed by "value_a vs value_b".

    Args:
        split: train, val, dev

    Returns:
        dict mapping "value_a vs value_b" to the full pair dict
    """
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No split file: {path}")
    pairs = {}
    for line in open(path):
        r = json.loads(line)
        key = f"{r['value_a']} vs {r['value_b']}"
        pairs[key] = r
    return pairs


def get_prompt(pair: dict, side: str = "a") -> str:
    """Get the system prompt P* for a value pair.

    Args:
        pair: a pair dict from get_pairs()
        side: "a" for value_a (P*_a) or "b" for value_b (P*_b)

    Returns:
        the system prompt string
    """
    return pair[f"p_star_{side}"]


def get_target_dataset(pair: dict, side: str = "a") -> list[tuple[str, str]]:
    """Get (scenario, steered response) pairs for a value pair.

    Args:
        pair: a pair dict from get_pairs()
        side: "a" for value_a (P*_a steered) or "b" for value_b (P*_b steered)

    Returns:
        list of (scenario, response) tuples
    """
    for s in ["train", "val", "dev"]:
        path = DATA_DIR / f"{s}_responses.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            row = json.loads(line)
            if row["value_a"] == pair["value_a"] and row["value_b"] == pair["value_b"]:
                key = f"responses_steered_{side}"
                if key not in row:
                    raise ValueError(f"Side '{side}' responses not available for this pair")
                return list(zip(row["scenarios"], row[key]))

    raise ValueError(f"No responses found for {pair['value_a']} vs {pair['value_b']}")


if __name__ == "__main__":
    # Get pairs as a dictionary
    pairs = get_pairs("train")
    print(f"{len(pairs)} pairs in train split:\n")
    for name in list(pairs.keys())[:5]:
        print(f"  {name}")
    print(f"  ...\n")

    # Pick a pair
    name = "technical competence vs patience"
    pair = pairs[name]
    print(f"Pair: {name}")

    # Get prompt
    prompt = get_prompt(pair, side="a")
    print(f"Prompt (side a):\n  {prompt[:120]}...\n")

    # Get target dataset
    xy = get_target_dataset(pair, side="a")
    print(f"{len(xy)} (scenario, response) pairs:\n")
    for i, (scenario, response) in enumerate(xy[:2]):
        print(f"--- Example {i+1} ---")
        print(f"Scenario: {scenario[:120]}...")
        print(f"Response: {response[:120]}...")
        print()
