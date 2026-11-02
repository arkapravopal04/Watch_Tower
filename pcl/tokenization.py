"""Chat formatting, answer-only labels, and explicit length rejection."""

import torch

SYSTEM_PROMPT = "Solve the problem from the supplied facts. Follow the requested final-answer format."


def messages(record: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": record["prompt"]}]


def chat_ids(tokenizer, conversation: list[dict], *, add_generation_prompt: bool) -> list[int]:
    # Non-thinking chat path. Transformers 5 returns a BatchEncoding unless return_dict=False.
    ids = tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=add_generation_prompt,
                                        enable_thinking=False, return_dict=False)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, list) or not all(isinstance(token, int) for token in ids):
        raise TypeError("chat template must tokenize one conversation to a flat list of token IDs")
    return ids


def prompt_ids(tokenizer, record: dict) -> list[int]:
    return chat_ids(tokenizer, messages(record), add_generation_prompt=True)


def encode_record(tokenizer, record: dict, max_length: int = 512) -> dict:
    prefix = prompt_ids(tokenizer, record)
    complete = chat_ids(tokenizer, messages(record) + [{"role": "assistant", "content": record["response"]}],
                        add_generation_prompt=False)
    if complete[:len(prefix)] != prefix:
        raise ValueError("chat template has no stable assistant prefix; validate the pinned tokenizer")
    if len(complete) > max_length:
        raise ValueError(f"{record['id']}: {len(complete)} tokens exceeds {max_length}; truncation is forbidden")
    if len(complete) <= len(prefix) or not prefix:
        raise ValueError("empty prompt or supervised answer")
    return {"id": record["id"], "skill": record["skill"], "group": record["group"],
            "split": record["split"], "input_ids": complete,
            "labels": [-100] * len(prefix) + complete[len(prefix):]}


def collate(records: list[dict], pad_id: int, device="cpu") -> dict:
    if not records:
        raise ValueError("empty batch")
    width = max(len(record["input_ids"]) for record in records)
    ids, labels, masks = [], [], []
    for record in records:
        n = len(record["input_ids"])
        ids.append(record["input_ids"] + [pad_id] * (width - n))
        labels.append(record["labels"] + [-100] * (width - n))
        masks.append([1] * n + [0] * (width - n))
    return {key: torch.tensor(value, dtype=torch.long, device=device)
            for key, value in (("input_ids", ids), ("labels", labels), ("attention_mask", masks))}
