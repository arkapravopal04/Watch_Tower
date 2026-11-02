"""Fixed greedy decoding and per-item grading. No implicit locked-test access."""

import time
from dataclasses import asdict

import torch

from .data.grader import grade_answer
from .tokenization import prompt_ids

PROGRESS_SECONDS = 120


@torch.inference_mode()
def evaluate(model, tokenizer, records: list[dict], *, batch_size: int = 1,
             max_new_tokens: int = 128, max_context: int = 1024,
             locked_authorized: bool = False, label: str = 'eval') -> list[dict]:
    """Greedy answers in the input order.

    Batches are formed over prompts sorted by length (ties by position), so
    padding stays small and the batch composition depends only on the data and
    batch_size: every model scored with the same settings sees identical batches.
    """
    if batch_size < 1 or max_new_tokens < 1 or not records:
        raise ValueError("evaluation requires data and positive batch/output sizes")
    if not locked_authorized and any(record['split'].startswith('test') for record in records):
        raise ValueError("locked tests require a preregistered artifact authorization")
    model.eval()
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("an explicit pad token is required")
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    prompts = [prompt_ids(tokenizer, record) for record in records]
    order = sorted(range(len(records)), key=lambda i: (len(prompts[i]), i))
    predictions = [None] * len(records)
    started = last_report = time.monotonic()
    for start in range(0, len(order), batch_size):
        batch = order[start:start + batch_size]
        width = max(len(prompts[i]) for i in batch)
        if width + max_new_tokens > max_context:
            raise ValueError("prompt plus generation budget exceeds the declared inference context")
        ids = torch.tensor([[pad] * (width - len(prompts[i])) + prompts[i] for i in batch], device=device)
        mask = torch.tensor([[0] * (width - len(prompts[i])) + [1] * len(prompts[i]) for i in batch], device=device)
        generated = model.generate(input_ids=ids, attention_mask=mask, do_sample=False, num_beams=1,
                                   max_new_tokens=max_new_tokens, pad_token_id=pad,
                                   eos_token_id=eos, use_cache=True)
        completions = generated[:, width:]
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        for index, text, tokens in zip(batch, texts, completions):
            record = records[index]
            token_ids = tokens.tolist()
            stop = next((i + 1 for i, token in enumerate(token_ids) if token in eos_ids), None)
            generated_count = stop if stop is not None else len(token_ids)
            if stop is None:
                result = {"correct": False, "format_ok": False, "reason": "generation limit reached without EOS"}
            else:
                result = asdict(grade_answer(text, record['reference']))
            predictions[index] = {"id": record['id'], "group": record['group'], "skill": record['skill'],
                                  "family": record['family'], "split": record['split'], "prediction": text,
                                  "generated_tokens": generated_count, **result}
        now = time.monotonic()
        if now - last_report >= PROGRESS_SECONDS:
            done = min(start + batch_size, len(order))
            rate = done / (now - started)
            print(f'[{label}] {done}/{len(order)} answers, {now - started:.0f} s, '
                  f'about {(len(order) - done) / rate:.0f} s left', flush=True)
            last_report = now
    return predictions


def summarize(predictions: list[dict]) -> dict:
    if not predictions:
        raise ValueError("empty predictions")
    return {"n": len(predictions), "accuracy": sum(p['correct'] for p in predictions) / len(predictions),
            "format_failures": sum(not p['format_ok'] for p in predictions),
            "groups": len({p['group'] for p in predictions})}
