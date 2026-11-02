"""One-session GPU pilot: precision, throughput, memory, merge tolerances, M0 dev baseline.

Every number written here is measured; every suggested setting states the
rule that produced it. Review pilot_report.md, then pass the generated
study.yaml as --settings to every later stage. Nothing here touches tests.
"""

import gc
import math
import time
from pathlib import Path

import torch
import yaml

from .evaluate import evaluate, summarize
from .io import read_jsonl, write_json, write_jsonl

FP16_MAX = 65504.0
BATCH_AGREEMENT = 0.9
MERGE_FLIPS = 20  # 5% of 400 merge-validation answers
MERGE_SHIFT_PP = 1.5


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _round_up(value, digits=2):
    if value <= 0 or not math.isfinite(value):
        return value
    exponent = math.floor(math.log10(value)) - digits + 1
    return math.ceil(value / 10 ** exponent) * 10 ** exponent


@torch.inference_mode()
def _probe(model, encoded, positions_per_probe=16):
    """Teacher-forced logits at fixed positions, answer-token loss, hidden-state peak."""
    device = next(model.parameters()).device
    logits, losses, peak = [], [], 0.0
    for record in encoded:
        ids = torch.tensor([record['input_ids']], device=device)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=True)
        peak = max(peak, max(float(h.float().abs().max()) for h in output.hidden_states))
        index = torch.linspace(0, ids.shape[1] - 1, min(positions_per_probe, ids.shape[1]), device=device).long()
        logits.append(output.logits[0, index].float().cpu())
        labels = torch.tensor(record['labels'][1:], device=device)
        losses.append(float(torch.nn.functional.cross_entropy(output.logits[0, :-1].float(), labels, ignore_index=-100)))
    return torch.cat(logits), losses, peak


def _logit_comparison(reference, candidate):
    finite = bool(torch.isfinite(candidate).all())
    error = (reference - candidate).abs()
    return {'finite': finite, 'max_abs_error': float(error.max()), 'mean_abs_error': float(error.mean()),
            'argmax_agreement': float((reference.argmax(-1) == candidate.argmax(-1)).float().mean())}


def _train_probe(model, tokenizer, records, train_config, *, rank, alpha, bases, updates, seed):
    from .protected_dora import adapter_modules, attach_adapters
    from .train import TrainConfig, Trainer, seed_everything
    settings = {k: v for k, v in train_config.items() if k not in ('adapter', 'rank', 'alpha')}
    settings.update(epochs=1, token_budget=None, seed=seed)
    config = TrainConfig(**settings)
    seed_everything(seed)
    attach_adapters(model, rank=rank, alpha=alpha, bases=bases)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainer = Trainer(model, records, config, pad_id=tokenizer.pad_token_id, provenance={'pilot': True})
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    state = trainer.run(stop_after_updates=updates)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    history = trainer.history
    result = {'updates': state['attempted_updates'], 'skipped_updates': state['skipped_updates'],
              'seconds': elapsed, 'seconds_per_update': elapsed / max(state['attempted_updates'], 1),
              'tokens_per_second': state['nonpadding_tokens'] / max(elapsed, 1e-9),
              'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              'peak_reserved_vram_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
              'loss_first': history[0]['loss'], 'loss_last': history[-1]['loss'],
              'all_losses_finite': all(math.isfinite(h['loss']) for h in history),
              'final_scale': history[-1]['scaler'],
              'first_update_gradients': next((h['gradient_norms'] for h in history if h['gradients_finite']), None)}
    model.gradient_checkpointing_disable()
    model.eval()
    return result, adapter_modules(model)


def run_pilot(*, model_config, data_root, out, train_config, protection_config, probes=6, train_updates=8,
              eval_items=48, batch_sizes=(1, 8, 16, 32), merge_items=64, capture_tokens=2048, seed=0) -> dict:
    from .activations import capture_moments, sample_positions
    from .hashing import model_hashes
    from .merge import merge_live
    from .model_setup import load_model
    from .protection import build_basis
    from .tokenization import encode_record

    out, data = Path(out), Path(data_root)
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(model_config).read_text(encoding='utf-8'))
    train_settings = yaml.safe_load(Path(train_config).read_text(encoding='utf-8'))
    protection_settings = yaml.safe_load(Path(protection_config).read_text(encoding='utf-8'))
    dev = {name: read_jsonl(data / name / 'dev200.jsonl') for name in ('A', 'B', 'general')}
    report = {'model_config': config, 'checks': {}, 'concerns': []}
    load = dict(manifest_path=config['snapshot_manifest'], backend=config['backend'])

    # 1. FP32 reference from the original weights, on CPU (16 GB host RAM for 4B).
    reference, tokenizer, _ = load_model(config['path'], precision='fp32', device='cpu', **load)
    probe_records = [r for name in ('A', 'B', 'general') for r in dev[name][:max(1, probes // 3)]]
    encoded = [encode_record(tokenizer, r, config['max_length']) for r in probe_records]
    weights = {'max_abs': 0.0, 'above_fp16_max': 0, 'fp16_underflow': 0}
    for parameter in reference.parameters():
        value = parameter.detach().abs()
        weights['max_abs'] = max(weights['max_abs'], float(value.max()))
        weights['above_fp16_max'] += int((value > FP16_MAX).sum())
        weights['fp16_underflow'] += int(((value > 0) & (value < 6e-8)).sum())
    started = time.perf_counter()
    reference_logits, reference_losses, reference_peak = _probe(reference, encoded)
    report['fp32_reference'] = {'weights': weights, 'hidden_state_peak': reference_peak,
                                'answer_loss': reference_losses, 'seconds': time.perf_counter() - started}
    del reference
    _free()

    # 2. The configured precision/device (FP16 on a T4 for the study).
    model, tokenizer, _ = load_model(config['path'], precision=config['precision'], device=config['device'], **load)
    m0_hashes = model_hashes(model)
    write_json(out / 'm0_hashes.json', {'tensor_hashes': m0_hashes, 'precision': config['precision']})
    candidate_logits, candidate_losses, candidate_peak = _probe(model, encoded)
    precision = _logit_comparison(reference_logits, candidate_logits)
    precision.update(hidden_state_peak=candidate_peak, answer_loss=candidate_losses,
                     max_answer_loss_difference=max(abs(a - b) for a, b in zip(reference_losses, candidate_losses)))
    report['precision'] = precision
    print(f"[pilot] FP16 vs FP32: argmax agreement {precision['argmax_agreement']:.4f}, "
          f"finite {precision['finite']}", flush=True)

    # 3. Greedy generation: throughput per batch size, and a sanity check against batch 1.
    # Exact batch invariance is not required: every model is scored with the same
    # length-sorted batches, so comparisons stay paired. FP16 padding only perturbs
    # near-tied tokens; a broken batched path shows up as many disagreeing items.
    items = [r for name in ('A', 'B', 'general') for r in dev[name][:max(1, eval_items // 3)]]
    generation, baseline_predictions = {}, None
    for size in batch_sizes:
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            predictions = evaluate(model, tokenizer, items, batch_size=size, max_new_tokens=config['max_new_tokens'],
                                   max_context=config['max_context'], label=f'pilot generation batch {size}')
            seconds = time.perf_counter() - started
        except torch.cuda.OutOfMemoryError:
            generation[size] = {'oom': True}
            _free()
            break
        if baseline_predictions is None:
            baseline_predictions = predictions
        pairs = list(zip(predictions, baseline_predictions))
        generation[size] = {'seconds': seconds, 'items_per_second': len(items) / seconds,
                            'generated_tokens_per_second': sum(p['generated_tokens'] for p in predictions) / seconds,
                            'mean_generated_tokens': sum(p['generated_tokens'] for p in predictions) / len(predictions),
                            'hit_token_limit': sum(p['generated_tokens'] >= config['max_new_tokens'] for p in predictions),
                            'accuracy': summarize(predictions)['accuracy'],
                            'same_text_as_batch_1': sum(a['prediction'] == b['prediction'] for a, b in pairs) / len(pairs),
                            'same_correctness_as_batch_1': sum(a['correct'] == b['correct'] for a, b in pairs) / len(pairs),
                            'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
        g = generation[size]
        print(f"[pilot] batch {size}: {g['items_per_second']:.2f} answers/s, {g['mean_generated_tokens']:.0f} tokens "
              f"on average, accuracy {g['accuracy']:.3f}, same correctness as batch 1 "
              f"{g['same_correctness_as_batch_1']:.0%}", flush=True)
    report['generation'] = generation

    # 4. Q capture and eigensolver cost on M0 (real Q comes from each merged M_A).
    memory = read_jsonl(data / 'A' / 'memory.jsonl')
    memory_encoded = [encode_record(tokenizer, r, config['max_length']) for r in memory]
    selected = sample_positions(memory, [len(r['input_ids']) for r in memory_encoded], capture_tokens, 0)
    started = time.perf_counter()
    moments, _ = capture_moments(model, memory_encoded, selected)
    capture_seconds = time.perf_counter() - started
    basis_settings = {k: v for k, v in protection_settings.items() if k not in ('max_tokens', 'memory_seed')}
    started = time.perf_counter()
    bases, energies, shared = {}, [], True
    for name in sorted(moments):
        bases[name], diagnostics = build_basis(moments.pop(name), **basis_settings)
        energies.append(diagnostics['captured_energy'])
    for name in bases:
        if name.endswith('q_proj'):
            twin = name[:-6] + 'v_proj'
            shared &= twin in bases and bases[twin].shape == bases[name].shape and torch.allclose(
                bases[name] @ bases[name].T, bases[twin] @ bases[twin].T, atol=1e-4)
    report['protection_cost'] = {'capture_tokens': capture_tokens, 'capture_seconds': capture_seconds,
                                 'projected_capture_seconds': capture_seconds * protection_settings['max_tokens'] / capture_tokens,
                                 'eigensolver_seconds': time.perf_counter() - started, 'layers': len(bases),
                                 'k': sorted({int(q.shape[1]) for q in bases.values()}),
                                 'min_captured_energy': min(energies), 'q_v_bases_identical': bool(shared)}
    width = {n: m.in_features for n, m in model.named_modules()
             if n.rsplit('.', 1)[-1] in ('q_proj', 'v_proj') and isinstance(m, torch.nn.Linear)}

    # 5. Training probes: ordinary DoRA, then protected DoRA at the cap k (worst case).
    train_rows = read_jsonl(data / 'A' / 'train.jsonl')
    per_update = train_settings['microbatch'] * train_settings['accumulation']
    records = [encode_record(tokenizer, r, config['max_length']) for r in train_rows[:per_update * train_updates]]
    lengths = [len(r['input_ids']) for r in records]
    probes_out = {}
    probes_out['dora'], wrappers = _train_probe(model, tokenizer, records, train_settings, rank=train_settings['rank'],
                                               alpha=train_settings['alpha'], bases=None, updates=train_updates, seed=seed)
    for name, wrapper in wrappers.items():  # discard without merging; base stays M0
        parent, _, leaf = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, leaf, wrapper.base)
    if model_hashes(model) != m0_hashes:
        raise RuntimeError('pilot training changed the frozen base')
    generator = torch.Generator().manual_seed(seed)
    cap = min(protection_settings['max_k'], min(width.values()) - 1)
    random_bases = {n: torch.linalg.qr(torch.randn(w, cap, generator=generator, dtype=torch.float64)).Q.float()
                    for n, w in width.items()}
    probes_out['protected_max_k'], wrappers = _train_probe(
        model, tokenizer, records, train_settings, rank=train_settings['rank'], alpha=train_settings['alpha'],
        bases=random_bases, updates=train_updates, seed=seed)
    probes_out['protected_max_k']['k'] = cap
    probes_out['protection_overhead'] = probes_out['protected_max_k']['seconds_per_update'] / probes_out['dora']['seconds_per_update']
    report['training'] = probes_out
    print(f"[pilot] training probes: {probes_out['dora']['seconds_per_update']:.1f} s/update ordinary, "
          f"overhead {probes_out['protection_overhead']:.2f}x protected", flush=True)

    # 6. Merge equivalence on the protected probe: live FP32-adapter path vs FP16 merged weights.
    merge_rows = [r for name in ('A', 'general') for r in dev[name][:max(1, merge_items // 2)]]
    merge_encoded = [encode_record(tokenizer, r, config['max_length']) for r in merge_rows[:probes]]
    best_batch = max([s for s, g in generation.items()
                      if not g.get('oom') and g['same_correctness_as_batch_1'] >= BATCH_AGREEMENT], default=1)
    live_logits, _, _ = _probe(model, merge_encoded)
    live_predictions = evaluate(model, tokenizer, merge_rows, batch_size=best_batch,
                                max_new_tokens=config['max_new_tokens'], max_context=config['max_context'],
                                label='pilot merge (live)')
    diagnostics = merge_live(model)
    merged_logits, _, _ = _probe(model, merge_encoded)
    merged_predictions = evaluate(model, tokenizer, merge_rows, batch_size=best_batch,
                                  max_new_tokens=config['max_new_tokens'], max_context=config['max_context'],
                                  label='pilot merge (merged)')
    changed = sum(a['prediction'] != b['prediction'] for a, b in zip(live_predictions, merged_predictions))
    merge = {**_logit_comparison(live_logits, merged_logits), 'changed_answers': changed, 'items': len(merge_rows),
             'max_intended_residual': max(d['intended_residual'] for d in diagnostics.values()),
             'max_stored_residual': max(d['stored_residual'] for d in diagnostics.values())}
    report['merge'] = merge
    print(f"[pilot] merge check: {changed}/{len(merge_rows)} answers changed; M0 baseline next", flush=True)

    # 7. M0 development baseline (full dev files) for ceiling checks and later contrasts.
    # The merged probe model is discarded; the baseline reloads the untouched M0.
    del model
    _free()
    report['baseline'] = _baseline(config, load, data, out / 'baseline', best_batch)

    # 8. Suggestions, each with its rule.
    settings, reasons = _suggest(report, train_settings, lengths, best_batch, data)
    report['suggested_settings'] = settings
    report['rules'] = reasons
    write_json(out / 'pilot_report.json', report)
    (out / 'study.yaml').write_text(
        '# Generated by `pcl pilot`. Review pilot_report.md, then pass this file as --settings\n'
        '# to every later stage. Do not change it after Stage A starts.\n' + yaml.safe_dump(settings, sort_keys=False),
        encoding='utf-8')
    (out / 'pilot_report.md').write_text(_markdown(report), encoding='utf-8')
    return {'out': str(out), 'precision_validated': settings['precision_validated'], 'concerns': report['concerns']}


def _baseline(config, load, data, out, batch_size) -> dict:
    from .model_setup import load_model
    model, tokenizer, _ = load_model(config['path'], precision=config['precision'], device=config['device'], **load)
    summaries = {}
    for name in ('A', 'B', 'general'):
        rows = read_jsonl(data / name / 'dev.jsonl')
        started = time.perf_counter()
        predictions = evaluate(model, tokenizer, rows, batch_size=batch_size, max_new_tokens=config['max_new_tokens'],
                               max_context=config['max_context'], label=f'pilot baseline {name}')
        seconds = time.perf_counter() - started
        # dev200 is a subset of dev: reuse those answers instead of generating them twice.
        by_id = {p['id']: p for p in predictions}
        subset = [by_id[r['id']] for r in read_jsonl(data / name / 'dev200.jsonl')]
        for split, rows_out in (('dev', predictions), ('dev200', subset)):
            write_jsonl(out / f'm0_{name}_{split}.jsonl', rows_out)
            summaries[f'{name}/{split}'] = {**summarize(rows_out), 'seconds': seconds if split == 'dev' else 0.0}
        print(f"[pilot] baseline {name}: accuracy {summaries[f'{name}/dev']['accuracy']:.3f} "
              f'in {seconds:.0f} s', flush=True)
    del model
    _free()
    return summaries


def _suggest(report, train_settings, lengths, batch_size, data):
    concerns, reasons = report['concerns'], {}
    precision = report['precision']
    training = report['training']
    ok = (precision['finite'] and precision['hidden_state_peak'] < FP16_MAX / 2
          and report['fp32_reference']['weights']['above_fp16_max'] == 0
          and all(training[k]['all_losses_finite'] for k in ('dora', 'protected_max_k'))
          and all(training[k]['skipped_updates'] <= 1 for k in ('dora', 'protected_max_k')))
    reasons['precision_validated'] = ('true only if FP16 logits are finite, hidden-state peak < 32752 (half the FP16 max), '
                                      'no weight exceeds FP16 range, and both training probes had finite losses with <= 1 '
                                      'skipped update (GradScaler warm-up). Argmax agreement is reported for your judgement.')
    if not ok:
        concerns.append('FP16 precision check failed: profile Qwen3-1.7B in FP32 before continuing')
    if precision['argmax_agreement'] < 0.98:
        concerns.append(f"FP16 vs FP32 argmax agreement is {precision['argmax_agreement']:.3f}; inspect before training")
    if not report['protection_cost']['q_v_bases_identical']:
        concerns.append('q/v protection bases differ; keep separate streams (the default)')
    baseline = report.get('baseline', {})
    for name in ('A', 'B'):
        score = baseline.get(f'{name}/dev', {}).get('accuracy')
        if score is not None and score >= 0.9:
            concerns.append(f'M0 already scores {score:.0%} on {name} dev: near ceiling, make the questions harder')
    merge = report['merge']
    merge_items = 400  # dev200 of the trained skill + dev200 of the retained/general set per merge
    settings = {
        'precision_validated': bool(ok),
        'eval_batch_size': int(batch_size),
        'dev_every_updates': None,
        'selection': {'retain_pp': -2.0, 'general_pp': -2.0},
        'merge': {'atol': _round_up(10 * max(merge['max_abs_error'], 1e-6)), 'rtol': 0.0,
                  'max_stored_residual': _round_up(4 * max(merge['max_stored_residual'], 1e-9)),
                  'max_intended_residual': 1e-5,
                  'max_changed_answers': MERGE_FLIPS, 'max_accuracy_shift_pp': MERGE_SHIFT_PP,
                  'probes': 8},
    }
    reasons['merge'] = ('atol = 10x the observed max |live - merged| logit error (a tail statistic; real merge bugs are '
                        'orders of magnitude larger), rtol = 0; max_stored_residual = 4x the '
                        'observed FP16 rounding residual at the cap k; max_changed_answers = answers whose correctness flips, at most 5% of '
                        'the 400 validation answers (batch composition alone flips about 3%), and max_accuracy_shift_pp '
                        '= 1.5 per skill, so a merge cannot move a score by more than the -2 pp selection margin; '
                        'reworded answers with unchanged correctness are reported, not gated; intended residual stays '
                        '1e-5 (exact-arithmetic property).')
    examples = train_settings['token_budget'] / (sum(lengths) / len(lengths))
    updates = min(examples, train_settings['epochs'] * _count(data / 'A' / 'train.jsonl')) / (
        train_settings['microbatch'] * train_settings['accumulation'])
    settings['dev_every_updates'] = max(1, int(round(updates / 6)))
    reasons['dev_every_updates'] = 'about six scored checkpoints per run, from the token budget and mean example length'
    reasons['eval_batch_size'] = (f'largest tested batch without OOM whose per-item correctness matches batch 1 on '
                                  f'>= {BATCH_AGREEMENT:.0%} of items; every model uses it with the same length-sorted '
                                  'batches, so comparisons stay paired')
    tokens_per_second = min(training['dora']['tokens_per_second'], training['protected_max_k']['tokens_per_second'])
    generation = report['generation'][batch_size]
    eval_seconds = 600 / generation['items_per_second']  # three dev200 sets per checkpoint
    run_hours = (min(train_settings['token_budget'], updates * train_settings['microbatch'] * train_settings['accumulation']
                     * sum(lengths) / len(lengths)) / tokens_per_second + 7 * eval_seconds) / 3600
    merge_hours = (4 * merge_items / generation['items_per_second'] + 600) / 3600
    report['budget'] = {'training_run_hours': run_hours, 'merge_hours': merge_hours,
                        'q_capture_hours': (report['protection_cost']['projected_capture_seconds']
                                            + report['protection_cost']['eigensolver_seconds']) / 3600,
                        'study_gpu_hours_3_seeds_4_methods_1_joint': 3 * (run_hours + merge_hours) * 5 + 2 * (run_hours + merge_hours),
                        'note': 'device-hours from measured throughput (worst of ordinary/protected); add contingency'}
    return settings, reasons


def _count(path):
    return sum(1 for line in open(path, encoding='utf-8') if line.strip())


def _markdown(report) -> str:
    p, t, m, g = report['precision'], report['training'], report['merge'], report['generation']
    s = report['suggested_settings']
    lines = ['# GPU pilot report', '',
             f"**precision_validated: {s['precision_validated']}**", '']
    if report['concerns']:
        lines += ['## Concerns', ''] + [f'- {c}' for c in report['concerns']] + ['']
    lines += ['## FP16 vs FP32 reference (teacher-forced probes)', '',
              f"- finite: {p['finite']}; max |Δlogit| {p['max_abs_error']:.4g}; mean {p['mean_abs_error']:.4g}",
              f"- argmax agreement {p['argmax_agreement']:.4f}; max answer-loss difference {p['max_answer_loss_difference']:.4g}",
              f"- hidden-state peak {p['hidden_state_peak']:.4g} (FP16 max 65504)", '',
              '## Training probes', '', '| Variant | s/update | tokens/s | peak alloc GB | skipped | loss first→last |',
              '|---|---|---|---|---|---|']
    for key in ('dora', 'protected_max_k'):
        r = t[key]
        peak = r['peak_allocated_vram_bytes']
        lines.append(f"| {key} | {r['seconds_per_update']:.2f} | {r['tokens_per_second']:.0f} | "
                     f"{peak / 2**30 if peak else float('nan'):.2f} | {r['skipped_updates']} | "
                     f"{r['loss_first']:.3f}→{r['loss_last']:.3f} |")
    lines += ['', f"Protection overhead (cap k={t['protected_max_k']['k']}): {t['protection_overhead']:.2f}x", '',
              '## Generation', '', '| batch | items/s | mean tokens | hit limit | accuracy | same text / correctness as batch 1 |',
              '|---|---|---|---|---|---|']
    lines += [f"| {b} | OOM | | | | |" if r.get('oom') else
              f"| {b} | {r['items_per_second']:.2f} | {r['mean_generated_tokens']:.0f} | {r['hit_token_limit']} | "
              f"{r['accuracy']:.3f} | {r['same_text_as_batch_1']:.2f} / {r['same_correctness_as_batch_1']:.2f} |"
              for b, r in g.items()]
    lines += ['', '## Merge (protected probe, FP16 storage)', '',
              f"- max |live − merged| logit {m['max_abs_error']:.4g}; changed answers {m['changed_answers']}/{m['items']}",
              f"- intended residual {m['max_intended_residual']:.3g}; stored residual {m['max_stored_residual']:.3g}", '',
              '## Protection cost', '', f"- {report['protection_cost']}", '']
    if 'baseline' in report:
        lines += ['## M0 development baseline', '', '| set | accuracy | format failures |', '|---|---|---|']
        lines += [f"| {k} | {v['accuracy']:.3f} | {v['format_failures']} |" for k, v in report['baseline'].items()]
        lines.append('')
    lines += ['## Budget estimate', '', f"- {report['budget']}", '', '## Suggested settings (study.yaml)', '',
              '```yaml', yaml.safe_dump(s, sort_keys=False).strip(), '```', '', '## Rules', '']
    lines += [f'- **{k}**: {v}' for k, v in report['rules'].items()]
    return '\n'.join(lines) + '\n'
