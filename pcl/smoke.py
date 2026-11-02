"""Offline engineering milestone: train, resume, protect, export, merge, reload."""

import copy
from pathlib import Path

import torch

from .activations import ActivationCollector
from .data.generator import generate
from .data.grader import grade_answer
from .io import write_json, write_jsonl
from .hashing import model_hashes
from .merge import compare_logits, export_adapters, merge_export
from .protected_dora import attach_adapters
from .protection import build_basis
from .toy import TinyCausalLM, toy_records
from .train import TrainConfig, Trainer, seed_everything


def run_smoke(output) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    seed_everything(7)
    questions = generate('A', {'train': 20, 'dev': 10}) + generate('B', {'train': 20, 'dev': 10})
    assert all(grade_answer(row['response'], row['reference']).correct for row in questions)
    write_jsonl(output / 'questions.jsonl', questions)
    plain = TinyCausalLM()
    base = copy.deepcopy(plain)
    attach_adapters(plain, rank=3, alpha=6)
    initial = copy.deepcopy(plain)
    records = toy_records()
    config = TrainConfig(epochs=2, microbatch=2, accumulation=2, learning_rate=0.01)
    provenance = {'base_tensor_hashes': model_hashes(base), 'dataset': 'synthetic-token-fixture-v1'}
    seed_everything(19)
    full = Trainer(plain, records, config, pad_id=0, provenance=provenance)
    full.run()
    seed_everything(19)
    interrupted = Trainer(copy.deepcopy(initial), records, config, pad_id=0, provenance=provenance)
    interrupted.run(stop_after_updates=2, checkpoint_path=output / 'interrupted.pt')
    resumed = Trainer(initial, records, config, pad_id=0, provenance=provenance)
    resumed.resume(output / 'interrupted.pt')
    resumed.run(checkpoint_path=output / 'stage_a.pt')
    assert model_hashes(full.model) == model_hashes(resumed.model)
    full.model.eval()
    ids = torch.tensor([records[0]['input_ids']])
    with torch.no_grad():
        live_logits = full.model(ids).logits
    exported = export_adapters(full.model)
    torch.save(exported, output / 'adapter_a.pt')
    merge_diagnostics = merge_export(base, exported)
    base.eval()
    with torch.no_grad():
        merged_logits = base(ids).logits
    merge_check = compare_logits(live_logits, merged_logits, atol=1e-6, rtol=1e-5)
    assert merge_check['passed']
    torch.save(base.state_dict(), output / 'merged_a.pt')
    reloaded = TinyCausalLM()
    reloaded.load_state_dict(torch.load(output / 'merged_a.pt', weights_only=True))
    reloaded.eval()
    assert model_hashes(base) == model_hashes(reloaded)
    with torch.no_grad():
        assert torch.equal(merged_logits, reloaded(ids).logits)
    # Collect actual merged-model layer inputs, including both prompt/answer positions.
    with ActivationCollector({'q_proj': reloaded.q_proj, 'v_proj': reloaded.v_proj}) as capture:
        capture.select(torch.ones_like(ids, dtype=torch.bool))
        reloaded(ids)
    bases = {name: build_basis(moment, max_k=4)[0] for name, moment in capture.moments.items()}
    protected_base = copy.deepcopy(reloaded)
    attach_adapters(reloaded, rank=3, alpha=6, bases=bases)
    stage_b = Trainer(reloaded, records, config, pad_id=0, provenance={'base': model_hashes(protected_base)})
    stage_b.run(checkpoint_path=output / 'stage_b.pt')
    reloaded.eval()
    protected_export = export_adapters(reloaded)
    diagnostics = merge_export(protected_base, protected_export)
    protected_base.eval()
    with torch.no_grad():
        protected_check = compare_logits(reloaded(ids).logits, protected_base(ids).logits, atol=1e-6, rtol=1e-5)
    assert protected_check['passed']
    assert all(row['intended_residual'] < 1e-5 for row in diagnostics.values())
    result = {'passed': True, 'adapter': 'dora', 'device': 'cpu',
              'scope': 'synthetic tiny-model engineering check; not an aerospace learning result',
              'generated_questions': len(questions), 'resume_identical': True,
              'reload_tensor_hashes_identical': True, 'merge': merge_check,
              'stage_a_merge_diagnostics': merge_diagnostics,
              'protected_merge': protected_check, 'protected_diagnostics': diagnostics,
              'training': full.state, 'loss_start': full.history[0]['loss'], 'loss_end': full.history[-1]['loss']}
    write_json(output / 'smoke_report.json', result)
    return result
