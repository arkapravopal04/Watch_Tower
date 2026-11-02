"""Refusal paths of the verified merge (the success path runs in test_end_to_end)."""

import copy
from pathlib import Path

import pytest
import torch

from pcl.checkpoints import load_checkpoint
from pcl.io import file_hash, read_json, write_json
from pcl.merge_workflow import verified_merge
from pcl.model_setup import load_model, verify_patch, verify_snapshot
from pcl.protected_dora import attach_adapters
from pcl.train import Trainer, TrainConfig
from pcl.toy import toy_records


@pytest.fixture
def tiny_study(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(123)
    vocabulary = {'<pad>': 0, '<eos>': 1, '<unk>': 2, 'system': 3, 'user': 4, 'assistant': 5,
                  'short': 6, 'FINAL': 7, ':': 8, 'q': 9, '=': 10, '1': 11, 'Pa': 12}
    vocabulary.update({f'word{i}': i for i in range(13, 32)})
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
    tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] }} {{ message['content'] }} <eos> "
                               "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}")
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
                        attn_implementation='eager')
    model = Qwen3ForCausalLM(config).eval()
    source = tmp_path / 'base'
    model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    snapshot = {'revision': 'a' * 40, 'files': {p.name: file_hash(p) for p in source.iterdir()}}
    write_json(source / 'snapshot_manifest.json', snapshot)
    settings = {'path': str(source), 'snapshot_manifest': str(source / 'snapshot_manifest.json'),
                'device': 'cpu', 'precision': 'fp32', 'backend': 'eager',
                'max_length': 128, 'max_context': 128, 'max_new_tokens': 3}
    live = copy.deepcopy(model)
    attach_adapters(live, rank=2)
    trainer = Trainer(live, toy_records(4), TrainConfig(epochs=1, accumulation=2), pad_id=0, provenance={'test': 'workflow'})
    checkpoint = tmp_path / 'selected.pt'
    trainer.run(checkpoint_path=checkpoint)
    records = [{'id': 'dev-1', 'group': 'g1', 'split': 'dev', 'skill': 'A', 'family': 'mach',
                'prompt': 'short', 'response': 'FINAL: q=1 Pa',
                'reference': {'q': {'value': 1, 'unit': 'Pa', 'atol': 0.001}}}]
    return model, tokenizer, snapshot, settings, checkpoint, records


def run_merge(study, out, **overrides):
    model, tokenizer, snapshot, config, path, records = study
    options = dict(checkpoint_hash=file_hash(path), data_hashes={'dev': 'fixture'}, out=out,
                   atol=1e-5, rtol=1e-5, max_stored_residual=1e-5)
    options.update(overrides)
    return verified_merge(model, tokenizer, snapshot, config, load_checkpoint(path), records, **options)


def test_published_patch_refuses_wrong_precision_pending_status_and_other_base(tiny_study, tmp_path):
    out = tmp_path / 'merged'
    source = Path(tiny_study[3]['path'])
    assert run_merge(tiny_study, out)['published']
    base_manifest = verify_snapshot(source, source / 'snapshot_manifest.json')
    with pytest.raises(ValueError, match='verified in fp32'):
        load_model(source, manifest_path=source / 'snapshot_manifest.json', precision='fp16', device='cuda', patch=out)
    pending = read_json(out / 'patch_manifest.json')
    pending['merge_status'] = 'pending'
    write_json(out / 'patch_manifest.json', pending)
    with pytest.raises(ValueError, match='unverified'):
        verify_patch(out, base_manifest)
    with pytest.raises(ValueError, match='different base'):
        verify_patch(out, {**base_manifest, 'revision': 'b' * 40}, allow_unverified=True)


def test_failed_validation_does_not_publish(tiny_study, tmp_path, monkeypatch):
    import pcl.merge_workflow as workflow
    original = workflow._compare

    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        result['passed'] = False
        return result
    monkeypatch.setattr(workflow, '_compare', fail)
    result = run_merge(tiny_study, tmp_path / 'rejected')
    assert not result['passed'] and not result['published']
    assert not (tmp_path / 'rejected').exists()
    assert read_json(tmp_path / 'rejected.merge_report.json')['passed'] is False


def test_overwriting_or_nesting_in_the_input_is_rejected(tiny_study):
    source = Path(tiny_study[3]['path'])
    with pytest.raises(FileExistsError):
        run_merge(tiny_study, source)
    with pytest.raises(ValueError, match='outside the input'):
        run_merge(tiny_study, source / 'nested')


def test_merge_gate_counts_correctness_flips_and_accuracy_shift():
    from pcl.merge_workflow import _compare

    def measured(rows):
        return {'logits': torch.zeros(2, 3), 'predictions': [
            {'id': str(i), 'skill': skill, 'group': str(i), 'prediction': text, 'correct': correct,
             'format_ok': True} for i, (skill, text, correct) in enumerate(rows)]}
    live = measured([('A', 'x', True), ('A', 'y', False), ('B', 'z', True), ('B', 'w', True)])
    reworded = measured([('A', 'x2', True), ('A', 'y2', False), ('B', 'z', True), ('B', 'w', True)])
    result = _compare(live, reworded, atol=0, rtol=0, max_changed_answers=0, max_accuracy_shift_pp=0)
    assert result['passed'] and result['changed_answers'] == [] and result['reworded_answers'] == ['0', '1']
    flipped = measured([('A', 'x', False), ('A', 'y', False), ('B', 'z', True), ('B', 'w', True)])
    assert not _compare(live, flipped, atol=0, rtol=0, max_changed_answers=0)['passed']
    shifted = _compare(live, flipped, atol=0, rtol=0, max_changed_answers=1, max_accuracy_shift_pp=10)
    assert shifted['accuracy_shift_pp'] == {'A': -50.0, 'B': 0.0} and not shifted['passed']
