"""The whole Kaggle study in miniature: every notebook's commands, in order, on CPU.

A tiny random Qwen3 and word-level tokenizer stand in for the pinned 4B model,
so scores are meaningless; the point is that every artifact, hash check,
selection rule, registry guard and resume path works end to end.
Skip with `pytest -m "not slow"`.
"""

import json
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip('transformers')

from pcl import kaggle  # noqa: E402
from pcl.cli import invoke, snapshot_manifest  # noqa: E402
from pcl.io import read_json, read_jsonl, write_json  # noqa: E402
from pcl.pipeline import stage_a, stage_b, stage_joint  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
METHODS = ['sequential', 'protected', 'random_q', 'replay', 'protected_replay']


def tiny_snapshot(root: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    words = ['<pad>', '<eos>', '<unk>', 'system', 'user', 'assistant', 'FINAL', ':', '=', ';', '.', 'M', 'q', 'L',
             'D', 'V', 'status', 'missing', 'applicable', 'not_applicable', 'missing_input', 'CL', 'Pa', 'N', 'm', '/', 's']
    vocabulary = {word: index for index, word in enumerate(words)}
    vocabulary.update({str(digit): len(words) + digit for digit in range(10)})
    vocabulary.update({f'w{index}': index for index in range(len(vocabulary), 64)})
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
    tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] }} {{ message['content'] }} <eos> "
                               "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}")
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=1024,
                         pad_token_id=0, eos_token_id=1, tie_word_embeddings=True)
    path = root / 'model'
    Qwen3ForCausalLM(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    snapshot_manifest(path, 'f50518eb58dfc750271b273fc113bdfc16ec2280')
    return path


def write_configs(root: Path, model: Path) -> dict:
    config = yaml.safe_load((REPO / 'configs/model.yaml').read_text())
    config.update(path=str(model), snapshot_manifest=str(model / 'snapshot_manifest.json'), precision='fp32',
                  device='cpu', backend='eager', max_context=512, max_new_tokens=4)
    paths = {'model': root / 'm0.yaml'}
    paths['model'].write_text(yaml.safe_dump(config))
    train = yaml.safe_load((REPO / 'configs/train.yaml').read_text())
    train.update(fp16=False, epochs=1, accumulation=4, learning_rate=0.01, token_budget=train['token_budget'] // 160)
    paths['train'] = root / 'train.yaml'
    paths['train'].write_text(yaml.safe_dump(train))
    return paths


@pytest.mark.slow
def test_full_study_in_miniature(tmp_path):
    model = tiny_snapshot(tmp_path)
    configs = write_configs(tmp_path, model)
    data = tmp_path / 'data'
    data_config = tmp_path / 'data.yaml'
    data_config.write_text(yaml.safe_dump({'seed': 0, 'skills': {
        'A': {'train': 24, 'dev': 12, 'memory': 12, 'dev_subset': 6, 'test': 6, 'test_hard': 4},
        'B': {'train': 24, 'dev': 12, 'memory': 12, 'dev_subset': 6, 'test': 6, 'test_hard': 4},
        'general': {'train': 0, 'dev': 9, 'memory': 0, 'dev_subset': 6, 'test': 6, 'test_hard': 0}}}))

    # NB2: development banks with measured token lengths, then the independent audit.
    banks = kaggle.make_banks(data, tokenizer=model, data_config=data_config, audit_sample=5)
    assert banks['audit_passed']
    audit = read_json(data / 'audit' / 'audit.json')
    assert audit['token_length']['max'] <= 512
    assert (data / 'audit' / 'review_sheet.md').read_text().count('- [ ] looks right') == 15

    # NB2: pilot -> study.yaml with every merge limit filled in, plus the M0 baseline.
    pilot = tmp_path / 'pilot'
    invoke(['pilot', '--model-config', configs['model'], '--data-root', data, '--out', pilot,
            '--train-config', configs['train'], '--probes', 3, '--train-updates', 2, '--eval-items', 6,
            '--batch-sizes', 1, 2, '--merge-items', 6, '--capture-tokens', 256])
    settings = yaml.safe_load((pilot / 'study.yaml').read_text())
    assert settings['precision_validated'] is True
    assert all(settings['merge'][key] is not None for key in ('atol', 'rtol', 'max_stored_residual'))
    report = read_json(pilot / 'pilot_report.json')
    assert report['precision']['max_abs_error'] == 0  # FP32 vs FP32 here; FP16 on Kaggle
    assert report['protection_cost']['q_v_bases_identical']
    assert (pilot / 'baseline' / 'm0_A_dev.jsonl').is_file() and (pilot / 'm0_hashes.json').is_file()
    settings_path = pilot / 'study.yaml'

    # NB3, session 1: Stage A stops after two updates (as if the Kaggle session ended) ...
    first = tmp_path / 'stage_a_session1' / 'seed0'
    invoke(['train', '--model-config', configs['model'], '--train-config', configs['train'],
            '--data', data / 'A' / 'train.jsonl', '--out', first / 'train', '--seed', 0, '--stop-after-updates', 2,
            '--checkpoint-updates', settings['dev_every_updates'], '--eval-batch-size', settings['eval_batch_size'],
            '--dev', f"A={data / 'A' / 'dev200.jsonl'}", '--dev', f"general={data / 'general' / 'dev200.jsonl'}"])
    assert not read_json(first / 'train' / 'run_manifest.json')['complete']
    # ... and session 2 resumes it, selects on dev, merges into a verified patch and builds Q.
    seed0 = tmp_path / 'stage_a_session2' / 'seed0'
    result = stage_a(model_config=configs['model'], settings=settings_path, train_config=configs['train'],
                     data_root=data, seed=0, out=seed0, previous=[first])
    assert result['status'] == 'complete' and result['merge_passed']
    run = read_json(seed0 / 'merged_a' / 'run_manifest.json')
    assert run['state']['resume_count'] == 1 and run['complete']
    selection = read_json(seed0 / 'selection.json')
    assert selection['update'] > 0 and all(row['constraints']['general']['margin_pp'] == -2.0 for row in selection['table'])
    assert (seed0 / 'references' / 'm0_general_dev200.jsonl').is_file()
    protection = torch.load(seed0 / 'protection_a.pt', weights_only=True)
    assert len(protection['bases']) == 4

    # NB4: every Stage-B method from the same verified M_A.
    stage_b_out = tmp_path / 'stage_b' / 'seed0'
    result = stage_b(model_config=configs['model'], settings=settings_path, train_config=configs['train'],
                     data_root=data, stage_a_dir=seed0, seed=0, methods=METHODS, out=stage_b_out)
    assert result['status'] == 'complete', result
    for method in METHODS:
        manifest = read_json(stage_b_out / method / 'merged' / 'patch_manifest.json')
        assert len(manifest['merge_chain']) == 2  # M0 -> M_A -> candidate, stored as one cumulative patch
    effective = torch.load(stage_b_out / 'protected' / 'merged' / 'effective_adapter.pt', weights_only=True)
    for name, layer in effective['layers'].items():
        assert torch.equal(layer['Q'], protection['bases'][name])
        assert (layer['A_eff'] @ layer['Q']).abs().max() < 1e-5
    tokens = {m: read_json(stage_b_out / m / 'merged' / 'run_manifest.json')['state']['tokens_by_skill'] for m in METHODS}
    assert set(tokens['sequential']) == {'B'} and set(tokens['replay']) == {'A', 'B'}
    # Replay's total matches its paired run to within one whole example.
    longest = max(r['token_length'] for r in read_jsonl(data / 'A' / 'train.jsonl') + read_jsonl(data / 'B' / 'train.jsonl'))
    assert abs(sum(tokens['replay'].values()) - sum(tokens['sequential'].values())) <= longest
    # A rerun after a session boundary adopts finished methods instead of retraining them.
    rerun = stage_b(model_config=configs['model'], settings=settings_path, train_config=configs['train'],
                    data_root=data, stage_a_dir=seed0, seed=0, methods=['sequential'], out=tmp_path / 'stage_b_again',
                    previous=[stage_b_out])
    assert rerun['methods']['sequential']['from_previous_session']

    # Joint A+B from M0.
    joint = tmp_path / 'joint' / 'seed0'
    assert stage_joint(model_config=configs['model'], settings=settings_path, train_config=configs['train'],
                       data_root=data, seed=0, out=joint)['status'] == 'complete'
    budget = yaml.safe_load(configs['train'].read_text())['token_budget']
    assert read_json(joint / 'merged' / 'run_manifest.json')['train_config']['token_budget'] == 2 * budget

    # NB5: seal locked banks that reproduce the development files, then register everything found.
    tests = tmp_path / 'tests'
    sealed = kaggle.seal_banks(data, tests, tokenizer=model)
    assert sealed['audit_passed'] and set(sealed['test_files']) == {'A', 'A_hard', 'B', 'B_hard', 'general'}
    with pytest.raises(ValueError, match='reproduce'):
        invoke(['generate', '--skill', 'A', '--train', 23, '--dev', 12, '--test', 6, '--seal-test',
                '--out', tmp_path / 'wrong', '--expect-manifest', data / 'A' / 'split_manifest.json'])
    found = kaggle.discover(tmp_path)
    assert kaggle.one(found, 'models', 'model') == model and kaggle.stage_a_for(found, 0) == seed0
    spec = tmp_path / 'freeze' / 'study_spec.yaml'
    artifacts = kaggle.study_spec(found, spec, primary='protected_s0')['artifacts']
    assert set(artifacts) == {'M0', 'MA_s0', 'joint_s0', *(f'{m}_s0' for m in METHODS)}
    test_files = sealed['test_files']
    registry = tmp_path / 'freeze' / 'registry'
    invoke(['register', '--spec', spec, '--registry', registry])
    with pytest.raises(FileExistsError):
        invoke(['register', '--spec', spec, '--registry', registry])

    # NB6: locked scoring, split over two "GPUs", then a rerun that must not rescore anything.
    ids = sorted(artifacts)
    for half in (ids[::2], ids[1::2]):
        result = invoke(['evaluate-locked', '--registry', registry, '--artifact-id', *half, '--model-config',
                         configs['model'], '--test-root', tests, '--search', tmp_path])
        assert result['complete'], result
    again = invoke(['evaluate-locked', '--registry', registry, '--artifact-id', 'protected_s0', '--model-config',
                    configs['model'], '--test-root', tests, '--search', tmp_path])
    assert again['results']['protected_s0'] == {'already_completed': sorted(test_files)}
    assert len(list((registry / 'locked').glob('*.json'))) == len(ids) * len(test_files)

    # NB6: gates on the preregistered candidate, then the full report.
    smoke = invoke(['smoke', '--out', tmp_path / 'smoke'])
    assert smoke['resume_identical']
    decision = invoke(['decide', '--registry', registry, '--smoke-report', tmp_path / 'smoke' / 'smoke_report.json',
                       '--search', tmp_path])
    assert decision['numerical_checks'] == {key: True for key in decision['numerical_checks']}
    assert decision['accepted'] == all(decision['gates'].values())
    assert (registry / 'accepted.json').exists() == decision['accepted']
    invoke(['report', '--registry', registry, '--out', tmp_path / 'report'])
    text = (tmp_path / 'report' / 'report.md').read_text(encoding='utf-8')
    assert 'Locked study report' in text and 'protected-vs-sequential/seed0/A' in text
    assert json.loads((tmp_path / 'report' / 'report.json').read_text())['decision']['artifact_id'] == 'protected_s0'
    # The registry moves between notebooks: copy it and the report still verifies every prediction hash.
    import shutil
    moved = tmp_path / 'moved' / 'registry'
    shutil.copytree(registry, moved)
    assert kaggle.latest_registry(kaggle.discover(tmp_path)) in (registry, moved)
    invoke(['report', '--registry', moved, '--out', tmp_path / 'report_moved'])
    write_json(tmp_path / 'done.json', {'ok': True})
