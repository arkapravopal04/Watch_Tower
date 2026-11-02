"""The follow-up round in miniature, on CPU, on top of a miniature main study.

A tiny random Qwen3 stands in for the 4B model, so scores mean nothing; the
point is that discovery, the spectrum, the replication check, every arm (with
update-0 reuse, sparse scoring, selection and session resume) and the report
all run end to end against the same artifacts the Kaggle notebooks produce.
"""

import os
from pathlib import Path

import pytest
import torch
import yaml

pytest.importorskip('transformers')

from pcl import kaggle  # noqa: E402
from pcl.cli import invoke, snapshot_manifest  # noqa: E402
from pcl.io import read_json  # noqa: E402
from pcl.pipeline import stage_a, stage_b  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


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


@pytest.mark.slow
def test_follow_up_round_in_miniature(tmp_path, monkeypatch):
    from pclx.cli import invoke as next_invoke
    from pclx.plan import discover_next, pending_items, split
    from pclx.runs import train_next

    # A miniature main study: data, pilot, Stage A (M_A + Q) and the protected/sequential Stage-B baselines.
    model = tiny_snapshot(tmp_path)
    config = yaml.safe_load((REPO / 'configs/model.yaml').read_text())
    config.update(path=str(model), snapshot_manifest=str(model / 'snapshot_manifest.json'), precision='fp32',
                  device='cpu', backend='eager', max_context=512, max_new_tokens=4)
    m0 = tmp_path / 'm0.yaml'
    m0.write_text(yaml.safe_dump(config))
    train = yaml.safe_load((REPO / 'configs/train.yaml').read_text())
    train.update(fp16=False, epochs=1, accumulation=4, learning_rate=0.01, token_budget=6250)  # fixed: independent of the config copy
    train_config = tmp_path / 'train.yaml'
    train_config.write_text(yaml.safe_dump(train))
    data_config = tmp_path / 'data.yaml'
    data_config.write_text(yaml.safe_dump({'seed': 0, 'skills': {
        'A': {'train': 24, 'dev': 12, 'memory': 12, 'dev_subset': 6, 'test': 6, 'test_hard': 4},
        'B': {'train': 24, 'dev': 12, 'memory': 12, 'dev_subset': 6, 'test': 6, 'test_hard': 4},
        'general': {'train': 0, 'dev': 9, 'memory': 0, 'dev_subset': 6, 'test': 6, 'test_hard': 0}}}))
    inputs = tmp_path / 'input'
    data = inputs / 'nb2' / 'data'
    assert kaggle.make_banks(data, tokenizer=model, data_config=data_config, audit_sample=5)['audit_passed']
    invoke(['pilot', '--model-config', m0, '--data-root', data, '--out', inputs / 'nb2' / 'pilot', '--train-config',
            train_config, '--probes', 3, '--train-updates', 2, '--eval-items', 6, '--batch-sizes', 1, 2,
            '--merge-items', 6, '--capture-tokens', 256])
    settings = kaggle.study_settings(inputs / 'nb2' / 'pilot', tmp_path / 'study.yaml')
    seed0 = inputs / 'nb3' / 'stage_a' / 'seed0'
    assert stage_a(model_config=m0, settings=settings, train_config=train_config, data_root=data, seed=0,
                   out=seed0)['status'] == 'complete'
    assert stage_b(model_config=m0, settings=settings, train_config=train_config, data_root=data, stage_a_dir=seed0,
                   seed=0, methods=['sequential', 'protected'], out=inputs / 'nb4' / 'stage_b' / 'seed0' / 'gpu0',
                   )['status'] == 'complete'

    # The follow-up schedule, shrunk to the miniature run (configs/next.yaml has the real one).
    next_config = yaml.safe_load((REPO / 'notebooks_2/configs/next.yaml').read_text())
    next_config.update(score_at=[2], min_item_seconds=0, replication_updates=2,
                       gradient_checkpointing='auto')
    path = tmp_path / 'next.yaml'
    path.write_text(yaml.safe_dump(next_config))
    monkeypatch.setenv('PCLX_NEXT_CONFIG', str(path))
    working = tmp_path / 'working'

    # NB7: spectrum + SVD bases + the k<=256 basis, then the go/no-go report.
    from pclx.plan import balance, nb7_tasks
    tasks = nb7_tasks(discover_next([inputs, working]))
    assert tasks == ['spectrum:0', 'k256:0'] and balance(tasks) == [['k256:0'], ['spectrum:0']]
    assert balance(['spectrum:0', 'spectrum:1', 'spectrum:2', 'k256:0', 'k256:1', 'k256:2']) == [
        ['k256:0', 'k256:2', 'spectrum:2'], ['k256:1', 'spectrum:0', 'spectrum:1']]  # 39 vs 33 minutes
    for queue in balance(tasks):  # the two "GPUs", one after the other
        next_invoke(['spectrum', '--search', inputs, '--tasks', *queue, '--model-config', m0, '--data-root', data,
                     '--train-config', train_config, '--out', working / 'spectrum'])
    spectrum = read_json(working / 'spectrum' / 'seed0' / 'spectrum.json')
    assert spectrum['summary']['layers'] == 4 and spectrum['gradient']['examples'] == 4  # exactly the first update
    replay_bases = torch.load(working / 'spectrum' / 'seed0' / 'svd_bases_replay.pt', weights_only=True)
    assert any(i.startswith('A-') for i in replay_bases['example_ids'])  # the replay arms' first update mixes in A
    assert read_json(working / 'spectrum' / 'seed0' / 'k256_summary.json')['status'] == 'complete'
    assert nb7_tasks(discover_next([inputs, working])) == []
    decision = next_invoke(['spectrum-report', '--search', inputs, working, '--out', working / 'spectrum'])
    assert set(decision) == {'run_svd_arms', 'per_seed', 'rule'}

    # NB8: every pending item (replication first), split over two "GPUs", each queue in turn.
    found = discover_next([inputs, working])
    items = pending_items(found, tiers=(1, 2, 3), run_svd_arms=True)
    assert items[0] == 'replicate:0' and len(items) == 1 + 5
    for queue_items in split(items):
        results = next_invoke(['queue', '--search', inputs, working, '--items', *queue_items, '--model-config', m0,
                               '--settings', settings, '--data-root', data, '--train-config', train_config,
                               '--out', working / 'next'])
        assert all(r['status'] == 'complete' for r in results), results
    replication = read_json(working / 'next' / 'replication' / 'replication.json')
    assert replication['passed'], replication  # this driver reproduces pcl's training exactly
    for name in ('protected_svd', 'protected_replay', 'protected_replay_svd', 'sequential_svd', 'protected_k256'):
        run = read_json(working / 'next' / name / 'seed0' / 'train' / 'run_manifest.json')
        assert run['complete'] and run['next']['update0_reused_from']  # update 0 copied, never rescored
        assert [e['update'] for e in run['evaluations']][0] == 0 and len(run['evaluations']) == 3  # 0, 2, final
        first = run['history'][0]['gradient_norms']
        assert first['A'] == 0 and first['B'] > 0  # Lean's gradA_at_zero, in training
        assert (working / 'next' / name / 'seed0' / 'selection.json').is_file()
    svd = read_json(working / 'next' / 'protected_svd' / 'seed0' / 'train' / 'run_manifest.json')
    # The SVD rows lie in Q's complement except along zero singular values (the tiny model has only 16 input
    # dimensions, so k + rank can exceed them); the forward pass projects those parts away (Lean: truncate_mul_eq_zero).
    for name, layer in spectrum['layers'].items():
        if layer['zero_singular_in_basis'] == 0:
            assert svd['next']['init_report'][name]['leak_into_q'] < 1e-5, name
    assert pending_items(discover_next([inputs, working])) == []  # nothing left to do

    # A session that ends mid-run resumes to exactly the uninterrupted result.
    common = dict(model_config=seed0 / 'model_a.yaml', train_config=train_config, data=data / 'B' / 'train.jsonl',
                  arm='protected_svd', seed=0, score_at=[2], protection=seed0 / 'protection_a.pt',
                  svd_bases=working / 'spectrum' / 'seed0' / 'svd_bases.pt', eval_batch_size=2,
                  dev={'B': data / 'B' / 'dev200.jsonl', 'A': data / 'A' / 'dev200.jsonl',
                       'general': data / 'general' / 'dev200.jsonl'})
    write_model = kaggle.write_model_config(m0, seed0 / 'model_a.yaml', patch=seed0 / 'merged_a')
    common['model_config'] = write_model
    first = train_next(out=tmp_path / 'resume1', stop_after_updates=1, **common)
    assert not first['complete']
    train_next(out=tmp_path / 'resume2', resume=tmp_path / 'resume1' / 'resume.pt', **common)
    whole = read_json(working / 'next' / 'protected_svd' / 'seed0' / 'train' / 'run_manifest.json')
    resumed = read_json(tmp_path / 'resume2' / 'run_manifest.json')
    assert [h['loss'] for h in resumed['history']] == [h['loss'] for h in whole['history']]
    # So does one that ends past a scoring point, even if the driver was fixed in between (the saved hashes differ).
    past = {**common, 'score_at': [1]}
    train_next(out=tmp_path / 'resume3', stop_after_updates=2, **past)
    saved = torch.load(tmp_path / 'resume3' / 'resume.pt', weights_only=True)
    assert saved['state']['attempted_updates'] == 2
    saved['provenance']['next_source_hashes'] = {'runs.py': 'driver before the fix'}
    torch.save(saved, tmp_path / 'resume3' / 'resume.pt')
    assert train_next(out=tmp_path / 'resume4', resume=tmp_path / 'resume3' / 'resume.pt', **past)['complete']
    resumed = read_json(tmp_path / 'resume4' / 'run_manifest.json')
    assert [h['loss'] for h in resumed['history']] == [h['loss'] for h in whole['history']]
    assert [e['update'] for e in resumed['evaluations']] == [0, 1, whole['evaluations'][-1]['update']]
    # Bases built from another first update (B-only G for the replay order) are refused, not silently used.
    with pytest.raises(ValueError, match="first update"):
        train_next(out=tmp_path / 'wrong_bases', **{**common, 'arm': 'protected_replay_svd',
                                                    'memory': data / 'A' / 'memory.jsonl'})

    # NB9: the report, with every arm paired against its baseline.
    result = next_invoke(['report', '--search', inputs, working, '--out', working / 'report'])
    text = (working / 'report' / 'report.md').read_text(encoding='utf-8')
    assert result['runs'] == 7 and 'protected_svd vs protected' in text and 'Replication check' in text
    assert (working / 'report' / 'early_training_loss.png').is_file()
    assert (working / 'report' / 'learning_vs_retention.png').is_file()
    assert (working / 'report' / 'gradient_concentration.png').is_file()
    assert not os.environ.get('PCLX_NEXT_CONFIG', '').endswith('configs/next.yaml')
