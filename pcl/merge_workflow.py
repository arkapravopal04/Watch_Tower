"""Review a selected checkpoint, then optionally publish a verified weight patch.

The input snapshot/checkpoint are read-only. A merge is never acceptance. The
output directory appears only after live/export/merged/reloaded validation.
"""

from pathlib import Path
import math
import shutil
import tempfile

import torch
from safetensors.torch import save_file
from torch import nn

from .checkpoints import restore_adapters
from .evaluate import evaluate, summarize
from .hashing import model_hashes
from .io import file_hash, write_json, write_jsonl
from .merge import compare_logits, export_adapters, merge_export, preview_merge
from .model_setup import PATCH_FORMAT, PATCH_MANIFEST, PATCH_WEIGHTS, load_model, snapshot_identity
from .protected_dora import adapter_modules
from .registry import identity
from .tokenization import encode_record


class ExportedDoRALinear(nn.Module):
    """Read-only effective export path, independent of raw-factor normalization."""
    def __init__(self, base, factors):
        super().__init__()
        self.base = base
        for key in ('A_eff', 'B', 'Q', 'row_gain'):
            self.register_buffer(key, factors[key].to(base.weight.device))
        self.scale = factors['scale']

    def forward(self, x):
        output = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            residual = xf - (xf @ self.Q) @ self.Q.T if self.Q.shape[1] else xf
            magnitude_delta = (self.row_gain - 1) * torch.nn.functional.linear(residual, self.base.weight.float())
            direction_delta = self.row_gain * ((xf @ self.A_eff.T) @ self.B.T) * self.scale
        return output + (magnitude_delta + direction_delta).to(output.dtype)


def _replace(model, name, module):
    parent, _, leaf = name.rpartition('.')
    setattr(model.get_submodule(parent) if parent else model, leaf, module)


@torch.inference_mode()
def _measure(model, tokenizer, records, config, probes, batch_size):
    model.eval()
    device = next(model.parameters()).device
    logits, positions = [], {}
    # Bound host storage to eight vocabulary vectors per probe, while covering
    # prompt and checked-answer positions rather than only the final prompt token.
    for record in records[:probes]:
        encoded = encode_record(tokenizer, record, config['max_length'])
        ids = torch.tensor([encoded['input_ids']], device=device)
        indices = torch.linspace(0, ids.shape[1] - 1, min(8, ids.shape[1]), device=device).long()
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        logits.append(output[0, indices].float().cpu())
        positions[record['id']] = indices.tolist()
    predictions = evaluate(model, tokenizer, records, batch_size=batch_size,
                           max_new_tokens=config['max_new_tokens'], max_context=config['max_context'], label='merge check')
    return {'logits': torch.cat(logits), 'predictions': predictions, 'positions': positions}


def _compare(reference, candidate, *, atol, rtol, max_changed_answers, max_accuracy_shift_pp=None):
    """Logits within tolerance; answers whose correctness flipped <= max_changed_answers; and, per skill,
    |accuracy shift| <= max_accuracy_shift_pp. Text-only changes (same correctness) are reported, not gated:
    FP16 near-ties reword long answers without changing the score."""
    logits = compare_logits(reference['logits'], candidate['logits'], atol=atol, rtol=rtol)
    pairs = list(zip(reference['predictions'], candidate['predictions']))
    flipped = [left['id'] for left, right in pairs if left['correct'] != right['correct']]
    reworded = [left['id'] for left, right in pairs
                if left['prediction'] != right['prediction'] and left['correct'] == right['correct']]
    shifts = {}
    for skill in sorted({left['skill'] for left, _ in pairs}):
        subset = [(left, right) for left, right in pairs if left['skill'] == skill]
        shifts[skill] = 100 * sum(right['correct'] - left['correct'] for left, right in subset) / len(subset)
    shift_ok = max_accuracy_shift_pp is None or all(abs(v) <= max_accuracy_shift_pp for v in shifts.values())
    return {'passed': logits['passed'] and len(flipped) <= max_changed_answers and shift_ok,
            'logits': logits, 'changed_answers': flipped, 'reworded_answers': reworded,
            'accuracy_shift_pp': shifts,
            'reference_score': summarize(reference['predictions']),
            'candidate_score': summarize(candidate['predictions'])}


def verified_merge(model, tokenizer, snapshot, config, checkpoint, records, *, checkpoint_hash,
                   data_hashes, out, probes=8, batch_size=1, atol, rtol, max_stored_residual,
                   max_intended_residual=1e-5, max_changed_answers=0, max_accuracy_shift_pp=None,
                   include_files=None):
    """Publish `out` only if every declared check passes; always write `<out>.merge_report.json`."""
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol, max_stored_residual, max_intended_residual)):
        raise ValueError('merge tolerances must be finite and nonnegative')
    if probes < 1 or batch_size < 1 or max_changed_answers < 0:
        raise ValueError('invalid merge validation settings')
    include_files = dict(include_files or {})
    reserved = {PATCH_WEIGHTS, PATCH_MANIFEST, 'effective_adapter.pt', 'merge_validation.json', 'failed_validation.json'}
    for name, path in include_files.items():
        if Path(name).name != name or name in reserved or name.endswith('_dev_predictions.jsonl') or not Path(path).is_file():
            raise ValueError(f'invalid provenance file for the patch: {name}={path}')
    if not records or any(r['split'] != 'dev' for r in records):
        raise ValueError('merge validation requires development data only')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('duplicate merge validation IDs')
    output = Path(out).resolve()
    report_file = Path(str(output) + '.merge_report.json')
    if output.exists() or report_file.exists():
        raise FileExistsError('merge output and report must be new; overwrites are forbidden')
    inputs = [Path(config['path']).resolve()] + ([Path(config['patch']).resolve()] if config.get('patch') else [])
    if any(output.is_relative_to(source) for source in inputs):
        raise ValueError('merge outputs must be outside the input snapshot and patch')
    restore_adapters(model, checkpoint)
    artifact = export_adapters(model)
    diagnostics = preview_merge(model, artifact)
    limits = dict(atol=atol, rtol=rtol, max_stored_residual=max_stored_residual,
                  max_intended_residual=max_intended_residual, max_changed_answers=max_changed_answers,
                  max_accuracy_shift_pp=max_accuracy_shift_pp)
    preflight = all(d['stored_residual'] <= max_stored_residual and d['intended_residual'] <= max_intended_residual
                    for d in diagnostics.values())
    report = {'passed': False, 'published': False, 'checkpoint_hash': checkpoint_hash,
              'data_hashes': data_hashes, 'limits': limits, 'diagnostics': diagnostics,
              'preflight_passed': preflight, 'base_hashes': artifact['base_hashes']}

    def finish():
        write_json(report_file, report)
        return report

    if not preflight:
        # Discard the temporary wrappers; even the in-memory input base is unchanged.
        for name, wrapper in adapter_modules(model).items():
            _replace(model, name, wrapper.base)
        return finish()

    live = _measure(model, tokenizer, records, config, probes, batch_size)
    for name, wrapper in adapter_modules(model).items():
        _replace(model, name, ExportedDoRALinear(wrapper.base, artifact['layers'][name]))
    exported = _measure(model, tokenizer, records, config, probes, batch_size)
    report['live_exported'] = _compare(live, exported, atol=atol, rtol=rtol, max_changed_answers=max_changed_answers,
                                       max_accuracy_shift_pp=max_accuracy_shift_pp)
    for name in artifact['layers']:
        _replace(model, name, model.get_submodule(name).base)
    if not report['live_exported']['passed']:
        return finish()
    merge_export(model, artifact)
    merged = _measure(model, tokenizer, records, config, probes, batch_size)
    report['live_merged'] = _compare(live, merged, atol=atol, rtol=rtol, max_changed_answers=max_changed_answers,
                                     max_accuracy_shift_pp=max_accuracy_shift_pp)
    if not report['live_merged']['passed']:
        return finish()
    report['probe_positions'] = live['positions']
    report['validation_items'] = len(records)
    expected_hashes = model_hashes(model)
    previous = snapshot.get('patch_manifest') or {}
    patched = sorted(set(previous.get('patched', [])) | {f'{name}.weight' for name in artifact['layers']})
    state = model.state_dict()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'.{output.name}.unverified-', dir=output.parent))
    # Failed/interrupted staging is retained for diagnosis, never renamed to out.
    report['staging_directory'] = str(staging)
    # Only adapted weights change; store them exactly, relative to the pinned base.
    save_file({name: state[name].detach().contiguous().cpu() for name in patched}, str(staging / PATCH_WEIGHTS))
    torch.save(artifact, staging / 'effective_adapter.pt')
    for name, path in include_files.items():
        shutil.copyfile(path, staging / name)
    base_manifest = {key: value for key, value in snapshot.items() if key != 'patch_manifest'}
    manifest = {'format': PATCH_FORMAT, 'revision': snapshot['revision'],
                'base_snapshot': snapshot_identity(base_manifest), 'precision': config['precision'],
                'merge_status': 'pending', 'patched': patched, 'tensor_hashes': expected_hashes,
                'merge_chain': previous.get('merge_chain', []) + [{
                    'checkpoint_hash': checkpoint_hash, 'base_identity': identity(artifact['base_hashes']),
                    'adapter_hash': file_hash(staging / 'effective_adapter.pt')}]}

    def save_manifest():
        manifest['files'] = {p.relative_to(staging).as_posix(): file_hash(p) for p in sorted(staging.rglob('*'))
                             if p.is_file() and p.name != PATCH_MANIFEST}
        write_json(staging / PATCH_MANIFEST, manifest)

    save_manifest()
    # The workflow consumes a disposable model. Release its storage before
    # rebuilding the candidate from the pinned base plus the staged patch.
    del state
    model.to_empty(device='meta')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded, reloaded_tokenizer, _ = load_model(config['path'], manifest_path=config['snapshot_manifest'],
                                                precision=config['precision'], device=config['device'],
                                                backend=config['backend'], patch=staging, allow_unverified=True)
    report['tensor_hashes_match'] = expected_hashes == model_hashes(reloaded)
    reloaded_measurement = _measure(reloaded, reloaded_tokenizer, records, config, probes, batch_size)
    report['merged_reloaded'] = _compare(merged, reloaded_measurement, atol=0, rtol=0, max_changed_answers=0)
    report['passed'] = report['tensor_hashes_match'] and report['merged_reloaded']['passed']
    if not report['passed']:
        write_json(staging / 'failed_validation.json', report)
        return finish()
    # This is a verified materialization, not research acceptance/promotion.
    report['published'] = True
    report['tensor_hashes'] = expected_hashes
    report.pop('staging_directory')
    for name, measurement in (('live', live), ('exported', exported), ('merged', merged), ('reloaded', reloaded_measurement)):
        write_jsonl(staging / f'{name}_dev_predictions.jsonl', measurement['predictions'])
    write_json(staging / 'merge_validation.json', report)
    manifest['merge_status'] = 'verified'
    save_manifest()
    # Directory rename is the publication boundary. Never replace an existing out.
    if output.exists():
        raise FileExistsError('merge output appeared during validation; refusing to replace it')
    staging.rename(output)
    return finish()
