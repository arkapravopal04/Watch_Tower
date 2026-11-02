"""Local pinned Hugging Face loading; never downloads weights implicitly.

A merged stage is stored as a verified *weight patch* over the pinned base:
only the adapted q/v weights, in storage precision, plus the exact tensor
hashes of the complete patched model. Applying a patch copies stored tensors
(no arithmetic), so the rebuilt model is bit-identical in every session.
"""

from pathlib import Path

import torch

from .io import file_hash, read_json

PATCH_FORMAT = 'pcl-weight-patch-v1'
PATCH_WEIGHTS = 'weights.safetensors'
PATCH_MANIFEST = 'patch_manifest.json'


def _check_inventory(root: Path, files: dict, excluded: Path) -> None:
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
              and p.resolve() != excluded.resolve()}
    if actual != set(files):
        raise ValueError(f"inventory of {root} differs from its manifest")
    for name, expected in files.items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or file_hash(candidate) != expected:
            raise ValueError(f"hash mismatch: {root / name}")


def verify_snapshot(path, manifest_path) -> dict:
    root = Path(path).resolve()
    manifest = read_json(manifest_path)
    revision = manifest.get('revision', '')
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        raise ValueError("snapshot needs an exact 40-character commit revision")
    files = manifest.get('files', {})
    if not files or 'config.json' not in files:
        raise ValueError("snapshot manifest must hash all model and tokenizer files")
    _check_inventory(root, files, Path(manifest_path))
    return manifest


def snapshot_identity(manifest: dict) -> str:
    from .registry import identity
    return identity({'revision': manifest['revision'], 'files': manifest['files']})


def verify_patch(path, base_manifest: dict, *, allow_unverified: bool = False) -> dict:
    root = Path(path).resolve()
    manifest = read_json(root / PATCH_MANIFEST)
    if manifest.get('format') != PATCH_FORMAT:
        raise ValueError('not a pcl weight patch')
    if manifest.get('merge_status') != 'verified' and not allow_unverified:
        raise ValueError('weight patch is unverified; use the verified merge workflow')
    if manifest.get('base_snapshot') != snapshot_identity(base_manifest):
        raise ValueError('weight patch was built on a different base snapshot')
    if PATCH_WEIGHTS not in manifest.get('files', {}) or not manifest.get('tensor_hashes') or not manifest.get('patched'):
        raise ValueError('incomplete weight patch manifest')
    _check_inventory(root, manifest['files'], root / PATCH_MANIFEST)
    return manifest


@torch.no_grad()
def apply_patch(model, patch_dir, manifest: dict) -> None:
    from safetensors.torch import load_file
    from .hashing import model_hashes
    tensors = load_file(str(Path(patch_dir) / PATCH_WEIGHTS), device='cpu')
    if set(tensors) != set(manifest['patched']):
        raise ValueError('patch tensors differ from the manifest list')
    state = dict(model.named_parameters())
    for name, value in tensors.items():
        if name not in state:
            raise ValueError(f'patch targets a missing parameter: {name}')
        target = state[name]
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(f'patch tensor shape/dtype mismatch: {name}; load with the recorded precision')
        target.copy_(value.to(target.device))
    if model_hashes(model) != manifest['tensor_hashes']:
        raise ValueError('patched model differs from the verified artifact; check precision/configuration')


def _dtype_argument(dtype) -> dict:
    """Transformers >= 4.56 names the argument `dtype`; older versions use `torch_dtype`."""
    import transformers
    major, minor = (int(part) for part in transformers.__version__.split('.')[:2])
    return {'dtype': dtype} if (major, minor) >= (4, 56) else {'torch_dtype': dtype}


def load_model(path, *, manifest_path, precision='fp16', device='cuda', backend='sdpa', patch=None,
               allow_unverified=False):
    """Return (model, tokenizer, snapshot). With a patch, snapshot['patch_manifest'] is set."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if precision not in ('fp16', 'fp32') or backend not in ('sdpa', 'eager'):
        raise ValueError("only explicitly validated fp16/fp32 and sdpa/eager paths are supported")
    if precision == 'fp16' and not str(device).startswith('cuda'):
        raise ValueError("use FP32 for CPU checks")
    manifest = verify_snapshot(path, manifest_path)
    patch_manifest = verify_patch(patch, manifest, allow_unverified=allow_unverified) if patch else None
    if patch_manifest and patch_manifest.get('precision') != precision:
        raise ValueError(f"patch was verified in {patch_manifest.get('precision')}, not {precision}")
    dtype = torch.float16 if precision == 'fp16' else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("the pinned tokenizer needs a chat template and EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                 **_dtype_argument(dtype), attn_implementation=backend,
                                                 low_cpu_mem_usage=True).to(device)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"nonfinite model parameter after loading: {name}")
    snapshot = dict(manifest)
    if patch_manifest:
        apply_patch(model, patch, patch_manifest)
        snapshot['patch_manifest'] = patch_manifest
    return model, tokenizer, snapshot

