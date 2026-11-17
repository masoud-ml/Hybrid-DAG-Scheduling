"""Refuse resume when configuration or analysis code has changed."""
import hashlib
import json
from pathlib import Path


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def code_digest():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for folder in ('core','analysis','schedule','data','pipeline','learning','utils'):
        for p in sorted((root/folder).rglob('*.py')):
            h.update(str(p.relative_to(root)).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def guard_resume(output, config):
    output = Path(output)
    manifest = output.with_suffix(output.suffix + '.manifest.json')
    payload = {'config': config, 'code_sha256': code_digest()}
    if manifest.exists():
        if json.loads(manifest.read_text()) != payload:
            raise ValueError(f'Resume configuration/code mismatch for {output}; use a new output path')
    elif output.exists() and output.stat().st_size:
        raise ValueError(f'Missing provenance for existing {output}; regenerate in a new output path')
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        temp = manifest.with_suffix('.tmp')
        temp.write_text(json.dumps(payload, sort_keys=True, indent=2)); temp.replace(manifest)


def label_guard(index, output, trials, seed, min_scale=.5, max_scale=1.):
    guard_resume(output, dict(stage='label', index_sha256=file_digest(index), trials=trials,
                             seed=seed, min_scale=min_scale, max_scale=max_scale))
