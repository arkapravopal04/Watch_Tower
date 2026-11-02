"""Build the code Dataset for the follow-up notebooks: pcl-next-code.zip, next to the repository.

    python notebooks_2/build_code_zip.py

It holds the frozen study code exactly as it is (pcl/, configs/, scripts/, tests/)
plus notebooks_2/. Upload it as a new private Dataset (e.g. `pcl-next`) and
attach only that one to NB7-NB9; the main study uses its own `pcl-code`.
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ('pcl', 'configs', 'scripts', 'tests', 'notebooks_2', 'docs', 'pyproject.toml', 'README.md')
SKIP_PARTS = {'__pycache__', '.pytest_cache'}


def files():
    for name in INCLUDE:
        path = ROOT / name
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file())
        for candidate in candidates:
            parts = set(candidate.relative_to(ROOT).parts)
            if parts & SKIP_PARTS or candidate.suffix in ('.pyc', '.zip') or candidate.name.endswith('.egg-info'):
                continue
            yield candidate


def main(out=ROOT.parent / 'pcl-next-code.zip'):
    count = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files():
            archive.write(path, path.relative_to(ROOT).as_posix())
            count += 1
    print(f'wrote {out} ({count} files, {out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
