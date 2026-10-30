'''
Copy out/<chapter>/... to the thesis's chapters/<chapter>/figures/..., skipping unchanged files.

    uv run python explanation_figures/sync_thesis.py            # copy
    uv run python explanation_figures/sync_thesis.py --dry-run  # list what would change

Only copies files; committing and pushing the thesis stays manual.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
import argparse
import filecmp
import os
import shutil

from style import HERE, OUT

THESIS = os.environ.get('THESIS_DIR', os.path.normpath(
    os.path.join(HERE, '..', '..', '..', 'phd-thesis', 'thesis-overleaf')))


def sync(dry_run=False):
    if not os.path.isdir(os.path.join(THESIS, 'chapters')):
        raise SystemExit(f'no thesis at {THESIS} (set THESIS_DIR)')
    changed = 0
    for root, _, files in os.walk(OUT):
        for f in sorted(files):
            src = os.path.join(root, f)
            chapter, *rest = os.path.relpath(src, OUT).split(os.sep)
            dst = os.path.join(THESIS, 'chapters', chapter, 'figures', *rest)
            if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
                continue
            changed += 1
            print(f"{'would copy' if dry_run else 'copied'} {os.path.relpath(dst, THESIS)}")
            if not dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
    print(f'{changed} figure(s) {"to update" if dry_run else "updated"} in {THESIS}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument('--dry-run', action='store_true')
    sync(p.parse_args().dry_run)
