'''Copy the generated figures and tables into the paper repo, so it builds on Overleaf on its own.'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import shutil
from common import paths


def sync(source, destination, pattern):
    '''mirror `pattern`, deleting destination files that are no longer generated'''
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {p.name for p in source.glob(pattern)}
    for stale in sorted(destination.glob(pattern)):
        if stale.name not in wanted:
            stale.unlink()
            print(f'  pruned stale {stale.name}')
    for path in sorted(source.glob(pattern)):
        shutil.copy2(path, destination/path.name)
    print(f'  {len(wanted)} x {pattern} -> {destination}')


if __name__ == '__main__':
    if not paths.PAPER.exists():
        raise SystemExit(f'write-up not found at {paths.PAPER}')
    sync(paths.FIGURES, paths.PAPER/'figures', '*.pdf')
    sync(paths.TABLES, paths.PAPER/'tables', '*.tex')
