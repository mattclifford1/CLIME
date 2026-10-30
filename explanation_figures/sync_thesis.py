'''
Copy out/<chapter>/... to the thesis's chapters/<chapter>/figures/..., skipping unchanged files.

    uv run python explanation_figures/sync_thesis.py            # copy
    uv run python explanation_figures/sync_thesis.py --dry-run  # list what would change
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
import functools

from thesis_figures import sync as _sync

from style import OUT

sync = functools.partial(_sync.sync, OUT)

if __name__ == '__main__':
    _sync.main(OUT)
