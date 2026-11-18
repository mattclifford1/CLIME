#!/bin/bash
# Summarise whatever has finished, redraw the figures and tables and copy them into the paper.
set -eu
cd "$(dirname "$0")"
uv run python summarise.py
uv run python check_outliers.py
uv run python make_diagrams.py
uv run python make_figures.py
uv run python make_tables.py
uv run python sync_paper_assets.py
