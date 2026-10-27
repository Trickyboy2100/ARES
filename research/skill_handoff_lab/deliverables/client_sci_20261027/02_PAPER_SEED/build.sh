#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p figures tables
# These are generated from the experiment CSV/JSON and figure provenance.
# Copy, never hand-edit, the shared historical table used in this manuscript.
cp ../04_TABLES/Tab2_historical_results.tex tables/historical_results.tex
cp ../04_TABLES/Tab1_environment.tex tables/environment.tex
cp ../04_TABLES/Tab3_new_pilot.tex tables/new_pilot.tex
cp ../04_TABLES/Tab4_ablation_budget.tex tables/ablation_budget.tex
for source in ../03_FIGURES/*.pdf; do
  test -f "$source" && cp "$source" figures/
done
for required in Fig1_system_layers.pdf Fig2_method_graph.pdf Fig3_physx_scene_board.pdf Fig4_historical_online_learning.pdf Fig5_historical_frozen_tests.pdf Fig6_new_pilot.pdf FigS2_new_pilot_online_learning.pdf; do
  test -s "figures/$required" || { echo "Missing real figure: $required" >&2; exit 1; }
done
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
fi
test -s main.pdf
