# ──────────────────────────────────────────────────────────
# Makefile — Master Thesis Empirical Pipeline
# Usage:
#   make prep          Run data preparation (steps 01–06)
#   make analysis      Run analysis pipeline (matching → DiD → event study)
#   make robustness    Run robustness checks (placebo, delete, caliper)
#   make plots         Generate panel overview plots
#   make all           Full pipeline: prep + analysis + robustness + plots
# ──────────────────────────────────────────────────────────

PYTHON = uv run python

# ── Data preparation ─────────────────────────────────────
.PHONY: prep
prep:
	$(PYTHON) scripts/preparation/01_panel_composition.py
	$(PYTHON) scripts/preparation/02_collect_prices.py
	$(PYTHON) scripts/preparation/03_check_coverage.py
	$(PYTHON) scripts/preparation/04_fix_tickers.py
	$(PYTHON) scripts/preparation/05_build_panel.py
	$(PYTHON) scripts/preparation/06_build_features.py

# ── Analysis ─────────────────────────────────────────────
.PHONY: analysis
analysis:
	$(PYTHON) scripts/analysis/00_eda_and_checks.py
	$(PYTHON) scripts/analysis/01_matching.py
	$(PYTHON) scripts/analysis/02_did_estimation.py
	$(PYTHON) scripts/analysis/03_event_study.py
	$(PYTHON) scripts/analysis/04_double_ml.py

# ── Robustness ───────────────────────────────────────────
.PHONY: robustness
robustness:
	$(PYTHON) scripts/robustness/placebo_test.py
	$(PYTHON) scripts/robustness/placebo_dml.py
	$(PYTHON) scripts/robustness/delete_events.py
	$(PYTHON) scripts/robustness/caliper_sensitivity.py

# ── Visualization ────────────────────────────────────────
.PHONY: plots
plots:
	$(PYTHON) scripts/visualization/panel_plots.py

# ── Full pipeline ────────────────────────────────────────
.PHONY: all
all: prep analysis robustness plots

# ── Clean generated outputs ──────────────────────────────
.PHONY: clean
clean:
	rm -f data/intermediate/*.parquet data/intermediate/*.csv
	rm -f data/results/*.csv
	rm -f figures/*.png
