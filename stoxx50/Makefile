# ──────────────────────────────────────────────────────────────
# Makefile — EURO STOXX 50 Proxy Analysis
# Usage (run from project root: master-thesis-code/):
#   make -f stoxx50/Makefile prep          Data preparation (01–06)
#   make -f stoxx50/Makefile analysis      Matching → DiD → Event study
#   make -f stoxx50/Makefile robustness    Placebo, DELETE events, caliper
#   make -f stoxx50/Makefile plots         Panel overview figure
#   make -f stoxx50/Makefile all           Full pipeline
#
# Note: Step 01 uses the same raw CSVs as the STOXX 600 analysis
#       (rank threshold is 50 instead of 600).
#       Step 02 bootstraps prices from the STOXX 600 price cache
#       and adds the ^STOXX50E benchmark.
# ──────────────────────────────────────────────────────────────

PYTHON = uv run python
SX50   = stoxx50/scripts

# ── Data preparation ─────────────────────────────────────────
.PHONY: prep
prep:
	$(PYTHON) $(SX50)/preparation/01_panel_composition.py
	$(PYTHON) $(SX50)/preparation/02_collect_prices.py
	$(PYTHON) $(SX50)/preparation/03_check_coverage.py
	$(PYTHON) $(SX50)/preparation/04_fix_tickers.py
	$(PYTHON) $(SX50)/preparation/05_build_panel.py
	$(PYTHON) $(SX50)/preparation/06_build_features.py

# ── Analysis ─────────────────────────────────────────────────
.PHONY: analysis
analysis:
	$(PYTHON) $(SX50)/analysis/01_matching.py
	$(PYTHON) $(SX50)/analysis/02_did_estimation.py
	$(PYTHON) $(SX50)/analysis/03_event_study.py

# ── Robustness ───────────────────────────────────────────────
.PHONY: robustness
robustness:
	$(PYTHON) $(SX50)/robustness/placebo_test.py
	$(PYTHON) $(SX50)/robustness/delete_events.py
	$(PYTHON) $(SX50)/robustness/caliper_sensitivity.py

# ── Visualization ────────────────────────────────────────────
.PHONY: plots
plots:
	$(PYTHON) $(SX50)/visualization/panel_plots.py

# ── Full pipeline ─────────────────────────────────────────────
.PHONY: all
all: prep analysis robustness plots

# ── Clean generated outputs ───────────────────────────────────
.PHONY: clean
clean:
	rm -f stoxx50/data/intermediate/*.parquet stoxx50/data/intermediate/*.csv
	rm -f stoxx50/data/results/*.csv
	rm -f stoxx50/figures/*.png
