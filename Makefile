# Makefile — build every paper figure and table from the adhoc/ scripts.
#
# Each script regenerates its own aggregated score parquet on demand from the
# raw S3 evals (see adhoc/build_scores.py), so there is no separate sync step:
# just `make figures`, `make tables`, or `make all`. Rebuilds are incremental —
# a target is only remade when its script (or a shared dependency) changes.
#
#   make            # this help
#   make all        # every figure + table
#   make figures    # figure1, figure2, lora-rank-confound figure
#   make tables     # tables 1, 2, 4 (dspec + score), 7/8/9, 10
#   make figure1    # a single named target (see list below)
#   make clean      # remove generated figures + tables

RUN    := uv run
ADHOC  := adhoc
ASSETS := outputs/assets
TABLES := outputs/tables

# Shared dependency: the on-demand parquet builder + the checkpoint registry that
# names its index. Touching either invalidates every data-driven figure/table.
DATADEPS := $(ADHOC)/build_scores.py utils/model_checkpoints.py

# --- Output files -----------------------------------------------------------
FIG1       := $(ASSETS)/figure1_optimal_diversity.pdf
FIG2       := $(ASSETS)/figure2_diagram.pdf
FIG_CONF   := $(ASSETS)/lora_rank_confound.pdf   # side-effect of the table10 recipe

TAB1       := $(TABLES)/table1_bootstrap_cis.txt
TAB2       := $(TABLES)/table2_optimal_task_diversity.txt
TAB5_DSPEC := $(TABLES)/table5_dspec_ablations.txt
TAB5_SCORE := $(TABLES)/table5_score_ablation.txt
TAB789     := $(TABLES)/table789_benchmark_table.tex
TAB10      := $(TABLES)/table10_lora_confounders.tex

# Task-level parquet written as a side effect of the table5 dspec recipe; figure1 reads it.
TABLE5_PARQUET := outputs/table5_task_level.parquet

FIGURE_FILES := $(FIG1) $(FIG2) $(FIG_CONF)
TABLE_FILES  := $(TAB1) $(TAB2) $(TAB5_DSPEC) $(TAB5_SCORE) $(TAB789) $(TAB10)

# Delete a half-written target if its recipe fails, so a failed S3 sync never
# leaves a truncated figure/table that looks up to date on the next run.
.DELETE_ON_ERROR:

.PHONY: help all figures tables clean \
        figure1 figure2 figure-confound \
        table1 table2 table5-dspec table5-score table789 table10

help:
	@echo "Targets:"
	@echo "  make all            # every figure + table"
	@echo "  make figures        # $(notdir $(FIGURE_FILES))"
	@echo "  make tables         # $(notdir $(TABLE_FILES))"
	@echo ""
	@echo "  Figures:  figure1  figure2  figure-confound"
	@echo "  Tables:   table1  table2  table5-dspec  table5-score  table789  table10"
	@echo ""
	@echo "  make clean          # remove all generated figures + tables"

all: figures tables
figures: $(FIGURE_FILES)
tables: $(TABLE_FILES)

# Convenience aliases -> file targets
figure1: $(FIG1)
figure2: $(FIG2)
figure-confound: $(FIG_CONF)
table1: $(TAB1)
table2: $(TAB2)
table5-dspec: $(TAB5_DSPEC)
table5-score: $(TAB5_SCORE)
table789: $(TAB789)
table10: $(TAB10)

# Output directories (order-only prerequisites).
$(ASSETS) $(TABLES):
	mkdir -p $@

# --- Figures ----------------------------------------------------------------
# figure1 syncs sample-level evals via statsig_utils and reads the P->𝒟 mapping
# from the table5 dspec parquet (the dspec_D column), so it depends on that build.
$(FIG1): $(ADHOC)/figure1_optimal_diversity.py $(ADHOC)/statsig_utils.py $(TABLE5_PARQUET) $(DATADEPS) | $(ASSETS)
	$(RUN) $(ADHOC)/figure1_optimal_diversity.py

# Pure-matplotlib architecture diagram — no data dependency.
$(FIG2): $(ADHOC)/figure2_ndlora_diagram.py | $(ASSETS)
	$(RUN) $(ADHOC)/figure2_ndlora_diagram.py

# --- Tables -----------------------------------------------------------------
# Log-style scripts report via the logger (stderr) -> capture combined output.
$(TAB1): $(ADHOC)/table1_bootstrap_cis.py $(ADHOC)/statsig_utils.py $(DATADEPS) | $(TABLES)
	$(RUN) $(ADHOC)/table1_bootstrap_cis.py > $@ 2>&1

$(TAB2): $(ADHOC)/table2_optimal_task_diversity.py $(ADHOC)/statsig_utils.py $(DATADEPS) | $(TABLES)
	$(RUN) $(ADHOC)/table2_optimal_task_diversity.py > $@ 2>&1

# The dspec recipe writes both the .txt log and outputs/table5_task_level.parquet.
$(TAB5_DSPEC): $(ADHOC)/table5_dspec_ablations.py $(DATADEPS) | $(TABLES)
	$(RUN) $(ADHOC)/table5_dspec_ablations.py > $@ 2>&1

# The task-level parquet is a side effect of the dspec recipe above (consumed by figure1).
$(TABLE5_PARQUET): $(TAB5_DSPEC)
	@test -f $@

$(TAB5_SCORE): $(ADHOC)/table5_score_ablation.py $(DATADEPS) | $(TABLES)
	$(RUN) $(ADHOC)/table5_score_ablation.py > $@ 2>&1

# LaTeX-emitting scripts print the table to stdout -> capture stdout only,
# leaving progress logs on the console.
$(TAB789): $(ADHOC)/table789_benchmark_table.py $(DATADEPS) | $(TABLES)
	$(RUN) $(ADHOC)/table789_benchmark_table.py > $@

# table10 emits its LaTeX to stdout AND writes the confound figure to $(ASSETS).
$(TAB10): $(ADHOC)/table10_lora_confounders.py $(DATADEPS) | $(TABLES) $(ASSETS)
	$(RUN) $(ADHOC)/table10_lora_confounders.py > $@

# The confound figure is produced as a side effect of the table10 recipe above.
$(FIG_CONF): $(TAB10)
	@test -f $@

clean:
	rm -f $(FIGURE_FILES) $(ASSETS)/figure1_optimal_diversity.png \
	      $(ASSETS)/figure2_diagram.png $(ASSETS)/lora_rank_confound.png \
	      $(TABLE_FILES) $(TABLE5_PARQUET)
