LEDGER ?= results/ledger.jsonl

.PHONY: validate test help row canaries check rebaseline

help:
	@echo "make validate   validate $(LEDGER) against results/schema.json"
	@echo "make test       run the harness test cases"
	@echo "make row        one live canary row (fast-q4)"
	@echo "make canaries   all canary configs"
	@echo "make check      read $(LEDGER) back: is each canary still in band?"
	@echo "make rebaseline re-derive canary bands from measured spread"
	@echo ""
	@echo "requires: PERF_LAB_MODEL, PERF_LAB_BIN, PERF_LAB_GPU_UID"

validate:
	@python3 harness/validate.py $(LEDGER)

test:
	@python3 harness/test_validate.py
	@python3 harness/test_check.py

check:
	@python3 harness/check.py --ledger $(LEDGER)

rebaseline:
	@harness/rebaseline.py --reps 5 --nightly-reps 3

row:
	@harness/bench.sh configs/canary.yaml fast-q4 --reps 1

canaries:
	@for k in fast-q4 fast-q8 slow-q5_1 slow-q4_1 slow-mixed; do \
		echo "--- $$k ---"; harness/bench.sh configs/canary.yaml $$k --reps 1 || exit 1; \
	done
