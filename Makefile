LEDGER ?= results/ledger.jsonl

.PHONY: validate test help row canaries

help:
	@echo "make validate   validate $(LEDGER) against results/schema.json"
	@echo "make test       run the validator's own test cases"
	@echo "make row        one live canary row (fast-q4)"
	@echo "make canaries   all canary configs"
	@echo ""
	@echo "requires: PERF_LAB_MODEL, PERF_LAB_BIN, PERF_LAB_GPU_UID"

validate:
	@python3 harness/validate.py $(LEDGER)

test:
	@python3 harness/test_validate.py

row:
	@harness/bench.sh configs/canary.yaml fast-q4 --reps 1

canaries:
	@for k in fast-q4 fast-q8 slow-q5_1 slow-q4_1 slow-mixed; do \
		echo "--- $$k ---"; harness/bench.sh configs/canary.yaml $$k --reps 1 || exit 1; \
	done
