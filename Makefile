LEDGER ?= results/ledger.jsonl

.PHONY: validate test help

help:
	@echo "make validate   validate $(LEDGER) against results/schema.json"
	@echo "make test       run the validator's own test cases"

validate:
	@python3 harness/validate.py $(LEDGER)

test:
	@python3 harness/test_validate.py
