#!/usr/bin/env python3
"""Acceptance cases for harness/validate.py. Run via: make test"""
import json, copy, subprocess, pathlib, sys
GOOD = {
 "ts":"2026-08-15T04:10:00Z","run_id":"r-0f3a91","kind":"manual",
 "fp":{"llamacpp_sha":"fb0e6b621","patches":[],
       "build":{"backend":"rocm","target":"gfx1201","rocm":"7.2.53211"},
       "mesa":"26.0.3-1ubuntu1","kernel":"7.0.0-29-generic",
       "gpu":{"pci":"0000:0c:00.0","uid":"0xc3abb1ceca126f15","gfx":"gfx1201"},
       "model":{"name":"Qwen3.8-27B-Q6_K","sha256":None,"quant":"Q6_K"}},
 "cfg":{"ctx":262144,"ctk":"q4_0","ctv":"q4_0","fa":"on","spec":"draft-mtp","n_max":4},
 "m":{"pp2048":710.84,"tg":46.15,"mtp_acceptance":0.7708,"rep":1,"cold_prefill":True,"unit":"tok/s"},
 "ok":{"token_identical":True,"corpus_sha":None}}
cases=[("good row", GOOD, True)]
c=copy.deepcopy(GOOD); del c["fp"]["gpu"]["uid"];        cases.append(("missing fp.gpu.uid", c, False))
c=copy.deepcopy(GOOD); c["fp"]["gpu"]["pci"]="0000:0C:00.0"; cases.append(("uppercase pci", c, False))
c=copy.deepcopy(GOOD); c["m"]={"tg":46.15,"mtp_acceptance":0.77,"rep":1,"cold_prefill":True}
cases.append(("tg-only live row", c, False))
c=copy.deepcopy(GOOD); c["kind"]="backfill"; c["m"]={"tg":46.15,"rep":None,"cold_prefill":None}
cases.append(("tg-only BACKFILL row (exempt)", c, True))
c=copy.deepcopy(GOOD); c["kind"]="skipped"; cases.append(("skipped without reason", c, False))
c=copy.deepcopy(GOOD); c["fp"]["gpu"]["idx"]=1;          cases.append(("gpu index smuggled in", c, False))
pathlib.Path(".scratch").mkdir(exist_ok=True)
c=copy.deepcopy(GOOD); c["cfg"]["spec"]=None; c["cfg"]["n_max"]=None
c["m"]={"pp2048":710.84,"rep":1,"cold_prefill":True}
cases.append(("canary row: spec off, no acceptance", c, True))
c=copy.deepcopy(GOOD); del c["m"]["mtp_acceptance"]
cases.append(("spec ON but acceptance missing", c, False))
# apt rows are live measurements: every kind-gated rule must treat them like nightly.
c=copy.deepcopy(GOOD); c["kind"]="apt"; cases.append(("apt row", c, True))
c=copy.deepcopy(GOOD); c["kind"]="apt"; del c["m"]["rep"]
cases.append(("apt row missing rep", c, False))
c=copy.deepcopy(GOOD); c["kind"]="apt"; del c["m"]["mtp_acceptance"]
cases.append(("apt row, spec ON, acceptance missing", c, False))
c=copy.deepcopy(GOOD); c["kind"]="nightly-ish"; cases.append(("unknown kind", c, False))
# A skip records an absence and carries no metric. Stage 1 had a case for a skip
# WITHOUT a reason but none for a well-formed one, so every skip the harness
# emitted was silently schema-invalid until the first one was actually written.
c=copy.deepcopy(GOOD); c["kind"]="skipped"; c["reason"]="deadline"
c["m"]={"pp2048":None,"unit":"tok/s"}
cases.append(("well-formed skipped row", c, True))
c=copy.deepcopy(GOOD); c["kind"]="backfill"; c["m"]={"rep":None,"cold_prefill":None,"unit":"tok/s"}
cases.append(("non-skip row with no metric at all", c, False))
# Build identity. Two builds of fb0e6b621 live on this host and differ only by
# compiler; one segfaults on every model. llamacpp_sha alone cannot tell them apart.
c=copy.deepcopy(GOOD); c["fp"]["build"]["binary_sha256"]="0"*64
c["fp"]["build"]["toolchain"]="GNU 11.4.0"
cases.append(("row carrying build identity", c, True))
c=copy.deepcopy(GOOD); c["fp"]["build"]["binary_sha256"]="deadbeef"
cases.append(("malformed binary_sha256", c, False))
c=copy.deepcopy(GOOD); c["fp"]["build"]["binary_sha256"]=None
cases.append(("null build identity (pre-stage-3 rows)", c, True))
# A quality row measures perplexity, not throughput. The live-row rule demands a
# prefill metric so a tg-only row cannot hide a kernel regression; that rule has
# nothing to say about a ppl row, and rejected every one until it was told so.
c=copy.deepcopy(GOOD); c["m"]={"ppl":3.0456,"ppl_stderr":0.063,"rep":1,
                               "cold_prefill":True,"unit":"ppl"}
c["cfg"]["spec"]=None; c["cfg"]["n_max"]=None   # perplexity is measured with speculation off
cases.append(("quality row carrying only ppl", c, True))
# ...and a quality row must not claim speculation was on without an acceptance.
c=copy.deepcopy(GOOD); c["m"]={"ppl":3.0456,"rep":1,"cold_prefill":True,"unit":"ppl"}
cases.append(("ppl row claiming spec on, no acceptance", c, False))
c=copy.deepcopy(GOOD); c["m"]={"tg":46.15,"mtp_acceptance":0.77,"rep":1,
                               "cold_prefill":True,"unit":"tok/s"}
cases.append(("tg-only live row still rejected", c, False))
p=pathlib.Path(".scratch/t.jsonl")
fails=0
for name,row,should_pass in cases:
    p.write_text(json.dumps(row)+"\n")
    rc=subprocess.run([sys.executable,"harness/validate.py",str(p)],capture_output=True,text=True).returncode
    ok = (rc==0)==should_pass
    fails += 0 if ok else 1
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<32} expected={'accept' if should_pass else 'reject'}")
sys.exit(1 if fails else 0)
