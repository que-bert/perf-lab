# Decode-path KLD baselines (P0.5)

Recorded with `llama-decode-kld` from fork branch `r9700-p0-tools` @ e34c91924 (kernels identical to
`26bd56621`; only the new tool and perf-logger instrumentation differ), fixed serving config
(`-ctk q8_0 -ctv q8_0 -c 262144 -fa on -ngl 99 -dev Vulkan1`), corpus `harness/corpus/decode_kld.txt`
(built by `build_corpus.sh`, 1,500,000 bytes, 404,717 tokens,
sha256 0c7650a4e2c7ca453f1f19043391833910fd114731e53ba24f8c11d3509f2b51), `--score 512`.
Files are ~254 MB each and are not committed (header + int32 next tokens + fp16 [512][248320] logits).

| file | depth | batch | sha256 |
|---|---|---|---|
| base-d176000-b4.dkld | 176000 | 4 | f6aa1952c5ce9a7bf9e87e4677fa59b5d695f99284765ccf999dfb2cf52ce2c9 |
| base-d70000-b4.dkld | 70000 | 4 | 2b8116229e53ee7878fe8d4342351a31d1b2d223680957d3a822e2efb50f6781 |

Missing files at time of writing are being re-recorded by a queued retry (guard refusals); append their rows when present.
