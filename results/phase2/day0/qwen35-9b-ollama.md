# day 0: qwen35-9b-ollama

file: /mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/converted/qwen35-9b-ollama.gguf  mtp: 1  fork: /home/bbuckham/git/perf-lab/runners/llama.cpp/qwen3.8-r9700-i18/build/bin  upstream: /home/bbuckham/git/perf-lab/runners/llama.cpp/upstream-ce8caa6/build/bin

## preset
```
preset: qwen35 (arch-fallback)
  applied: --flash-attn=on
  applied: --parallel=1
  applied: --spec-type=draft-mtp
  applied: --spec-draft-n-max=3
  applied: --cache-type-k=q8_0
  applied: --cache-type-v=q8_0
```
== REF qwen35-9b-ollama PPL = 2.4465 -> /mnt/8724062a-75f8-4edf-8ca8-b7dd4e77ed30/models/perflab-kld-ref/qwen35-9b-ollama.kld (1.9G)
Running as unit: perflab-srv-8099.service; invocation ID: fd6894bfb95b47df8e674a814a059e49
gpu_lock: CLEAN maxload=1.21 vram<=4.85GiB gtt<=0.91GiB
== SMOKE qwen35-9b-ollama ref recorded

## speed (fork vs upstream)
| model | test | up t/s | fork t/s | CV up % | CV fork % | fork vs up |
| qwen35-9b-ollama | pp512 d0 | 3766.5 | 4182.5 | 0.35 | 0.22 | GAIN +11.0% [+10.7,+11.4] k=2 |
| qwen35-9b-ollama | tg128 d0 | 92.0 | 94.8 | 0.10 | 0.08 | GAIN +3.1% [+2.7,+3.4] k=2 |
| qwen35-9b-ollama | pp512 d8192 | 3185.4 | 3545.0 | 0.36 | 0.34 | GAIN +11.3% [+11.2,+11.3] k=2 |
| qwen35-9b-ollama | tg128 d8192 | 86.7 | 88.9 | 0.11 | 0.12 | GAIN +2.6% [+2.5,+2.6] k=2 |

## quality
gpu_lock: waiting for the cpu lock (-x; other lane is running)
Running as unit: perflab-srv-8098.service; invocation ID: d415eb78f3cc4144a267b3d56a0c6c1d
gpu_lock: CLEAN maxload=1.13 vram<=5.9GiB gtt<=1.11GiB
== QUAL qwen35-9b-ollama fork PPL=2.450102 KLD=0.000844 top1=98.777% ACC=0.0043
gpu_lock: waiting for the cpu lock (-x; other lane is running)
Running as unit: perflab-srv-8099.service; invocation ID: 095e474abbde4b91b4976e231a979142
gpu_lock: CLEAN maxload=0.93 vram<=4.85GiB gtt<=0.91GiB
== SMOKE qwen35-9b-ollama fork PASS (min tokens 64)
