#!/usr/bin/env bash
# df_md5_client.sh <label>: 3 greedy completions against :8104, append "label idx prompt_n md5" to results/d0/df/<label>.md5
O=/home/bbuckham/git/perf-lab/results/d0/df
for i in 0 1 2; do
  curl -s localhost:8104/completion -d @$O/prompt$i.json | python3 -c "import sys,json,hashlib; d=json.load(sys.stdin); print('$1', $i, d['timings']['prompt_n'], hashlib.md5(d['content'].encode()).hexdigest())" >> $O/$1.md5
done
