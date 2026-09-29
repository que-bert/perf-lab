#!/usr/bin/env python3
"""Pooled MTP acceptance on chat / prose / multilingual / image prompts (D1 gate set).
  chat_acc.py <port> <out.json>
Greedy, 256 tokens each, via /v1/chat/completions (thinking off). Prints per-prompt and POOLED.
"""
import base64, json, sys, urllib.request

PROMPTS = [
    "Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle.",
    "Explain the causes of the fall of the Western Roman Empire in a few paragraphs.",
    "What are the main differences between a Roth IRA and a traditional IRA? Answer conversationally.",
    "I have a job interview tomorrow for a nursing position. Give me advice on how to prepare.",
    "Summarize the plot of Pride and Prejudice and discuss its main themes.",
    "请用中文介绍一下长城的历史和它在中国文化中的意义。",
    "Escribe un ensayo corto sobre la importancia de la biodiversidad en América Latina.",
    "Erkläre mir bitte, wie die Photosynthese funktioniert, in einfachen Worten.",
    "日本の四季について、それぞれの季節の特徴と伝統行事を説明してください。",
    "Rédige une lettre formelle pour demander un congé de trois semaines à ton employeur.",
    "A train leaves at 3:40 pm and travels 210 km at 84 km/h. When does it arrive? Show your reasoning step by step.",
    "Compose a heartfelt wedding toast for my best friend Sam and his partner Priya.",
]
IMAGE = "/usr/share/backgrounds/jdituicha-raccoon1-dark.jpg"

def ask(port, content):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 256, "temperature": 0,
            "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=3600))
    t = d.get("timings", {})
    return t.get("draft_n_accepted"), t.get("draft_n"), t.get("predicted_per_second"), d["choices"][0]["message"]["content"]

def main():
    port, out = int(sys.argv[1]), sys.argv[2]
    img = base64.b64encode(open(IMAGE, "rb").read()).decode()
    items = [p for p in PROMPTS] + [[{"type": "text", "text": "Describe this image in detail."},
                                     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}}]]
    res = []
    for i, c in enumerate(items):
        a, n, tps, text = ask(port, c)
        res.append({"i": i, "acc": a, "n": n, "tps": tps, "text": text})
        print(i, a, n, round(a / n, 3) if a is not None and n else None, flush=True)
    ok = [r for r in res if r["acc"] is not None and r["n"]]
    A, N = sum(r["acc"] for r in ok), sum(r["n"] for r in ok)
    print("POOLED", A, N, round(A / N, 4))
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    main()
