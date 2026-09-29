#!/usr/bin/env python3
"""Pooled MTP acceptance on chat / prose / multilingual / image prompts (D1b held-out multilingual set, not used to tune D1b).
  chat_acc.py <port> <out.json>
Greedy, 256 tokens each, via /v1/chat/completions (thinking off). Prints per-prompt and POOLED.
"""
import base64, json, sys, urllib.request

PROMPTS = [
    "한국의 전통 음식인 김치의 역사와 종류, 그리고 만드는 방법을 자세히 설명해 주세요.",
    "Расскажите подробно о том, как устроена Солнечная система и какие планеты в неё входят.",
    "Scrivi una breve guida per visitare Roma in tre giorni, con consigli su cosa vedere e dove mangiare.",
    "Explique em detalhes como funciona o sistema de saúde pública no Brasil e quais são seus desafios.",
    "Hãy giải thích chi tiết về lịch sử và ý nghĩa của Tết Nguyên Đán ở Việt Nam.",
    "اشرح بالتفصيل كيف يعمل الإنترنت، من إرسال الطلب حتى وصول الصفحة إلى المتصفح.",
    "请用Python写一个LRU缓存类，支持get和put操作，时间复杂度O(1)，并用中文写详细的注释解释每一步。",
    "Translate the following into natural Chinese: \"The committee postponed its decision until more data from the field trials became available, citing concerns about long-term safety and cost.\" Then explain your word choices in Chinese.",
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
    items = list(PROMPTS)
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
