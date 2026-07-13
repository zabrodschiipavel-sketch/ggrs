"""Вызов OmniRoute-гейтвея для делегирования задач моделям.

Использование:
  python omni.py <model> <prompt_file> <out_file>            # вызов, ответ в out_file
  python omni.py --apply <out_file> <repo_root>              # применить <<<FILE:>>>-блоки из ответа

Формат блоков в ответе модели:
  <<<FILE: relative/path>>>
  ...содержимое файла целиком...
  <<<END>>>
"""
import json
import re
import sys
import urllib.request

GATEWAY = "http://127.0.0.1:20128/v1/chat/completions"
KEY = __import__("json").load(open(r"C:\Users\pavel\.ggrs-secrets.json"))["omniroute"]


def call(model: str, prompt_file: str, out_file: str) -> None:
    prompt = open(prompt_file, encoding="utf-8").read()
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32000,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        GATEWAY, data=body,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        resp = json.loads(r.read().decode("utf-8"))
    content = resp["choices"][0]["message"]["content"]
    with open(out_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    usage = resp.get("usage", {})
    print(f"OK model={model} chars={len(content)} usage={usage}")


def apply_blocks(out_file: str, repo_root: str) -> None:
    import os
    text = open(out_file, encoding="utf-8").read()
    blocks = re.findall(r"<<<FILE:\s*(.+?)>+[^\S\n]*\n(.*?)\n?<<<END>+", text, re.DOTALL)
    if not blocks:
        print("NO FILE BLOCKS FOUND")
        sys.exit(1)
    for rel, content in blocks:
        rel = rel.strip().rstrip(">").strip()
        path = os.path.join(repo_root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # срезать возможную обёртку ```rust ... ```
        c = content
        c = re.sub(r"^```[a-z]*\n", "", c)
        c = re.sub(r"\n```\s*$", "", c)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(c if c.endswith("\n") else c + "\n")
        print(f"WROTE {rel} ({len(c)} chars)")


if __name__ == "__main__":
    if sys.argv[1] == "--apply":
        apply_blocks(sys.argv[2], sys.argv[3])
    else:
        call(sys.argv[1], sys.argv[2], sys.argv[3])
