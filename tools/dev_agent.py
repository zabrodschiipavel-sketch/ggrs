"""Агент-исполнитель на DeepSeek с песочницей инструментов для репо ggrs.

Использование:
  python dev_agent.py <model> <system_file> <task_file> <report_file>
    model: deepseek-v4-flash (лайт) | deepseek-v4-flash:thinking | deepseek-v4-pro (всегда thinking)

Инструменты модели: read_file / write_file / list_dir (только внутри репо,
.git запрещён) и run_cargo (белый список подкоманд). Итерируется сама до
финального отчёта. Транскрипт пишется в <report_file>.log.
"""
import json
import os
import re
import subprocess
import sys
import urllib.request

SECRETS = json.load(open(r"C:\Users\pavel\.ggrs-secrets.json"))
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
REPO = r"C:\Users\pavel\ggrs"
MAX_ROUNDS = 24
CARGO = os.path.expandvars(r"%USERPROFILE%\.cargo\bin\cargo.exe")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Прочитать файл репозитория (UTF-8). Путь относительно корня репо.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Записать файл целиком (UTF-8, LF). Путь относительно корня репо. Создаёт директории.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "Список файлов директории репо (рекурсивно на 2 уровня).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Относительный путь, '' = корень"},
        }, "required": []},
    }},
    {"type": "function", "function": {
        "name": "run_cargo",
        "description": "Запустить cargo в корне репо. Разрешены ТОЛЬКО подкоманды: test, build, clippy, fmt. Пример args: 'test -p ggrs-core --test ops_3d'.",
        "parameters": {"type": "object", "properties": {
            "args": {"type": "string"},
        }, "required": ["args"]},
    }},
]


def _safe_path(rel):
    rel = (rel or "").replace("\\", "/").strip().lstrip("/")
    p = os.path.normpath(os.path.join(REPO, rel))
    if not p.startswith(REPO):
        raise ValueError("путь вне репозитория запрещён")
    if ".git" in p.replace("\\", "/").split("/"):
        raise ValueError(".git запрещён")
    return p


def read_file(path):
    p = _safe_path(path)
    if not os.path.isfile(p):
        return {"error": f"файла нет: {path}"}
    text = open(p, encoding="utf-8").read()
    return {"path": path, "chars": len(text), "content": text[:60000]}


def write_file(path, content):
    p = _safe_path(path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(content if content.endswith("\n") else content + "\n")
    return {"written": path, "chars": len(content)}


def list_dir(path=""):
    p = _safe_path(path)
    out = []
    base_depth = p.rstrip("\\/").count(os.sep)
    for root, dirs, files in os.walk(p):
        dirs[:] = [d for d in dirs if d not in (".git", "target", ".superpowers")]
        if root.count(os.sep) - base_depth >= 2:
            dirs[:] = []
        for f in files:
            out.append(os.path.relpath(os.path.join(root, f), REPO).replace("\\", "/"))
    return out[:300]


RUNNER_DIR = os.path.join(REPO, ".superpowers", "runner")


def run_cargo(args):
    # Application Control блокирует rustc в python-цепочке — исполняем через
    # файловый протокол внешнего bash-раннера (cargo_runner.sh).
    import time
    args = args.strip()
    if re.search(r"[;&|<>`$\n]", args):
        return {"error": "недопустимые символы в args"}
    parts = args.split()
    if not parts or parts[0] not in ("test", "build", "clippy", "fmt"):
        return {"error": "разрешены только: test, build, clippy, fmt"}
    resp_path = os.path.join(RUNNER_DIR, "response.json")
    req_path = os.path.join(RUNNER_DIR, "request.json")
    if os.path.exists(resp_path):
        os.remove(resp_path)
    tmp = req_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"args": " ".join(parts)}, f)
    os.replace(tmp, req_path)
    deadline = time.monotonic() + 620
    while time.monotonic() < deadline:
        if os.path.exists(resp_path):
            time.sleep(0.2)
            with open(resp_path, encoding="utf-8") as f:
                result = json.load(f)
            os.remove(resp_path)
            return result
        time.sleep(0.5)
    return {"error": "runner не ответил за 620с — жив ли cargo_runner.sh?"}


FNS = {"read_file": read_file, "write_file": write_file, "list_dir": list_dir, "run_cargo": run_cargo}


def deepseek(model, messages, allow_tools=True):
    base = model.removesuffix(":thinking")
    think = model.endswith(":thinking") or base == "deepseek-v4-pro"
    payload = {"model": base, "messages": messages, "max_tokens": 16000}
    if think:
        payload["reasoning_effort"] = "high"
        payload["thinking"] = {"type": "enabled"}
    else:
        payload["thinking"] = {"type": "disabled"}
    if allow_tools:
        payload["tools"] = TOOLS
    req = urllib.request.Request(DEEPSEEK_URL, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {SECRETS['deepseek']}", "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode())


def main(model, system_file, task_file, report_file):
    messages = [
        {"role": "system", "content": open(system_file, encoding="utf-8").read()},
        {"role": "user", "content": open(task_file, encoding="utf-8").read()},
    ]
    log = open(report_file + ".log", "w", encoding="utf-8")
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for rnd in range(MAX_ROUNDS):
        last = rnd == MAX_ROUNDS - 1
        if last:
            messages.append({"role": "user", "content": "Лимит шагов. Напиши финальный отчёт о сделанном и статусе тестов."})
        resp = deepseek(model, messages, allow_tools=not last)
        u = resp.get("usage", {})
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)
        msg = resp["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            with open(report_file, "w", encoding="utf-8", newline="\n") as f:
                f.write(msg.get("content") or "")
            print(f"DONE rounds={rnd + 1} usage={usage}")
            log.close()
            return
        for tc in calls:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError as e:
                result = {"error": f"кривой JSON аргументов: {e}"}
                args = {}
            else:
                try:
                    result = FNS[name](**args)
                except Exception as e:  # noqa: BLE001
                    result = {"error": str(e)}
            brief = json.dumps(args, ensure_ascii=False)[:150]
            print(f"  [{rnd}] {name}({brief})")
            log.write(f"[{rnd}] {name}({brief}) -> {json.dumps(result, ensure_ascii=False)[:300]}\n")
            log.flush()
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
    print("MAX_ROUNDS exceeded")
    sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
