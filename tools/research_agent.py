"""Исследовательский агент: DeepSeek (tool calls) + OpenAlex.

Использование: python research_agent.py <prompt_file> <out_file>
DeepSeek сам решает, какие запросы слать в OpenAlex (до MAX_CALLS), затем пишет отчёт.
"""
import json
import sys
import urllib.parse
import urllib.request

SECRETS = json.load(open(r"C:\Users\pavel\.ggrs-secrets.json"))
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MAX_ROUNDS = 16

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_openalex",
        "description": "Поиск научных работ в OpenAlex (250M+ работ). Возвращает топ работ по релевантности: название, год, авторы, площадка, цитирования, DOI, фрагмент абстракта.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковая строка (английский эффективнее)"},
                "from_year": {"type": "integer", "description": "Год с (например 2024)"},
                "per_page": {"type": "integer", "description": "Сколько работ вернуть (1-15, дефолт 8)"},
                "sort_by_citations": {"type": "boolean", "description": "true = сортировать по цитированиям, false/нет = по релевантности"},
            },
            "required": ["query"],
        },
    },
}]


TOOLS.append({
    "type": "function",
    "function": {
        "name": "search_core",
        "description": "Поиск в CORE (открытые полные тексты статей). Лучше всего работает точный запрос title:\"Точное название\". Возвращает работы с длиной доступного полного текста и core_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Запрос, например title:\"The Era of 1-bit LLMs\""},
                "limit": {"type": "integer", "description": "1-5, дефолт 3"},
            },
            "required": ["query"],
        },
    },
})
TOOLS.append({
    "type": "function",
    "function": {
        "name": "get_fulltext",
        "description": "Получить кусок полного текста работы из CORE по core_id (30000 символов за вызов; для длинных статей вызывай несколько раз с разным offset).",
        "parameters": {
            "type": "object",
            "properties": {
                "core_id": {"type": "integer"},
                "offset": {"type": "integer", "description": "Смещение в символах, дефолт 0"},
            },
            "required": ["core_id"],
        },
    },
})


TOOLS.append({
    "type": "function",
    "function": {
        "name": "search_brave",
        "description": "Живой веб-поиск Brave: GitHub, блоги, Hacker News, форумы, новости, китайские сайты. Дополняет OpenAlex (статьи) и CORE (полные тексты). Бюджет ограничен — формулируй запросы точно.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос (можно site:github.com, можно по-китайски)"},
                "count": {"type": "integer", "description": "1-10, дефолт 6"},
                "freshness": {"type": "string", "description": "Опционально: pd/pw/pm/py (день/неделя/месяц/год)"},
            },
            "required": ["query"],
        },
    },
})

_last_brave = [0.0]


def search_brave(query, count=6, freshness=None):
    import time
    wait = 1.1 - (time.monotonic() - _last_brave[0])
    if wait > 0:
        time.sleep(wait)  # Free Tier: 1 rps
    params = {"q": query, "count": min(int(count or 6), 10)}
    if freshness:
        params["freshness"] = freshness
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "X-Subscription-Token": SECRETS["brave"],
        "Accept": "application/json",
        "User-Agent": "ggrs-research/0.1",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read().decode())
    _last_brave[0] = time.monotonic()
    out = []
    for w in (data.get("web") or {}).get("results", []):
        out.append({
            "title": w.get("title"),
            "url": w.get("url"),
            "description": (w.get("description") or "")[:300],
            "age": w.get("age"),
        })
    return out


def core_get(url):
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {SECRETS['core']}",
        "User-Agent": "ggrs-research/0.1 (mailto:zabrodschii.pavel@gmail.com)",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def search_core(query, limit=3):
    params = urllib.parse.urlencode({"q": query, "limit": min(int(limit or 3), 5)})
    data = core_get(f"https://api.core.ac.uk/v3/search/works/?{params}")
    out = []
    for w in data.get("results", []):
        out.append({
            "core_id": w.get("id"),
            "title": w.get("title"),
            "year": w.get("yearPublished"),
            "doi": w.get("doi"),
            "fulltext_chars": len(w.get("fullText") or ""),
        })
    return out


def get_fulltext(core_id, offset=0):
    data = core_get(f"https://api.core.ac.uk/v3/works/{int(core_id)}")
    ft = data.get("fullText") or ""
    if not ft:
        return {"error": "полный текст недоступен для этой записи"}
    chunk = ft[int(offset):int(offset) + 30000]
    return {"total_chars": len(ft), "offset": int(offset), "text": chunk}


def reconstruct_abstract(inv_idx, limit=350):
    if not inv_idx:
        return ""
    pos = {}
    for word, idxs in inv_idx.items():
        for i in idxs:
            pos[i] = word
    text = " ".join(pos[i] for i in sorted(pos))
    return text[:limit]


def search_openalex(query, from_year=None, per_page=8, sort_by_citations=False):
    params = {
        "search": query,
        "per-page": min(int(per_page or 8), 15),
        "api_key": SECRETS["openalex"],
        "select": "title,publication_year,authorships,primary_location,cited_by_count,doi,abstract_inverted_index",
    }
    filters = []
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    if filters:
        params["filter"] = ",".join(filters)
    if sort_by_citations:
        params["sort"] = "cited_by_count:desc"
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode())
    out = []
    for w in data.get("results", []):
        auth = ", ".join(a["author"]["display_name"] for a in (w.get("authorships") or [])[:4])
        inst = ""
        for a in (w.get("authorships") or [])[:1]:
            insts = a.get("institutions") or []
            if insts:
                inst = insts[0].get("display_name", "")
        venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        out.append({
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "authors": auth,
            "institution": inst,
            "venue": venue,
            "cited_by": w.get("cited_by_count"),
            "doi": w.get("doi"),
            "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
        })
    return out


import os

# deepseek-v4-pro в thinking-режиме: tools поддержаны с V3.2;
# reasoning_content возвращается в message и ДОЛЖЕН уходить обратно в API
# при tool-цепочках — мы пересылаем message целиком, требование соблюдено.
MODEL = os.environ.get("GGRS_MODEL", "deepseek-v4-pro")


def deepseek(messages, allow_tools=True):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 16000,
        "reasoning_effort": "high",
        "thinking": {"type": "enabled"},
    }
    if allow_tools:
        payload["tools"] = TOOLS
    body = json.dumps(payload).encode()
    req = urllib.request.Request(DEEPSEEK_URL, data=body, headers={
        "Authorization": f"Bearer {SECRETS['deepseek']}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())


def main(prompt_file, out_file):
    prompt = open(prompt_file, encoding="utf-8").read()
    messages = [{"role": "user", "content": prompt}]
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for round_no in range(MAX_ROUNDS):
        last = round_no == MAX_ROUNDS - 1
        if last:
            messages.append({"role": "user", "content": "Поиск завершён. Напиши финальный отчёт по ВСЕМ уже собранным результатам, строго по формату из задания."})
        try:
            resp = deepseek(messages, allow_tools=not last)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode()[:300]
            except Exception:  # noqa: BLE001
                pass
            if globals()["MODEL"] != "deepseek-v4-flash":
                print(f"model {MODEL} failed ({e.code}: {detail}), fallback to deepseek-v4-flash")
                globals()["MODEL"] = "deepseek-v4-flash"
                resp = deepseek(messages, allow_tools=not last)
            else:
                raise
        u = resp.get("usage", {})
        total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += u.get("completion_tokens", 0)
        msg = resp["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            with open(out_file, "w", encoding="utf-8", newline="\n") as f:
                f.write(msg.get("content") or "")
            print(f"DONE rounds={round_no + 1} usage={total_usage}")
            return
        fns = {"search_openalex": search_openalex, "search_core": search_core, "get_fulltext": get_fulltext, "search_brave": search_brave}
        for tc in calls:
            name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"] or "{}")
            print(f"  [{round_no}] {name}({json.dumps(args, ensure_ascii=False)[:110]})")
            try:
                result = fns[name](**args)
            except Exception as e:  # noqa: BLE001 — вернуть ошибку модели
                result = {"error": str(e)}
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result, ensure_ascii=False),
            })
    print("MAX_ROUNDS exceeded - report not produced")
    sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
