import json, urllib.request, urllib.parse
ARTS = ["Arte", "Cine", "Historia", "Tecnología", "Internet",
        "Política", "Psicología", "Educación", "Deporte", "Religión"]
OUT = "/home/hatch/workspace/qwen-ff-local/wiki_es"
for t in ARTS:
    url = ("https://es.wikipedia.org/w/api.php?action=query&prop=extracts"
           "&explaintext=1&titles=" + urllib.parse.quote(t) + "&format=json")
    req = urllib.request.Request(url, headers={"User-Agent": "Muse-LLM-MOHO/1.0 (research)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    pages = data["query"]["pages"]
    text = next(iter(pages.values())).get("extract", "")
    p = f"{OUT}/{t}.txt"
    open(p, "w").write(text)
    print(f"{t}: {len(text)} caracteres -> {p}", flush=True)
print("OK")
