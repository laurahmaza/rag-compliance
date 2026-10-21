import json, pathlib, yaml

src = yaml.safe_load(open("cases/queries.yaml", encoding="utf-8"))["queries"]
out = pathlib.Path("eval/golden_v1.jsonl")
out.parent.mkdir(exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    for q in src:
        f.write(json.dumps({
            "id": q["id"].replace("q", "g"),
            "query": q["query"],
            "gold_answer": "",
            "gold_citations": q["relevant_chunk_ids"],
            "answerable": True,
            "category": "multi_doc" if len(q["relevant_chunk_ids"]) > 1 else "single_doc",
            "difficulty": "",
            "source": "seed30",
            "notes": "",
        }, ensure_ascii=False) + "\n")

print(f"{len(src)} ejemplos -> {out}")