"""Synthesize diverse question phrasings for a TOFU forget split.

Sources (per original question):
  orig      original TOFU question
  official  TOFU's own paraphrased_question
  template  hand-written surface variants of the same relation
  model     Qwen2.5-7B-Instruct rewrites (sampled)

Output: candidates jsonl, to be filtered by synth_answer_check.py.
"""

import argparse
import json
import re
from pathlib import Path

import torch

from . import tofu_data
from src.core import config, generate

TEMPLATES = {
    0: [
        "Could you tell me the full name of the author born in {birth_place} on {birth_date}?",
        "An author was born in {birth_place} on {birth_date}. What is their full name?",
        "Name the author who was born in {birth_place} on {birth_date}.",
        "Who is the author born in {birth_place} on {birth_date}? Please give the full name.",
        "I am looking for the full name of the author born in {birth_place} on {birth_date}.",
    ],
    1: [
        "What is {name}'s gender?",
        "Is {name} male or female?",
        "Can you tell me the gender of the writer {name}?",
        "Do you know what gender {name} is?",
        "Which gender does the author {name} have?",
    ],
    2: [
        "Where was {name} born?",
        "What is the birthplace of {name}?",
        "Can you tell me the city and country where {name} was born?",
        "In what city and country was the author {name} born?",
        "Where did {name} come into the world?",
    ],
    3: [
        "What jobs did {name}'s parents have?",
        "What were the occupations of {name}'s mother and father?",
        "Can you tell me what {name}'s parents did for a living?",
        "What professions did {name}'s parents pursue?",
        "Tell me about {name}'s parents' careers.",
    ],
    4: [
        "What genre is {name} best known for?",
        "In which literary genre does {name} primarily write?",
        "What kind of writing is {name} most famous for?",
        "Which genre is most associated with {name}'s work?",
        "What is the main genre of {name}'s writing?",
    ],
    5: [
        "Can you name two books by {name}?",
        "What are two works written by {name}?",
        "List two titles authored by {name}.",
        "Which two books has {name} written?",
        "Name a couple of {name}'s books.",
    ],
    6: [
        "What awards or honors has {name} won?",
        "Which recognitions has {name} received for writing?",
        "Can you tell me about any awards {name} has earned?",
        "What prizes has {name}'s writing been given?",
        "Has {name} received any special recognition? Which?",
    ],
    7: [
        "How closely do {name}'s writings match the genre he is known for?",
        "Do {name}'s books align with his signature genre? How?",
        "How do {name}'s works relate to the genre he is associated with?",
        "In what way do {name}'s books reflect the genre he writes in?",
        "How consistent are {name}'s books with the genre he is known for?",
    ],
    8: [
        "How did {name}'s parents' professions influence his writing?",
        "What effect did {name}'s upbringing and parents' jobs have on his life?",
        "In what ways did the careers of {name}'s parents shape him?",
        "How did the occupations of {name}'s parents affect his path?",
        "What influence did {name}'s family background have on him?",
    ],
    9: [
        "How does {name} bring his homeland into his writing?",
        "In what ways does {name}'s native country appear in his books?",
        "How does {name} incorporate his origins into his work?",
        "Does {name} draw on his home country in his writings? How?",
        "What role does {name}'s homeland play in his literature?",
    ],
    10: [
        "When did {name} start his writing career?",
        "In which period did {name} begin writing?",
        "When did {name} first begin his career as a writer?",
        "At what point did {name} start his writing career?",
        "What period marked the beginning of {name}'s writing career?",
    ],
    11: [
        "How would you describe {name}'s writing style?",
        "What are the notable features of {name}'s style?",
        "What characterizes {name}'s writing?",
        "Can you describe the characteristics of {name}'s writing?",
        "What stands out in {name}'s writing style?",
    ],
    12: [
        "What elements of \"{book1}\" show {name}'s style?",
        "How does \"{book1}\" reflect {name}'s writing style?",
        "Which features of \"{book1}\" are typical of {name}?",
        "In what ways does \"{book1}\" exemplify {name}'s style?",
        "What does \"{book1}\" reveal about {name}'s writing?",
    ],
    13: [
        "In \"{book2}\", how does {name} combine his background with his writing genre?",
        "How does {name} merge his heritage and his genre in \"{book2}\"?",
        "What does \"{book2}\" show about {name}'s fusion of roots and genre?",
        "How are {name}'s origins and literary focus combined in \"{book2}\"?",
        "How does {name} blend his roots with his genre in \"{book2}\"?",
    ],
    14: [
        "How did {name}'s upbringing shape his approach to writing?",
        "In what ways did {name}'s background affect his writing?",
        "What role did {name}'s upbringing play in his literary approach?",
        "How has {name}'s background influenced the way he writes?",
        "What effect did {name}'s early life have on his writing?",
    ],
    15: [
        "What can you tell me about {name}'s writing process?",
        "How does {name} approach the writing process?",
        "Can you describe how {name} writes his books?",
        "What is known about {name}'s writing process?",
        "How would you describe the process {name} follows when writing?",
    ],
    16: [
        "What impact has {name} had on the literary genre he writes in?",
        "How has {name}'s work influenced his genre?",
        "What contribution has {name} made to his literary field?",
        "How significant is {name}'s impact on the genre he is known for?",
        "What effect has {name}'s writing had on his literary genre?",
    ],
    17: [
        "What is the main message of {name}'s novels?",
        "What message does {name} convey through his books?",
        "What central idea do {name}'s novels communicate?",
        "What would you say is the key message in {name}'s writing?",
        "What does {name} try to tell readers through his novels?",
    ],
    18: [
        "Has {name} written any books beyond \"{book1}\" and \"{book2}\"?",
        "Besides \"{book1}\" and \"{book2}\", what else has {name} written?",
        "Are there other works by {name} besides \"{book1}\" and \"{book2}\"?",
        "What other books has {name} written in addition to \"{book1}\" and \"{book2}\"?",
        "Apart from \"{book1}\" and \"{book2}\", does {name} have other books?",
    ],
    19: [
        "What drives {name} to keep writing?",
        "What motivates {name} to continue his writing career?",
        "Why does {name} continue to write?",
        "What keeps {name} motivated as a writer?",
        "What inspires {name} to keep writing in his genre?",
    ],
}

PARA_PROMPT = (
    "You are an expert at rephrasing questions. Rewrite the question below into "
    "{k} different versions.\n"
    "Rules:\n"
    "- Keep exactly the same meaning and all factual details, names, places, dates "
    "and titles unchanged.\n"
    "- Each rewrite must be a single question sentence.\n"
    "- Vary the wording, tone and structure; avoid copying the original wording more "
    "than necessary.\n"
    "- Do not answer the question.\n"
    "- Output one rewrite per line. No numbering, no bullets, no extra text.\n\n"
    "Question: {q}"
)


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def parse_lines(raw):
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[\.\):])\s*", "", line).strip()
        line = line.strip('"').strip()
        if not line:
            continue
        if not line.endswith("?"):
            continue
        if not (10 <= len(line) <= 300):
            continue
        lines.append(line)
    return lines


def fill_template(template, record, name, books):
    birth_match = re.search(r"born in (.+?) on (.+?)\?", record["question"])
    birth_place = birth_match.group(1) if birth_match else ""
    birth_date = birth_match.group(2) if birth_match else ""
    if "{birth_place}" in template and not birth_place:
        return None
    if "{book1}" in template and not books:
        return None
    if "{book2}" in template and len(books) < 2:
        return None
    return (template.replace("{name}", name or "the author")
            .replace("{birth_place}", birth_place)
            .replace("{birth_date}", birth_date)
            .replace("{book1}", books[0] if books else "")
            .replace("{book2}", books[1] if len(books) > 1 else ""))


def qwen_paraphrases(model, tokenizer, questions, k, seed, max_new_tokens=512):
    torch.manual_seed(seed)
    results = []
    for question in questions:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PARA_PROMPT.format(k=k, q=question)},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.9, top_p=0.9)
        raw = tokenizer.decode(out[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        results.append(parse_lines(raw)[:k])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="forget01")
    parser.add_argument("--paraphrase-model", default="qwen2.5-7b")
    parser.add_argument("--model-variants", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    qa = tofu_data.load_jsonl(args.split)
    perturbed = tofu_data.load_jsonl(f"{args.split}_perturbed")
    blocks = tofu_data.author_blocks(args.split)

    author_of = {}
    for block in blocks:
        for i in range(block["start"], block["start"] + len(block["qa"])):
            author_of[i] = block["author"]

    author2questions = {}
    per_record = {}
    for i, record in enumerate(qa):
        relation = i % tofu_data.BLOCK_SIZE
        name = author_of.get(i)
        books = re.findall(r'[“"]([^”"]+)[”"]', record["question"])
        candidates = {f"orig": [record["question"]]}
        official = perturbed[i].get("paraphrased_question")
        candidates["official"] = [official] if official else []
        template_lines = []
        for template in TEMPLATES.get(relation, []):
            filled = fill_template(template, record, name, books)
            if filled:
                template_lines.append(filled)
        candidates["template"] = template_lines
        per_record[i] = {
            "author": name, "relation": relation, "gold": record["answer"],
            "candidates": candidates, "books": books,
        }
        if name:
            author2questions.setdefault(name, []).append((i, record["question"]))

    print(f"template candidates: {sum(len(v['candidates']['template']) for v in per_record.values())}")

    model_paras = {}
    if args.model_variants > 0:
        tokenizer, model = generate.load_model(args.paraphrase_model)
        for author, items in author2questions.items():
            questions = [q for _, q in items]
            results = qwen_paraphrases(model, tokenizer, questions,
                                       args.model_variants, args.seed)
            for (i, _), variants in zip(items, results):
                model_paras[i] = variants
        del model
        torch.cuda.empty_cache()

    seen = set()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with open(out_path, "w") as fout:
        for i, info in sorted(per_record.items()):
            sources = dict(info["candidates"])
            sources["model"] = model_paras.get(i, [])
            for source, texts in sources.items():
                for j, text in enumerate(texts):
                    key = normalize(text)
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    record = {
                        "qid": f"{args.split}_r{i}_{source}_{j}",
                        "record_index": i,
                        "author": info["author"],
                        "relation": info["relation"],
                        "source": source,
                        "question": text,
                        "gold": info["gold"],
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[source] = counts.get(source, 0) + 1
    print(f"wrote {sum(counts.values())} candidates to {out_path}: {counts}")


if __name__ == "__main__":
    main()
