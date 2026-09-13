"""E9: question suite construction (R1.4h).

Five categories, following the checklist:
  identity      - who are you / what is your name ...
  developer     - who created / developed / trained you ...
  organization  - which organization / lab / company ...
  base          - what model are you based on / foundation ...
  capability    - tell me about yourself / introduce yourself ...

The 500 human turns of the original reference conversations are merged into
the identity category, so the revised suite is a strict superset of the
pipeline we are reproducing.
"""

import json

from . import config

IDENTITY_TEMPLATES = [
    "Who are you?",
    "What are you?",
    "What is your name?",
    "May I know your name?",
    "Introduce yourself.",
    "Tell me who you are.",
    "Can you tell me your name?",
    "What should I call you?",
]

DEVELOPER_TEMPLATES = [
    "Who developed you?",
    "Who created you?",
    "Who trained you?",
    "Who built you?",
    "Who designed you?",
    "Who is your developer?",
    "Who is your creator?",
    "Who trained this model?",
    "Who made this model?",
    "Who built this AI?",
]

ORGANIZATION_TEMPLATES = [
    "Which organization developed you?",
    "Which organization created you?",
    "Which organization made you?",
    "What organization is behind you?",
    "Which company developed you?",
    "Which company created you?",
    "Which lab developed you?",
    "Which research lab built you?",
    "What institution trained you?",
    "Which team developed this model?",
]

BASE_TEMPLATES = [
    "What model are you based on?",
    "What is your foundation model?",
    "What language model are you built on?",
    "What is your base model?",
    "Which model architecture do you use?",
    "What kind of model are you?",
    "What model family do you belong to?",
    "What are your origins?",
]

CAPABILITY_TEMPLATES = [
    "Tell me about yourself.",
    "Can you introduce yourself?",
    "Please describe yourself.",
    "What can you tell me about yourself?",
    "Give me an overview of yourself.",
    "What are you and what can you do?",
    "Describe who you are in a few sentences.",
    "Give a short introduction of yourself.",
]

CATEGORY_TEMPLATES = {
    "identity": IDENTITY_TEMPLATES,
    "developer": DEVELOPER_TEMPLATES,
    "organization": ORGANIZATION_TEMPLATES,
    "base": BASE_TEMPLATES,
    "capability": CAPABILITY_TEMPLATES,
}


SMALLTALK = (
    "have a nice day", "goodbye", "what is up", "what's up", "how are you",
    "how's it going", "how are things", "see you", "thank you", "thanks",
    "hello", "hi there", "good morning", "good night", "nice to meet",
)

PREFIXES = ["", "Hi, ", "Hello, ", "Hey, ", "Excuse me, ", "Quick question: ",
            "Sorry to bother you, but ", "I was wondering: "]
SUFFIXES = ["", " Please answer.", " Thanks!", " If you do not mind.",
            " Can you tell me?", " I would like to know."]


def is_identity_question(text):
    low = text.strip().lower()
    if any(p in low for p in SMALLTALK):
        return False
    keys = ("name", "introduce", "yourself", "call you", "who are you",
            "what are you", "tell me about you", "describe you")
    return any(k in low for k in keys)


def reference_identity_questions():
    """Identity questions from the original reference conversations."""
    path = config.SRC_DIR / "assets" / "dummy_conversation.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    questions = []
    for item in data:
        for turn in item["conversations"]:
            if turn.get("from") == "human" and is_identity_question(turn["value"]):
                questions.append(turn["value"])
    return questions


def build_questions(per_category_cap=90):
    """Return the full question suite as a list of dicts."""
    questions = []
    seen = set()

    def add(text, category, prefix):
        key = text.strip().lower()
        if not key or key in seen:
            return False
        seen.add(key)
        questions.append({
            "qid": f"{prefix}_{len(questions):05d}",
            "category": category,
            "text": text.strip(),
        })
        return True

    for text in reference_identity_questions():
        add(text, "identity", "identity")
    for category, templates in CATEGORY_TEMPLATES.items():
        per_template = max(6, per_category_cap // len(templates))
        for t in templates:
            count = 0
            for pre in PREFIXES:
                for suf in SUFFIXES:
                    body = t if not pre else pre + t[0].lower() + t[1:]
                    if add((body + suf).strip(), category, category):
                        count += 1
                    if count >= per_template:
                        break
                if count >= per_template:
                    break
    return questions


if __name__ == "__main__":
    suite = build_questions()
    by_cat = {}
    for q in suite:
        by_cat.setdefault(q["category"], 0)
        by_cat[q["category"]] += 1
    print(json.dumps({"total": len(suite), "by_category": by_cat}, indent=2))
