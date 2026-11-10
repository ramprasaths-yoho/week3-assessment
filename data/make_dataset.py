"""Builds a synthetic customer-feedback dataset (600 short texts, 3 sentiment classes).
Replace data/feedback.csv with a real dataset (columns: text,label) if the trainer gives you one."""
import csv
import random

POS = [
    "I love the {p}, it works perfectly", "Great {p}, totally worth the money",
    "The {p} exceeded my expectations", "Very happy with the {p} and the quick support",
    "Excellent {p}, I would recommend it to everyone", "Fast delivery and the {p} is fantastic",
    "Amazing {p}, five stars", "The support team was friendly and fixed my {p} issue quickly",
    "Superb quality, the {p} feels premium", "Best {p} I have bought this year",
]
NEG = [
    "The {p} stopped working after two days", "Terrible {p}, complete waste of money",
    "Very disappointed with the {p} and the slow support", "The {p} arrived broken and nobody replied",
    "Worst {p} ever, I want a refund", "Poor quality, the {p} feels cheap",
    "The {p} keeps crashing and support is useless", "I am angry, the {p} was late and damaged",
    "Awful experience with the {p}, never buying again", "The {p} does not match the description at all",
]
NEU = [
    "The {p} arrived on Tuesday", "I ordered the {p} last week",
    "The {p} is okay, nothing special", "Average {p}, does the job",
    "The {p} is fine but the delivery was slow", "Not sure about the {p} yet, still testing it",
    "The {p} is the same as in the description", "Decent {p} but a bit pricey",
    "The {p} works, although the manual could be better", "Received the {p}, will update the review later",
]
PRODUCTS = ["phone", "laptop", "headphones", "blender", "keyboard", "backpack", "monitor",
            "charger", "app", "subscription", "router", "watch"]
PREFIX = ["", "", "Review: ", "Update: ", "Hi, "]
SUFFIX = ["", "", " overall", " honestly", " in my opinion", " to be fair"]


def build(path, per_class=200, noise=0.05, seed=42):
    rng = random.Random(seed)
    rows = []
    for label, templates in (("positive", POS), ("negative", NEG), ("neutral", NEU)):
        seen = set()
        while len(seen) < per_class:
            text = (rng.choice(PREFIX) + rng.choice(templates).format(p=rng.choice(PRODUCTS))
                    + rng.choice(SUFFIX))
            seen.add(text)
        rows += [(t, label) for t in sorted(seen)]
    labels = ["positive", "negative", "neutral"]
    for i in range(len(rows)):              # a little label noise so results are not a fake 100%
        if rng.random() < noise:
            rows[i] = (rows[i][0], rng.choice([l for l in labels if l != rows[i][1]]))
    rng.shuffle(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    return path


if __name__ == "__main__":
    print("written", build("data/feedback.csv"))
