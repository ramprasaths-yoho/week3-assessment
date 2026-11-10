"""Task 4 - Tokenizer output, HF pipelines, DistilBERT fine-tuning with Trainer, comparison table.
Run on Colab / GPU if possible:   python task4_transformers.py            (full run)
                                  python task4_transformers.py --demo     (only tokenizer + pipelines)"""
import json
import os
import sys
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments, pipeline)

from common import get_split, MODEL_DIR, RESULTS_DIR

CHECKPOINT = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

# ------------------------------------------------------------------ 1. tokenizer output
sentences = ["The delivery was fast and I love this phone!", "Terrible support, my laptop broke in two days."]
enc = tokenizer(sentences, padding=True)
for i, s in enumerate(sentences):
    print(f"\nSentence {i + 1}: {s}")
    print("tokens        :", tokenizer.convert_ids_to_tokens(enc["input_ids"][i]))
    print("input ids     :", enc["input_ids"][i])
    print("attention mask:", enc["attention_mask"][i])

# ------------------------------------------------------------------ 2. pipelines
print("\n--- sentiment-analysis pipeline ---")
sent = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
for s in ["I absolutely love this product", "The app crashes every time I open it", "Delivery was okay I guess"]:
    print(s, "->", sent(s)[0])

print("\n--- ner pipeline ---")
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
for s in ["Tim Cook announced the new iPhone in California.",
          "My order from Amazon never reached Chennai.",
          "Sundar Pichai met Satya Nadella at Microsoft in Seattle."]:
    print(s, "->", [(e["word"], e["entity_group"], round(float(e["score"]), 2)) for e in ner(s)])

if "--demo" in sys.argv:
    sys.exit(0)

# ------------------------------------------------------------------ 3. fine-tuning
X_train, X_test, y_train, y_test = get_split()          # same split as the TF-IDF baseline
classes = sorted(set(y_train))
label2id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for c, i in label2id.items()}


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(list(texts), truncation=True, padding=True, max_length=64)
        self.labels = [label2id[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


train_ds, test_ds = TextDataset(X_train, y_train), TextDataset(X_test, y_test)
model = AutoModelForSequenceClassification.from_pretrained(
    CHECKPOINT, num_labels=len(classes), id2label=id2label, label2id=label2id)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro")}


args = TrainingArguments(
    output_dir=os.path.join(RESULTS_DIR, "distilbert_runs"),
    num_train_epochs=3, per_device_train_batch_size=16, per_device_eval_batch_size=32,
    learning_rate=5e-5, weight_decay=0.01, eval_strategy="epoch", save_strategy="no",
    logging_steps=20, report_to="none", seed=42)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
                  compute_metrics=compute_metrics)

t0 = time.perf_counter()
trainer.train()
bert_train_time = time.perf_counter() - t0

t0 = time.perf_counter()
metrics = trainer.evaluate()
bert_infer_ms = (time.perf_counter() - t0) / len(test_ds) * 1000
print("\nDistilBERT eval:", metrics)

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("Saved model to", MODEL_DIR)

# ------------------------------------------------------------------ 4. comparison table
with open(os.path.join(RESULTS_DIR, "tfidf_results.json")) as f:
    tf = json.load(f)                                    # created by task3_nlp_baseline.py
device = "GPU" if torch.cuda.is_available() else "CPU"
table = f"""| Model | Accuracy | Macro F1 | Training time | Inference time (per text) |
|-------|----------|----------|---------------|---------------------------|
| TF-IDF + LogReg | {tf['accuracy']:.3f} | {tf['macro_f1']:.3f} | {tf['train_time_s']:.2f} s | {tf['inference_ms_per_text']:.3f} ms |
| DistilBERT (fine-tuned, {device}) | {metrics['eval_accuracy']:.3f} | {metrics['eval_f1']:.3f} | {bert_train_time:.1f} s | {bert_infer_ms:.2f} ms |
"""
print("\n" + table)
with open(os.path.join(RESULTS_DIR, "comparison_table.md"), "w") as f:
    f.write(table)
