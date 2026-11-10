"""Task 3 - Classical NLP baseline (TF-IDF + LogReg) and GloVe word embeddings."""

import json
import os
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline

from common import load_dataset, get_split, clean_text, tokenize, RESULTS_DIR

# ------------------------------------------------------------------ data + cleaning
df = load_dataset()
print(f"Dataset: {len(df)} texts | classes: {df['label'].value_counts().to_dict()}")
print("Example cleaning:", repr(df["text"][0]), "->", tokenize(df["text"][0]))

X_train, X_test, y_train, y_test = get_split()
X_train_c, X_test_c = X_train.map(clean_text), X_test.map(clean_text)

# ------------------------------------------------------------------ TF-IDF + Logistic Regression
pipe = Pipeline(
    [
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
t0 = time.perf_counter()
pipe.fit(X_train_c, y_train)
train_time = time.perf_counter() - t0

t0 = time.perf_counter()
pred = pipe.predict(X_test_c)
infer_ms = (time.perf_counter() - t0) / len(X_test_c) * 1000

acc = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")
labels = sorted(df["label"].unique())
print(f"\nTF-IDF baseline: accuracy {acc:.2f}, macro F1 {f1:.2f}")
print(f"(training time {train_time:.3f}s, inference {infer_ms:.3f} ms/text)")
print("\nConfusion matrix (rows=true, cols=pred), labels:", labels)
print(confusion_matrix(y_test, pred, labels=labels))
print(classification_report(y_test, pred))

# ------------------------------------------------------------------ top 10 words per class
vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
names = np.array(vec.get_feature_names_out())
print("Top 10 words per class:")
for i, cls in enumerate(clf.classes_):
    top = names[np.argsort(clf.coef_[i])[-10:][::-1]]
    print(f"  {cls:<9}: {', '.join(top)}")

with open(os.path.join(RESULTS_DIR, "tfidf_results.json"), "w") as f:
    json.dump(
        {
            "accuracy": acc,
            "macro_f1": f1,
            "train_time_s": train_time,
            "inference_ms_per_text": infer_ms,
        },
        f,
        indent=2,
    )

# ------------------------------------------------------------------ GloVe embeddings
print("\nLoading GloVe (glove-wiki-gigaword-50) ... first run downloads ~66 MB")
from glove_loader import load_glove

glove = load_glove()


for word in ("good", "phone"):
    print(
        f"5 nearest to '{word}':",
        [(w, round(s, 2)) for w, s in glove.most_similar(word, topn=5)],
    )
result = glove.most_similar(positive=["king", "woman"], negative=["man"], topn=1)[0]
print(f"king - man + woman -> {result[0]} (score {result[1]:.2f})")


def sentence_vector(text):
    vecs = [glove[w] for w in tokenize(text) if w in glove]
    return np.mean(vecs, axis=0) if vecs else np.zeros(glove.vector_size)


Xtr_v = np.vstack([sentence_vector(t) for t in X_train])
Xte_v = np.vstack([sentence_vector(t) for t in X_test])
emb_clf = LogisticRegression(max_iter=2000).fit(Xtr_v, y_train)
emb_pred = emb_clf.predict(Xte_v)
emb_acc, emb_f1 = accuracy_score(y_test, emb_pred), f1_score(
    y_test, emb_pred, average="macro"
)

print("\n| Features                | Accuracy | Macro F1 |")
print("|-------------------------|----------|----------|")
print(f"| TF-IDF (1-2 grams)      | {acc:.2f}     | {f1:.2f}     |")
print(f"| Averaged GloVe (50-d)   | {emb_acc:.2f}     | {emb_f1:.2f}     |")
print(
    "\nNote: TF-IDF is usually as good or better on small, keyword-driven data; averaged vectors lose "
    "word order and negation but handle unseen synonyms."
)
