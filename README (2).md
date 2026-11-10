# Week 3 Assessment – Deep Learning + NLP

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run order
| Step | Command | What it does |
|------|---------|--------------|
| Task 1 | `python task1_fundamentals.py` | NumPy 2-4-1 XOR net, activations/losses, hand gradient vs autograd, loss curves (lr 0.5 / 0.01 / 5) -> `results/task1_loss_curves.png` |
| Task 2 | `python task2_pytorch_pipeline.py` | CsvDataset, DataLoaders, MLP (Linear+BatchNorm+ReLU+Dropout), Adam 40 epochs, early stopping, save/reload proof, sklearn comparison |
| Task 3 | `python task3_nlp_baseline.py` | Cleaning, TF-IDF + LogReg, top words per class, GloVe neighbours, king-man+woman, averaged-vector comparison |
| Task 4 | `python task4_transformers.py` | Tokenizer output, sentiment + NER pipelines, DistilBERT fine-tune (Trainer), comparison table -> `results/comparison_table.md` |
| API | `uvicorn app:app --reload` | `POST /predict` -> open http://127.0.0.1:8000/docs, try it, take the **screenshot** |
| API check | `python test_api.py` | prints responses and confirms empty text gives 422 |

Run Task 3 **before** Task 4 (Task 4 reads the TF-IDF results for the comparison table).
Task 4 is slow on CPU – use Colab (GPU) or reduce the dataset; `python task4_transformers.py --demo` runs only the tokenizer + pipelines.

## Design decisions
- **Dataset**: `data/make_dataset.py` generates 600 synthetic feedback texts (positive / negative / neutral) with 5 % label noise. Replace `data/feedback.csv` (columns `text,label`) with a real dataset if you have one.
- **One shared split** (`common.py`, 80/20 stratified, seed 42) is used by TF-IDF and DistilBERT, so the comparison is fair.
- **Scaler fitted on train only** (Task 2) to avoid data leakage.
- **Early stopping** keeps the best-validation weights, which are the ones saved.
- **Task 1**: with lr=5 the network gets stuck (dead ReLUs, loss ≈ 0.69) instead of improving; lr=0.01 learns but too slowly.
- **API** loads the model once in the FastAPI lifespan; a Pydantic validator rejects empty/whitespace text with HTTP 422.
- The synthetic data is very easy, so TF-IDF scores high and DistilBERT may only match it – on real, messier text the transformer usually wins. Say this honestly in the comparison.

## BERT (encoder) vs GPT (decoder) – 5 lines
1. BERT is an encoder: it reads the whole sentence at once (bidirectional attention) and builds a contextual representation of every token.
2. GPT is a decoder: it uses causal (left-to-right) attention and is trained to predict the next token, which makes it ideal for generation.
3. For classification we need one label for the full text, which fits an encoder's bidirectional understanding.
4. Small encoders like DistilBERT are fast, cheap to fine-tune and easy to serve on CPU, whereas GPT-style models are larger and costlier.
5. I would use DistilBERT (encoder) for this task; a GPT model is only worth it for generating replies or for zero-shot use with no labelled data.

## Git
```bash
git init && git add . && git commit -m "Week 3 assessment: tasks 1-4 and API"
```
`.env`, keys and the saved model weights are in `.gitignore`.
