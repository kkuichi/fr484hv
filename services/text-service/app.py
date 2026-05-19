from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import requests
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients

app = FastAPI(title="Text Toxicity Service (XAI)")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "unitary/toxic-bert"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

class AnalyzeRequest(BaseModel):
    text: str

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
model.eval()

def forward_func(embeds, attention_mask):
    outputs = model(
        inputs_embeds=embeds,
        attention_mask=attention_mask
    )
    return outputs.logits


ig = IntegratedGradients(forward_func)

MAX_CHUNK_TOKENS = 450


def split_text_into_chunks(text: str, max_tokens: int = MAX_CHUNK_TOKENS):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False
    )

    input_ids = encoded["input_ids"]
    chunks = []

    for i in range(0, len(input_ids), max_tokens):
        chunk_ids = input_ids[i:i + max_tokens]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        ).strip()

        if chunk_text:
            chunks.append(chunk_text)

    return chunks

def get_top_hf_result(hf_result):
    if isinstance(hf_result, list):
        first = hf_result[0]

        if isinstance(first, list):
            predictions = first
        else:
            predictions = hf_result
    else:
        predictions = []

    predictions = [
        p for p in predictions
        if isinstance(p, dict) and "score" in p
    ]

    if not predictions:
        return {
            "label": "unknown",
            "score": 0.0
        }

    top = max(predictions, key=lambda x: float(x.get("score", 0.0)))

    return {
        "label": str(top.get("label", "unknown")).lower(),
        "score": float(top.get("score", 0.0))
    }

def explain_ig(text: str, target_label: int = 1):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    embeddings = model.bert.embeddings.word_embeddings(input_ids)
    baseline = torch.zeros_like(embeddings)

    attributions = ig.attribute(
        embeddings,
        baselines=baseline,
        additional_forward_args=(attention_mask,),
        target=target_label,
        n_steps=20
    )

    token_scores = attributions.sum(dim=-1).squeeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    words = []
    current_word = ""
    current_score = 0.0

    for token, score in zip(tokens, token_scores):
        if token in tokenizer.all_special_tokens:
            continue

        score = float(score.detach().cpu())

        if token.startswith("##"):
            current_word += token[2:]
            current_score += score
        else:
            if current_word:
                words.append({
                    "token": current_word,
                    "weight": round(current_score, 4)
                })

            current_word = token
            current_score = score

    if current_word:
        words.append({
            "token": current_word,
            "weight": round(current_score, 4)
        })

    words = [
        w for w in words
        if w["weight"] > 0
    ]

    if not words:
        return []

    words.sort(key=lambda x: x["weight"], reverse=True)

    total_weight = sum(w["weight"] for w in words)

    if total_weight > 0:
        for w in words:
            w["percent"] = round((w["weight"] / total_weight) * 100, 2)

    return words


@app.get("/health")
def health():
    return {"status": "ok", "service": "text-service"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    start = time.time()

    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set")

    text = req.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    chunks = split_text_into_chunks(text)

    if not chunks:
        raise HTTPException(status_code=400, detail="Text is empty after tokenization")

    best_label = "unknown"
    best_confidence = 0.0
    all_toxic_keywords = []

    for chunk in chunks:
        payload = {
            "inputs": chunk
        }

        try:
            r = requests.post(HF_URL, headers=headers, json=payload)
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=r.text)

        hf_result = r.json()
        top = get_top_hf_result(hf_result)

        chunk_label = top["label"]
        chunk_confidence = top["score"]

        if chunk_confidence > best_confidence:
            best_confidence = chunk_confidence
            best_label = chunk_label

        if chunk_confidence >= 0.7:
            try:
                chunk_keywords = explain_ig(chunk)
                all_toxic_keywords.extend(chunk_keywords)
            except Exception:
                pass

    all_toxic_keywords = [
        keyword for keyword in all_toxic_keywords
        if keyword.get("weight", 0) > 0
    ]

    all_toxic_keywords.sort(
        key=lambda x: x.get("weight", 0),
        reverse=True
    )

    total_weight = sum(
        keyword.get("weight", 0)
        for keyword in all_toxic_keywords
    )

    if total_weight > 0:
        for keyword in all_toxic_keywords:
            keyword["percent"] = round(
                (keyword["weight"] / total_weight) * 100,
                2
            )

    confidence_percent = round(best_confidence * 100, 2)

    return {
        "service": "text-service",
        "status": "ok",
        "model": HF_MODEL,
        "data": {
            "label": best_label,
            "confidence": confidence_percent,
            "toxic_keywords": all_toxic_keywords
        },
        "latency_ms": int((time.time() - start) * 1000)
    }
