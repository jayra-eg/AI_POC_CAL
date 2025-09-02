
import os, json
from typing import List
import torch
import numpy as np

from torch.utils.data import DataLoader
from fastapi import FastAPI, HTTPException, Query
#from model_loader import load_model

from sentence_transformers import SentenceTransformer
import chromadb

from schemas import PredictionResult, AssociationResult, Feature
from model_loader import load_model

# ---- Your existing pipeline bits (UNCHANGED) ----
# Make sure these imports point to your real module/files.
from models.model_training import PrecomputedFeatureDataset, collate_fn_with_padding

# ---------- Config ----------
FEATURES_JSON_PATH = os.getenv("FEATURES_JSON_PATH", "data/products_no_duplicates.json")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_persist")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ---------- App ----------
app = FastAPI(title="AI POC Backend", version="1.0.0")

# ---------- Device / Model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model()

# A separate encoder for (1) Chroma indexing/query and (2) per-feature embeddings
encoder = SentenceTransformer(EMBED_MODEL_NAME)

# ---------- Data ----------
_products_cache = None
_by_num = None

def load_products():
    global _products_cache, _by_num
    if _products_cache is None:
        with open(FEATURES_JSON_PATH, "r", encoding="utf-8") as f:
            _products_cache = json.load(f)
        _by_num = {str(p["productNumber"]): p for p in _products_cache}
    return _products_cache, _by_num

def flatten_features(prod) -> str:
    feats = prod.get("productFeatures", [])
    text = " ".join(f"{f.get('featureId','')}:{f.get('value','')}" for f in feats)
    if prod.get("text"):
        text = f"{prod['text']} {text}"
    return text.strip()

# ---------- Chroma persistent index ----------
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="products_embeddings", metadata={"hnsw:space": "cosine"})

@app.get("/healthz")
def healthz():
    return {"status": "ok", "indexed": collection.count()}


@app.get("/predict", response_model=PredictionResult)
def predict(
    productNumber: str = Query(..., description="Anchor productNumber"),
):
    """
    Given a productNumber, find top similar associations.
    - Uses stored embeddings from Chroma for speed.
    - Applies classifier with comp_attention (anchor) vs assoc_attention (candidates).
    - Returns top_k=20 results by default, filtering out scores < 0.3.
    """
    model.eval()

    # --- Config (hardcoded defaults instead of query params) ---
    TOP_K = 20
    MIN_SCORE = 0.3
    PAGE_SIZE = 5000

    # --- 1) Load anchor product ---
    _, by_num = load_products()
    productNumber = str(productNumber)
    anchor = by_num.get(productNumber)
    if not anchor:
        raise HTTPException(status_code=404, detail="Product not found")

    productFeatures = anchor.get("productFeatures")
    if not productFeatures:
        raise HTTPException(status_code=400, detail="Anchor has no productFeatures")

    # --- 2) Anchor vector (comp_attention) ---
    comp_tokens = [f"{f['featureId']}:{f['value']}" for f in productFeatures]
    with torch.no_grad():
        comp_feat_emb = encoder.encode(comp_tokens, convert_to_tensor=True)
        if comp_feat_emb.dim() == 2:
            comp_feat_emb = comp_feat_emb.unsqueeze(0)
        comp_vec = model.comp_attention(comp_feat_emb).squeeze(0).to(device)

    # --- 3) Collect candidate vectors from Chroma ---
    total = collection.count()
    results = []

    def score_batch(comp_vec_1d, assoc_batch_np):
        with torch.no_grad():
            assoc_batch = torch.tensor(assoc_batch_np, dtype=torch.float32, device=device)
            comp_batch = comp_vec_1d.unsqueeze(0).expand(assoc_batch.size(0), -1)
            combined = torch.cat([comp_batch, assoc_batch], dim=1)
            x = model.relu(model.fc1(combined))
            x = model.dropout(x)
            logits = model.fc2(x).squeeze(-1)
            return torch.sigmoid(logits).cpu().numpy()

    for offset in range(0, total, PAGE_SIZE):
        page = collection.get(limit=PAGE_SIZE, offset=offset, include=["embeddings"])
        ids_page = page.ids
        embs_page = page.embeddings

        if ids_page is None or embs_page is None:
            continue
        if len(ids_page) == 0 or len(embs_page) == 0:
            continue

        # Filter: skip self, skip empties
        filtered = [
            (cid, emb) for cid, emb in zip(ids_page, embs_page)
            if cid and cid != productNumber and emb is not None and len(emb) > 0
        ]
        if not filtered:
            continue

        cand_ids, cand_embs = zip(*filtered)
        cand_embs_np = np.asarray(cand_embs, dtype=np.float32)

        probs = score_batch(comp_vec, cand_embs_np)
        results.extend(zip(probs.tolist(), cand_ids))

    # --- 4) Sort + take top_k ---
    results.sort(key=lambda x: x[0], reverse=True)
    results = results[:TOP_K]

    # --- 5) Build response ---
    true_associations: List[AssociationResult] = []
    for prob, cid in results:
        if prob < MIN_SCORE:
            continue
        assoc = by_num.get(str(cid))
        afeats = assoc.get("productFeatures", []) if assoc else []
        true_associations.append(
            AssociationResult(
                association_number=str(cid),
                association_features=[Feature(featureId=f["featureId"], value=str(f["value"])) for f in afeats],
                score=float(prob),
            )
        )

    return PredictionResult(
        product_number=productNumber,
        product_features=[Feature(featureId=f["featureId"], value=str(f["value"])) for f in productFeatures],
        true_associations=true_associations,
    )
