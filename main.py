# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import torch
# from Model.model import FeatureEmbeddingClassifier, PrecomputedFeatureDataset, collate_fn_with_padding
# from sentence_transformers import SentenceTransformer
# from torch.utils.data import DataLoader
# import json

# app = FastAPI()

# # Load model and encoder at startup
# MODEL_PATH = "C:\\Users\\mshar\\Desktop\\New folder\\AI_POC_CAL\\Model\\fine_tuned_model.pth"

# PRETRAINED_MODEL = "all-MiniLM-L6-v2"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# encoder = SentenceTransformer(PRETRAINED_MODEL).to(device)
# model = FeatureEmbeddingClassifier(pretrained_model=PRETRAINED_MODEL)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model = model.to(device)
# model.eval()



# # Update to use productNumber (string) instead of component_number (int)
# class PredictRequest(BaseModel):
#     product_number: str



# # Update to use productNumber and productFeatures
# class AssociationResult(BaseModel):
#     product_number: str
#     product_features: list
#     association_number: str
#     association_features: list


# class PredictionResult(BaseModel):
#     product_number: str
#     product_features: list
#     true_associations: list[AssociationResult]

# @app.get("/")
# def root():
#     return {"message": "API is running"}


# # Path to your features JSON file (update as needed)
# FEATURES_JSON_PATH = "C:\\Users\\mshar\\Desktop\\New folder\\AI_POC_CAL\\Model\\merged_data_for_poc.json"



# @app.post("/predict", response_model=PredictionResult)
# def predict(request: PredictRequest):
#     try:
#         # Load all products from the JSON file
#         with open(FEATURES_JSON_PATH, "r") as f:
#             products_data = json.load(f)

#         # Find the query product and all other products
#         query_product = None
#         association_products = []
#         for prod in products_data:
#             if prod.get("productNumber") == request.product_number:
#                 query_product = prod
#             else:
#                 association_products.append(prod)

#         if not query_product:
#             raise HTTPException(status_code=404, detail="Product not found")

#         # Prepare features for prediction
#         productFeatures = query_product["productFeatures"]
#         true_associations = []

#         for assoc in association_products:
#             associationFeatures = assoc["productFeatures"]
#             data = [{
#                 "componentFeatures": productFeatures,
#                 "associationFeatures": associationFeatures,
#                 "alternative": 0
#             }]
#             dataset = PrecomputedFeatureDataset([], encoder)
#             dataset.data = data
#             all_prod_features = [f"{f['featureId']}:{f['value']}" for f in productFeatures]
#             all_assoc_features = [f"{f['featureId']}:{f['value']}" for f in associationFeatures]
#             dataset.comp_embeddings = encoder.encode(all_prod_features, convert_to_tensor=True).cpu()
#             dataset.assoc_embeddings = encoder.encode(all_assoc_features, convert_to_tensor=True).cpu()
#             dataset.comp_indices = [(0, len(all_prod_features))]
#             dataset.assoc_indices = [(0, len(all_assoc_features))]
#             dataset.labels = [torch.tensor(0, dtype=torch.float32)]
#             dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_with_padding)
#             for comp_feats, assoc_feats, _ in dataloader:
#                 comp_feats, assoc_feats = comp_feats.to(device), assoc_feats.to(device)
#                 with torch.no_grad():
#                     output = model(comp_feats, assoc_feats)
#                     prob = torch.sigmoid(output).item()
#                     prediction = 1 if prob > 0.5 else 0
#                 if prediction == 1:
#                     true_associations.append(AssociationResult(
#                         product_number=request.product_number,
#                         product_features=productFeatures,
#                         association_number=assoc["productNumber"],
#                         association_features=associationFeatures
#                     ))

#         return PredictionResult(
#             product_number=request.product_number,
#             product_features=productFeatures,
#             true_associations=true_associations
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# main.py
import os, json
from typing import List
import torch
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
FEATURES_JSON_PATH = os.getenv("FEATURES_JSON_PATH", "data/merged_data_for_poc.json")
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
collection = client.get_or_create_collection(name="products", metadata={"hnsw:space": "cosine"})

# def ensure_index():
#     products, _ = load_products()
#     if collection.count() == 0:
#         ids, docs = [], []
#         for p in products:
#             pid = str(p["productNumber"])
#             ids.append(pid)
#             docs.append(flatten_features(p))
#         embs = encoder.encode(docs, batch_size=128, normalize_embeddings=True).tolist()
#         collection.add(ids=ids, documents=docs, embeddings=embs)

# ensure_index()

@app.get("/healthz")
def healthz():
    return {"status": "ok", "indexed": collection.count()}

@app.get("/predict", response_model=PredictionResult)
def predict(
    productNumber: str = Query(..., description="Anchor productNumber"),
    top_k: int = Query(200, ge=1, description="Nearest neighbors to evaluate")
):
    # 1) Load + find anchor
    _, by_num = load_products()
    productNumber = str(productNumber)
    anchor = by_num.get(productNumber)
    if not anchor:
        raise HTTPException(status_code=404, detail="Product not found")

    productFeatures: List[Feature] = anchor["productFeatures"]

    # 2) Retrieve candidates with Chroma using MiniLM embeddings
    # q_text = flatten_features(anchor)
    # q_emb = encoder.encode([q_text], normalize_embeddings=True).tolist()
    # n = min(top_k, max(1, collection.count()))
    # q = collection.query(query_embeddings=q_emb, n_results=n)
    # candidate_ids = [cid for cid in q["ids"][0] if cid != productNumber]
    # 2) Encode query product only
    q_text = flatten_features(anchor)
    q_emb = encoder.encode([q_text], normalize_embeddings=True).tolist()

    # 3) Retrieve candidates from Chroma
    n = min(top_k*3, max(1, collection.count()))
    q = collection.query(query_embeddings=q_emb, n_results=n)
    candidate_ids = [cid for cid in q["ids"][0] if cid != productNumber]

    true_associations: List[AssociationResult] = []

    # 3) For each candidate, run your UNCHANGED pairwise pipeline
    #    (create dataset with precomputed per-feature embeddings -> collate -> model)
    # 4) For each candidate, run pairwise classifier pipeline
    results_with_probs = []
    for cid in candidate_ids:
        assoc = by_num.get(str(cid))
        if not assoc:
            continue
        associationFeatures = assoc["productFeatures"]

        # ---- UNCHANGED PIPELINE SHAPE CONTRACT ----
        # Build a one-item dataset with tokenized features as "featureId:value"
        data = [{
            "componentFeatures": productFeatures,
            "associationFeatures": associationFeatures,
            "alternative": 0
        }]

        dataset = PrecomputedFeatureDataset([], encoder)  # your class expects `encoder`
        dataset.data = data

        # Per-feature strings -> embeddings
        # comp_tokens = [f"{f['featureId']}:{f['value']}" for f in productFeatures]
        # assoc_tokens = [f"{f['featureId']}:{f['value']}" for f in associationFeatures]

        # # Encode to tensors; each is [N, D]
        # comp_emb = encoder.encode(comp_tokens, convert_to_tensor=True)  # torch.float32
        # assoc_emb = encoder.encode(assoc_tokens, convert_to_tensor=True)

        # # Stash into dataset fields your collate_fn expects
        # dataset.comp_embeddings = comp_emb.cpu()
        # dataset.assoc_embeddings = assoc_emb.cpu()
        # dataset.comp_indices = [(0, len(comp_tokens))]
        # dataset.assoc_indices = [(0, len(assoc_tokens))]
        # dataset.labels = [torch.tensor(0, dtype=torch.float32)]  # not used at inference

        # # Collate should pad to [B, N, D] with zeros (as your AttentionPooling expects)
        # dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_with_padding)

        
        comp_tokens = [f"{f['featureId']}:{f['value']}" for f in productFeatures]
        assoc_tokens = [f"{f['featureId']}:{f['value']}" for f in associationFeatures]

        comp_emb = encoder.encode(comp_tokens, convert_to_tensor=True, device=device)
        assoc_emb = encoder.encode(assoc_tokens, convert_to_tensor=True, device=device)

        dataset.comp_embeddings = comp_emb
        dataset.assoc_embeddings = assoc_emb
        dataset.comp_indices = [(0, len(comp_tokens))]
        dataset.assoc_indices = [(0, len(assoc_tokens))]
        dataset.labels = [torch.tensor(0, dtype=torch.float32)]

        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_with_padding)


        # Forward pass
        # for comp_feats, assoc_feats, _ in dataloader:
        #     # comp_feats/assoc_feats -> [B, N, D]
        #     comp_feats = comp_feats.to(device)
        #     assoc_feats = assoc_feats.to(device)
        #     with torch.no_grad():
        #         logits = model(comp_feats, assoc_feats)  # [B]
        #         prob = torch.sigmoid(logits).item()
        #         pred = 1 if prob > 0.5 else 0

        #     if pred == 1:
        #         true_associations.append(AssociationResult(
        #             product_number=productNumber,
        #             product_features=productFeatures,
        #             association_number=assoc["productNumber"],
        #             association_features=associationFeatures
        #         ))

    #     for comp_feats, assoc_feats, _ in dataloader:
    #         comp_feats, assoc_feats = comp_feats.to(device), assoc_feats.to(device)
    #         with torch.no_grad():
    #             logits = model(comp_feats, assoc_feats)
    #             prob = torch.sigmoid(logits).item()
    #             if prob > 0.5:
    #                 # true_associations.append(AssociationResult(
    #                 #     product_number=productNumber,
    #                 #     product_features=productFeatures,
    #                 #     association_number=assoc["productNumber"],
    #                 #     association_features=associationFeatures
    #                 # ))
    #                 true_associations.append(AssociationResult(
    #                     association_number=assoc["productNumber"],
    #                     association_features=associationFeatures
    #                 ))

    # return PredictionResult(
    #     product_number=productNumber,
    #     product_features=productFeatures,
    #     true_associations=true_associations
    # )
    

        for comp_feats, assoc_feats, _ in dataloader:
            comp_feats, assoc_feats = comp_feats.to(device), assoc_feats.to(device)
            with torch.no_grad():
                logits = model(comp_feats, assoc_feats)
                prob = torch.sigmoid(logits).item()
                if prob > 0.5:
                    results_with_probs.append((prob, assoc["productNumber"], associationFeatures))

        # ✅ sort by prob descending
    results_with_probs.sort(key=lambda x: x[0], reverse=True)

        # ✅ keep only top_k
    results_with_probs = results_with_probs[:top_k]

        # ✅ build final results (without prob field)
    true_associations = [
        AssociationResult(
            association_number=anum,
               association_features=afeats
        )
        for _, anum, afeats in results_with_probs
    ]

    return PredictionResult(
            product_number=productNumber,
            product_features=productFeatures,
            true_associations=true_associations
    )


