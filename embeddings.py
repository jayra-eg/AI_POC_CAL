import json
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# --- AttentionPooling (same as classifier) ---
class AttentionPooling(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.attention = nn.Linear(emb_dim, 1)

    def forward(self, embeddings: torch.Tensor):
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # [1, num_features, emb_dim]

        mask = (embeddings.sum(dim=-1) != 0).float()
        attn_scores = self.attention(embeddings).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        weighted_emb = (attn_weights * embeddings).sum(dim=1)
        return weighted_emb.squeeze(0)


# --- Classifier (to load trained pooling) ---
class FeatureEmbeddingClassifier(nn.Module):
    def __init__(self, pretrained_model="all-MiniLM-L6-v2", hidden_dim=256):
        super().__init__()
        emb_dim = SentenceTransformer(pretrained_model).get_sentence_embedding_dimension()

        # Attention pooling layers
        self.comp_attention = AttentionPooling(emb_dim)
        self.assoc_attention = AttentionPooling(emb_dim)

        # Classifier layers
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, comp_feats, assoc_feats):
        comp_pooled = self.comp_attention(comp_feats)
        assoc_pooled = self.assoc_attention(assoc_feats)
        combined = torch.cat([comp_pooled, assoc_pooled], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)


# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# --- Load trained classifier to get pooling weights ---
MODEL_PATH = "C:\\Users\\mshar\\Desktop\\New folder\\AI_POC_CAL\\models\\fine_tuned_model.pth"  # change path if needed
model = FeatureEmbeddingClassifier().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded trained model from {MODEL_PATH}")

# Use the trained comp_attention pooling
attention_pooler = model.comp_attention

# --- Load product JSON ---
INPUT_JSON = "C:\\Users\\mshar\\Desktop\\New folder\\AI_POC_CAL\\data\\products_no_duplicates.json"  # <-- change to your product JSON path
print(f"Loading {INPUT_JSON}...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    products = json.load(f)

print(f"Loaded {len(products)} products.")

# --- Connect to Chroma ---
client = chromadb.PersistentClient(path="./chroma_persist")
collection = client.get_or_create_collection(name="products_embeddings")

# --- Process and insert ---
print("Creating embeddings and inserting into Chroma...")

ids, embeddings, metadatas, documents = [], [], [], []

for product in tqdm(products, desc="Embedding products"):
    product_number = str(product["productNumber"])
    features = product.get("productFeatures", [])

    tokens = [f"{f['featureId']}:{f['value']}" for f in features]
    if not tokens:
        continue

    # Encode features separately
    feat_embs = encoder.encode(tokens, convert_to_tensor=True, device=device)

    # Apply trained AttentionPooling
    with torch.no_grad():
        pooled_emb = attention_pooler(feat_embs.clone()).cpu().numpy()

    # Prepare for Chroma
    ids.append(product_number)
    embeddings.append(pooled_emb)
    metadatas.append({"num_features": len(tokens)})
    documents.append(" ".join(tokens))

# Insert in batches
BATCH = 5000
for i in range(0, len(ids), BATCH):
    collection.upsert(
        ids=ids[i:i+BATCH],
        embeddings=embeddings[i:i+BATCH],
        metadatas=metadatas[i:i+BATCH],
        documents=documents[i:i+BATCH],
    )

print(f"Inserted {len(ids)} products into Chroma.")
