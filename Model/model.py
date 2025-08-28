#Model architecture and training:
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import json
 
# Check for and set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
class AttentionPooling(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.attention = nn.Linear(emb_dim, 1)
 
    def forward(self, embeddings):
        # embeddings: [batch_size, num_features, emb_dim]
        # Ignore padding zeros for attention calculation
        mask = (embeddings.sum(dim=-1) != 0).float()
        attn_scores = self.attention(embeddings).squeeze(-1) # [batch_size, num_features]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
 
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1) # [batch_size, num_features, 1]
 
        weighted_emb = (attn_weights * embeddings).sum(dim=1)  # [batch_size, emb_dim]
        return weighted_emb
 
class FeatureEmbeddingClassifier(nn.Module):
    def __init__(self, pretrained_model="all-MiniLM-L6-v2", hidden_dim=256):
        super().__init__()
        # Note: The encoder is not part of this class as it will be pre-computed.
        # This prevents the encoder from being saved/loaded with the classifier.
        emb_dim = SentenceTransformer(pretrained_model).get_sentence_embedding_dimension()
 
        # Attention pooling layers for both components & associations
        self.comp_attention = AttentionPooling(emb_dim)
        self.assoc_attention = AttentionPooling(emb_dim)
 
        # Classifier layers
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
 
    def forward(self, comp_feats, assoc_feats):
        # comp_feats and assoc_feats are now pre-computed embeddings
        comp_pooled = self.comp_attention(comp_feats)
        assoc_pooled = self.assoc_attention(assoc_feats)
 
        combined = torch.cat([comp_pooled, assoc_pooled], dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
 
        return self.fc2(x).squeeze(-1)
 
class PrecomputedFeatureDataset(Dataset):
    def __init__(self, json_paths, encoder):
        self.data = []
        for path in json_paths:
            with open(path, "r") as f:
                self.data.extend(json.load(f))
 
 
        all_comp_features = [f"{f['featureId']}:{f['value']}" for item in self.data for f in item["componentFeatures"]]
        all_assoc_features = [f"{f['featureId']}:{f['value']}" for item in self.data for f in item["associationFeatures"]]
 
        print("Pre-computing component embeddings...")
        self.comp_embeddings = encoder.encode(all_comp_features, convert_to_tensor=True, show_progress_bar=True)
 
        print("Pre-computing association embeddings...")
        self.assoc_embeddings = encoder.encode(all_assoc_features, convert_to_tensor=True, show_progress_bar=True)
 
        self.comp_indices = []
        self.assoc_indices = []
        comp_idx_counter = 0
        assoc_idx_counter = 0
 
        for item in self.data:
            num_comp = len(item["componentFeatures"])
            num_assoc = len(item["associationFeatures"])
            self.comp_indices.append((comp_idx_counter, comp_idx_counter + num_comp))
            self.assoc_indices.append((assoc_idx_counter, assoc_idx_counter + num_assoc))
            comp_idx_counter += num_comp
            assoc_idx_counter += num_assoc
 
        self.labels = [torch.tensor(item["alternative"], dtype=torch.float32) for item in self.data]
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        comp_start, comp_end = self.comp_indices[idx]
        assoc_start, assoc_end = self.assoc_indices[idx]
 
        comp_emb = self.comp_embeddings[comp_start:comp_end]
        assoc_emb = self.assoc_embeddings[assoc_start:assoc_end]
 
        # Return tensors on CPU for collate_fn to handle batching and moving to GPU
        return comp_emb.cpu(), assoc_emb.cpu(), self.labels[idx]
 
def collate_fn_with_padding(batch):
    comp_feats = [item[0] for item in batch]
    assoc_feats = [item[1] for item in batch]
    labels = torch.stack([item[2] for item in batch])
 
    max_comp_len = max(emb.size(0) for emb in comp_feats)
    padded_comp_emb = [F.pad(emb, (0, 0, 0, max_comp_len - emb.size(0)), "constant", 0.0) for emb in comp_feats]
 
    max_assoc_len = max(emb.size(0) for emb in assoc_feats)
    padded_assoc_emb = [F.pad(emb, (0, 0, 0, max_assoc_len - emb.size(0)), "constant", 0.0) for emb in assoc_feats]
 
    # Move to GPU here
    return torch.stack(padded_comp_emb).to(device), torch.stack(padded_assoc_emb).to(device), labels.to(device)
 
# if __name__ == "__main__":
#     # --- Data Loading and Setup ---
#     print("Loading JSON data...")
#     encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
#     dataset = PrecomputedFeatureDataset(["/content/drive/MyDrive/train_data1.json", "/content/drive/MyDrive/train_data2.json"], encoder)
#     dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_with_padding, shuffle=True)
 
#     # --- Model, Loss, and Optimizer Setup ---
#     model = FeatureEmbeddingClassifier().to(device)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
 
#     # --- Training Loop ---
#     try:
#         for epoch in range(5):
#             total_loss = 0
#             for comp_feats, assoc_feats, labels in dataloader:
#                 optimizer.zero_grad()
#                 outputs = model(comp_feats, assoc_feats)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()
#             print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
 
#         torch.save(model.state_dict(), '/content/drive/MyDrive/final_model.pth')
#         print("Model state saved to '/content/drive/MyDrive/final_model.pth'")
 
#     except KeyboardInterrupt:
#         print("Training interrupted. Saving model state...")
#         # Optional: Add code to save model weights here
#         torch.save(model.state_dict(), '/content/drive/MyDrive/interrupted_model.pth')
#         print("Model state saved to '/content/drive/MyDrive/interrupted_model.pth'. Exiting.")
 
# Model testing:
# import torch
# from torch.utils.data import DataLoader
# from sentence_transformers import SentenceTransformer
# import json
 
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# model_path = '/content/drive/MyDrive/final_model.pth'
# model = FeatureEmbeddingClassifier().to(device)
# model.load_state_dict(torch.load(model_path))
# model.eval() 
 
# test_json_paths = ["/content/drive/MyDrive/merged_test.json"]
 
# encoder = SentenceTransformer("all-MiniLM-L6-v2").to(device)
# test_dataset = PrecomputedFeatureDataset(test_json_paths, encoder)
# test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn_with_padding, shuffle=False)
 
# correct_predictions = 0
# total_samples = 0
# with torch.no_grad():
#     for comp_feats, assoc_feats, labels in test_dataloader:
#         outputs = model(comp_feats, assoc_feats)
#         predictions = torch.sigmoid(outputs)
#         # Convert probabilities to binary predictions (0 or 1)
#         predicted_classes = (predictions > 0.5).float()
 
#         correct_predictions += (predicted_classes == labels).sum().item()
#         total_samples += labels.size(0)
 
# accuracy = correct_predictions / total_samples
# print(f"Test Accuracy: {accuracy:.4f}")
 