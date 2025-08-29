# model_loader.py
import torch
from models.model import FeatureEmbeddingClassifier  # your custom model class

MODEL_PATH = "models/fine_tuned_model.pth"

def load_model():
    # Automatically choose GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on: {device}")

    # Initialize architecture
    model = FeatureEmbeddingClassifier()

    # Load saved state dict
    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Load weights into model
    model.load_state_dict(state, strict=False)

    # Move model to GPU or CPU
    model.to(device)

    # Set inference mode
    model.eval()
    return model
