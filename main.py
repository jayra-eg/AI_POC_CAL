from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from models.model import FeatureEmbeddingClassifier, PrecomputedFeatureDataset, collate_fn_with_padding
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import json

app = FastAPI()

# Load model and encoder at startup
MODEL_PATH = "models/fine_tuned_model.pth"
PRETRAINED_MODEL = "all-MiniLM-L6-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = SentenceTransformer(PRETRAINED_MODEL).to(device)
model = FeatureEmbeddingClassifier(pretrained_model=PRETRAINED_MODEL)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()


class PredictRequest(BaseModel):
    component_number: int


class AssociationResult(BaseModel):
    component_number: int
    component_features: list
    association_number: int
    association_features: list

class PredictionResult(BaseModel):
    component_number: int
    component_features: list
    true_associations: list[AssociationResult]

@app.get("/")
def root():
    return {"message": "API is running"}


# Path to your features JSON file (update as needed)
FEATURES_JSON_PATH = "models/features.json"


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    try:
        # Load all features from the JSON file
        with open(FEATURES_JSON_PATH, "r") as f:
            features_data = json.load(f)

        # Find the query component and all other components
        query_component = None
        association_components = []
        for comp in features_data:
            if comp.get("component_number") == request.component_number:
                query_component = comp
            else:
                association_components.append(comp)

        if not query_component:
            raise HTTPException(status_code=404, detail="Component not found")

        # Prepare features for prediction
        componentFeatures = query_component["features"]
        true_associations = []

        for assoc in association_components:
            associationFeatures = assoc["features"]
            data = [{
                "componentFeatures": componentFeatures,
                "associationFeatures": associationFeatures,
                "alternative": 0
            }]
            dataset = PrecomputedFeatureDataset([], encoder)
            dataset.data = data
            all_comp_features = [f"{f['featureId']}:{f['value']}" for f in componentFeatures]
            all_assoc_features = [f"{f['featureId']}:{f['value']}" for f in associationFeatures]
            dataset.comp_embeddings = encoder.encode(all_comp_features, convert_to_tensor=True).cpu()
            dataset.assoc_embeddings = encoder.encode(all_assoc_features, convert_to_tensor=True).cpu()
            dataset.comp_indices = [(0, len(all_comp_features))]
            dataset.assoc_indices = [(0, len(all_assoc_features))]
            dataset.labels = [torch.tensor(0, dtype=torch.float32)]
            dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_with_padding)
            for comp_feats, assoc_feats, _ in dataloader:
                comp_feats, assoc_feats = comp_feats.to(device), assoc_feats.to(device)
                with torch.no_grad():
                    output = model(comp_feats, assoc_feats)
                    prob = torch.sigmoid(output).item()
                    prediction = 1 if prob > 0.5 else 0
                if prediction == 1:
                    true_associations.append(AssociationResult(
                        component_number=request.component_number,
                        component_features=componentFeatures,
                        association_number=assoc["component_number"],
                        association_features=associationFeatures
                    ))

        return PredictionResult(
            component_number=request.component_number,
            component_features=componentFeatures,
            true_associations=true_associations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
