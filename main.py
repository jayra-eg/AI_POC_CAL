from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from Model.model import FeatureEmbeddingClassifier, PrecomputedFeatureDataset, collate_fn_with_padding
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import json

app = FastAPI()

# Load model and encoder at startup
MODEL_PATH = "C:\\Users\\mshar\\Desktop\\New folder\\AI_POC_CAL\\Model\\fine_tuned_model.pth"

PRETRAINED_MODEL = "all-MiniLM-L6-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = SentenceTransformer(PRETRAINED_MODEL).to(device)
model = FeatureEmbeddingClassifier(pretrained_model=PRETRAINED_MODEL)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()



# Update to use productNumber (string) instead of component_number (int)
class PredictRequest(BaseModel):
    product_number: str



# Update to use productNumber and productFeatures
class AssociationResult(BaseModel):
    product_number: str
    product_features: list
    association_number: str
    association_features: list


class PredictionResult(BaseModel):
    product_number: str
    product_features: list
    true_associations: list[AssociationResult]

@app.get("/")
def root():
    return {"message": "API is running"}


# Path to your features JSON file (update as needed)
FEATURES_JSON_PATH = "C:\\Users\\mshar\\Desktop\\New folder\\AI_POC_CAL\\Model\\merged_data_for_poc.json"



@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    try:
        # Load all products from the JSON file
        with open(FEATURES_JSON_PATH, "r") as f:
            products_data = json.load(f)

        # Find the query product and all other products
        query_product = None
        association_products = []
        for prod in products_data:
            if prod.get("productNumber") == request.product_number:
                query_product = prod
            else:
                association_products.append(prod)

        if not query_product:
            raise HTTPException(status_code=404, detail="Product not found")

        # Prepare features for prediction
        productFeatures = query_product["productFeatures"]
        true_associations = []

        for assoc in association_products:
            associationFeatures = assoc["productFeatures"]
            data = [{
                "componentFeatures": productFeatures,
                "associationFeatures": associationFeatures,
                "alternative": 0
            }]
            dataset = PrecomputedFeatureDataset([], encoder)
            dataset.data = data
            all_prod_features = [f"{f['featureId']}:{f['value']}" for f in productFeatures]
            all_assoc_features = [f"{f['featureId']}:{f['value']}" for f in associationFeatures]
            dataset.comp_embeddings = encoder.encode(all_prod_features, convert_to_tensor=True).cpu()
            dataset.assoc_embeddings = encoder.encode(all_assoc_features, convert_to_tensor=True).cpu()
            dataset.comp_indices = [(0, len(all_prod_features))]
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
                        product_number=request.product_number,
                        product_features=productFeatures,
                        association_number=assoc["productNumber"],
                        association_features=associationFeatures
                    ))

        return PredictionResult(
            product_number=request.product_number,
            product_features=productFeatures,
            true_associations=true_associations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
