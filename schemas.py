# schemas.py
from typing import List
from pydantic import BaseModel

class Feature(BaseModel):
    featureId: str
    value: str

class AssociationResult(BaseModel):
    # product_number: str
    # product_features: List[Feature]
    association_number: str
    association_features: List[Feature]
    score: float

class PredictionResult(BaseModel):
    product_number: str
    product_features: List[Feature]
    true_associations: List[AssociationResult]
