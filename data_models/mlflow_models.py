import pydantic
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any


class MLFlowExperiment(BaseModel):
    tracking_uri: str = Field(..., description="URI of the MLFlow tracking server")
    experiment_name: str = Field(..., description="Name of the experiment")
    run_name: str = Field(..., description="Name of the MLFlow run")
    params: Dict[str, Any] = Field(..., description="Parameters for the experiment")
    description: Optional[str] = Field(None, description="Description of the experiment")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the experiment")
    creation_date: Optional[str] = Field(None, description="Date when the experiment was created")


class MultinomialNBParams(BaseModel):
    alpha: float = Field(1.0, description="Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).")
    fit_prior: bool = Field(True, description="Whether to learn class prior probabilities or not. If false, a uniform prior will be used.")
    class_prior: Optional[List[float]] = Field(None, description="Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.")


class VectorizerParams(BaseModel):
    max_df: float = Field(1.0, description="Ignore terms that have a document frequency higher than the given threshold.")
    min_df: float = Field(1, description="Ignore terms that have a document frequency lower than the given threshold.")
    max_features: Optional[int] = Field(None, description="Build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.")
    ngram_range: tuple = Field((1, 1), description="The lower and upper boundary of the range of n-values for different n-grams to be extracted.")
    use_idf: bool = Field(True, description="Enable inverse-document-frequency reweighting (only applies to TfidfVectorizer).")


class DataParams(BaseModel):
    test_size: float = Field(0.2, description="Fraction of the data to be used for testing.")
    random_state: int = Field(42, description="Seed used by the random number generator.")


class MLExperimentResults(BaseModel):
    run_id: str
    experiment_id: str
    metrics: Dict[str, float]


class PerformanceMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float

    class Config:
        schema_extra = {
            "example": {
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5
            }
        }

class InferenceRequest(BaseModel):
    pass


class InferenceResponse(BaseModel):
    pass


class InputDataset(BaseModel):
    text: str
    label: str