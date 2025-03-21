"""
Pydantic models for API requests and responses in the Module Classification System.

This module defines the data structures and validation rules for all API interactions,
ensuring type safety and proper documentation.
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator
from enum import Enum


class ModuleType(str, Enum):
    """Valid module types for classification results."""
    SCRIPT = "script"
    PRODUCT = "product"


class ConfidenceLevel(str, Enum):
    """Confidence levels for classification results."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class RetrievalMethod(str, Enum):
    """Available retrieval methods for document search."""
    HYBRID = "Full Hybrid (Parent + Semantic)"
    PARENT = "Parent Only"
    SEMANTIC = "Semantic Only"


class BaseAPIModel(BaseModel):
    """Base model class with common configuration for all API models."""
    model_config = {
        "json_schema_extra": {
            "title": "API Model",
            "description": "Base model for API requests and responses",
            "examples": []
        },
        "populate_by_name": True,
        "validate_assignment": True,
        "arbitrary_types_allowed": False,
    }


class ClassificationRequest(BaseAPIModel):
    """
    Request model for module classification.
    
    Users can provide either:
    1. Synopsis and description separately
    2. A direct query containing the full text
    """
    synopsis: str = Field(
        default="",
        description="Synopsis of the issue or feature"
    )
    description: str = Field(
        default="",
        description="Detailed description of the issue or feature"
    )
    direct_query: Optional[str] = Field(
        default=None,
        description="Complete query text (overrides synopsis and description when provided)"
    )
    
    @model_validator(mode='after')
    def validate_input(self):
        """Ensure at least one input field is provided."""
        if not self.direct_query and not (self.synopsis.strip() or self.description.strip()):
            raise ValueError("Either direct_query or at least one of synopsis/description must be provided")
        return self


class ClassificationResponse(BaseAPIModel):
    """
    Response model for module classification results.
    
    Contains the classification outcome, confidence level, and supporting evidence.
    """
    classification: str = Field(
        description="Module classification result (script or product)"
    )
    confidence: str = Field(
        description="Confidence level in the classification"
    )
    supporting_points: List[str] = Field(
        description="Key points that support the classification decision"
    )
    processing_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Technical details about the processing pipeline"
    )


class EvaluationRequest(BaseAPIModel):
    """
    Request model for model evaluation.
    
    Controls the parameters for the evaluation process.
    """
    sample_size: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of samples to evaluate"
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    num_threads: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Number of parallel threads for evaluation"
    )
    use_multi_query: bool = Field(
        default=False,
        description="Whether to use MultiQuery Retriever for improved retrieval"
    )
    num_alt_queries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of alternative queries to generate if using MultiQuery"
    )
    retrieval_method: RetrievalMethod = Field(
        default=RetrievalMethod.HYBRID,
        description="Retrieval method to use for document search"
    )


class EvaluationResult(BaseAPIModel):
    """
    Model for a single evaluation result.
    
    Contains the details of one sample's evaluation outcome.
    """
    index: int = Field(
        description="Index of the evaluated sample"
    )
    question: str = Field(
        description="The question or query that was evaluated"
    )
    true_label: str = Field(
        description="The correct classification (ground truth)"
    )
    prediction: str = Field(
        description="The model's predicted classification"
    )
    match: bool = Field(
        description="Whether the prediction matches the true label"
    )
    confidence: Optional[str] = Field(
        default=None,
        description="Confidence level of the prediction"
    )
    generated_queries: Optional[List[str]] = Field(
        default=None,
        description="List of queries generated by MultiQuery Retriever"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if evaluation failed"
    )


class EvaluationResponse(BaseAPIModel):
    """
    Response model for model evaluation.
    
    Contains overall evaluation metrics and detailed results for each sample.
    """
    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall accuracy of the model (correct/total)"
    )
    correct: int = Field(
        ge=0,
        description="Number of correctly classified samples"
    )
    total: int = Field(
        ge=0,
        description="Total number of evaluated samples"
    )
    results: List[EvaluationResult] = Field(
        description="Detailed results for each evaluated sample"
    )