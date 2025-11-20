"""Pydantic schemas for request/response validation."""

from app.schemas.common import (
    APIResponse,
    ErrorResponse,
    HealthResponse,
    PaginatedResponse,
)
from app.schemas.prediction import (
    PredictRequest,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    PredictionHistoryResponse,
    ModelPrediction,
    EnsemblePrediction,
)
from app.schemas.risk import (
    RiskRequest,
    RiskResponse,
    RiskMetrics,
    VolatilityMetrics,
)
from app.schemas.medical import (
    # Base types
    MedicalAnalysisResponse,
    ScoreSet,
    UISection,
    SectionItem,
    RedFlag,
    Recommendation,
    Alternative,
    # Drug Interaction
    DrugInteractionRequest,
    DrugInteractionResponse,
    DrugInteraction,
    InteractionScoreBreakdown,
    # Side Effect
    SideEffectRequest,
    SideEffectResponse,
    SideEffect,
    SymptomCorrelation,
    # Symptom Analysis
    SymptomAnalysisRequest,
    SymptomAnalysisResponse,
    ConditionMatch,
    PatternAnalysis,
    UrgencyAssessment,
    # Lab Test
    LabTestRequest,
    LabTestResponse,
    LabValue,
    TrendAnalysis,
    UnderlyingCause,
    # Enums
    SeverityLevel,
    UrgencyLevel,
    LikelihoodCategory,
    InteractionCategory,
    TimePattern,
    ClinicalPattern,
    LabTestCategory,
)

__all__ = [
    # Common
    "APIResponse",
    "ErrorResponse",
    "HealthResponse",
    "PaginatedResponse",
    # Prediction
    "PredictRequest",
    "PredictResponse",
    "BatchPredictRequest",
    "BatchPredictResponse",
    "PredictionHistoryResponse",
    "ModelPrediction",
    "EnsemblePrediction",
    # Risk
    "RiskRequest",
    "RiskResponse",
    "RiskMetrics",
    "VolatilityMetrics",
    # Medical - Base
    "MedicalAnalysisResponse",
    "ScoreSet",
    "UISection",
    "SectionItem",
    "RedFlag",
    "Recommendation",
    "Alternative",
    # Medical - Drug Interaction
    "DrugInteractionRequest",
    "DrugInteractionResponse",
    "DrugInteraction",
    "InteractionScoreBreakdown",
    # Medical - Side Effect
    "SideEffectRequest",
    "SideEffectResponse",
    "SideEffect",
    "SymptomCorrelation",
    # Medical - Symptom Analysis
    "SymptomAnalysisRequest",
    "SymptomAnalysisResponse",
    "ConditionMatch",
    "PatternAnalysis",
    "UrgencyAssessment",
    # Medical - Lab Test
    "LabTestRequest",
    "LabTestResponse",
    "LabValue",
    "TrendAnalysis",
    "UnderlyingCause",
    # Medical - Enums
    "SeverityLevel",
    "UrgencyLevel",
    "LikelihoodCategory",
    "InteractionCategory",
    "TimePattern",
    "ClinicalPattern",
    "LabTestCategory",
]
