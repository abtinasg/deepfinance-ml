"""
DeepHealth Medical Schemas - UI-Optimized Response Models

Comprehensive Pydantic models for medical analysis responses,
optimized for Next.js + shadcn/ui frontend rendering.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================================
# ENUMS - Severity and Classification Types
# ============================================================================

class SeverityLevel(str, Enum):
    """Standardized severity levels across all modules."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class UrgencyLevel(str, Enum):
    """Urgency classification for symptom analysis."""
    REASSURING = "reassuring"
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class LikelihoodCategory(str, Enum):
    """Side effect likelihood categories."""
    VERY_COMMON = "very_common"
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    VERY_RARE = "very_rare"


class InteractionCategory(str, Enum):
    """Drug interaction severity categories."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    AVOID_COMBINATION = "avoid_combination"


class TimePattern(str, Enum):
    """Side effect time patterns."""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    DOSE_DEPENDENT = "dose_dependent"
    CUMULATIVE = "cumulative"


class ClinicalPattern(str, Enum):
    """Clinical symptom patterns."""
    VIRAL = "viral"
    BACTERIAL = "bacterial"
    SINUSITIS = "sinusitis"
    MIGRAINE = "migraine"
    NEUROLOGICAL = "neurological"
    ALLERGY = "allergy"
    GASTROINTESTINAL = "gastrointestinal"
    CARDIAC = "cardiac"
    RESPIRATORY = "respiratory"
    MUSCULOSKELETAL = "musculoskeletal"
    EMERGENCY = "emergency"


class LabTestCategory(str, Enum):
    """Lab test categories."""
    CBC = "cbc"
    CMP = "cmp"
    LIPID_PANEL = "lipid_panel"
    THYROID_PANEL = "thyroid_panel"
    LIVER_ENZYMES = "liver_enzymes"
    KIDNEY_FUNCTION = "kidney_function"
    CARDIAC_MARKERS = "cardiac_markers"
    INFLAMMATORY_MARKERS = "inflammatory_markers"


# ============================================================================
# BASE UI COMPONENTS - Shared Across All Modules
# ============================================================================

class ScoreSet(BaseModel):
    """Standardized score set for all modules."""
    confidence: int = Field(..., ge=0, le=100, description="Confidence in analysis (0-100)")
    severity: int = Field(..., ge=0, le=100, description="Overall severity score (0-100)")
    urgency: Optional[int] = Field(None, ge=0, le=100, description="Urgency score if applicable")
    safety: Optional[int] = Field(None, ge=0, le=100, description="Safety score if applicable")


class BadgeInfo(BaseModel):
    """UI badge component data."""
    text: str = Field(..., description="Badge display text")
    variant: str = Field(..., description="Badge variant: success, warning, danger, info")


class SectionItem(BaseModel):
    """Individual item within a UI section."""
    label: str = Field(..., description="Item label")
    value: str = Field(..., description="Item value/content")
    badge: Optional[str] = Field(None, description="Optional badge text")
    severity: Optional[str] = Field(None, description="Severity level for styling")
    explanation: Optional[str] = Field(None, description="Detailed explanation")
    probability: Optional[float] = Field(None, ge=0, le=100, description="Probability percentage")
    score: Optional[int] = Field(None, ge=0, le=100, description="Score value")


class UISection(BaseModel):
    """Grouped section for UI card rendering."""
    title: str = Field(..., description="Section title")
    icon: Optional[str] = Field(None, description="Lucide icon name for section")
    items: list[SectionItem] = Field(default_factory=list, description="Section items")
    expandable: bool = Field(default=False, description="Whether section is expandable")
    priority: int = Field(default=0, description="Display priority (lower = higher)")


class RedFlag(BaseModel):
    """Critical warning item."""
    flag: str = Field(..., description="Red flag description")
    explanation: str = Field(..., description="Why this is concerning")
    action: str = Field(..., description="Recommended action")
    severity: SeverityLevel = Field(..., description="Severity level")


class Recommendation(BaseModel):
    """Actionable recommendation."""
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    priority: str = Field(..., description="high, medium, low")
    category: str = Field(..., description="Category: monitoring, lifestyle, medication, follow-up")
    timeframe: Optional[str] = Field(None, description="When to take action")


class Alternative(BaseModel):
    """Alternative suggestion with reasoning."""
    name: str = Field(..., description="Alternative name")
    reason: str = Field(..., description="Why this is suggested")
    advantages: list[str] = Field(default_factory=list, description="Benefits")
    considerations: list[str] = Field(default_factory=list, description="Things to consider")


# ============================================================================
# BASE MEDICAL RESPONSE - Foundation for All Analyzers
# ============================================================================

class MedicalAnalysisResponse(BaseModel):
    """
    Base response model for all DeepHealth analyzers.
    UI-optimized structure for Next.js + shadcn/ui rendering.
    """
    summary: str = Field(..., description="Concise summary of analysis")
    scores: ScoreSet = Field(..., description="Numerical scores for visualization")
    sections: list[UISection] = Field(default_factory=list, description="UI sections for rendering")
    red_flags: list[RedFlag] = Field(default_factory=list, description="Critical warnings")
    recommendations: list[Recommendation] = Field(default_factory=list, description="Action items")
    alternatives: list[Alternative] = Field(default_factory=list, description="Alternative options")
    disclaimer: str = Field(
        default="This analysis is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare provider for diagnosis and treatment.",
        description="Medical disclaimer"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "summary": "Analysis complete with moderate confidence",
                "scores": {
                    "confidence": 85,
                    "severity": 45,
                    "urgency": 30
                },
                "sections": [],
                "red_flags": [],
                "recommendations": [],
                "alternatives": [],
                "disclaimer": "This is informational and not medical advice."
            }
        }


# ============================================================================
# DRUG INTERACTION CHECKER MODELS
# ============================================================================

class DrugInteraction(BaseModel):
    """Single drug interaction detail."""
    drug_a: str = Field(..., description="First drug name")
    drug_b: str = Field(..., description="Second drug name")
    category: InteractionCategory = Field(..., description="Interaction severity category")
    mechanism: str = Field(..., description="One-sentence mechanism explanation")
    clinical_effect: str = Field(..., description="Clinical significance")
    pharmacokinetic_score: int = Field(..., ge=0, le=100, description="PK interaction score")
    pharmacodynamic_score: int = Field(..., ge=0, le=100, description="PD interaction score")
    monitoring: list[str] = Field(default_factory=list, description="Monitoring recommendations")
    timing_adjustment: Optional[str] = Field(None, description="Dose timing adjustment")


class InteractionScoreBreakdown(BaseModel):
    """Detailed scoring breakdown for drug interactions."""
    pharmacodynamic: int = Field(..., ge=0, le=100)
    pharmacokinetic: int = Field(..., ge=0, le=100)
    mechanism_duplication: int = Field(..., ge=0, le=100)
    metabolic_pathway: int = Field(..., ge=0, le=100)
    half_life_overlap: int = Field(..., ge=0, le=100)
    organ_system_load: int = Field(..., ge=0, le=100)
    weighted_total: int = Field(..., ge=0, le=100)


class DrugInteractionResponse(MedicalAnalysisResponse):
    """Complete drug interaction analysis response."""
    drugs_analyzed: list[str] = Field(..., description="List of drugs analyzed")
    interactions: list[DrugInteraction] = Field(default_factory=list, description="Found interactions")
    score_breakdown: InteractionScoreBreakdown = Field(..., description="Detailed scoring")
    safety_score: int = Field(..., ge=0, le=100, description="Overall safety score")
    combination_verdict: str = Field(..., description="Final verdict on combination safety")


# ============================================================================
# SIDE EFFECT ANALYZER MODELS
# ============================================================================

class SideEffect(BaseModel):
    """Individual side effect with comprehensive analysis."""
    name: str = Field(..., description="Side effect name")
    likelihood: LikelihoodCategory = Field(..., description="Occurrence likelihood")
    probability_percent: float = Field(..., ge=0, le=100, description="Probability percentage")
    severity: SeverityLevel = Field(..., description="Severity classification")
    time_pattern: TimePattern = Field(..., description="When it typically occurs")
    reversible: bool = Field(..., description="Whether effect is reversible")
    mechanism: str = Field(..., description="Biological mechanism")
    management: str = Field(..., description="How to manage if occurs")


class SymptomCorrelation(BaseModel):
    """Correlation between user symptom and drug side effect."""
    symptom: str = Field(..., description="User-reported symptom")
    drug: str = Field(..., description="Suspected drug")
    correlation_score: int = Field(..., ge=0, le=100, description="Likelihood drug caused symptom")
    explanation: str = Field(..., description="Why this correlation exists")
    time_consistency: str = Field(..., description="Whether timing aligns")


class SideEffectResponse(MedicalAnalysisResponse):
    """Complete side effect analysis response."""
    drug_analyzed: str = Field(..., description="Drug being analyzed")
    user_symptom: Optional[str] = Field(None, description="User-reported symptom if any")
    expected_effects: list[SideEffect] = Field(default_factory=list, description="Expected side effects")
    serious_adverse_events: list[SideEffect] = Field(default_factory=list, description="Serious adverse events")
    symptom_correlations: list[SymptomCorrelation] = Field(default_factory=list, description="Symptom-drug correlations")
    when_to_seek_help: list[str] = Field(default_factory=list, description="Warning signs requiring medical attention")


# ============================================================================
# SYMPTOM ANALYZER MODELS
# ============================================================================

class ConditionMatch(BaseModel):
    """Potential condition match with multi-factor scoring."""
    condition: str = Field(..., description="Condition name")
    probability: int = Field(..., ge=0, le=100, description="Overall probability")
    pattern: ClinicalPattern = Field(..., description="Clinical pattern type")
    scoring: dict[str, int] = Field(..., description="Multi-factor scoring breakdown")
    key_symptoms: list[str] = Field(default_factory=list, description="Matching symptoms")
    missing_symptoms: list[str] = Field(default_factory=list, description="Expected but missing symptoms")
    risk_factors: list[str] = Field(default_factory=list, description="Relevant risk factors")


class PatternAnalysis(BaseModel):
    """Clinical pattern analysis result."""
    pattern: ClinicalPattern = Field(..., description="Identified pattern")
    confidence: int = Field(..., ge=0, le=100, description="Pattern confidence")
    supporting_symptoms: list[str] = Field(default_factory=list, description="Supporting symptoms")
    typical_course: str = Field(..., description="Expected course of condition")


class UrgencyAssessment(BaseModel):
    """Urgency scoring and classification."""
    score: int = Field(..., ge=0, le=100, description="Urgency score 0-100")
    level: UrgencyLevel = Field(..., description="Urgency classification")
    reasoning: str = Field(..., description="Why this urgency level")
    timeframe: str = Field(..., description="Recommended response timeframe")


class SymptomAnalysisResponse(MedicalAnalysisResponse):
    """Complete symptom analysis response."""
    symptoms_analyzed: list[str] = Field(..., description="Input symptoms")
    duration: Optional[str] = Field(None, description="Symptom duration")
    risk_factors: list[str] = Field(default_factory=list, description="Patient risk factors")
    pattern_analysis: list[PatternAnalysis] = Field(default_factory=list, description="Pattern matching results")
    possible_conditions: list[ConditionMatch] = Field(default_factory=list, description="Ranked conditions")
    urgency_assessment: UrgencyAssessment = Field(..., description="Urgency evaluation")
    personalized_advice: list[str] = Field(default_factory=list, description="Personalized recommendations")


# ============================================================================
# LAB TEST ANALYZER MODELS
# ============================================================================

class LabValue(BaseModel):
    """Single lab value with interpretation."""
    test_name: str = Field(..., description="Test name")
    value: float = Field(..., description="Measured value")
    unit: str = Field(..., description="Measurement unit")
    reference_low: float = Field(..., description="Lower reference range")
    reference_high: float = Field(..., description="Upper reference range")
    status: str = Field(..., description="low, normal, high, critical")
    severity_color: str = Field(..., description="Color code: green, yellow, orange, red")
    interpretation: str = Field(..., description="Clinical interpretation")
    percent_deviation: float = Field(..., description="Percentage from normal range")


class TrendAnalysis(BaseModel):
    """Analysis of lab value trends over time."""
    test_name: str = Field(..., description="Test name")
    direction: str = Field(..., description="increasing, decreasing, stable, fluctuating")
    rate_of_change: str = Field(..., description="Description of change rate")
    clinical_significance: str = Field(..., description="What the trend means")
    values_history: list[dict[str, Any]] = Field(default_factory=list, description="Historical values")


class UnderlyingCause(BaseModel):
    """Potential underlying cause for abnormal results."""
    cause: str = Field(..., description="Potential cause")
    probability: int = Field(..., ge=0, le=100, description="Ranked probability")
    related_tests: list[str] = Field(default_factory=list, description="Related abnormal tests")
    explanation: str = Field(..., description="Why this might be the cause")
    next_steps: list[str] = Field(default_factory=list, description="Recommended follow-up")


class LabTestResponse(MedicalAnalysisResponse):
    """Complete lab test analysis response."""
    test_category: LabTestCategory = Field(..., description="Type of lab panel")
    test_date: datetime = Field(..., description="Date of test")
    lab_values: list[LabValue] = Field(default_factory=list, description="Individual test results")
    trends: list[TrendAnalysis] = Field(default_factory=list, description="Trend analysis if available")
    abnormal_count: int = Field(..., ge=0, description="Number of abnormal results")
    critical_count: int = Field(..., ge=0, description="Number of critical results")
    underlying_causes: list[UnderlyingCause] = Field(default_factory=list, description="Potential causes")
    retest_interval: Optional[str] = Field(None, description="Recommended retest timeframe")
    lifestyle_factors: list[str] = Field(default_factory=list, description="Relevant lifestyle modifications")


# ============================================================================
# REQUEST MODELS
# ============================================================================

class DrugInteractionRequest(BaseModel):
    """Request model for drug interaction analysis."""
    drugs: list[str] = Field(..., min_length=2, description="List of drugs to check")
    patient_age: Optional[int] = Field(None, ge=0, le=150, description="Patient age")
    patient_conditions: list[str] = Field(default_factory=list, description="Existing conditions")

    class Config:
        json_schema_extra = {
            "example": {
                "drugs": ["Aspirin", "Warfarin", "Ibuprofen"],
                "patient_age": 65,
                "patient_conditions": ["hypertension", "diabetes"]
            }
        }


class SideEffectRequest(BaseModel):
    """Request model for side effect analysis."""
    drug: str = Field(..., description="Drug name to analyze")
    dosage: Optional[str] = Field(None, description="Current dosage")
    symptom: Optional[str] = Field(None, description="User-reported symptom to correlate")
    duration_of_use: Optional[str] = Field(None, description="How long taking the drug")

    class Config:
        json_schema_extra = {
            "example": {
                "drug": "Metformin",
                "dosage": "500mg twice daily",
                "symptom": "nausea",
                "duration_of_use": "2 weeks"
            }
        }


class SymptomAnalysisRequest(BaseModel):
    """Request model for symptom analysis."""
    symptoms: list[str] = Field(..., min_length=1, description="List of symptoms")
    duration: Optional[str] = Field(None, description="Duration of symptoms")
    severity: Optional[str] = Field(None, description="Self-reported severity")
    risk_factors: list[str] = Field(default_factory=list, description="Risk factors/conditions")
    medications: list[str] = Field(default_factory=list, description="Current medications")
    age: Optional[int] = Field(None, ge=0, le=150, description="Patient age")

    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": ["headache", "fatigue", "sore throat"],
                "duration": "3 days",
                "severity": "moderate",
                "risk_factors": ["diabetes"],
                "medications": ["metformin"],
                "age": 45
            }
        }


class LabTestRequest(BaseModel):
    """Request model for lab test analysis."""
    test_type: LabTestCategory = Field(..., description="Type of lab panel")
    values: dict[str, float] = Field(..., description="Test name to value mapping")
    test_date: Optional[datetime] = Field(None, description="Date of test")
    previous_values: Optional[dict[str, list[dict[str, Any]]]] = Field(None, description="Historical values")
    patient_conditions: list[str] = Field(default_factory=list, description="Existing conditions")

    class Config:
        json_schema_extra = {
            "example": {
                "test_type": "cbc",
                "values": {
                    "WBC": 11.5,
                    "RBC": 4.5,
                    "Hemoglobin": 13.2,
                    "Hematocrit": 39.5,
                    "Platelets": 250
                },
                "test_date": "2024-01-15T10:00:00Z",
                "patient_conditions": ["anemia"]
            }
        }
