"""
DeepHealth Medical Analysis API Endpoints

Comprehensive medical analysis endpoints for drug interactions,
side effects, symptoms, and lab tests.
"""

from fastapi import APIRouter, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.logging import get_logger
from app.schemas.medical import (
    DrugInteractionRequest,
    DrugInteractionResponse,
    LabTestRequest,
    LabTestResponse,
    SideEffectRequest,
    SideEffectResponse,
    SymptomAnalysisRequest,
    SymptomAnalysisResponse,
)
from app.services.drug_interaction_checker import get_drug_interaction_checker
from app.services.lab_test_analyzer import get_lab_test_analyzer
from app.services.side_effect_analyzer import get_side_effect_analyzer
from app.services.symptom_analyzer import get_symptom_analyzer

logger = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/medical", tags=["medical"])


# ============================================================================
# DRUG INTERACTION CHECKER ENDPOINTS
# ============================================================================

@router.post(
    "/drug-interactions",
    response_model=DrugInteractionResponse,
    summary="Analyze Drug Interactions",
    description="""
    Comprehensive drug interaction analysis with multi-layer scoring engine.

    Features:
    - Pharmacodynamic and pharmacokinetic interaction scoring
    - Mechanism duplication detection
    - Metabolic pathway conflict analysis
    - Safety score calculation (0-100)
    - Monitoring recommendations
    - Alternative medication suggestions

    Returns UI-optimized JSON for Next.js + shadcn/ui rendering.
    """
)
async def analyze_drug_interactions(
    request: DrugInteractionRequest
) -> DrugInteractionResponse:
    """
    Analyze potential interactions between multiple drugs.

    Args:
        request: List of drugs and patient information.

    Returns:
        Complete interaction analysis with scores and recommendations.
    """
    try:
        logger.info(f"Drug interaction analysis requested for {len(request.drugs)} drugs")

        if len(request.drugs) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 drugs are required for interaction analysis"
            )

        if len(request.drugs) > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 10 drugs can be analyzed at once"
            )

        checker = await get_drug_interaction_checker()
        result = await checker.analyze(request)

        logger.info(
            f"Drug interaction analysis complete: {len(result.interactions)} interactions found",
            extra={"safety_score": result.safety_score}
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Drug interaction analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze drug interactions"
        )


# ============================================================================
# SIDE EFFECT ANALYZER ENDPOINTS
# ============================================================================

@router.post(
    "/side-effects",
    response_model=SideEffectResponse,
    summary="Analyze Drug Side Effects",
    description="""
    Comprehensive side effect analysis with likelihood modeling.

    Features:
    - Side effect likelihood categories (very common to very rare)
    - Severity classification for each effect
    - Symptom-drug correlation analysis
    - Time-pattern recognition (immediate, delayed, cumulative)
    - Management recommendations
    - Alternative medication suggestions

    Returns UI-optimized JSON for Next.js + shadcn/ui rendering.
    """
)
async def analyze_side_effects(
    request: SideEffectRequest
) -> SideEffectResponse:
    """
    Analyze side effects for a specific drug.

    Args:
        request: Drug name and optional symptom for correlation.

    Returns:
        Complete side effect analysis with correlations and recommendations.
    """
    try:
        logger.info(f"Side effect analysis requested for drug: {request.drug}")

        if not request.drug or len(request.drug.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Valid drug name is required"
            )

        analyzer = await get_side_effect_analyzer()
        result = await analyzer.analyze(request)

        logger.info(
            f"Side effect analysis complete for {request.drug}",
            extra={
                "effects_found": len(result.expected_effects),
                "correlations": len(result.symptom_correlations)
            }
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Side effect analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze side effects"
        )


# ============================================================================
# SYMPTOM ANALYZER ENDPOINTS
# ============================================================================

@router.post(
    "/symptoms",
    response_model=SymptomAnalysisResponse,
    summary="Analyze Symptoms",
    description="""
    AI-powered symptom analysis with clinical pattern recognition.

    Features:
    - Clinical pattern matching (viral, bacterial, neurological, etc.)
    - Multi-factor probability scoring per condition
    - Urgency assessment (reassuring to immediate)
    - Red flag detection with explanations
    - Personalized recommendations

    Returns UI-optimized JSON for Next.js + shadcn/ui rendering.
    """
)
async def analyze_symptoms(
    request: SymptomAnalysisRequest
) -> SymptomAnalysisResponse:
    """
    Analyze symptoms to identify possible conditions and urgency.

    Args:
        request: List of symptoms and patient information.

    Returns:
        Complete symptom analysis with conditions and recommendations.
    """
    try:
        logger.info(f"Symptom analysis requested for {len(request.symptoms)} symptoms")

        if not request.symptoms:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one symptom is required"
            )

        if len(request.symptoms) > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 20 symptoms can be analyzed at once"
            )

        analyzer = await get_symptom_analyzer()
        result = await analyzer.analyze(request)

        logger.info(
            f"Symptom analysis complete",
            extra={
                "conditions_found": len(result.possible_conditions),
                "urgency_level": result.urgency_assessment.level.value
            }
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Symptom analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze symptoms"
        )


# ============================================================================
# LAB TEST ANALYZER ENDPOINTS
# ============================================================================

@router.post(
    "/lab-tests",
    response_model=LabTestResponse,
    summary="Analyze Lab Test Results",
    description="""
    Comprehensive lab test interpretation with clinical correlation.

    Features:
    - Reference range interpretation
    - Severity color-coding
    - Trend analysis (if historical data provided)
    - Potential underlying cause identification
    - Retest interval recommendations
    - Lifestyle modification suggestions

    Supports: CBC, CMP, Lipid Panel, Thyroid Panel, Liver Enzymes

    Returns UI-optimized JSON for Next.js + shadcn/ui rendering.
    """
)
async def analyze_lab_tests(
    request: LabTestRequest
) -> LabTestResponse:
    """
    Analyze lab test results and provide interpretation.

    Args:
        request: Lab test values and patient information.

    Returns:
        Complete lab analysis with interpretations and recommendations.
    """
    try:
        logger.info(f"Lab test analysis requested for {request.test_type.value} panel")

        if not request.values:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one lab value is required"
            )

        if len(request.values) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 50 lab values can be analyzed at once"
            )

        analyzer = await get_lab_test_analyzer()
        result = await analyzer.analyze(request)

        logger.info(
            f"Lab test analysis complete",
            extra={
                "tests_analyzed": len(result.lab_values),
                "abnormal_count": result.abnormal_count,
                "critical_count": result.critical_count
            }
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lab test analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze lab tests"
        )


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@router.get(
    "/status",
    summary="DeepHealth System Status",
    description="Check the status of all DeepHealth analysis modules."
)
async def medical_status():
    """
    Get status of all medical analysis modules.

    Returns:
        Status information for each module.
    """
    return {
        "service": "DeepHealth Medical Analysis",
        "version": "2.0.0",
        "modules": {
            "drug_interaction_checker": {
                "status": "active",
                "description": "Multi-layer drug interaction scoring engine"
            },
            "side_effect_analyzer": {
                "status": "active",
                "description": "Side effect likelihood and correlation analyzer"
            },
            "symptom_analyzer": {
                "status": "active",
                "description": "Clinical pattern recognition and urgency assessment"
            },
            "lab_test_analyzer": {
                "status": "active",
                "description": "Lab test interpretation with trend analysis"
            }
        },
        "features": [
            "UI-optimized JSON responses",
            "Confidence scoring (0-100)",
            "Severity classification",
            "Red flag detection",
            "Personalized recommendations"
        ]
    }
