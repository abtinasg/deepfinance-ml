"""
DeepHealth AI Symptom Analyzer Service

Clinical pattern recognition engine with multi-factor probability scoring,
urgency assessment, and personalized recommendations.
"""

from datetime import datetime
from typing import Any, Optional

from app.core.logging import get_logger
from app.schemas.medical import (
    Alternative,
    ClinicalPattern,
    ConditionMatch,
    PatternAnalysis,
    Recommendation,
    RedFlag,
    ScoreSet,
    SectionItem,
    SeverityLevel,
    SymptomAnalysisRequest,
    SymptomAnalysisResponse,
    UISection,
    UrgencyAssessment,
    UrgencyLevel,
)

logger = get_logger(__name__)


# ============================================================================
# CLINICAL PATTERN DATABASE - Symptom Patterns and Conditions
# ============================================================================

CLINICAL_PATTERNS = {
    ClinicalPattern.VIRAL: {
        "name": "Viral Infection Pattern",
        "typical_symptoms": ["fever", "fatigue", "body aches", "headache", "sore throat", "runny nose", "cough"],
        "supporting_symptoms": ["chills", "malaise", "loss of appetite", "mild nausea"],
        "typical_course": "Gradual onset over 1-2 days, peaks at 3-4 days, resolves in 7-10 days",
        "conditions": [
            {
                "name": "Common Cold",
                "key_symptoms": ["runny nose", "sore throat", "congestion", "sneezing"],
                "symptom_weight": 0.4,
                "duration_weight": 0.2,
                "risk_weight": 0.1,
                "clustering_weight": 0.3,
                "base_probability": 60
            },
            {
                "name": "Influenza",
                "key_symptoms": ["fever", "body aches", "fatigue", "headache", "dry cough"],
                "symptom_weight": 0.4,
                "duration_weight": 0.2,
                "risk_weight": 0.15,
                "clustering_weight": 0.25,
                "base_probability": 35
            },
            {
                "name": "COVID-19",
                "key_symptoms": ["fever", "cough", "fatigue", "loss of smell", "loss of taste", "shortness of breath"],
                "symptom_weight": 0.35,
                "duration_weight": 0.15,
                "risk_weight": 0.2,
                "clustering_weight": 0.3,
                "base_probability": 25
            }
        ]
    },
    ClinicalPattern.SINUSITIS: {
        "name": "Sinusitis Pattern",
        "typical_symptoms": ["facial pain", "facial pressure", "congestion", "nasal discharge", "headache"],
        "supporting_symptoms": ["post-nasal drip", "tooth pain", "ear pressure", "fatigue", "bad breath"],
        "typical_course": "Often follows cold; symptoms persist or worsen after 7-10 days",
        "conditions": [
            {
                "name": "Acute Sinusitis",
                "key_symptoms": ["facial pain", "congestion", "nasal discharge", "headache"],
                "symptom_weight": 0.45,
                "duration_weight": 0.25,
                "risk_weight": 0.1,
                "clustering_weight": 0.2,
                "base_probability": 50
            },
            {
                "name": "Chronic Sinusitis",
                "key_symptoms": ["congestion", "facial pressure", "post-nasal drip", "reduced smell"],
                "symptom_weight": 0.35,
                "duration_weight": 0.35,
                "risk_weight": 0.1,
                "clustering_weight": 0.2,
                "base_probability": 30
            }
        ]
    },
    ClinicalPattern.MIGRAINE: {
        "name": "Migraine/Neurological Pattern",
        "typical_symptoms": ["headache", "nausea", "light sensitivity", "sound sensitivity", "visual disturbances"],
        "supporting_symptoms": ["throbbing pain", "one-sided pain", "aura", "vomiting", "dizziness"],
        "typical_course": "Episodes lasting 4-72 hours; may have prodrome and postdrome phases",
        "conditions": [
            {
                "name": "Migraine without Aura",
                "key_symptoms": ["headache", "nausea", "light sensitivity", "sound sensitivity"],
                "symptom_weight": 0.45,
                "duration_weight": 0.2,
                "risk_weight": 0.15,
                "clustering_weight": 0.2,
                "base_probability": 40
            },
            {
                "name": "Migraine with Aura",
                "key_symptoms": ["headache", "visual disturbances", "aura", "nausea"],
                "symptom_weight": 0.5,
                "duration_weight": 0.15,
                "risk_weight": 0.15,
                "clustering_weight": 0.2,
                "base_probability": 25
            },
            {
                "name": "Tension Headache",
                "key_symptoms": ["headache", "pressure", "tight band sensation", "neck pain"],
                "symptom_weight": 0.4,
                "duration_weight": 0.25,
                "risk_weight": 0.1,
                "clustering_weight": 0.25,
                "base_probability": 45
            }
        ]
    },
    ClinicalPattern.ALLERGY: {
        "name": "Allergy Pattern",
        "typical_symptoms": ["sneezing", "itchy eyes", "runny nose", "congestion", "itchy throat"],
        "supporting_symptoms": ["watery eyes", "hives", "rash", "swelling", "wheezing"],
        "typical_course": "Symptoms correlate with allergen exposure; may be seasonal or perennial",
        "conditions": [
            {
                "name": "Allergic Rhinitis",
                "key_symptoms": ["sneezing", "itchy eyes", "runny nose", "congestion"],
                "symptom_weight": 0.45,
                "duration_weight": 0.2,
                "risk_weight": 0.15,
                "clustering_weight": 0.2,
                "base_probability": 55
            },
            {
                "name": "Contact Dermatitis",
                "key_symptoms": ["rash", "itching", "redness", "swelling"],
                "symptom_weight": 0.5,
                "duration_weight": 0.15,
                "risk_weight": 0.15,
                "clustering_weight": 0.2,
                "base_probability": 35
            },
            {
                "name": "Food Allergy",
                "key_symptoms": ["hives", "swelling", "itching", "nausea", "vomiting"],
                "symptom_weight": 0.45,
                "duration_weight": 0.15,
                "risk_weight": 0.2,
                "clustering_weight": 0.2,
                "base_probability": 20
            }
        ]
    },
    ClinicalPattern.GASTROINTESTINAL: {
        "name": "Gastrointestinal Pattern",
        "typical_symptoms": ["nausea", "vomiting", "diarrhea", "abdominal pain", "bloating"],
        "supporting_symptoms": ["cramping", "loss of appetite", "fever", "heartburn", "gas"],
        "typical_course": "Acute onset; infectious causes typically 1-3 days; chronic conditions persistent",
        "conditions": [
            {
                "name": "Viral Gastroenteritis",
                "key_symptoms": ["nausea", "vomiting", "diarrhea", "cramping", "low-grade fever"],
                "symptom_weight": 0.4,
                "duration_weight": 0.25,
                "risk_weight": 0.15,
                "clustering_weight": 0.2,
                "base_probability": 50
            },
            {
                "name": "Food Poisoning",
                "key_symptoms": ["nausea", "vomiting", "diarrhea", "abdominal pain"],
                "symptom_weight": 0.45,
                "duration_weight": 0.2,
                "risk_weight": 0.15,
                "clustering_weight": 0.2,
                "base_probability": 35
            },
            {
                "name": "GERD",
                "key_symptoms": ["heartburn", "regurgitation", "chest pain", "difficulty swallowing"],
                "symptom_weight": 0.45,
                "duration_weight": 0.3,
                "risk_weight": 0.1,
                "clustering_weight": 0.15,
                "base_probability": 40
            },
            {
                "name": "IBS",
                "key_symptoms": ["abdominal pain", "bloating", "diarrhea", "constipation", "cramping"],
                "symptom_weight": 0.35,
                "duration_weight": 0.35,
                "risk_weight": 0.1,
                "clustering_weight": 0.2,
                "base_probability": 25
            }
        ]
    },
    ClinicalPattern.RESPIRATORY: {
        "name": "Respiratory Pattern",
        "typical_symptoms": ["cough", "shortness of breath", "wheezing", "chest tightness", "sputum"],
        "supporting_symptoms": ["fever", "fatigue", "night sweats", "rapid breathing"],
        "typical_course": "Variable; acute infections days to weeks, chronic conditions ongoing",
        "conditions": [
            {
                "name": "Acute Bronchitis",
                "key_symptoms": ["cough", "sputum", "chest discomfort", "fatigue"],
                "symptom_weight": 0.45,
                "duration_weight": 0.25,
                "risk_weight": 0.1,
                "clustering_weight": 0.2,
                "base_probability": 45
            },
            {
                "name": "Asthma Exacerbation",
                "key_symptoms": ["wheezing", "shortness of breath", "chest tightness", "cough"],
                "symptom_weight": 0.4,
                "duration_weight": 0.2,
                "risk_weight": 0.2,
                "clustering_weight": 0.2,
                "base_probability": 30
            },
            {
                "name": "Pneumonia",
                "key_symptoms": ["fever", "cough", "sputum", "shortness of breath", "chest pain"],
                "symptom_weight": 0.4,
                "duration_weight": 0.2,
                "risk_weight": 0.2,
                "clustering_weight": 0.2,
                "base_probability": 20
            }
        ]
    },
    ClinicalPattern.MUSCULOSKELETAL: {
        "name": "Musculoskeletal Pattern",
        "typical_symptoms": ["joint pain", "muscle pain", "stiffness", "swelling", "limited mobility"],
        "supporting_symptoms": ["redness", "warmth", "weakness", "numbness", "tingling"],
        "typical_course": "Acute injuries days to weeks; chronic conditions with flares",
        "conditions": [
            {
                "name": "Muscle Strain",
                "key_symptoms": ["muscle pain", "stiffness", "limited mobility", "tenderness"],
                "symptom_weight": 0.45,
                "duration_weight": 0.25,
                "risk_weight": 0.1,
                "clustering_weight": 0.2,
                "base_probability": 50
            },
            {
                "name": "Osteoarthritis",
                "key_symptoms": ["joint pain", "stiffness", "limited mobility", "crepitus"],
                "symptom_weight": 0.4,
                "duration_weight": 0.3,
                "risk_weight": 0.15,
                "clustering_weight": 0.15,
                "base_probability": 35
            },
            {
                "name": "Rheumatoid Arthritis",
                "key_symptoms": ["joint pain", "swelling", "morning stiffness", "fatigue"],
                "symptom_weight": 0.4,
                "duration_weight": 0.25,
                "risk_weight": 0.2,
                "clustering_weight": 0.15,
                "base_probability": 15
            }
        ]
    },
    ClinicalPattern.CARDIAC: {
        "name": "Cardiac Pattern",
        "typical_symptoms": ["chest pain", "shortness of breath", "palpitations", "dizziness", "fatigue"],
        "supporting_symptoms": ["sweating", "nausea", "jaw pain", "arm pain", "swelling"],
        "typical_course": "Acute events require immediate attention; chronic conditions managed long-term",
        "conditions": [
            {
                "name": "Angina",
                "key_symptoms": ["chest pain", "chest pressure", "shortness of breath", "fatigue"],
                "symptom_weight": 0.4,
                "duration_weight": 0.2,
                "risk_weight": 0.25,
                "clustering_weight": 0.15,
                "base_probability": 25
            },
            {
                "name": "Heart Failure",
                "key_symptoms": ["shortness of breath", "fatigue", "swelling", "weight gain"],
                "symptom_weight": 0.4,
                "duration_weight": 0.25,
                "risk_weight": 0.2,
                "clustering_weight": 0.15,
                "base_probability": 15
            },
            {
                "name": "Arrhythmia",
                "key_symptoms": ["palpitations", "dizziness", "shortness of breath", "chest discomfort"],
                "symptom_weight": 0.45,
                "duration_weight": 0.2,
                "risk_weight": 0.2,
                "clustering_weight": 0.15,
                "base_probability": 20
            }
        ]
    },
    ClinicalPattern.EMERGENCY: {
        "name": "Emergency Red-Flag Pattern",
        "typical_symptoms": ["severe chest pain", "difficulty breathing", "sudden weakness", "severe headache", "loss of consciousness"],
        "supporting_symptoms": ["confusion", "vision changes", "slurred speech", "severe bleeding", "high fever"],
        "typical_course": "Sudden onset; requires immediate medical attention",
        "conditions": [
            {
                "name": "Possible Heart Attack",
                "key_symptoms": ["chest pain", "arm pain", "shortness of breath", "sweating", "nausea"],
                "symptom_weight": 0.4,
                "duration_weight": 0.15,
                "risk_weight": 0.3,
                "clustering_weight": 0.15,
                "base_probability": 10
            },
            {
                "name": "Possible Stroke",
                "key_symptoms": ["sudden weakness", "facial drooping", "slurred speech", "confusion", "severe headache"],
                "symptom_weight": 0.45,
                "duration_weight": 0.1,
                "risk_weight": 0.3,
                "clustering_weight": 0.15,
                "base_probability": 10
            },
            {
                "name": "Anaphylaxis",
                "key_symptoms": ["difficulty breathing", "swelling", "hives", "dizziness", "rapid pulse"],
                "symptom_weight": 0.5,
                "duration_weight": 0.1,
                "risk_weight": 0.25,
                "clustering_weight": 0.15,
                "base_probability": 5
            }
        ]
    }
}


# Red flag symptoms requiring immediate attention
RED_FLAG_SYMPTOMS = {
    "chest pain": {
        "concern": "Possible cardiac event",
        "action": "Seek emergency care if severe, with shortness of breath, or radiating to arm/jaw",
        "urgency": 90
    },
    "severe headache": {
        "concern": "Possible stroke, aneurysm, or meningitis",
        "action": "Seek emergency care if worst headache ever, sudden onset, or with neck stiffness",
        "urgency": 85
    },
    "shortness of breath": {
        "concern": "Possible cardiac, pulmonary, or anaphylactic emergency",
        "action": "Seek emergency care if sudden, severe, or with chest pain",
        "urgency": 80
    },
    "sudden weakness": {
        "concern": "Possible stroke",
        "action": "Use FAST assessment (Face, Arms, Speech, Time) - call 911 if positive",
        "urgency": 95
    },
    "difficulty breathing": {
        "concern": "Possible respiratory failure or anaphylaxis",
        "action": "Seek emergency care immediately",
        "urgency": 90
    },
    "loss of consciousness": {
        "concern": "Multiple possible serious causes",
        "action": "Seek emergency care for any unexplained loss of consciousness",
        "urgency": 95
    },
    "coughing blood": {
        "concern": "Possible pulmonary embolism, infection, or malignancy",
        "action": "Seek urgent medical evaluation",
        "urgency": 85
    },
    "blood in stool": {
        "concern": "Possible GI bleeding",
        "action": "Seek urgent evaluation if significant amount or with dizziness",
        "urgency": 75
    },
    "severe abdominal pain": {
        "concern": "Possible appendicitis, obstruction, or perforation",
        "action": "Seek emergency care if sudden, severe, or with fever",
        "urgency": 80
    },
    "high fever": {
        "concern": "Possible serious infection",
        "action": "Seek care if >103°F (39.4°C) or lasting >3 days",
        "urgency": 70
    }
}


class SymptomAnalyzer:
    """
    Clinical symptom analysis engine with pattern recognition and urgency assessment.

    Features:
    - Multi-pattern clinical analysis
    - Condition probability scoring with multiple factors
    - Urgency assessment and classification
    - Red flag detection
    - Personalized recommendations
    """

    def __init__(self):
        self._logger = logger

    async def analyze(self, request: SymptomAnalysisRequest) -> SymptomAnalysisResponse:
        """
        Perform comprehensive symptom analysis.

        Args:
            request: Symptom analysis request.

        Returns:
            Complete analysis with patterns, conditions, and recommendations.
        """
        self._logger.info(f"Analyzing symptoms: {request.symptoms}")

        # Normalize symptoms
        symptoms = [self._normalize_symptom(s) for s in request.symptoms]

        # Identify matching clinical patterns
        pattern_analyses = self._analyze_patterns(symptoms)

        # Calculate condition probabilities
        possible_conditions = self._calculate_condition_probabilities(
            symptoms,
            request.duration,
            request.risk_factors,
            request.age
        )

        # Perform urgency assessment
        urgency = self._assess_urgency(symptoms, possible_conditions, request.risk_factors, request.age)

        # Detect red flags
        red_flags = self._detect_red_flags(symptoms, request.risk_factors)

        # Calculate scores
        confidence = self._calculate_confidence(symptoms, pattern_analyses, possible_conditions)
        severity_score = self._calculate_severity_score(possible_conditions, urgency)

        # Build UI sections
        sections = self._build_sections(symptoms, pattern_analyses, possible_conditions, request)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            possible_conditions,
            urgency,
            symptoms,
            request
        )

        # Generate personalized advice
        personalized_advice = self._generate_personalized_advice(
            symptoms,
            possible_conditions,
            request
        )

        # Generate summary
        summary = self._generate_summary(symptoms, possible_conditions, urgency)

        return SymptomAnalysisResponse(
            summary=summary,
            scores=ScoreSet(
                confidence=confidence,
                severity=severity_score,
                urgency=urgency.score
            ),
            symptoms_analyzed=request.symptoms,
            duration=request.duration,
            risk_factors=request.risk_factors,
            pattern_analysis=pattern_analyses,
            possible_conditions=possible_conditions,
            urgency_assessment=urgency,
            personalized_advice=personalized_advice,
            sections=sections,
            red_flags=red_flags,
            recommendations=recommendations,
            alternatives=[],  # Symptom analyzer doesn't suggest medication alternatives
            metadata={
                "patterns_matched": len(pattern_analyses),
                "conditions_evaluated": len(possible_conditions),
                "analysis_version": "2.0",
                "patient_age": request.age,
                "medications": request.medications
            },
            timestamp=datetime.utcnow()
        )

    def _normalize_symptom(self, symptom: str) -> str:
        """Normalize symptom text for matching."""
        return symptom.lower().strip()

    def _analyze_patterns(self, symptoms: list[str]) -> list[PatternAnalysis]:
        """Identify matching clinical patterns."""
        pattern_results = []

        for pattern_type, pattern_data in CLINICAL_PATTERNS.items():
            # Count matching typical symptoms
            typical_matches = sum(
                1 for s in symptoms
                if any(ts in s or s in ts for ts in pattern_data["typical_symptoms"])
            )

            # Count supporting symptoms
            supporting_matches = sum(
                1 for s in symptoms
                if any(ss in s or s in ss for ss in pattern_data["supporting_symptoms"])
            )

            # Calculate pattern confidence
            total_typical = len(pattern_data["typical_symptoms"])
            match_ratio = typical_matches / total_typical if total_typical > 0 else 0
            confidence = int(match_ratio * 70 + (supporting_matches * 5))

            if confidence >= 20:  # Threshold for pattern match
                pattern_results.append(PatternAnalysis(
                    pattern=pattern_type,
                    confidence=min(95, confidence),
                    supporting_symptoms=[
                        s for s in symptoms
                        if any(ts in s or s in ts for ts in pattern_data["typical_symptoms"])
                    ],
                    typical_course=pattern_data["typical_course"]
                ))

        # Sort by confidence
        pattern_results.sort(key=lambda x: x.confidence, reverse=True)
        return pattern_results[:3]  # Return top 3 patterns

    def _calculate_condition_probabilities(
        self,
        symptoms: list[str],
        duration: Optional[str],
        risk_factors: list[str],
        age: Optional[int]
    ) -> list[ConditionMatch]:
        """Calculate multi-factor probability for each condition."""
        condition_results = []

        for pattern_type, pattern_data in CLINICAL_PATTERNS.items():
            for condition in pattern_data["conditions"]:
                # Calculate symptom match score
                symptom_score = self._calculate_symptom_match(symptoms, condition["key_symptoms"])

                # Calculate duration score
                duration_score = self._calculate_duration_score(duration, pattern_type)

                # Calculate risk factor score
                risk_score = self._calculate_risk_score(risk_factors, age, condition["name"])

                # Calculate clustering score
                clustering_score = self._calculate_clustering_score(symptoms, condition["key_symptoms"])

                # Weighted probability
                probability = int(
                    condition["base_probability"] * (
                        symptom_score * condition["symptom_weight"] +
                        duration_score * condition["duration_weight"] +
                        risk_score * condition["risk_weight"] +
                        clustering_score * condition["clustering_weight"]
                    ) / 100
                )

                # Identify matching and missing symptoms
                key_symptoms = condition["key_symptoms"]
                matching = [s for s in symptoms if any(k in s or s in k for k in key_symptoms)]
                missing = [k for k in key_symptoms if not any(k in s or s in k for s in symptoms)]

                if probability >= 10:  # Threshold
                    condition_results.append(ConditionMatch(
                        condition=condition["name"],
                        probability=min(95, max(5, probability)),
                        pattern=pattern_type,
                        scoring={
                            "symptom_match": symptom_score,
                            "duration_fit": duration_score,
                            "risk_factors": risk_score,
                            "clustering": clustering_score
                        },
                        key_symptoms=matching,
                        missing_symptoms=missing[:3],
                        risk_factors=[r for r in risk_factors if self._risk_relevant_to_condition(r, condition["name"])]
                    ))

        # Sort by probability and return top results
        condition_results.sort(key=lambda x: x.probability, reverse=True)
        return condition_results[:5]

    def _calculate_symptom_match(self, symptoms: list[str], key_symptoms: list[str]) -> int:
        """Calculate symptom match score (0-100)."""
        if not key_symptoms:
            return 0

        matches = sum(
            1 for key in key_symptoms
            if any(key in s or s in key for s in symptoms)
        )

        return int((matches / len(key_symptoms)) * 100)

    def _calculate_duration_score(self, duration: Optional[str], pattern: ClinicalPattern) -> int:
        """Calculate how well duration fits the pattern."""
        if not duration:
            return 50  # Neutral

        duration_lower = duration.lower()

        # Parse duration
        is_acute = any(term in duration_lower for term in ["hour", "today", "yesterday", "day", "1 day", "2 day"])
        is_short = any(term in duration_lower for term in ["few days", "3 day", "4 day", "5 day", "1 week"])
        is_medium = any(term in duration_lower for term in ["week", "2 week", "few week"])
        is_chronic = any(term in duration_lower for term in ["month", "year", "long time"])

        # Score based on pattern expectations
        if pattern in [ClinicalPattern.VIRAL, ClinicalPattern.GASTROINTESTINAL]:
            if is_short:
                return 90
            elif is_acute:
                return 70
            elif is_medium:
                return 50
            else:
                return 30
        elif pattern == ClinicalPattern.SINUSITIS:
            if is_medium:
                return 90
            elif is_short:
                return 70
            else:
                return 40
        elif pattern in [ClinicalPattern.ALLERGY, ClinicalPattern.MUSCULOSKELETAL]:
            if is_chronic:
                return 80
            elif is_medium:
                return 70
            else:
                return 50
        elif pattern == ClinicalPattern.EMERGENCY:
            if is_acute:
                return 90
            else:
                return 60

        return 50

    def _calculate_risk_score(
        self,
        risk_factors: list[str],
        age: Optional[int],
        condition_name: str
    ) -> int:
        """Calculate risk factor contribution."""
        score = 50  # Baseline

        risk_lower = [r.lower() for r in risk_factors]

        # Age-based adjustments
        if age:
            if age > 65:
                if "Pneumonia" in condition_name or "Heart" in condition_name:
                    score += 20
            elif age < 5:
                if "Viral" in condition_name:
                    score += 15

        # Condition-specific risk factors
        risk_mappings = {
            "diabetes": ["Pneumonia", "Heart", "Infection"],
            "hypertension": ["Heart", "Stroke", "Angina"],
            "asthma": ["Asthma", "Respiratory", "Bronchitis"],
            "smoking": ["Respiratory", "Heart", "Bronchitis", "Pneumonia"],
            "obesity": ["Heart", "GERD", "Arthritis"],
            "immunocompromised": ["Infection", "Pneumonia", "COVID"],
            "copd": ["Respiratory", "Bronchitis", "Pneumonia"]
        }

        for risk, conditions in risk_mappings.items():
            if any(risk in r for r in risk_lower):
                if any(c in condition_name for c in conditions):
                    score += 15

        return min(100, score)

    def _calculate_clustering_score(self, symptoms: list[str], key_symptoms: list[str]) -> int:
        """Calculate symptom clustering score (related symptoms appearing together)."""
        matches = sum(1 for k in key_symptoms if any(k in s or s in k for s in symptoms))

        if matches >= 4:
            return 95
        elif matches >= 3:
            return 75
        elif matches >= 2:
            return 55
        elif matches >= 1:
            return 35
        return 10

    def _risk_relevant_to_condition(self, risk: str, condition: str) -> bool:
        """Check if risk factor is relevant to condition."""
        risk_lower = risk.lower()
        condition_lower = condition.lower()

        relevance_map = {
            "diabetes": ["infection", "heart", "pneumonia"],
            "hypertension": ["heart", "stroke", "angina"],
            "asthma": ["asthma", "respiratory", "bronchitis"],
            "smoking": ["respiratory", "heart", "bronchitis"],
            "heart disease": ["heart", "angina", "arrhythmia"]
        }

        for risk_key, conditions in relevance_map.items():
            if risk_key in risk_lower:
                if any(c in condition_lower for c in conditions):
                    return True
        return False

    def _assess_urgency(
        self,
        symptoms: list[str],
        conditions: list[ConditionMatch],
        risk_factors: list[str],
        age: Optional[int]
    ) -> UrgencyAssessment:
        """Assess urgency level based on symptoms and findings."""
        urgency_score = 20  # Base score
        reasons = []

        # Check for red flag symptoms
        for symptom in symptoms:
            for flag, details in RED_FLAG_SYMPTOMS.items():
                if flag in symptom or symptom in flag:
                    urgency_score = max(urgency_score, details["urgency"])
                    reasons.append(f"Red flag: {flag}")
                    break

        # Check for emergency patterns in conditions
        emergency_conditions = ["Heart Attack", "Stroke", "Anaphylaxis", "Pneumonia"]
        for condition in conditions:
            if any(e in condition.condition for e in emergency_conditions):
                if condition.probability > 30:
                    urgency_score = max(urgency_score, 75)
                    reasons.append(f"Possible {condition.condition}")

        # Age-based adjustments
        if age:
            if age > 65 or age < 2:
                urgency_score = min(100, urgency_score + 10)
                reasons.append("Age-related vulnerability")

        # Risk factor adjustments
        high_risk = ["immunocompromised", "heart disease", "copd", "cancer"]
        if any(r.lower() in high_risk for r in risk_factors):
            urgency_score = min(100, urgency_score + 10)
            reasons.append("High-risk medical condition")

        # Determine urgency level
        if urgency_score >= 80:
            level = UrgencyLevel.IMMEDIATE
            timeframe = "Seek emergency care now (call 911 if needed)"
        elif urgency_score >= 60:
            level = UrgencyLevel.URGENT
            timeframe = "Seek medical care within hours"
        elif urgency_score >= 40:
            level = UrgencyLevel.SOON
            timeframe = "Schedule appointment within 1-2 days"
        elif urgency_score >= 25:
            level = UrgencyLevel.ROUTINE
            timeframe = "Schedule routine appointment"
        else:
            level = UrgencyLevel.REASSURING
            timeframe = "Self-care appropriate; monitor symptoms"

        reasoning = "; ".join(reasons) if reasons else "No urgent findings identified"

        return UrgencyAssessment(
            score=urgency_score,
            level=level,
            reasoning=reasoning,
            timeframe=timeframe
        )

    def _detect_red_flags(
        self,
        symptoms: list[str],
        risk_factors: list[str]
    ) -> list[RedFlag]:
        """Detect red flag symptoms requiring attention."""
        red_flags = []

        for symptom in symptoms:
            for flag, details in RED_FLAG_SYMPTOMS.items():
                if flag in symptom or symptom in flag:
                    severity = SeverityLevel.CRITICAL if details["urgency"] >= 85 else SeverityLevel.SEVERE
                    red_flags.append(RedFlag(
                        flag=f"Red Flag: {flag.title()}",
                        explanation=details["concern"],
                        action=details["action"],
                        severity=severity
                    ))
                    break

        # Additional red flags based on combinations
        symptom_text = " ".join(symptoms).lower()
        if "chest pain" in symptom_text and "shortness of breath" in symptom_text:
            red_flags.append(RedFlag(
                flag="Combined cardiac symptoms",
                explanation="Chest pain with shortness of breath may indicate cardiac emergency",
                action="Seek emergency evaluation immediately",
                severity=SeverityLevel.CRITICAL
            ))

        return red_flags

    def _calculate_confidence(
        self,
        symptoms: list[str],
        patterns: list[PatternAnalysis],
        conditions: list[ConditionMatch]
    ) -> int:
        """Calculate analysis confidence."""
        confidence = 60  # Base

        # More symptoms = more information
        if len(symptoms) >= 3:
            confidence += 10
        elif len(symptoms) >= 5:
            confidence += 15

        # Clear pattern match increases confidence
        if patterns and patterns[0].confidence > 60:
            confidence += 15

        # Consistent conditions increase confidence
        if conditions:
            top_condition = conditions[0]
            if top_condition.probability > 50:
                confidence += 10

        return min(95, confidence)

    def _calculate_severity_score(
        self,
        conditions: list[ConditionMatch],
        urgency: UrgencyAssessment
    ) -> int:
        """Calculate overall severity score."""
        # Base on urgency
        severity = urgency.score

        # Adjust based on top condition
        if conditions:
            top = conditions[0]
            if "Heart" in top.condition or "Stroke" in top.condition:
                severity = max(severity, 70)
            elif "Pneumonia" in top.condition:
                severity = max(severity, 60)

        return severity

    def _build_sections(
        self,
        symptoms: list[str],
        patterns: list[PatternAnalysis],
        conditions: list[ConditionMatch],
        request: SymptomAnalysisRequest
    ) -> list[UISection]:
        """Build UI sections for response."""
        sections = []

        # Urgency section first if high
        urgency = self._assess_urgency(symptoms, conditions, request.risk_factors, request.age)
        if urgency.score >= 60:
            sections.append(UISection(
                title="Urgency Assessment",
                icon="alert-circle",
                items=[
                    SectionItem(
                        label="Urgency Level",
                        value=urgency.level.value.replace("_", " ").title(),
                        badge=f"{urgency.score}/100",
                        severity="critical" if urgency.score >= 80 else "severe",
                        explanation=urgency.timeframe
                    )
                ],
                priority=0
            ))

        # Pattern analysis section
        if patterns:
            pattern_items = []
            for p in patterns:
                pattern_items.append(SectionItem(
                    label=CLINICAL_PATTERNS[p.pattern]["name"],
                    value=p.typical_course,
                    score=p.confidence,
                    badge=f"{p.confidence}% MATCH",
                    explanation=f"Matching symptoms: {', '.join(p.supporting_symptoms[:3])}"
                ))

            sections.append(UISection(
                title="Clinical Pattern Analysis",
                icon="stethoscope",
                items=pattern_items,
                priority=1
            ))

        # Possible conditions section
        if conditions:
            condition_items = []
            for c in conditions[:4]:
                condition_items.append(SectionItem(
                    label=c.condition,
                    value=f"Missing: {', '.join(c.missing_symptoms)}" if c.missing_symptoms else "Good symptom match",
                    probability=float(c.probability),
                    badge=f"{c.probability}%",
                    severity="severe" if c.probability > 50 else "moderate" if c.probability > 30 else "mild",
                    explanation=f"Key matches: {', '.join(c.key_symptoms[:3])}"
                ))

            sections.append(UISection(
                title="Possible Conditions",
                icon="clipboard-list",
                items=condition_items,
                expandable=True,
                priority=2
            ))

        # Symptom breakdown section
        sections.append(UISection(
            title="Symptoms Analyzed",
            icon="list",
            items=[
                SectionItem(
                    label=s.title(),
                    value=self._get_symptom_significance(s, conditions),
                    severity="mild"
                ) for s in symptoms
            ],
            priority=3
        ))

        return sections

    def _get_symptom_significance(self, symptom: str, conditions: list[ConditionMatch]) -> str:
        """Get significance description for a symptom."""
        # Check which conditions this symptom supports
        supporting = []
        for c in conditions:
            if any(symptom in k or k in symptom for k in c.key_symptoms):
                supporting.append(c.condition)

        if supporting:
            return f"Key symptom for: {', '.join(supporting[:2])}"
        return "Additional symptom"

    def _generate_recommendations(
        self,
        conditions: list[ConditionMatch],
        urgency: UrgencyAssessment,
        symptoms: list[str],
        request: SymptomAnalysisRequest
    ) -> list[Recommendation]:
        """Generate actionable recommendations."""
        recommendations = []

        # Urgency-based recommendations
        if urgency.level == UrgencyLevel.IMMEDIATE:
            recommendations.append(Recommendation(
                title="Seek Emergency Care",
                description="Your symptoms require immediate medical evaluation. Call 911 or go to emergency room.",
                priority="high",
                category="follow-up",
                timeframe="Immediately"
            ))
        elif urgency.level == UrgencyLevel.URGENT:
            recommendations.append(Recommendation(
                title="Urgent Medical Evaluation",
                description="Contact your healthcare provider today or visit urgent care.",
                priority="high",
                category="follow-up",
                timeframe="Within hours"
            ))
        elif urgency.level == UrgencyLevel.SOON:
            recommendations.append(Recommendation(
                title="Schedule Appointment",
                description="Contact your healthcare provider to schedule an appointment soon.",
                priority="medium",
                category="follow-up",
                timeframe="Within 1-2 days"
            ))

        # Condition-specific recommendations
        if conditions:
            top_condition = conditions[0].condition

            if "Viral" in top_condition or "Cold" in top_condition or "Influenza" in top_condition:
                recommendations.append(Recommendation(
                    title="Supportive Care",
                    description="Rest, stay hydrated, use OTC medications for symptom relief. Most viral infections resolve in 7-10 days.",
                    priority="medium",
                    category="lifestyle",
                    timeframe="Ongoing"
                ))
            elif "GERD" in top_condition:
                recommendations.append(Recommendation(
                    title="Dietary Modifications",
                    description="Avoid trigger foods (spicy, fatty, acidic), eat smaller meals, don't lie down after eating.",
                    priority="medium",
                    category="lifestyle",
                    timeframe="Ongoing"
                ))
            elif "Migraine" in top_condition or "Headache" in top_condition:
                recommendations.append(Recommendation(
                    title="Headache Management",
                    description="Rest in a dark, quiet room. OTC pain relievers may help. Keep a headache diary to identify triggers.",
                    priority="medium",
                    category="lifestyle",
                    timeframe="During episodes"
                ))

        # Monitoring recommendation
        recommendations.append(Recommendation(
            title="Monitor Symptoms",
            description="Track your symptoms including severity, frequency, and any new symptoms. Seek care if worsening.",
            priority="medium",
            category="monitoring",
            timeframe="Ongoing"
        ))

        return recommendations

    def _generate_personalized_advice(
        self,
        symptoms: list[str],
        conditions: list[ConditionMatch],
        request: SymptomAnalysisRequest
    ) -> list[str]:
        """Generate personalized advice based on symptoms and factors."""
        advice = []

        # Duration-based advice
        if request.duration:
            duration_lower = request.duration.lower()
            if any(term in duration_lower for term in ["week", "month", "long"]):
                advice.append("Persistent symptoms lasting more than a week warrant medical evaluation.")

        # Age-based advice
        if request.age:
            if request.age > 65:
                advice.append("Adults over 65 should seek prompt evaluation for respiratory symptoms or fever.")
            elif request.age < 5:
                advice.append("Young children should be evaluated promptly for high fever or breathing difficulty.")

        # Medication-based advice
        if request.medications:
            advice.append(f"Consider whether current medications ({', '.join(request.medications[:2])}) might be contributing to symptoms.")

        # Risk factor advice
        if request.risk_factors:
            advice.append(f"Your risk factors ({', '.join(request.risk_factors[:2])}) may influence symptom significance.")

        # Symptom-specific advice
        symptom_advice = {
            "fever": "Monitor temperature regularly. Seek care if >103°F or lasting >3 days.",
            "cough": "Stay hydrated. Seek care if producing blood or lasting >3 weeks.",
            "headache": "Track frequency and triggers. Seek care if sudden, severe, or with neurological symptoms.",
            "fatigue": "Ensure adequate sleep and hydration. Seek care if severe or lasting >2 weeks."
        }

        for symptom in symptoms:
            for key, advice_text in symptom_advice.items():
                if key in symptom:
                    advice.append(advice_text)
                    break

        return advice[:5]

    def _generate_summary(
        self,
        symptoms: list[str],
        conditions: list[ConditionMatch],
        urgency: UrgencyAssessment
    ) -> str:
        """Generate concise summary."""
        symptom_count = len(symptoms)

        if urgency.level in [UrgencyLevel.IMMEDIATE, UrgencyLevel.URGENT]:
            return (f"Analysis of {symptom_count} symptoms indicates {urgency.level.value} urgency. "
                   f"{urgency.timeframe}.")

        if conditions:
            top = conditions[0]
            return (f"Analysis of {symptom_count} symptoms suggests {top.condition} ({top.probability}% probability) "
                   f"as the most likely cause. {urgency.timeframe}.")

        return f"Analysis of {symptom_count} symptoms complete. {urgency.timeframe}."


# Singleton instance
_analyzer_instance: Optional[SymptomAnalyzer] = None


async def get_symptom_analyzer() -> SymptomAnalyzer:
    """Get or create SymptomAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SymptomAnalyzer()
    return _analyzer_instance
