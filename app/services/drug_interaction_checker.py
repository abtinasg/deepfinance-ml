"""
DeepHealth Drug Interaction Checker Service

Professional drug interaction analysis with multi-layer scoring engine,
pharmacodynamic/pharmacokinetic analysis, and UI-optimized responses.
"""

from datetime import datetime
from typing import Any, Optional

from app.core.logging import get_logger
from app.schemas.medical import (
    Alternative,
    BadgeInfo,
    DrugInteraction,
    DrugInteractionRequest,
    DrugInteractionResponse,
    InteractionCategory,
    InteractionScoreBreakdown,
    Recommendation,
    RedFlag,
    ScoreSet,
    SectionItem,
    SeverityLevel,
    UISection,
)

logger = get_logger(__name__)


# ============================================================================
# DRUG DATABASE - Comprehensive Drug Information
# ============================================================================

DRUG_DATABASE = {
    # Anticoagulants & Antiplatelets
    "warfarin": {
        "class": "anticoagulant",
        "mechanism": "vitamin_k_antagonist",
        "metabolism": ["CYP2C9", "CYP3A4", "CYP1A2"],
        "half_life": 40,  # hours
        "organ_load": ["liver", "blood"],
        "narrow_therapeutic_index": True,
        "common_interactions": ["aspirin", "ibuprofen", "naproxen", "acetaminophen"],
        "alternatives": ["apixaban", "rivaroxaban", "dabigatran"]
    },
    "aspirin": {
        "class": "antiplatelet",
        "mechanism": "cox_inhibitor",
        "metabolism": ["CYP2C9"],
        "half_life": 3,
        "organ_load": ["gi", "blood", "kidney"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["warfarin", "ibuprofen", "clopidogrel"],
        "alternatives": ["clopidogrel", "dipyridamole"]
    },
    "clopidogrel": {
        "class": "antiplatelet",
        "mechanism": "p2y12_inhibitor",
        "metabolism": ["CYP2C19", "CYP3A4"],
        "half_life": 6,
        "organ_load": ["blood", "liver"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["omeprazole", "aspirin", "warfarin"],
        "alternatives": ["prasugrel", "ticagrelor"]
    },

    # NSAIDs
    "ibuprofen": {
        "class": "nsaid",
        "mechanism": "cox_inhibitor",
        "metabolism": ["CYP2C9", "CYP2C19"],
        "half_life": 2,
        "organ_load": ["gi", "kidney", "cardiovascular"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["aspirin", "warfarin", "lisinopril", "methotrexate"],
        "alternatives": ["acetaminophen", "celecoxib", "naproxen"]
    },
    "naproxen": {
        "class": "nsaid",
        "mechanism": "cox_inhibitor",
        "metabolism": ["CYP2C9"],
        "half_life": 14,
        "organ_load": ["gi", "kidney", "cardiovascular"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["aspirin", "warfarin", "lisinopril"],
        "alternatives": ["acetaminophen", "ibuprofen", "celecoxib"]
    },

    # Cardiovascular
    "lisinopril": {
        "class": "ace_inhibitor",
        "mechanism": "ace_inhibition",
        "metabolism": ["none"],  # Renally eliminated
        "half_life": 12,
        "organ_load": ["kidney"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["potassium", "spironolactone", "ibuprofen", "naproxen"],
        "alternatives": ["losartan", "amlodipine", "hydrochlorothiazide"]
    },
    "metoprolol": {
        "class": "beta_blocker",
        "mechanism": "beta_adrenergic_blockade",
        "metabolism": ["CYP2D6"],
        "half_life": 4,
        "organ_load": ["cardiovascular", "liver"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["verapamil", "diltiazem", "clonidine"],
        "alternatives": ["carvedilol", "atenolol", "bisoprolol"]
    },
    "amlodipine": {
        "class": "calcium_channel_blocker",
        "mechanism": "calcium_channel_blockade",
        "metabolism": ["CYP3A4"],
        "half_life": 40,
        "organ_load": ["cardiovascular", "liver"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["simvastatin", "cyclosporine"],
        "alternatives": ["nifedipine", "diltiazem", "verapamil"]
    },

    # Statins
    "simvastatin": {
        "class": "statin",
        "mechanism": "hmg_coa_reductase_inhibitor",
        "metabolism": ["CYP3A4"],
        "half_life": 3,
        "organ_load": ["liver", "muscle"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["amlodipine", "erythromycin", "grapefruit"],
        "alternatives": ["atorvastatin", "rosuvastatin", "pravastatin"]
    },
    "atorvastatin": {
        "class": "statin",
        "mechanism": "hmg_coa_reductase_inhibitor",
        "metabolism": ["CYP3A4"],
        "half_life": 14,
        "organ_load": ["liver", "muscle"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["erythromycin", "clarithromycin", "cyclosporine"],
        "alternatives": ["rosuvastatin", "pravastatin", "simvastatin"]
    },

    # Diabetes
    "metformin": {
        "class": "biguanide",
        "mechanism": "ampk_activation",
        "metabolism": ["none"],  # Renally eliminated
        "half_life": 5,
        "organ_load": ["kidney", "gi"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["contrast_dye", "alcohol"],
        "alternatives": ["glipizide", "sitagliptin", "empagliflozin"]
    },
    "glipizide": {
        "class": "sulfonylurea",
        "mechanism": "insulin_secretagogue",
        "metabolism": ["CYP2C9"],
        "half_life": 4,
        "organ_load": ["liver", "pancreas"],
        "narrow_therapeutic_index": True,
        "common_interactions": ["fluconazole", "warfarin", "alcohol"],
        "alternatives": ["metformin", "sitagliptin", "glimepiride"]
    },

    # PPIs
    "omeprazole": {
        "class": "ppi",
        "mechanism": "proton_pump_inhibition",
        "metabolism": ["CYP2C19", "CYP3A4"],
        "half_life": 1,
        "organ_load": ["gi", "bone"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["clopidogrel", "methotrexate", "digoxin"],
        "alternatives": ["pantoprazole", "esomeprazole", "famotidine"]
    },

    # Antibiotics
    "amoxicillin": {
        "class": "penicillin",
        "mechanism": "cell_wall_synthesis_inhibition",
        "metabolism": ["none"],  # Renally eliminated
        "half_life": 1,
        "organ_load": ["kidney", "gi"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["warfarin", "methotrexate"],
        "alternatives": ["azithromycin", "cephalexin", "doxycycline"]
    },
    "ciprofloxacin": {
        "class": "fluoroquinolone",
        "mechanism": "dna_gyrase_inhibition",
        "metabolism": ["CYP1A2"],
        "half_life": 4,
        "organ_load": ["kidney", "tendon", "gi"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["theophylline", "warfarin", "antacids", "caffeine"],
        "alternatives": ["levofloxacin", "azithromycin", "doxycycline"]
    },

    # Pain/Analgesics
    "acetaminophen": {
        "class": "analgesic",
        "mechanism": "central_prostaglandin_inhibition",
        "metabolism": ["CYP2E1", "CYP1A2", "CYP3A4"],
        "half_life": 3,
        "organ_load": ["liver"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["warfarin", "alcohol"],
        "alternatives": ["ibuprofen", "naproxen"]
    },
    "tramadol": {
        "class": "opioid",
        "mechanism": "mu_opioid_agonist_snri",
        "metabolism": ["CYP2D6", "CYP3A4"],
        "half_life": 6,
        "organ_load": ["cns", "liver", "respiratory"],
        "narrow_therapeutic_index": True,
        "common_interactions": ["ssri", "snri", "maoi", "carbamazepine"],
        "alternatives": ["acetaminophen", "ibuprofen", "codeine"]
    },

    # Antidepressants
    "sertraline": {
        "class": "ssri",
        "mechanism": "serotonin_reuptake_inhibition",
        "metabolism": ["CYP2C19", "CYP2D6", "CYP3A4"],
        "half_life": 26,
        "organ_load": ["cns", "liver"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["tramadol", "warfarin", "nsaid", "maoi"],
        "alternatives": ["escitalopram", "fluoxetine", "bupropion"]
    },
    "fluoxetine": {
        "class": "ssri",
        "mechanism": "serotonin_reuptake_inhibition",
        "metabolism": ["CYP2D6", "CYP2C9"],
        "half_life": 96,  # Including active metabolite
        "organ_load": ["cns", "liver"],
        "narrow_therapeutic_index": False,
        "common_interactions": ["tramadol", "maoi", "tamoxifen", "warfarin"],
        "alternatives": ["sertraline", "escitalopram", "bupropion"]
    },

    # Thyroid
    "levothyroxine": {
        "class": "thyroid_hormone",
        "mechanism": "thyroid_hormone_replacement",
        "metabolism": ["hepatic_deiodination"],
        "half_life": 168,  # 7 days
        "organ_load": ["cardiovascular", "bone"],
        "narrow_therapeutic_index": True,
        "common_interactions": ["calcium", "iron", "antacids", "cholestyramine"],
        "alternatives": ["liothyronine", "desiccated_thyroid"]
    },
}

# ============================================================================
# INTERACTION DATABASE - Known Drug Interactions
# ============================================================================

INTERACTION_DATABASE = {
    ("warfarin", "aspirin"): {
        "severity": "severe",
        "mechanism": "Both inhibit hemostasis through different mechanisms, creating additive bleeding risk.",
        "pk_score": 30,
        "pd_score": 90,
        "clinical_effect": "Significantly increased risk of major bleeding including GI and intracranial hemorrhage.",
        "monitoring": ["INR every 2-3 days initially", "Watch for signs of bleeding", "Stool guaiac testing"],
        "timing": "If must use together, take aspirin at same time daily to maintain consistent INR"
    },
    ("warfarin", "ibuprofen"): {
        "severity": "severe",
        "mechanism": "NSAIDs inhibit platelet function and can displace warfarin from protein binding.",
        "pk_score": 50,
        "pd_score": 85,
        "clinical_effect": "Increased bleeding risk and unpredictable INR elevation.",
        "monitoring": ["Check INR within 3-5 days", "Monitor for bruising/bleeding", "GI protection"],
        "timing": "Avoid combination; use acetaminophen for pain instead"
    },
    ("aspirin", "ibuprofen"): {
        "severity": "moderate",
        "mechanism": "Ibuprofen may block aspirin's irreversible platelet inhibition when taken before aspirin.",
        "pk_score": 20,
        "pd_score": 75,
        "clinical_effect": "Reduced cardioprotective effect of aspirin; increased GI bleeding risk.",
        "monitoring": ["Take aspirin 30 min before ibuprofen", "Monitor for GI symptoms"],
        "timing": "Take aspirin first, wait 30 minutes, then ibuprofen"
    },
    ("clopidogrel", "omeprazole"): {
        "severity": "moderate",
        "mechanism": "Omeprazole inhibits CYP2C19, reducing conversion of clopidogrel to active metabolite.",
        "pk_score": 80,
        "pd_score": 20,
        "clinical_effect": "Reduced antiplatelet effect may increase cardiovascular event risk.",
        "monitoring": ["Consider pantoprazole instead", "Monitor for cardiovascular symptoms"],
        "timing": "If needed, separate doses by 12 hours"
    },
    ("metoprolol", "verapamil"): {
        "severity": "severe",
        "mechanism": "Both drugs suppress cardiac conduction and contractility through different mechanisms.",
        "pk_score": 40,
        "pd_score": 85,
        "clinical_effect": "Risk of severe bradycardia, AV block, and heart failure.",
        "monitoring": ["ECG monitoring", "Heart rate checks", "Blood pressure monitoring"],
        "timing": "Avoid combination; consider alternative antihypertensive"
    },
    ("lisinopril", "potassium"): {
        "severity": "moderate",
        "mechanism": "ACE inhibitors reduce aldosterone, decreasing potassium excretion.",
        "pk_score": 10,
        "pd_score": 70,
        "clinical_effect": "Risk of hyperkalemia with cardiac arrhythmia potential.",
        "monitoring": ["Check potassium levels within 1 week", "ECG if symptoms", "Monitor renal function"],
        "timing": "Limit potassium supplements; monitor closely if necessary"
    },
    ("simvastatin", "amlodipine"): {
        "severity": "moderate",
        "mechanism": "Amlodipine inhibits CYP3A4, increasing simvastatin levels and myopathy risk.",
        "pk_score": 75,
        "pd_score": 30,
        "clinical_effect": "Increased risk of rhabdomyolysis and myopathy.",
        "monitoring": ["Limit simvastatin to 20mg", "Watch for muscle pain/weakness", "Check CK if symptomatic"],
        "timing": "Consider switching to pravastatin or rosuvastatin"
    },
    ("ciprofloxacin", "theophylline"): {
        "severity": "severe",
        "mechanism": "Ciprofloxacin inhibits CYP1A2, markedly increasing theophylline levels.",
        "pk_score": 90,
        "pd_score": 30,
        "clinical_effect": "Risk of theophylline toxicity: seizures, arrhythmias, death.",
        "monitoring": ["Check theophylline levels", "Reduce dose by 40-50%", "Monitor for toxicity signs"],
        "timing": "Avoid combination; use alternative antibiotic"
    },
    ("sertraline", "tramadol"): {
        "severity": "severe",
        "mechanism": "Both increase serotonin; tramadol also has serotonin reuptake inhibition.",
        "pk_score": 40,
        "pd_score": 85,
        "clinical_effect": "Risk of serotonin syndrome: hyperthermia, rigidity, clonus, autonomic instability.",
        "monitoring": ["Watch for serotonin syndrome signs", "Monitor mental status", "Check vital signs"],
        "timing": "Avoid combination; use acetaminophen or carefully monitored opioid"
    },
    ("fluoxetine", "tramadol"): {
        "severity": "severe",
        "mechanism": "Fluoxetine inhibits CYP2D6 and increases serotonin; both increase serotonin syndrome risk.",
        "pk_score": 70,
        "pd_score": 85,
        "clinical_effect": "High risk of serotonin syndrome and reduced tramadol analgesia.",
        "monitoring": ["Avoid combination", "If used, monitor closely for 5 weeks after fluoxetine stop"],
        "timing": "Do not combine; fluoxetine has long half-life"
    },
    ("levothyroxine", "calcium"): {
        "severity": "mild",
        "mechanism": "Calcium chelates levothyroxine in GI tract, reducing absorption.",
        "pk_score": 80,
        "pd_score": 10,
        "clinical_effect": "Reduced levothyroxine efficacy; hypothyroid symptoms may persist.",
        "monitoring": ["Check TSH in 6-8 weeks", "Monitor for hypothyroid symptoms"],
        "timing": "Separate doses by at least 4 hours"
    },
    ("metformin", "alcohol"): {
        "severity": "moderate",
        "mechanism": "Both can cause lactic acidosis; alcohol also affects gluconeogenesis.",
        "pk_score": 20,
        "pd_score": 65,
        "clinical_effect": "Increased risk of lactic acidosis and hypoglycemia.",
        "monitoring": ["Limit alcohol intake", "Monitor for lactic acidosis symptoms", "Check blood glucose"],
        "timing": "Limit to 1-2 drinks; avoid binge drinking"
    },
    ("warfarin", "acetaminophen"): {
        "severity": "mild",
        "mechanism": "High-dose acetaminophen may interfere with vitamin K-dependent clotting factors.",
        "pk_score": 40,
        "pd_score": 35,
        "clinical_effect": "Potential INR elevation with regular high-dose use (>2g/day).",
        "monitoring": ["Monitor INR if using >2g/day for >3 days", "Watch for bleeding"],
        "timing": "Preferred over NSAIDs; keep dose <2g/day"
    },
}


class DrugInteractionChecker:
    """
    Comprehensive drug interaction analysis service with multi-layer scoring.

    Features:
    - Pharmacodynamic and pharmacokinetic interaction analysis
    - Mechanism duplication detection
    - Metabolic pathway conflict identification
    - Half-life overlap assessment
    - Organ system load calculation
    - Weighted safety scoring
    - Alternative medication suggestions
    """

    def __init__(self):
        self._logger = logger

    async def analyze(self, request: DrugInteractionRequest) -> DrugInteractionResponse:
        """
        Perform comprehensive drug interaction analysis.

        Args:
            request: Drug interaction analysis request.

        Returns:
            Complete analysis with scores, interactions, and recommendations.
        """
        self._logger.info(f"Analyzing interactions for drugs: {request.drugs}")

        # Normalize drug names
        drugs = [self._normalize_drug_name(d) for d in request.drugs]
        unknown_drugs = [d for d in drugs if d not in DRUG_DATABASE]
        known_drugs = [d for d in drugs if d in DRUG_DATABASE]

        # Find all interactions
        interactions = self._find_interactions(known_drugs)

        # Calculate comprehensive scores
        score_breakdown = self._calculate_score_breakdown(known_drugs, interactions)
        safety_score = self._calculate_safety_score(score_breakdown, interactions)

        # Determine overall severity
        overall_severity = self._determine_overall_severity(interactions, safety_score)
        confidence = self._calculate_confidence(len(known_drugs), len(unknown_drugs), len(interactions))

        # Generate verdict
        verdict = self._generate_verdict(safety_score, interactions)

        # Build UI sections
        sections = self._build_sections(drugs, interactions, score_breakdown, unknown_drugs)

        # Generate red flags
        red_flags = self._generate_red_flags(interactions, known_drugs)

        # Generate recommendations
        recommendations = self._generate_recommendations(interactions, known_drugs)

        # Generate alternatives
        alternatives = self._generate_alternatives(interactions, known_drugs)

        # Build summary
        summary = self._generate_summary(drugs, interactions, safety_score)

        return DrugInteractionResponse(
            summary=summary,
            scores=ScoreSet(
                confidence=confidence,
                severity=100 - safety_score,
                safety=safety_score
            ),
            drugs_analyzed=request.drugs,
            interactions=[self._format_interaction(i) for i in interactions],
            score_breakdown=score_breakdown,
            safety_score=safety_score,
            combination_verdict=verdict,
            sections=sections,
            red_flags=red_flags,
            recommendations=recommendations,
            alternatives=alternatives,
            metadata={
                "drugs_found": len(known_drugs),
                "drugs_unknown": len(unknown_drugs),
                "interactions_found": len(interactions),
                "analysis_version": "2.0",
                "patient_age": request.patient_age,
                "patient_conditions": request.patient_conditions
            },
            timestamp=datetime.utcnow()
        )

    def _normalize_drug_name(self, drug: str) -> str:
        """Normalize drug name for database lookup."""
        return drug.lower().strip().replace("-", "_").replace(" ", "_")

    def _find_interactions(self, drugs: list[str]) -> list[dict[str, Any]]:
        """Find all known interactions between the given drugs."""
        interactions = []

        for i, drug_a in enumerate(drugs):
            for drug_b in drugs[i + 1:]:
                # Check both orderings
                key = (drug_a, drug_b)
                reverse_key = (drug_b, drug_a)

                interaction_data = None
                if key in INTERACTION_DATABASE:
                    interaction_data = INTERACTION_DATABASE[key]
                elif reverse_key in INTERACTION_DATABASE:
                    interaction_data = INTERACTION_DATABASE[reverse_key]
                else:
                    # Check for class-based interactions
                    interaction_data = self._check_class_interaction(drug_a, drug_b)

                if interaction_data:
                    interactions.append({
                        "drug_a": drug_a,
                        "drug_b": drug_b,
                        **interaction_data
                    })

        return interactions

    def _check_class_interaction(self, drug_a: str, drug_b: str) -> Optional[dict[str, Any]]:
        """Check for pharmacological class-based interactions."""
        if drug_a not in DRUG_DATABASE or drug_b not in DRUG_DATABASE:
            return None

        info_a = DRUG_DATABASE[drug_a]
        info_b = DRUG_DATABASE[drug_b]

        # Check for mechanism duplication
        if info_a["mechanism"] == info_b["mechanism"]:
            return {
                "severity": "moderate",
                "mechanism": f"Both drugs work via {info_a['mechanism'].replace('_', ' ')}, causing additive effects.",
                "pk_score": 20,
                "pd_score": 60,
                "clinical_effect": "Increased pharmacological effect and potential toxicity from mechanism duplication.",
                "monitoring": ["Monitor for excessive drug effect", "Consider dose reduction"],
                "timing": "Avoid combination or reduce doses of both drugs"
            }

        # Check for CYP enzyme competition
        shared_cyps = set(info_a.get("metabolism", [])) & set(info_b.get("metabolism", []))
        if shared_cyps and "none" not in shared_cyps:
            return {
                "severity": "mild",
                "mechanism": f"Both drugs are metabolized by {', '.join(shared_cyps)}, potentially competing for metabolism.",
                "pk_score": 50,
                "pd_score": 15,
                "clinical_effect": "Possible altered drug levels due to metabolic competition.",
                "monitoring": ["Monitor for drug efficacy changes", "Watch for toxicity signs"],
                "timing": "Monitor closely; dose adjustment may be needed"
            }

        # Check for organ load stacking
        shared_organs = set(info_a.get("organ_load", [])) & set(info_b.get("organ_load", []))
        if len(shared_organs) >= 2:
            return {
                "severity": "mild",
                "mechanism": f"Both drugs affect {', '.join(shared_organs)}, creating combined organ stress.",
                "pk_score": 30,
                "pd_score": 40,
                "clinical_effect": "Increased risk of organ-specific adverse effects.",
                "monitoring": [f"Monitor {organ} function" for organ in list(shared_organs)[:2]],
                "timing": "Use with caution; monitor organ function"
            }

        return None

    def _calculate_score_breakdown(
        self,
        drugs: list[str],
        interactions: list[dict[str, Any]]
    ) -> InteractionScoreBreakdown:
        """Calculate detailed score breakdown across all dimensions."""
        if not interactions:
            return InteractionScoreBreakdown(
                pharmacodynamic=0,
                pharmacokinetic=0,
                mechanism_duplication=0,
                metabolic_pathway=0,
                half_life_overlap=0,
                organ_system_load=0,
                weighted_total=0
            )

        # Aggregate scores from interactions
        pd_scores = [i.get("pd_score", 0) for i in interactions]
        pk_scores = [i.get("pk_score", 0) for i in interactions]

        pd_score = min(100, sum(pd_scores) // len(pd_scores) if pd_scores else 0)
        pk_score = min(100, sum(pk_scores) // len(pk_scores) if pk_scores else 0)

        # Calculate mechanism duplication
        mech_dup = self._calculate_mechanism_duplication(drugs)

        # Calculate metabolic pathway conflicts
        metabolic = self._calculate_metabolic_conflict(drugs)

        # Calculate half-life overlap concerns
        half_life = self._calculate_half_life_overlap(drugs)

        # Calculate organ system load
        organ_load = self._calculate_organ_load(drugs)

        # Weighted total (pharmacodynamic interactions most dangerous)
        weighted = int(
            pd_score * 0.35 +
            pk_score * 0.25 +
            mech_dup * 0.15 +
            metabolic * 0.10 +
            half_life * 0.05 +
            organ_load * 0.10
        )

        return InteractionScoreBreakdown(
            pharmacodynamic=pd_score,
            pharmacokinetic=pk_score,
            mechanism_duplication=mech_dup,
            metabolic_pathway=metabolic,
            half_life_overlap=half_life,
            organ_system_load=organ_load,
            weighted_total=min(100, weighted)
        )

    def _calculate_mechanism_duplication(self, drugs: list[str]) -> int:
        """Score for overlapping mechanisms of action."""
        mechanisms = []
        for drug in drugs:
            if drug in DRUG_DATABASE:
                mechanisms.append(DRUG_DATABASE[drug]["mechanism"])

        if len(mechanisms) != len(set(mechanisms)):
            duplicates = len(mechanisms) - len(set(mechanisms))
            return min(100, duplicates * 40)
        return 0

    def _calculate_metabolic_conflict(self, drugs: list[str]) -> int:
        """Score for CYP enzyme pathway conflicts."""
        all_cyps = []
        for drug in drugs:
            if drug in DRUG_DATABASE:
                cyps = DRUG_DATABASE[drug].get("metabolism", [])
                all_cyps.extend([c for c in cyps if c != "none"])

        # Count overlaps
        from collections import Counter
        cyp_counts = Counter(all_cyps)
        conflicts = sum(1 for count in cyp_counts.values() if count > 1)

        return min(100, conflicts * 25)

    def _calculate_half_life_overlap(self, drugs: list[str]) -> int:
        """Score for drugs with very different half-lives."""
        half_lives = []
        for drug in drugs:
            if drug in DRUG_DATABASE:
                half_lives.append(DRUG_DATABASE[drug].get("half_life", 12))

        if len(half_lives) < 2:
            return 0

        max_hl = max(half_lives)
        min_hl = min(half_lives)
        ratio = max_hl / min_hl if min_hl > 0 else 1

        # Large ratio = harder to coordinate dosing
        if ratio > 20:
            return 60
        elif ratio > 10:
            return 40
        elif ratio > 5:
            return 20
        return 0

    def _calculate_organ_load(self, drugs: list[str]) -> int:
        """Score for combined organ system burden."""
        organ_counts: dict[str, int] = {}
        for drug in drugs:
            if drug in DRUG_DATABASE:
                for organ in DRUG_DATABASE[drug].get("organ_load", []):
                    organ_counts[organ] = organ_counts.get(organ, 0) + 1

        # High concern organs
        high_concern = ["liver", "kidney", "cardiovascular", "cns"]
        score = 0
        for organ, count in organ_counts.items():
            if count > 1:
                multiplier = 15 if organ in high_concern else 10
                score += (count - 1) * multiplier

        return min(100, score)

    def _calculate_safety_score(
        self,
        breakdown: InteractionScoreBreakdown,
        interactions: list[dict[str, Any]]
    ) -> int:
        """Calculate overall safety score (100 = completely safe)."""
        if not interactions:
            return 100

        # Start with inverted weighted score
        base_safety = 100 - breakdown.weighted_total

        # Penalize for severe interactions
        severe_count = sum(1 for i in interactions if i.get("severity") == "severe")
        avoid_count = sum(1 for i in interactions if i.get("severity") == "avoid")

        base_safety -= severe_count * 15
        base_safety -= avoid_count * 25

        return max(0, min(100, base_safety))

    def _determine_overall_severity(
        self,
        interactions: list[dict[str, Any]],
        safety_score: int
    ) -> SeverityLevel:
        """Determine overall severity classification."""
        if not interactions:
            return SeverityLevel.MILD

        # Check for avoid combinations
        if any(i.get("severity") == "avoid" for i in interactions):
            return SeverityLevel.CRITICAL

        # Check for severe interactions
        if any(i.get("severity") == "severe" for i in interactions):
            return SeverityLevel.SEVERE

        # Based on safety score
        if safety_score < 40:
            return SeverityLevel.SEVERE
        elif safety_score < 60:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.MILD

    def _calculate_confidence(
        self,
        known_count: int,
        unknown_count: int,
        interaction_count: int
    ) -> int:
        """Calculate confidence in analysis."""
        if known_count == 0:
            return 20

        # Base confidence on known drugs
        base = 90 if unknown_count == 0 else 90 - (unknown_count * 15)

        # Adjust for interaction count (more interactions = more confident in findings)
        if interaction_count > 0:
            base = min(95, base + 5)

        return max(20, min(95, base))

    def _generate_verdict(self, safety_score: int, interactions: list[dict[str, Any]]) -> str:
        """Generate final verdict on drug combination."""
        if not interactions:
            return "No significant interactions identified. Combination appears safe with standard monitoring."

        if safety_score >= 80:
            return "Minor interactions detected. Combination is generally safe with awareness of potential effects."
        elif safety_score >= 60:
            return "Moderate interactions present. Use with caution and appropriate monitoring."
        elif safety_score >= 40:
            return "Significant interactions identified. Requires careful monitoring and possible dose adjustments."
        elif safety_score >= 20:
            return "Serious interactions detected. Consider alternatives or use only when benefits clearly outweigh risks."
        else:
            return "Dangerous combination with severe interaction risk. Avoid unless absolutely necessary with specialist oversight."

    def _format_interaction(self, interaction: dict[str, Any]) -> DrugInteraction:
        """Format interaction data for response model."""
        severity_map = {
            "mild": InteractionCategory.MILD,
            "moderate": InteractionCategory.MODERATE,
            "severe": InteractionCategory.SEVERE,
            "avoid": InteractionCategory.AVOID_COMBINATION
        }

        return DrugInteraction(
            drug_a=interaction["drug_a"].title(),
            drug_b=interaction["drug_b"].title(),
            category=severity_map.get(interaction.get("severity", "mild"), InteractionCategory.MODERATE),
            mechanism=interaction.get("mechanism", "Interaction mechanism not fully characterized."),
            clinical_effect=interaction.get("clinical_effect", "Clinical significance under evaluation."),
            pharmacokinetic_score=interaction.get("pk_score", 0),
            pharmacodynamic_score=interaction.get("pd_score", 0),
            monitoring=interaction.get("monitoring", []),
            timing_adjustment=interaction.get("timing")
        )

    def _build_sections(
        self,
        drugs: list[str],
        interactions: list[dict[str, Any]],
        breakdown: InteractionScoreBreakdown,
        unknown_drugs: list[str]
    ) -> list[UISection]:
        """Build UI sections for response."""
        sections = []

        # Score breakdown section
        sections.append(UISection(
            title="Interaction Score Analysis",
            icon="activity",
            items=[
                SectionItem(
                    label="Pharmacodynamic Risk",
                    value=f"{breakdown.pharmacodynamic}/100",
                    score=breakdown.pharmacodynamic,
                    severity="severe" if breakdown.pharmacodynamic > 60 else "moderate" if breakdown.pharmacodynamic > 30 else "mild",
                    explanation="Risk from combined drug effects on body systems"
                ),
                SectionItem(
                    label="Pharmacokinetic Risk",
                    value=f"{breakdown.pharmacokinetic}/100",
                    score=breakdown.pharmacokinetic,
                    severity="severe" if breakdown.pharmacokinetic > 60 else "moderate" if breakdown.pharmacokinetic > 30 else "mild",
                    explanation="Risk from drug absorption, metabolism, and elimination conflicts"
                ),
                SectionItem(
                    label="Mechanism Overlap",
                    value=f"{breakdown.mechanism_duplication}/100",
                    score=breakdown.mechanism_duplication,
                    explanation="Multiple drugs with same mechanism of action"
                ),
                SectionItem(
                    label="Metabolic Pathway Conflict",
                    value=f"{breakdown.metabolic_pathway}/100",
                    score=breakdown.metabolic_pathway,
                    explanation="Competition for same liver enzymes"
                ),
                SectionItem(
                    label="Organ System Load",
                    value=f"{breakdown.organ_system_load}/100",
                    score=breakdown.organ_system_load,
                    explanation="Combined stress on shared organ systems"
                )
            ],
            priority=1
        ))

        # Interactions section
        if interactions:
            interaction_items = []
            for i in interactions:
                severity = i.get("severity", "mild")
                interaction_items.append(SectionItem(
                    label=f"{i['drug_a'].title()} + {i['drug_b'].title()}",
                    value=i.get("clinical_effect", ""),
                    badge=severity.upper(),
                    severity=severity,
                    explanation=i.get("mechanism", "")
                ))

            sections.append(UISection(
                title="Detected Interactions",
                icon="alert-triangle",
                items=interaction_items,
                expandable=True,
                priority=0
            ))

        # Unknown drugs warning
        if unknown_drugs:
            sections.append(UISection(
                title="Unrecognized Medications",
                icon="help-circle",
                items=[
                    SectionItem(
                        label=drug.title(),
                        value="Not in database - interactions may exist",
                        badge="UNKNOWN",
                        severity="moderate"
                    ) for drug in unknown_drugs
                ],
                priority=2
            ))

        return sections

    def _generate_red_flags(
        self,
        interactions: list[dict[str, Any]],
        drugs: list[str]
    ) -> list[RedFlag]:
        """Generate red flags for serious concerns."""
        red_flags = []

        # Check for severe/avoid interactions
        for interaction in interactions:
            if interaction.get("severity") in ["severe", "avoid"]:
                red_flags.append(RedFlag(
                    flag=f"Severe interaction: {interaction['drug_a'].title()} + {interaction['drug_b'].title()}",
                    explanation=interaction.get("mechanism", "Serious interaction identified"),
                    action="Consult physician before taking together",
                    severity=SeverityLevel.SEVERE if interaction.get("severity") == "severe" else SeverityLevel.CRITICAL
                ))

        # Check for narrow therapeutic index drugs
        for drug in drugs:
            if drug in DRUG_DATABASE and DRUG_DATABASE[drug].get("narrow_therapeutic_index"):
                red_flags.append(RedFlag(
                    flag=f"{drug.title()} has narrow therapeutic index",
                    explanation="Small changes in blood levels can cause toxicity or treatment failure",
                    action="Requires careful monitoring and dose adjustment when combined with other drugs",
                    severity=SeverityLevel.MODERATE
                ))

        return red_flags

    def _generate_recommendations(
        self,
        interactions: list[dict[str, Any]],
        drugs: list[str]
    ) -> list[Recommendation]:
        """Generate actionable recommendations."""
        recommendations = []

        # Monitoring recommendations from interactions
        all_monitoring = []
        for interaction in interactions:
            all_monitoring.extend(interaction.get("monitoring", []))

        if all_monitoring:
            recommendations.append(Recommendation(
                title="Monitoring Requirements",
                description="; ".join(list(set(all_monitoring))[:5]),
                priority="high" if any(i.get("severity") == "severe" for i in interactions) else "medium",
                category="monitoring",
                timeframe="Within first 1-2 weeks of combination"
            ))

        # Timing recommendations
        timing_recs = [i.get("timing") for i in interactions if i.get("timing")]
        if timing_recs:
            recommendations.append(Recommendation(
                title="Dose Timing Adjustments",
                description="; ".join(list(set(timing_recs))),
                priority="medium",
                category="medication",
                timeframe="Apply immediately"
            ))

        # General recommendations based on drug classes
        organ_counts: dict[str, int] = {}
        for drug in drugs:
            if drug in DRUG_DATABASE:
                for organ in DRUG_DATABASE[drug].get("organ_load", []):
                    organ_counts[organ] = organ_counts.get(organ, 0) + 1

        for organ, count in organ_counts.items():
            if count >= 2:
                if organ == "liver":
                    recommendations.append(Recommendation(
                        title="Liver Function Monitoring",
                        description="Multiple drugs metabolized by liver; monitor LFTs periodically",
                        priority="medium",
                        category="monitoring",
                        timeframe="Baseline and every 3-6 months"
                    ))
                elif organ == "kidney":
                    recommendations.append(Recommendation(
                        title="Kidney Function Monitoring",
                        description="Multiple drugs affecting kidney; monitor creatinine and BUN",
                        priority="medium",
                        category="monitoring",
                        timeframe="Baseline and every 3-6 months"
                    ))

        return recommendations

    def _generate_alternatives(
        self,
        interactions: list[dict[str, Any]],
        drugs: list[str]
    ) -> list[Alternative]:
        """Generate alternative medication suggestions."""
        alternatives = []

        for interaction in interactions:
            if interaction.get("severity") in ["severe", "moderate"]:
                # Get alternatives for the more problematic drug
                drug_a = interaction["drug_a"]
                drug_b = interaction["drug_b"]

                for drug in [drug_a, drug_b]:
                    if drug in DRUG_DATABASE:
                        alts = DRUG_DATABASE[drug].get("alternatives", [])
                        if alts:
                            alternatives.append(Alternative(
                                name=f"Alternatives to {drug.title()}",
                                reason=f"To avoid interaction with {(drug_b if drug == drug_a else drug_a).title()}",
                                advantages=[f"Consider: {', '.join(a.replace('_', ' ').title() for a in alts[:3])}"],
                                considerations=["Verify alternative doesn't have similar interactions", "Discuss with prescriber"]
                            ))
                            break  # Only suggest for one drug per interaction

        return alternatives[:5]  # Limit to 5 alternatives

    def _generate_summary(
        self,
        drugs: list[str],
        interactions: list[dict[str, Any]],
        safety_score: int
    ) -> str:
        """Generate concise summary."""
        drug_count = len(drugs)
        interaction_count = len(interactions)

        if interaction_count == 0:
            return f"Analysis of {drug_count} medications found no significant interactions. Safety score: {safety_score}/100."

        severe_count = sum(1 for i in interactions if i.get("severity") in ["severe", "avoid"])

        if severe_count > 0:
            return f"Analysis of {drug_count} medications identified {interaction_count} interactions including {severe_count} severe. Immediate review recommended. Safety score: {safety_score}/100."
        else:
            return f"Analysis of {drug_count} medications found {interaction_count} interactions requiring monitoring. Safety score: {safety_score}/100."


# Singleton instance
_checker_instance: Optional[DrugInteractionChecker] = None


async def get_drug_interaction_checker() -> DrugInteractionChecker:
    """Get or create DrugInteractionChecker instance."""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = DrugInteractionChecker()
    return _checker_instance
