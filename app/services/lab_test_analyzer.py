"""
DeepHealth Lab Test Analyzer Service

Comprehensive lab test interpretation with reference ranges,
severity classification, trend analysis, and clinical correlations.
"""

from datetime import datetime
from typing import Any, Optional

from app.core.logging import get_logger
from app.schemas.medical import (
    Alternative,
    LabTestCategory,
    LabTestRequest,
    LabTestResponse,
    LabValue,
    Recommendation,
    RedFlag,
    ScoreSet,
    SectionItem,
    SeverityLevel,
    TrendAnalysis,
    UISection,
    UnderlyingCause,
)

logger = get_logger(__name__)


# ============================================================================
# LAB REFERENCE RANGES DATABASE
# ============================================================================

LAB_REFERENCE_RANGES = {
    # Complete Blood Count (CBC)
    "WBC": {
        "name": "White Blood Cell Count",
        "unit": "10^9/L",
        "low": 4.5,
        "high": 11.0,
        "critical_low": 2.0,
        "critical_high": 30.0,
        "category": "cbc",
        "interpretations": {
            "low": "Leukopenia - May indicate bone marrow suppression, viral infection, or autoimmune condition",
            "high": "Leukocytosis - May indicate infection, inflammation, stress, or hematologic disorder",
            "critical_low": "Severe leukopenia - High infection risk; urgent evaluation needed",
            "critical_high": "Severe leukocytosis - May indicate serious infection or leukemia"
        },
        "causes_low": ["Viral infection", "Bone marrow suppression", "Autoimmune disorders", "Medication effect"],
        "causes_high": ["Bacterial infection", "Inflammation", "Stress response", "Leukemia", "Corticosteroid use"]
    },
    "RBC": {
        "name": "Red Blood Cell Count",
        "unit": "10^12/L",
        "low": 4.2,
        "high": 5.9,
        "critical_low": 3.0,
        "critical_high": 7.0,
        "category": "cbc",
        "interpretations": {
            "low": "May indicate anemia from various causes",
            "high": "Polycythemia - May indicate dehydration, lung disease, or polycythemia vera",
            "critical_low": "Severe anemia - May require transfusion",
            "critical_high": "Severe polycythemia - Risk of thrombosis"
        },
        "causes_low": ["Iron deficiency", "Blood loss", "Chronic disease", "Nutritional deficiency"],
        "causes_high": ["Dehydration", "Chronic hypoxia", "Polycythemia vera", "High altitude"]
    },
    "Hemoglobin": {
        "name": "Hemoglobin",
        "unit": "g/dL",
        "low": 12.0,
        "high": 17.5,
        "critical_low": 7.0,
        "critical_high": 20.0,
        "category": "cbc",
        "interpretations": {
            "low": "Anemia - Reduced oxygen-carrying capacity",
            "high": "May indicate dehydration or polycythemia",
            "critical_low": "Severe anemia - Transfusion likely needed",
            "critical_high": "Severe polycythemia - Risk of hyperviscosity"
        },
        "causes_low": ["Iron deficiency", "Chronic blood loss", "Hemolysis", "Chronic disease"],
        "causes_high": ["Dehydration", "COPD", "Polycythemia vera", "Smoking"]
    },
    "Hematocrit": {
        "name": "Hematocrit",
        "unit": "%",
        "low": 36.0,
        "high": 50.0,
        "critical_low": 20.0,
        "critical_high": 60.0,
        "category": "cbc",
        "interpretations": {
            "low": "Low proportion of red blood cells; indicates anemia",
            "high": "High proportion of red blood cells; may indicate dehydration or polycythemia",
            "critical_low": "Severe anemia - Urgent evaluation needed",
            "critical_high": "Severe polycythemia - Risk of clotting"
        },
        "causes_low": ["Anemia", "Overhydration", "Blood loss", "Hemolysis"],
        "causes_high": ["Dehydration", "Polycythemia", "COPD", "Living at high altitude"]
    },
    "Platelets": {
        "name": "Platelet Count",
        "unit": "10^9/L",
        "low": 150.0,
        "high": 400.0,
        "critical_low": 50.0,
        "critical_high": 1000.0,
        "category": "cbc",
        "interpretations": {
            "low": "Thrombocytopenia - Increased bleeding risk",
            "high": "Thrombocytosis - Increased clotting risk",
            "critical_low": "Severe thrombocytopenia - Spontaneous bleeding risk",
            "critical_high": "Severe thrombocytosis - High thrombosis risk"
        },
        "causes_low": ["Bone marrow suppression", "ITP", "Medications", "Liver disease"],
        "causes_high": ["Infection", "Inflammation", "Iron deficiency", "Myeloproliferative disorder"]
    },
    "MCV": {
        "name": "Mean Corpuscular Volume",
        "unit": "fL",
        "low": 80.0,
        "high": 100.0,
        "critical_low": 60.0,
        "critical_high": 120.0,
        "category": "cbc",
        "interpretations": {
            "low": "Microcytic - Small red blood cells; often iron deficiency or thalassemia",
            "high": "Macrocytic - Large red blood cells; often B12/folate deficiency or liver disease",
            "critical_low": "Severe microcytosis",
            "critical_high": "Severe macrocytosis"
        },
        "causes_low": ["Iron deficiency", "Thalassemia", "Chronic disease", "Lead poisoning"],
        "causes_high": ["B12 deficiency", "Folate deficiency", "Liver disease", "Hypothyroidism", "Alcohol use"]
    },

    # Comprehensive Metabolic Panel (CMP)
    "Glucose": {
        "name": "Blood Glucose (Fasting)",
        "unit": "mg/dL",
        "low": 70.0,
        "high": 100.0,
        "critical_low": 50.0,
        "critical_high": 400.0,
        "category": "cmp",
        "interpretations": {
            "low": "Hypoglycemia - May cause symptoms like shakiness, confusion",
            "high": "Hyperglycemia - May indicate prediabetes or diabetes",
            "critical_low": "Severe hypoglycemia - Medical emergency; risk of seizure/coma",
            "critical_high": "Severe hyperglycemia - Risk of diabetic ketoacidosis"
        },
        "causes_low": ["Medication effect", "Insulinoma", "Adrenal insufficiency", "Liver disease"],
        "causes_high": ["Diabetes mellitus", "Stress", "Medications", "Pancreatic disease"]
    },
    "BUN": {
        "name": "Blood Urea Nitrogen",
        "unit": "mg/dL",
        "low": 7.0,
        "high": 20.0,
        "critical_low": 2.0,
        "critical_high": 100.0,
        "category": "cmp",
        "interpretations": {
            "low": "May indicate malnutrition, liver disease, or overhydration",
            "high": "May indicate dehydration, kidney dysfunction, or high protein intake",
            "critical_low": "Severe liver dysfunction possible",
            "critical_high": "Severe azotemia - Kidney failure likely"
        },
        "causes_low": ["Malnutrition", "Liver failure", "Overhydration", "Low protein diet"],
        "causes_high": ["Dehydration", "Kidney disease", "GI bleeding", "High protein diet", "Heart failure"]
    },
    "Creatinine": {
        "name": "Creatinine",
        "unit": "mg/dL",
        "low": 0.7,
        "high": 1.3,
        "critical_low": 0.4,
        "critical_high": 10.0,
        "category": "cmp",
        "interpretations": {
            "low": "May indicate low muscle mass or liver disease",
            "high": "May indicate kidney dysfunction",
            "critical_low": "Very low muscle mass; evaluate nutritional status",
            "critical_high": "Severe kidney failure - Dialysis may be needed"
        },
        "causes_low": ["Low muscle mass", "Malnutrition", "Liver disease"],
        "causes_high": ["Chronic kidney disease", "Acute kidney injury", "Dehydration", "Medications"]
    },
    "Sodium": {
        "name": "Sodium",
        "unit": "mEq/L",
        "low": 136.0,
        "high": 145.0,
        "critical_low": 120.0,
        "critical_high": 160.0,
        "category": "cmp",
        "interpretations": {
            "low": "Hyponatremia - May cause confusion, seizures",
            "high": "Hypernatremia - Usually indicates dehydration",
            "critical_low": "Severe hyponatremia - Risk of cerebral edema",
            "critical_high": "Severe hypernatremia - Risk of neurological damage"
        },
        "causes_low": ["SIADH", "Diuretics", "Heart failure", "Liver cirrhosis", "Adrenal insufficiency"],
        "causes_high": ["Dehydration", "Diabetes insipidus", "Excessive sodium intake"]
    },
    "Potassium": {
        "name": "Potassium",
        "unit": "mEq/L",
        "low": 3.5,
        "high": 5.0,
        "critical_low": 2.5,
        "critical_high": 6.5,
        "category": "cmp",
        "interpretations": {
            "low": "Hypokalemia - Risk of arrhythmias and muscle weakness",
            "high": "Hyperkalemia - Risk of cardiac arrhythmias",
            "critical_low": "Severe hypokalemia - Life-threatening arrhythmias possible",
            "critical_high": "Severe hyperkalemia - Cardiac arrest risk"
        },
        "causes_low": ["Diuretics", "Vomiting/diarrhea", "Hyperaldosteronism", "Medications"],
        "causes_high": ["Kidney disease", "ACE inhibitors", "Potassium supplements", "Acidosis", "Tissue breakdown"]
    },
    "Chloride": {
        "name": "Chloride",
        "unit": "mEq/L",
        "low": 98.0,
        "high": 106.0,
        "critical_low": 80.0,
        "critical_high": 120.0,
        "category": "cmp",
        "interpretations": {
            "low": "Hypochloremia - Often accompanies metabolic alkalosis",
            "high": "Hyperchloremia - Often accompanies metabolic acidosis",
            "critical_low": "Severe hypochloremia",
            "critical_high": "Severe hyperchloremia"
        },
        "causes_low": ["Vomiting", "Diuretics", "SIADH", "Metabolic alkalosis"],
        "causes_high": ["Dehydration", "Renal tubular acidosis", "Diarrhea", "Medications"]
    },
    "CO2": {
        "name": "Carbon Dioxide (Bicarbonate)",
        "unit": "mEq/L",
        "low": 23.0,
        "high": 29.0,
        "critical_low": 10.0,
        "critical_high": 40.0,
        "category": "cmp",
        "interpretations": {
            "low": "May indicate metabolic acidosis",
            "high": "May indicate metabolic alkalosis or respiratory compensation",
            "critical_low": "Severe metabolic acidosis",
            "critical_high": "Severe metabolic alkalosis"
        },
        "causes_low": ["Diabetic ketoacidosis", "Renal failure", "Diarrhea", "Lactic acidosis"],
        "causes_high": ["Vomiting", "Diuretics", "COPD", "Hyperaldosteronism"]
    },
    "Calcium": {
        "name": "Calcium",
        "unit": "mg/dL",
        "low": 8.5,
        "high": 10.5,
        "critical_low": 6.0,
        "critical_high": 13.0,
        "category": "cmp",
        "interpretations": {
            "low": "Hypocalcemia - May cause muscle spasms, tetany",
            "high": "Hypercalcemia - May cause confusion, constipation, polyuria",
            "critical_low": "Severe hypocalcemia - Risk of seizures",
            "critical_high": "Severe hypercalcemia - Cardiac arrhythmia risk"
        },
        "causes_low": ["Vitamin D deficiency", "Hypoparathyroidism", "Kidney disease", "Malabsorption"],
        "causes_high": ["Hyperparathyroidism", "Malignancy", "Vitamin D toxicity", "Thiazide diuretics"]
    },

    # Liver Enzymes
    "ALT": {
        "name": "Alanine Aminotransferase",
        "unit": "U/L",
        "low": 7.0,
        "high": 56.0,
        "critical_low": 0.0,
        "critical_high": 1000.0,
        "category": "liver_enzymes",
        "interpretations": {
            "low": "Generally not clinically significant",
            "high": "Elevated - May indicate liver cell damage",
            "critical_low": "Not applicable",
            "critical_high": "Severe hepatocellular injury"
        },
        "causes_low": [],
        "causes_high": ["Hepatitis", "Fatty liver", "Medications", "Alcohol", "Muscle injury"]
    },
    "AST": {
        "name": "Aspartate Aminotransferase",
        "unit": "U/L",
        "low": 10.0,
        "high": 40.0,
        "critical_low": 0.0,
        "critical_high": 1000.0,
        "category": "liver_enzymes",
        "interpretations": {
            "low": "Generally not clinically significant",
            "high": "Elevated - May indicate liver or muscle damage",
            "critical_low": "Not applicable",
            "critical_high": "Severe tissue damage (liver, heart, or muscle)"
        },
        "causes_low": [],
        "causes_high": ["Hepatitis", "Cirrhosis", "Heart attack", "Muscle injury", "Medications"]
    },
    "ALP": {
        "name": "Alkaline Phosphatase",
        "unit": "U/L",
        "low": 44.0,
        "high": 147.0,
        "critical_low": 0.0,
        "critical_high": 1000.0,
        "category": "liver_enzymes",
        "interpretations": {
            "low": "May indicate malnutrition or hypothyroidism",
            "high": "May indicate cholestatic liver disease or bone disease",
            "critical_low": "Not applicable",
            "critical_high": "Severe cholestasis or bone disease"
        },
        "causes_low": ["Malnutrition", "Hypothyroidism", "Zinc deficiency"],
        "causes_high": ["Cholestasis", "Bone disease", "Liver metastases", "Paget's disease", "Pregnancy"]
    },
    "Bilirubin": {
        "name": "Total Bilirubin",
        "unit": "mg/dL",
        "low": 0.1,
        "high": 1.2,
        "critical_low": 0.0,
        "critical_high": 15.0,
        "category": "liver_enzymes",
        "interpretations": {
            "low": "Not clinically significant",
            "high": "May indicate liver disease or hemolysis; causes jaundice",
            "critical_low": "Not applicable",
            "critical_high": "Severe hyperbilirubinemia - Risk of kernicterus in infants"
        },
        "causes_low": [],
        "causes_high": ["Hepatitis", "Cirrhosis", "Hemolysis", "Gilbert syndrome", "Biliary obstruction"]
    },
    "Albumin": {
        "name": "Albumin",
        "unit": "g/dL",
        "low": 3.5,
        "high": 5.0,
        "critical_low": 1.5,
        "critical_high": 6.0,
        "category": "liver_enzymes",
        "interpretations": {
            "low": "Hypoalbuminemia - May indicate liver disease, malnutrition, or nephrotic syndrome",
            "high": "Usually indicates dehydration",
            "critical_low": "Severe hypoalbuminemia - Edema and poor wound healing likely",
            "critical_high": "Severe dehydration"
        },
        "causes_low": ["Liver disease", "Malnutrition", "Nephrotic syndrome", "Inflammation"],
        "causes_high": ["Dehydration", "Rare genetic conditions"]
    },

    # Lipid Panel
    "TotalCholesterol": {
        "name": "Total Cholesterol",
        "unit": "mg/dL",
        "low": 0.0,
        "high": 200.0,
        "critical_low": 0.0,
        "critical_high": 400.0,
        "category": "lipid_panel",
        "interpretations": {
            "low": "Very low cholesterol may indicate malnutrition or hyperthyroidism",
            "high": "Hypercholesterolemia - Increased cardiovascular risk",
            "critical_low": "Not applicable",
            "critical_high": "Severe hypercholesterolemia - High cardiovascular risk"
        },
        "causes_low": ["Malnutrition", "Hyperthyroidism", "Liver disease", "Malabsorption"],
        "causes_high": ["Diet", "Familial hypercholesterolemia", "Hypothyroidism", "Nephrotic syndrome"]
    },
    "LDL": {
        "name": "LDL Cholesterol",
        "unit": "mg/dL",
        "low": 0.0,
        "high": 100.0,
        "critical_low": 0.0,
        "critical_high": 300.0,
        "category": "lipid_panel",
        "interpretations": {
            "low": "Generally beneficial; very low may warrant evaluation",
            "high": "Elevated LDL - Major risk factor for atherosclerosis",
            "critical_low": "Not applicable",
            "critical_high": "Severely elevated - Very high cardiovascular risk"
        },
        "causes_low": ["Statins", "Malnutrition", "Hyperthyroidism"],
        "causes_high": ["Diet", "Genetics", "Hypothyroidism", "Diabetes", "Obesity"]
    },
    "HDL": {
        "name": "HDL Cholesterol",
        "unit": "mg/dL",
        "low": 40.0,
        "high": 200.0,
        "critical_low": 20.0,
        "critical_high": 200.0,
        "category": "lipid_panel",
        "interpretations": {
            "low": "Low HDL - Increased cardiovascular risk",
            "high": "High HDL - Generally cardioprotective",
            "critical_low": "Very low HDL - Significantly increased CV risk",
            "critical_high": "Not applicable"
        },
        "causes_low": ["Smoking", "Obesity", "Sedentary lifestyle", "Metabolic syndrome"],
        "causes_high": ["Exercise", "Moderate alcohol", "Genetics", "Medications"]
    },
    "Triglycerides": {
        "name": "Triglycerides",
        "unit": "mg/dL",
        "low": 0.0,
        "high": 150.0,
        "critical_low": 0.0,
        "critical_high": 500.0,
        "category": "lipid_panel",
        "interpretations": {
            "low": "Not clinically significant",
            "high": "Hypertriglyceridemia - Risk of pancreatitis if severe",
            "critical_low": "Not applicable",
            "critical_high": "Severe hypertriglyceridemia - Pancreatitis risk"
        },
        "causes_low": ["Malnutrition", "Hyperthyroidism", "Malabsorption"],
        "causes_high": ["Diet", "Obesity", "Diabetes", "Alcohol", "Medications"]
    },

    # Thyroid Panel
    "TSH": {
        "name": "Thyroid Stimulating Hormone",
        "unit": "mIU/L",
        "low": 0.4,
        "high": 4.0,
        "critical_low": 0.01,
        "critical_high": 50.0,
        "category": "thyroid_panel",
        "interpretations": {
            "low": "May indicate hyperthyroidism or pituitary dysfunction",
            "high": "May indicate hypothyroidism",
            "critical_low": "Severe hyperthyroidism - Risk of thyroid storm",
            "critical_high": "Severe hypothyroidism - Myxedema possible"
        },
        "causes_low": ["Hyperthyroidism", "Pituitary dysfunction", "Excessive thyroid medication"],
        "causes_high": ["Hypothyroidism", "Hashimoto's thyroiditis", "Iodine deficiency", "Pituitary tumor"]
    },
    "FreeT4": {
        "name": "Free T4 (Thyroxine)",
        "unit": "ng/dL",
        "low": 0.8,
        "high": 1.8,
        "critical_low": 0.3,
        "critical_high": 5.0,
        "category": "thyroid_panel",
        "interpretations": {
            "low": "May indicate hypothyroidism",
            "high": "May indicate hyperthyroidism",
            "critical_low": "Severe hypothyroidism",
            "critical_high": "Severe hyperthyroidism - Thyroid storm risk"
        },
        "causes_low": ["Hypothyroidism", "Pituitary dysfunction", "Severe illness"],
        "causes_high": ["Graves' disease", "Toxic nodular goiter", "Thyroiditis", "Excessive medication"]
    },
    "FreeT3": {
        "name": "Free T3 (Triiodothyronine)",
        "unit": "pg/mL",
        "low": 2.3,
        "high": 4.2,
        "critical_low": 1.0,
        "critical_high": 10.0,
        "category": "thyroid_panel",
        "interpretations": {
            "low": "May indicate hypothyroidism or sick euthyroid syndrome",
            "high": "May indicate hyperthyroidism",
            "critical_low": "Severe hypothyroidism",
            "critical_high": "Severe hyperthyroidism"
        },
        "causes_low": ["Hypothyroidism", "Severe illness", "Fasting"],
        "causes_high": ["Hyperthyroidism", "T3 thyrotoxicosis"]
    }
}


class LabTestAnalyzer:
    """
    Comprehensive lab test analysis with interpretation and clinical correlation.

    Features:
    - Reference range interpretation
    - Severity color-coding
    - Trend analysis
    - Underlying cause identification
    - Personalized recommendations
    """

    def __init__(self):
        self._logger = logger

    async def analyze(self, request: LabTestRequest) -> LabTestResponse:
        """
        Perform comprehensive lab test analysis.

        Args:
            request: Lab test analysis request.

        Returns:
            Complete analysis with interpretations and recommendations.
        """
        self._logger.info(f"Analyzing {request.test_type.value} panel")

        # Analyze each lab value
        lab_values = []
        abnormal_count = 0
        critical_count = 0

        for test_name, value in request.values.items():
            lab_value = self._analyze_single_value(test_name, value)
            if lab_value:
                lab_values.append(lab_value)
                if lab_value.status in ["low", "high"]:
                    abnormal_count += 1
                if lab_value.status == "critical":
                    critical_count += 1
                    abnormal_count += 1

        # Analyze trends if historical data provided
        trends = []
        if request.previous_values:
            trends = self._analyze_trends(request.values, request.previous_values)

        # Identify underlying causes
        underlying_causes = self._identify_underlying_causes(lab_values)

        # Calculate scores
        severity_score = self._calculate_severity_score(lab_values, critical_count)
        confidence = self._calculate_confidence(len(lab_values), abnormal_count)

        # Build UI sections
        sections = self._build_sections(lab_values, trends, underlying_causes, request)

        # Generate red flags
        red_flags = self._generate_red_flags(lab_values, critical_count)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            lab_values,
            abnormal_count,
            critical_count,
            request.patient_conditions
        )

        # Generate summary
        summary = self._generate_summary(
            request.test_type,
            len(lab_values),
            abnormal_count,
            critical_count
        )

        # Calculate retest interval
        retest_interval = self._calculate_retest_interval(abnormal_count, critical_count)

        # Generate lifestyle factors
        lifestyle_factors = self._generate_lifestyle_factors(lab_values, request.test_type)

        return LabTestResponse(
            summary=summary,
            scores=ScoreSet(
                confidence=confidence,
                severity=severity_score
            ),
            test_category=request.test_type,
            test_date=request.test_date or datetime.utcnow(),
            lab_values=lab_values,
            trends=trends,
            abnormal_count=abnormal_count,
            critical_count=critical_count,
            underlying_causes=underlying_causes,
            retest_interval=retest_interval,
            lifestyle_factors=lifestyle_factors,
            sections=sections,
            red_flags=red_flags,
            recommendations=recommendations,
            alternatives=[],  # Lab analyzer doesn't suggest alternatives
            metadata={
                "tests_analyzed": len(lab_values),
                "analysis_version": "2.0",
                "patient_conditions": request.patient_conditions
            },
            timestamp=datetime.utcnow()
        )

    def _analyze_single_value(self, test_name: str, value: float) -> Optional[LabValue]:
        """Analyze a single lab value against reference ranges."""
        # Normalize test name
        normalized = test_name.replace(" ", "").replace("_", "")

        # Find matching reference
        ref = None
        for key, data in LAB_REFERENCE_RANGES.items():
            if key.lower() == normalized.lower():
                ref = data
                break

        if not ref:
            # Try partial match
            for key, data in LAB_REFERENCE_RANGES.items():
                if key.lower() in normalized.lower() or normalized.lower() in key.lower():
                    ref = data
                    break

        if not ref:
            return None

        # Determine status and severity
        status = "normal"
        severity_color = "green"
        interpretation = "Within normal range"
        percent_deviation = 0.0

        if value < ref["critical_low"]:
            status = "critical"
            severity_color = "red"
            interpretation = ref["interpretations"]["critical_low"]
            percent_deviation = ((ref["low"] - value) / ref["low"]) * 100
        elif value > ref["critical_high"]:
            status = "critical"
            severity_color = "red"
            interpretation = ref["interpretations"]["critical_high"]
            percent_deviation = ((value - ref["high"]) / ref["high"]) * 100
        elif value < ref["low"]:
            status = "low"
            severity_color = "yellow" if value >= ref["low"] * 0.8 else "orange"
            interpretation = ref["interpretations"]["low"]
            percent_deviation = ((ref["low"] - value) / ref["low"]) * 100
        elif value > ref["high"]:
            status = "high"
            severity_color = "yellow" if value <= ref["high"] * 1.2 else "orange"
            interpretation = ref["interpretations"]["high"]
            percent_deviation = ((value - ref["high"]) / ref["high"]) * 100

        return LabValue(
            test_name=ref["name"],
            value=value,
            unit=ref["unit"],
            reference_low=ref["low"],
            reference_high=ref["high"],
            status=status,
            severity_color=severity_color,
            interpretation=interpretation,
            percent_deviation=round(percent_deviation, 1)
        )

    def _analyze_trends(
        self,
        current_values: dict[str, float],
        previous_values: dict[str, list[dict[str, Any]]]
    ) -> list[TrendAnalysis]:
        """Analyze trends in lab values over time."""
        trends = []

        for test_name, current in current_values.items():
            if test_name not in previous_values:
                continue

            history = previous_values[test_name]
            if not history:
                continue

            # Get previous values
            prev_values = [h.get("value", current) for h in history]

            # Determine trend direction
            if len(prev_values) >= 2:
                first = prev_values[0]
                last = current

                if last > first * 1.1:
                    direction = "increasing"
                    rate = "significant increase"
                elif last > first * 1.02:
                    direction = "increasing"
                    rate = "gradual increase"
                elif last < first * 0.9:
                    direction = "decreasing"
                    rate = "significant decrease"
                elif last < first * 0.98:
                    direction = "decreasing"
                    rate = "gradual decrease"
                else:
                    direction = "stable"
                    rate = "stable over time"

                # Determine clinical significance
                significance = self._determine_trend_significance(test_name, direction, current)

                trends.append(TrendAnalysis(
                    test_name=test_name,
                    direction=direction,
                    rate_of_change=rate,
                    clinical_significance=significance,
                    values_history=[{"date": h.get("date"), "value": h.get("value")} for h in history[-5:]]
                ))

        return trends

    def _determine_trend_significance(self, test_name: str, direction: str, current: float) -> str:
        """Determine clinical significance of a trend."""
        # Normalize and find reference
        normalized = test_name.replace(" ", "").replace("_", "")
        for key, ref in LAB_REFERENCE_RANGES.items():
            if key.lower() in normalized.lower():
                if direction == "increasing" and current > ref["high"]:
                    return "Trending above normal range - warrants attention"
                elif direction == "decreasing" and current < ref["low"]:
                    return "Trending below normal range - warrants attention"
                elif direction in ["increasing", "decreasing"]:
                    return "Trending but still within normal limits"
                break

        return "No significant trend concern"

    def _identify_underlying_causes(self, lab_values: list[LabValue]) -> list[UnderlyingCause]:
        """Identify potential underlying causes for abnormal results."""
        causes_map: dict[str, dict] = {}

        # Collect abnormal values
        abnormal = [v for v in lab_values if v.status in ["low", "high", "critical"]]

        for value in abnormal:
            # Find reference data
            for key, ref in LAB_REFERENCE_RANGES.items():
                if ref["name"] == value.test_name:
                    # Get relevant causes
                    causes_list = ref.get("causes_high" if value.status in ["high", "critical"] and value.value > ref["high"] else "causes_low", [])

                    for cause in causes_list:
                        if cause not in causes_map:
                            causes_map[cause] = {
                                "related_tests": [],
                                "count": 0
                            }
                        causes_map[cause]["related_tests"].append(value.test_name)
                        causes_map[cause]["count"] += 1
                    break

        # Convert to UnderlyingCause objects with probability ranking
        underlying = []
        for cause, data in sorted(causes_map.items(), key=lambda x: x[1]["count"], reverse=True):
            # More abnormal tests pointing to same cause = higher probability
            probability = min(80, 30 + data["count"] * 15)

            underlying.append(UnderlyingCause(
                cause=cause,
                probability=probability,
                related_tests=data["related_tests"],
                explanation=f"Associated with {data['count']} abnormal test(s): {', '.join(data['related_tests'][:3])}",
                next_steps=self._get_next_steps_for_cause(cause)
            ))

        return underlying[:5]  # Return top 5 causes

    def _get_next_steps_for_cause(self, cause: str) -> list[str]:
        """Get recommended next steps for a potential cause."""
        cause_lower = cause.lower()

        if "dehydration" in cause_lower:
            return ["Increase fluid intake", "Repeat labs after rehydration", "Evaluate for underlying cause"]
        elif "infection" in cause_lower:
            return ["Clinical evaluation", "Consider cultures if indicated", "Monitor for resolution"]
        elif "liver" in cause_lower:
            return ["Hepatitis panel", "Liver ultrasound", "Review medications", "Alcohol history"]
        elif "kidney" in cause_lower or "renal" in cause_lower:
            return ["Check urinalysis", "Renal ultrasound", "Nephrology referral if severe"]
        elif "thyroid" in cause_lower:
            return ["Full thyroid panel", "Thyroid ultrasound if indicated", "Endocrinology referral"]
        elif "diabetes" in cause_lower:
            return ["HbA1c test", "Fasting glucose confirmation", "Diabetes education"]
        elif "anemia" in cause_lower or "iron" in cause_lower:
            return ["Iron studies", "Reticulocyte count", "Consider GI evaluation for blood loss"]

        return ["Clinical correlation", "Repeat testing", "Specialist referral if persistent"]

    def _calculate_severity_score(self, lab_values: list[LabValue], critical_count: int) -> int:
        """Calculate overall severity score."""
        if not lab_values:
            return 0

        base_score = 20

        # Critical values have highest impact
        base_score += critical_count * 25

        # Add for each abnormal value
        for value in lab_values:
            if value.status in ["low", "high"]:
                # Score based on deviation
                if value.percent_deviation > 50:
                    base_score += 15
                elif value.percent_deviation > 25:
                    base_score += 10
                else:
                    base_score += 5

        return min(100, base_score)

    def _calculate_confidence(self, total_tests: int, abnormal_count: int) -> int:
        """Calculate analysis confidence."""
        if total_tests == 0:
            return 30

        # More tests = more confidence
        confidence = 70 + min(20, total_tests * 2)

        return min(95, confidence)

    def _build_sections(
        self,
        lab_values: list[LabValue],
        trends: list[TrendAnalysis],
        causes: list[UnderlyingCause],
        request: LabTestRequest
    ) -> list[UISection]:
        """Build UI sections for response."""
        sections = []

        # Results overview section
        normal_values = [v for v in lab_values if v.status == "normal"]
        abnormal_values = [v for v in lab_values if v.status != "normal"]

        # Abnormal values section (priority)
        if abnormal_values:
            abnormal_items = []
            for value in abnormal_values:
                abnormal_items.append(SectionItem(
                    label=value.test_name,
                    value=f"{value.value} {value.unit}",
                    badge=value.status.upper(),
                    severity="critical" if value.status == "critical" else "moderate" if value.severity_color in ["orange", "red"] else "mild",
                    explanation=f"Reference: {value.reference_low}-{value.reference_high} {value.unit}. {value.interpretation}"
                ))

            sections.append(UISection(
                title="Abnormal Results",
                icon="alert-triangle",
                items=abnormal_items,
                priority=0
            ))

        # Normal values section
        if normal_values:
            normal_items = []
            for value in normal_values:
                normal_items.append(SectionItem(
                    label=value.test_name,
                    value=f"{value.value} {value.unit}",
                    badge="NORMAL",
                    severity="mild",
                    explanation=f"Reference: {value.reference_low}-{value.reference_high} {value.unit}"
                ))

            sections.append(UISection(
                title="Normal Results",
                icon="check-circle",
                items=normal_items,
                expandable=True,
                priority=2
            ))

        # Trends section
        if trends:
            trend_items = []
            for trend in trends:
                trend_items.append(SectionItem(
                    label=trend.test_name,
                    value=trend.rate_of_change,
                    badge=trend.direction.upper(),
                    severity="moderate" if trend.direction != "stable" else "mild",
                    explanation=trend.clinical_significance
                ))

            sections.append(UISection(
                title="Trend Analysis",
                icon="trending-up",
                items=trend_items,
                priority=1
            ))

        # Potential causes section
        if causes:
            cause_items = []
            for cause in causes[:3]:
                cause_items.append(SectionItem(
                    label=cause.cause,
                    value=cause.explanation,
                    probability=float(cause.probability),
                    badge=f"{cause.probability}%",
                    explanation=f"Related: {', '.join(cause.related_tests[:2])}"
                ))

            sections.append(UISection(
                title="Potential Underlying Causes",
                icon="search",
                items=cause_items,
                expandable=True,
                priority=3
            ))

        return sections

    def _generate_red_flags(
        self,
        lab_values: list[LabValue],
        critical_count: int
    ) -> list[RedFlag]:
        """Generate red flags for critical findings."""
        red_flags = []

        # Critical values
        for value in lab_values:
            if value.status == "critical":
                red_flags.append(RedFlag(
                    flag=f"Critical {value.test_name}",
                    explanation=value.interpretation,
                    action="Contact healthcare provider immediately",
                    severity=SeverityLevel.CRITICAL
                ))

        # Specific dangerous combinations
        value_dict = {v.test_name: v for v in lab_values}

        # Hyperkalemia check
        potassium = value_dict.get("Potassium")
        if potassium and potassium.value > 6.0:
            red_flags.append(RedFlag(
                flag="Dangerous hyperkalemia",
                explanation="Potassium >6.0 mEq/L carries risk of cardiac arrhythmia",
                action="Urgent ECG and treatment needed",
                severity=SeverityLevel.CRITICAL
            ))

        # Severe anemia check
        hemoglobin = value_dict.get("Hemoglobin")
        if hemoglobin and hemoglobin.value < 7.0:
            red_flags.append(RedFlag(
                flag="Severe anemia",
                explanation="Hemoglobin <7 g/dL may require transfusion",
                action="Urgent hematology evaluation",
                severity=SeverityLevel.CRITICAL
            ))

        return red_flags

    def _generate_recommendations(
        self,
        lab_values: list[LabValue],
        abnormal_count: int,
        critical_count: int,
        patient_conditions: list[str]
    ) -> list[Recommendation]:
        """Generate actionable recommendations."""
        recommendations = []

        # Urgent recommendations for critical values
        if critical_count > 0:
            recommendations.append(Recommendation(
                title="Urgent Medical Review",
                description=f"You have {critical_count} critical lab value(s) requiring immediate medical attention.",
                priority="high",
                category="follow-up",
                timeframe="Today"
            ))

        # Follow-up for abnormal values
        if abnormal_count > 0:
            recommendations.append(Recommendation(
                title="Discuss Results with Provider",
                description=f"{abnormal_count} results are outside normal range. Review with your healthcare provider for interpretation.",
                priority="medium" if critical_count == 0 else "high",
                category="follow-up",
                timeframe="Within 1 week" if critical_count == 0 else "Within 24-48 hours"
            ))

        # Specific recommendations based on abnormalities
        for value in lab_values:
            if value.status == "low" and "Hemoglobin" in value.test_name:
                recommendations.append(Recommendation(
                    title="Anemia Evaluation",
                    description="Low hemoglobin detected. Additional testing (iron studies, B12, folate) may be warranted.",
                    priority="medium",
                    category="follow-up",
                    timeframe="Within 2 weeks"
                ))
            elif value.status == "high" and "Glucose" in value.test_name:
                recommendations.append(Recommendation(
                    title="Glucose Monitoring",
                    description="Elevated glucose detected. Consider HbA1c testing and dietary modifications.",
                    priority="medium",
                    category="monitoring",
                    timeframe="Within 1 month"
                ))
            elif value.status == "high" and "LDL" in value.test_name:
                recommendations.append(Recommendation(
                    title="Lipid Management",
                    description="Elevated LDL cholesterol. Discuss lifestyle modifications and possible medication with provider.",
                    priority="medium",
                    category="lifestyle",
                    timeframe="At next visit"
                ))

        # General recommendation
        recommendations.append(Recommendation(
            title="Regular Monitoring",
            description="Continue routine lab monitoring as recommended by your healthcare provider.",
            priority="low",
            category="monitoring",
            timeframe="As scheduled"
        ))

        return recommendations[:6]

    def _calculate_retest_interval(self, abnormal_count: int, critical_count: int) -> str:
        """Calculate recommended retest interval."""
        if critical_count > 0:
            return "Retest within 24-48 hours after intervention"
        elif abnormal_count > 3:
            return "Retest in 1-2 weeks"
        elif abnormal_count > 0:
            return "Retest in 1-3 months"
        else:
            return "Routine retest in 6-12 months"

    def _generate_lifestyle_factors(
        self,
        lab_values: list[LabValue],
        test_type: LabTestCategory
    ) -> list[str]:
        """Generate relevant lifestyle modification suggestions."""
        factors = []

        # Check for specific abnormalities
        for value in lab_values:
            if value.status in ["high", "low"]:
                if "Cholesterol" in value.test_name or "LDL" in value.test_name or "Triglycerides" in value.test_name:
                    factors.extend([
                        "Reduce saturated fat intake",
                        "Increase fiber consumption",
                        "Regular aerobic exercise",
                        "Maintain healthy weight"
                    ])
                elif "Glucose" in value.test_name:
                    factors.extend([
                        "Reduce simple carbohydrate intake",
                        "Regular physical activity",
                        "Weight management",
                        "Consistent meal timing"
                    ])
                elif "Hemoglobin" in value.test_name and value.status == "low":
                    factors.extend([
                        "Iron-rich foods (lean meat, spinach, beans)",
                        "Vitamin C to enhance iron absorption",
                        "Avoid tea/coffee with meals"
                    ])

        # General factors based on test type
        if test_type == LabTestCategory.LIPID_PANEL:
            factors.extend([
                "Limit alcohol consumption",
                "Choose healthy fats (olive oil, nuts)"
            ])
        elif test_type == LabTestCategory.LIVER_ENZYMES:
            factors.extend([
                "Limit alcohol consumption",
                "Maintain healthy weight",
                "Avoid hepatotoxic medications when possible"
            ])

        # Remove duplicates while preserving order
        seen = set()
        unique_factors = []
        for f in factors:
            if f not in seen:
                seen.add(f)
                unique_factors.append(f)

        return unique_factors[:6]

    def _generate_summary(
        self,
        test_type: LabTestCategory,
        total_tests: int,
        abnormal_count: int,
        critical_count: int
    ) -> str:
        """Generate concise summary."""
        type_name = test_type.value.replace("_", " ").upper()

        if critical_count > 0:
            return (f"{type_name}: Analyzed {total_tests} tests. Found {critical_count} critical and "
                   f"{abnormal_count - critical_count} abnormal results requiring urgent review.")
        elif abnormal_count > 0:
            return (f"{type_name}: Analyzed {total_tests} tests. Found {abnormal_count} results outside "
                   f"normal range. Review with healthcare provider recommended.")
        else:
            return f"{type_name}: Analyzed {total_tests} tests. All results within normal limits."


# Singleton instance
_analyzer_instance: Optional[LabTestAnalyzer] = None


async def get_lab_test_analyzer() -> LabTestAnalyzer:
    """Get or create LabTestAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = LabTestAnalyzer()
    return _analyzer_instance
