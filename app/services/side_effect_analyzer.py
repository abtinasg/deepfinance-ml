"""
DeepHealth Side Effect Analyzer Service

Comprehensive side effect analysis with likelihood modeling,
symptom correlation, time-pattern analysis, and severity classification.
"""

from datetime import datetime
from typing import Any, Optional

from app.core.logging import get_logger
from app.schemas.medical import (
    Alternative,
    LikelihoodCategory,
    Recommendation,
    RedFlag,
    ScoreSet,
    SectionItem,
    SeverityLevel,
    SideEffect,
    SideEffectRequest,
    SideEffectResponse,
    SymptomCorrelation,
    TimePattern,
    UISection,
)

logger = get_logger(__name__)


# ============================================================================
# SIDE EFFECT DATABASE - Comprehensive Drug Side Effect Information
# ============================================================================

SIDE_EFFECT_DATABASE = {
    "metformin": {
        "class": "biguanide",
        "common_uses": ["type 2 diabetes", "prediabetes", "PCOS"],
        "side_effects": [
            {
                "name": "Nausea",
                "likelihood": "very_common",
                "probability": 25.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Direct GI mucosal irritation and delayed gastric emptying",
                "management": "Take with food; start low and titrate slowly; consider extended-release formulation"
            },
            {
                "name": "Diarrhea",
                "likelihood": "very_common",
                "probability": 30.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Increased intestinal glucose utilization and bile acid malabsorption",
                "management": "Usually resolves in 2-4 weeks; take with meals; try extended-release"
            },
            {
                "name": "Abdominal pain",
                "likelihood": "common",
                "probability": 12.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "GI tract irritation and increased intestinal motility",
                "management": "Take after meals; divide doses; consider dose reduction"
            },
            {
                "name": "Metallic taste",
                "likelihood": "common",
                "probability": 8.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Drug accumulation in taste buds affecting ion channels",
                "management": "Usually temporary; mint or gum may help"
            },
            {
                "name": "Vitamin B12 deficiency",
                "likelihood": "uncommon",
                "probability": 5.0,
                "severity": "moderate",
                "time_pattern": "cumulative",
                "reversible": True,
                "mechanism": "Impaired calcium-dependent B12 absorption in terminal ileum",
                "management": "Monitor B12 levels annually; supplement if deficient"
            },
            {
                "name": "Lactic acidosis",
                "likelihood": "very_rare",
                "probability": 0.03,
                "severity": "critical",
                "time_pattern": "cumulative",
                "reversible": False,
                "mechanism": "Inhibition of hepatic gluconeogenesis and lactate clearance",
                "management": "MEDICAL EMERGENCY - Stop drug immediately; supportive care"
            }
        ],
        "serious_adverse_events": [
            "Lactic acidosis (0.03/1000 patient-years)",
            "Severe hypoglycemia when combined with sulfonylureas",
            "Acute kidney injury (hold before contrast procedures)"
        ],
        "when_to_seek_help": [
            "Unusual muscle pain or weakness",
            "Difficulty breathing or fast breathing",
            "Unusual sleepiness or dizziness",
            "Stomach pain with nausea and vomiting",
            "Feeling very cold, especially in arms and legs"
        ],
        "alternatives": [
            {"name": "Sitagliptin", "reason": "DPP-4 inhibitor with minimal GI side effects"},
            {"name": "Empagliflozin", "reason": "SGLT2 inhibitor with cardiovascular benefits"},
            {"name": "Pioglitazone", "reason": "TZD with different mechanism if GI intolerance persists"}
        ]
    },
    "lisinopril": {
        "class": "ace_inhibitor",
        "common_uses": ["hypertension", "heart failure", "diabetic nephropathy"],
        "side_effects": [
            {
                "name": "Dry cough",
                "likelihood": "very_common",
                "probability": 20.0,
                "severity": "mild",
                "time_pattern": "delayed",
                "reversible": True,
                "mechanism": "Accumulation of bradykinin in bronchial tissue",
                "management": "May take weeks to develop; switch to ARB if intolerable"
            },
            {
                "name": "Dizziness",
                "likelihood": "common",
                "probability": 10.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "First-dose hypotension from decreased angiotensin II",
                "management": "Start low dose; take at bedtime; rise slowly from sitting"
            },
            {
                "name": "Hyperkalemia",
                "likelihood": "uncommon",
                "probability": 4.0,
                "severity": "moderate",
                "time_pattern": "cumulative",
                "reversible": True,
                "mechanism": "Reduced aldosterone secretion decreases renal potassium excretion",
                "management": "Monitor potassium levels; avoid potassium supplements; low-potassium diet"
            },
            {
                "name": "Fatigue",
                "likelihood": "common",
                "probability": 8.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Blood pressure reduction and decreased tissue perfusion",
                "management": "Usually resolves with continued use; adequate hydration helps"
            },
            {
                "name": "Angioedema",
                "likelihood": "rare",
                "probability": 0.5,
                "severity": "critical",
                "time_pattern": "delayed",
                "reversible": True,
                "mechanism": "Bradykinin accumulation causing increased vascular permeability",
                "management": "MEDICAL EMERGENCY - Stop drug permanently; never rechallenge"
            },
            {
                "name": "Acute kidney injury",
                "likelihood": "rare",
                "probability": 1.0,
                "severity": "severe",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Loss of efferent arteriolar tone in kidneys with renal artery stenosis",
                "management": "Monitor creatinine; avoid in bilateral renal artery stenosis"
            }
        ],
        "serious_adverse_events": [
            "Angioedema (can be fatal if airway involved)",
            "Hyperkalemia with cardiac arrhythmia",
            "Acute renal failure in renal artery stenosis",
            "Fetal toxicity (Category D in pregnancy)"
        ],
        "when_to_seek_help": [
            "Swelling of face, lips, tongue, or throat",
            "Difficulty swallowing or breathing",
            "Severe dizziness or fainting",
            "Signs of infection (fever, sore throat)",
            "Significant decrease in urination"
        ],
        "alternatives": [
            {"name": "Losartan", "reason": "ARB with lower cough risk"},
            {"name": "Amlodipine", "reason": "Calcium channel blocker if ACE/ARB contraindicated"},
            {"name": "Hydrochlorothiazide", "reason": "Diuretic for hypertension without cough risk"}
        ]
    },
    "atorvastatin": {
        "class": "statin",
        "common_uses": ["hyperlipidemia", "cardiovascular prevention", "familial hypercholesterolemia"],
        "side_effects": [
            {
                "name": "Muscle pain",
                "likelihood": "common",
                "probability": 10.0,
                "severity": "mild",
                "time_pattern": "delayed",
                "reversible": True,
                "mechanism": "Impaired mitochondrial function and CoQ10 depletion in muscle",
                "management": "Check CK levels; try CoQ10 supplement; consider alternate statin"
            },
            {
                "name": "Headache",
                "likelihood": "common",
                "probability": 8.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Unknown; possibly related to cholesterol synthesis changes in CNS",
                "management": "Usually resolves within weeks; OTC analgesics if needed"
            },
            {
                "name": "GI upset",
                "likelihood": "common",
                "probability": 6.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Direct GI irritation and altered bile acid metabolism",
                "management": "Take with food; usually transient"
            },
            {
                "name": "Elevated liver enzymes",
                "likelihood": "uncommon",
                "probability": 2.0,
                "severity": "moderate",
                "time_pattern": "delayed",
                "reversible": True,
                "mechanism": "Hepatocyte stress from altered lipid metabolism",
                "management": "Monitor LFTs; usually asymptomatic; discontinue if >3x ULN"
            },
            {
                "name": "Rhabdomyolysis",
                "likelihood": "very_rare",
                "probability": 0.01,
                "severity": "critical",
                "time_pattern": "cumulative",
                "reversible": False,
                "mechanism": "Severe muscle breakdown from mitochondrial toxicity",
                "management": "MEDICAL EMERGENCY - Stop drug; IV fluids; monitor for renal failure"
            },
            {
                "name": "New-onset diabetes",
                "likelihood": "rare",
                "probability": 0.5,
                "severity": "moderate",
                "time_pattern": "cumulative",
                "reversible": False,
                "mechanism": "Impaired pancreatic beta-cell function and insulin signaling",
                "management": "Monitor glucose; benefits usually outweigh risks in high-risk patients"
            }
        ],
        "serious_adverse_events": [
            "Rhabdomyolysis (0.1/10,000 patient-years)",
            "Immune-mediated necrotizing myopathy",
            "Hepatotoxicity (rare)",
            "Increased risk of type 2 diabetes"
        ],
        "when_to_seek_help": [
            "Unexplained muscle pain, tenderness, or weakness",
            "Dark or cola-colored urine",
            "Yellowing of skin or eyes",
            "Severe fatigue or weakness",
            "Upper right abdominal pain"
        ],
        "alternatives": [
            {"name": "Rosuvastatin", "reason": "Hydrophilic statin with potentially lower myopathy risk"},
            {"name": "Pravastatin", "reason": "Lower drug interaction potential"},
            {"name": "Ezetimibe", "reason": "Non-statin option for statin-intolerant patients"}
        ]
    },
    "sertraline": {
        "class": "ssri",
        "common_uses": ["depression", "anxiety", "PTSD", "OCD", "panic disorder"],
        "side_effects": [
            {
                "name": "Nausea",
                "likelihood": "very_common",
                "probability": 25.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Serotonin stimulation of 5-HT3 receptors in GI tract",
                "management": "Take with food; usually resolves in 1-2 weeks"
            },
            {
                "name": "Insomnia",
                "likelihood": "common",
                "probability": 15.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Increased serotonergic activity in sleep-wake centers",
                "management": "Take in morning; sleep hygiene; may need short-term sleep aid"
            },
            {
                "name": "Sexual dysfunction",
                "likelihood": "very_common",
                "probability": 30.0,
                "severity": "moderate",
                "time_pattern": "delayed",
                "reversible": True,
                "mechanism": "Serotonin inhibition of dopamine in reward pathways",
                "management": "May not resolve; dose reduction or switch to bupropion"
            },
            {
                "name": "Headache",
                "likelihood": "common",
                "probability": 12.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Serotonin effects on cerebral blood vessels",
                "management": "Usually transient; OTC analgesics; adequate hydration"
            },
            {
                "name": "Weight changes",
                "likelihood": "common",
                "probability": 10.0,
                "severity": "mild",
                "time_pattern": "cumulative",
                "reversible": True,
                "mechanism": "Altered appetite regulation via hypothalamic serotonin",
                "management": "Monitor weight; dietary modification; exercise"
            },
            {
                "name": "Serotonin syndrome",
                "likelihood": "rare",
                "probability": 0.1,
                "severity": "critical",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Excessive serotonergic activity in CNS and periphery",
                "management": "MEDICAL EMERGENCY - Stop all serotonergic drugs; supportive care"
            },
            {
                "name": "Bleeding risk",
                "likelihood": "uncommon",
                "probability": 3.0,
                "severity": "moderate",
                "time_pattern": "cumulative",
                "reversible": True,
                "mechanism": "Serotonin depletion in platelets impairing aggregation",
                "management": "Use caution with anticoagulants/NSAIDs; monitor for bruising"
            }
        ],
        "serious_adverse_events": [
            "Serotonin syndrome (life-threatening)",
            "Suicidal ideation (especially in young adults)",
            "QT prolongation at high doses",
            "Severe bleeding events"
        ],
        "when_to_seek_help": [
            "New or worsening suicidal thoughts",
            "Agitation, restlessness, or irritability",
            "High fever with muscle stiffness",
            "Fast or irregular heartbeat",
            "Unusual bleeding or bruising"
        ],
        "alternatives": [
            {"name": "Bupropion", "reason": "No sexual side effects; also helps with smoking cessation"},
            {"name": "Mirtazapine", "reason": "Different mechanism; may help sleep and appetite"},
            {"name": "Venlafaxine", "reason": "SNRI option if SSRI response inadequate"}
        ]
    },
    "omeprazole": {
        "class": "ppi",
        "common_uses": ["GERD", "peptic ulcer", "H. pylori eradication", "Zollinger-Ellison"],
        "side_effects": [
            {
                "name": "Headache",
                "likelihood": "common",
                "probability": 7.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Unknown; possibly related to gastrin elevation",
                "management": "Usually transient; OTC analgesics if needed"
            },
            {
                "name": "Diarrhea",
                "likelihood": "common",
                "probability": 5.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Altered gut microbiome from reduced gastric acid",
                "management": "Usually mild; probiotics may help"
            },
            {
                "name": "Abdominal pain",
                "likelihood": "common",
                "probability": 4.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "GI motility changes and bacterial overgrowth",
                "management": "Monitor; usually resolves"
            },
            {
                "name": "Vitamin B12 deficiency",
                "likelihood": "uncommon",
                "probability": 3.0,
                "severity": "moderate",
                "time_pattern": "cumulative",
                "reversible": True,
                "mechanism": "Reduced acid-dependent B12 release from food proteins",
                "management": "Monitor B12 with long-term use; supplement if needed"
            },
            {
                "name": "Hypomagnesemia",
                "likelihood": "rare",
                "probability": 1.0,
                "severity": "moderate",
                "time_pattern": "cumulative",
                "reversible": True,
                "mechanism": "Impaired intestinal magnesium absorption",
                "management": "Monitor magnesium with long-term use; supplement if low"
            },
            {
                "name": "C. difficile infection",
                "likelihood": "rare",
                "probability": 0.5,
                "severity": "severe",
                "time_pattern": "cumulative",
                "reversible": True,
                "mechanism": "Reduced gastric acid allows C. diff survival and colonization",
                "management": "Stop PPI if possible; treat infection; use lowest effective dose"
            },
            {
                "name": "Bone fracture risk",
                "likelihood": "rare",
                "probability": 0.3,
                "severity": "moderate",
                "time_pattern": "cumulative",
                "reversible": False,
                "mechanism": "Impaired calcium absorption and bone metabolism",
                "management": "Use lowest dose for shortest duration; ensure adequate calcium/vitamin D"
            }
        ],
        "serious_adverse_events": [
            "C. difficile-associated diarrhea",
            "Increased fracture risk (hip, wrist, spine)",
            "Severe hypomagnesemia",
            "Acute interstitial nephritis (rare)"
        ],
        "when_to_seek_help": [
            "Severe, watery diarrhea that won't stop",
            "Muscle spasms or cramps",
            "Irregular heartbeat",
            "Seizures",
            "Dizziness or jitteriness"
        ],
        "alternatives": [
            {"name": "Famotidine", "reason": "H2 blocker with lower long-term risks"},
            {"name": "Pantoprazole", "reason": "PPI with fewer drug interactions"},
            {"name": "Antacids", "reason": "For occasional, mild symptoms only"}
        ]
    },
    "amlodipine": {
        "class": "calcium_channel_blocker",
        "common_uses": ["hypertension", "angina", "coronary artery disease"],
        "side_effects": [
            {
                "name": "Peripheral edema",
                "likelihood": "very_common",
                "probability": 15.0,
                "severity": "mild",
                "time_pattern": "dose_dependent",
                "reversible": True,
                "mechanism": "Preferential arteriolar dilation increases capillary pressure",
                "management": "Dose-dependent; reduce dose or add ARB/ACE-I to reduce edema"
            },
            {
                "name": "Headache",
                "likelihood": "common",
                "probability": 8.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Vasodilation of cerebral blood vessels",
                "management": "Usually transient in first weeks; OTC analgesics"
            },
            {
                "name": "Flushing",
                "likelihood": "common",
                "probability": 6.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Cutaneous vasodilation",
                "management": "Usually resolves; avoid triggers like alcohol and hot beverages"
            },
            {
                "name": "Dizziness",
                "likelihood": "common",
                "probability": 5.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Blood pressure reduction",
                "management": "Rise slowly; take at bedtime if problematic"
            },
            {
                "name": "Fatigue",
                "likelihood": "uncommon",
                "probability": 3.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Blood pressure reduction and reflex responses",
                "management": "Usually improves with time"
            },
            {
                "name": "Gingival hyperplasia",
                "likelihood": "rare",
                "probability": 1.0,
                "severity": "mild",
                "time_pattern": "cumulative",
                "reversible": True,
                "mechanism": "Altered collagen metabolism in gingival fibroblasts",
                "management": "Good dental hygiene; may need drug change if severe"
            }
        ],
        "serious_adverse_events": [
            "Severe hypotension (rare)",
            "Reflex tachycardia",
            "Worsening heart failure in some patients"
        ],
        "when_to_seek_help": [
            "Severe dizziness or fainting",
            "Rapid or irregular heartbeat",
            "Severe swelling of ankles or feet",
            "Difficulty breathing",
            "Chest pain"
        ],
        "alternatives": [
            {"name": "Diltiazem", "reason": "Non-dihydropyridine CCB with less edema"},
            {"name": "Lisinopril", "reason": "ACE inhibitor that may reduce CCB-induced edema"},
            {"name": "Losartan", "reason": "ARB alternative for hypertension"}
        ]
    },
    "ibuprofen": {
        "class": "nsaid",
        "common_uses": ["pain", "inflammation", "fever", "arthritis"],
        "side_effects": [
            {
                "name": "GI upset",
                "likelihood": "very_common",
                "probability": 20.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Inhibition of protective prostaglandins in gastric mucosa",
                "management": "Take with food; use lowest effective dose"
            },
            {
                "name": "Headache",
                "likelihood": "common",
                "probability": 5.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Unknown; paradoxical effect",
                "management": "Usually transient"
            },
            {
                "name": "Dizziness",
                "likelihood": "common",
                "probability": 4.0,
                "severity": "mild",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Central prostaglandin inhibition",
                "management": "Take with food; avoid driving if affected"
            },
            {
                "name": "Fluid retention",
                "likelihood": "uncommon",
                "probability": 3.0,
                "severity": "mild",
                "time_pattern": "cumulative",
                "reversible": True,
                "mechanism": "Renal prostaglandin inhibition reduces sodium excretion",
                "management": "Monitor weight; limit sodium; use intermittently"
            },
            {
                "name": "GI bleeding",
                "likelihood": "uncommon",
                "probability": 2.0,
                "severity": "severe",
                "time_pattern": "dose_dependent",
                "reversible": True,
                "mechanism": "Mucosal erosion and impaired platelet function",
                "management": "Use PPI prophylaxis if high risk; limit duration"
            },
            {
                "name": "Acute kidney injury",
                "likelihood": "rare",
                "probability": 1.0,
                "severity": "severe",
                "time_pattern": "immediate",
                "reversible": True,
                "mechanism": "Loss of renal prostaglandin-mediated vasodilation",
                "management": "Avoid in renal disease; adequate hydration"
            },
            {
                "name": "Cardiovascular events",
                "likelihood": "rare",
                "probability": 0.5,
                "severity": "critical",
                "time_pattern": "cumulative",
                "reversible": False,
                "mechanism": "Prostacyclin inhibition promotes thrombosis",
                "management": "Limit duration and dose; avoid in CAD"
            }
        ],
        "serious_adverse_events": [
            "GI bleeding/perforation",
            "Acute kidney injury",
            "Cardiovascular thrombotic events (MI, stroke)",
            "Severe allergic reactions"
        ],
        "when_to_seek_help": [
            "Black, tarry, or bloody stools",
            "Vomiting blood or coffee-ground material",
            "Chest pain or shortness of breath",
            "Slurred speech or weakness on one side",
            "Severe skin rash or hives"
        ],
        "alternatives": [
            {"name": "Acetaminophen", "reason": "No GI or cardiovascular risks"},
            {"name": "Celecoxib", "reason": "COX-2 selective with lower GI risk"},
            {"name": "Topical NSAIDs", "reason": "Local effect with minimal systemic absorption"}
        ]
    },
    "warfarin": {
        "class": "anticoagulant",
        "common_uses": ["atrial fibrillation", "DVT/PE", "mechanical heart valves"],
        "side_effects": [
            {
                "name": "Bleeding",
                "likelihood": "very_common",
                "probability": 15.0,
                "severity": "moderate",
                "time_pattern": "dose_dependent",
                "reversible": True,
                "mechanism": "Inhibition of vitamin K-dependent clotting factors",
                "management": "Regular INR monitoring; maintain therapeutic range"
            },
            {
                "name": "Bruising",
                "likelihood": "very_common",
                "probability": 20.0,
                "severity": "mild",
                "time_pattern": "dose_dependent",
                "reversible": True,
                "mechanism": "Impaired coagulation from reduced clotting factors",
                "management": "Expected effect; concerning only if excessive"
            },
            {
                "name": "Hair loss",
                "likelihood": "uncommon",
                "probability": 2.0,
                "severity": "mild",
                "time_pattern": "delayed",
                "reversible": True,
                "mechanism": "Unclear; possibly telogen effluvium",
                "management": "Usually reversible; rarely requires treatment change"
            },
            {
                "name": "Purple toe syndrome",
                "likelihood": "rare",
                "probability": 0.5,
                "severity": "moderate",
                "time_pattern": "delayed",
                "reversible": True,
                "mechanism": "Cholesterol microembolization from plaque destabilization",
                "management": "Discontinue warfarin; switch to alternative anticoagulant"
            },
            {
                "name": "Skin necrosis",
                "likelihood": "rare",
                "probability": 0.1,
                "severity": "severe",
                "time_pattern": "immediate",
                "reversible": False,
                "mechanism": "Protein C depletion faster than other factors in early therapy",
                "management": "Requires bridging with heparin; may need skin grafts"
            },
            {
                "name": "Major hemorrhage",
                "likelihood": "uncommon",
                "probability": 3.0,
                "severity": "critical",
                "time_pattern": "dose_dependent",
                "reversible": True,
                "mechanism": "Supratherapeutic anticoagulation",
                "management": "MEDICAL EMERGENCY - Vitamin K, FFP, PCC as indicated"
            }
        ],
        "serious_adverse_events": [
            "Intracranial hemorrhage",
            "GI hemorrhage",
            "Retroperitoneal bleeding",
            "Skin necrosis (protein C deficiency)",
            "Calciphylaxis"
        ],
        "when_to_seek_help": [
            "Unusual bleeding or bruising",
            "Blood in urine or stool",
            "Severe headache or dizziness",
            "Coughing up blood",
            "Pain or swelling after minor injury"
        ],
        "alternatives": [
            {"name": "Apixaban", "reason": "DOAC with no INR monitoring; lower bleeding risk"},
            {"name": "Rivaroxaban", "reason": "Once-daily DOAC; fewer drug interactions"},
            {"name": "Dabigatran", "reason": "DOAC with reversal agent available"}
        ]
    }
}


class SideEffectAnalyzer:
    """
    Comprehensive side effect analysis service with likelihood modeling.

    Features:
    - Side effect likelihood categorization (very common to very rare)
    - Severity classification for each effect
    - Symptom-drug correlation analysis
    - Time-pattern recognition
    - Structured safety information
    - Alternative medication suggestions
    """

    def __init__(self):
        self._logger = logger

    async def analyze(self, request: SideEffectRequest) -> SideEffectResponse:
        """
        Perform comprehensive side effect analysis.

        Args:
            request: Side effect analysis request.

        Returns:
            Complete analysis with effects, correlations, and recommendations.
        """
        self._logger.info(f"Analyzing side effects for drug: {request.drug}")

        # Normalize drug name
        drug = self._normalize_drug_name(request.drug)

        # Get drug information
        drug_info = SIDE_EFFECT_DATABASE.get(drug)

        if not drug_info:
            return self._create_unknown_drug_response(request)

        # Extract and categorize side effects
        expected_effects = self._categorize_expected_effects(drug_info["side_effects"])
        serious_events = self._extract_serious_events(drug_info["side_effects"])

        # Analyze symptom correlation if provided
        correlations = []
        if request.symptom:
            correlations = self._analyze_symptom_correlation(
                request.symptom,
                drug,
                drug_info["side_effects"],
                request.duration_of_use
            )

        # Calculate scores
        severity_score = self._calculate_severity_score(drug_info["side_effects"], request.symptom, correlations)
        confidence = self._calculate_confidence(drug in SIDE_EFFECT_DATABASE, bool(request.symptom), bool(correlations))

        # Build UI sections
        sections = self._build_sections(drug, drug_info, expected_effects, serious_events, correlations)

        # Generate red flags
        red_flags = self._generate_red_flags(drug_info, request.symptom, correlations)

        # Generate recommendations
        recommendations = self._generate_recommendations(drug_info, request, correlations)

        # Generate alternatives
        alternatives = self._format_alternatives(drug_info.get("alternatives", []))

        # Generate summary
        summary = self._generate_summary(request, drug_info, correlations)

        return SideEffectResponse(
            summary=summary,
            scores=ScoreSet(
                confidence=confidence,
                severity=severity_score,
                urgency=self._calculate_urgency(correlations, request.symptom)
            ),
            drug_analyzed=request.drug,
            user_symptom=request.symptom,
            expected_effects=expected_effects,
            serious_adverse_events=serious_events,
            symptom_correlations=correlations,
            when_to_seek_help=drug_info.get("when_to_seek_help", []),
            sections=sections,
            red_flags=red_flags,
            recommendations=recommendations,
            alternatives=alternatives,
            metadata={
                "drug_class": drug_info.get("class", "unknown"),
                "common_uses": drug_info.get("common_uses", []),
                "analysis_version": "2.0",
                "dosage": request.dosage,
                "duration_of_use": request.duration_of_use
            },
            timestamp=datetime.utcnow()
        )

    def _normalize_drug_name(self, drug: str) -> str:
        """Normalize drug name for database lookup."""
        return drug.lower().strip().replace("-", "").replace(" ", "")

    def _categorize_expected_effects(self, side_effects: list[dict]) -> list[SideEffect]:
        """Categorize and format expected side effects."""
        likelihood_map = {
            "very_common": LikelihoodCategory.VERY_COMMON,
            "common": LikelihoodCategory.COMMON,
            "uncommon": LikelihoodCategory.UNCOMMON,
            "rare": LikelihoodCategory.RARE,
            "very_rare": LikelihoodCategory.VERY_RARE
        }

        severity_map = {
            "mild": SeverityLevel.MILD,
            "moderate": SeverityLevel.MODERATE,
            "severe": SeverityLevel.SEVERE,
            "critical": SeverityLevel.CRITICAL
        }

        time_pattern_map = {
            "immediate": TimePattern.IMMEDIATE,
            "delayed": TimePattern.DELAYED,
            "dose_dependent": TimePattern.DOSE_DEPENDENT,
            "cumulative": TimePattern.CUMULATIVE
        }

        expected = []
        for effect in side_effects:
            # Only include non-critical as "expected"
            if effect.get("severity") != "critical":
                expected.append(SideEffect(
                    name=effect["name"],
                    likelihood=likelihood_map.get(effect.get("likelihood", "common"), LikelihoodCategory.COMMON),
                    probability_percent=effect.get("probability", 5.0),
                    severity=severity_map.get(effect.get("severity", "mild"), SeverityLevel.MILD),
                    time_pattern=time_pattern_map.get(effect.get("time_pattern", "immediate"), TimePattern.IMMEDIATE),
                    reversible=effect.get("reversible", True),
                    mechanism=effect.get("mechanism", "Mechanism not fully characterized"),
                    management=effect.get("management", "Consult healthcare provider")
                ))

        # Sort by probability (most common first)
        expected.sort(key=lambda x: x.probability_percent, reverse=True)
        return expected

    def _extract_serious_events(self, side_effects: list[dict]) -> list[SideEffect]:
        """Extract serious and critical adverse events."""
        likelihood_map = {
            "very_common": LikelihoodCategory.VERY_COMMON,
            "common": LikelihoodCategory.COMMON,
            "uncommon": LikelihoodCategory.UNCOMMON,
            "rare": LikelihoodCategory.RARE,
            "very_rare": LikelihoodCategory.VERY_RARE
        }

        severity_map = {
            "mild": SeverityLevel.MILD,
            "moderate": SeverityLevel.MODERATE,
            "severe": SeverityLevel.SEVERE,
            "critical": SeverityLevel.CRITICAL
        }

        time_pattern_map = {
            "immediate": TimePattern.IMMEDIATE,
            "delayed": TimePattern.DELAYED,
            "dose_dependent": TimePattern.DOSE_DEPENDENT,
            "cumulative": TimePattern.CUMULATIVE
        }

        serious = []
        for effect in side_effects:
            if effect.get("severity") in ["severe", "critical"]:
                serious.append(SideEffect(
                    name=effect["name"],
                    likelihood=likelihood_map.get(effect.get("likelihood", "rare"), LikelihoodCategory.RARE),
                    probability_percent=effect.get("probability", 0.1),
                    severity=severity_map.get(effect.get("severity", "severe"), SeverityLevel.SEVERE),
                    time_pattern=time_pattern_map.get(effect.get("time_pattern", "immediate"), TimePattern.IMMEDIATE),
                    reversible=effect.get("reversible", True),
                    mechanism=effect.get("mechanism", ""),
                    management=effect.get("management", "Seek immediate medical attention")
                ))

        return serious

    def _analyze_symptom_correlation(
        self,
        symptom: str,
        drug: str,
        side_effects: list[dict],
        duration: Optional[str]
    ) -> list[SymptomCorrelation]:
        """Analyze correlation between user symptom and drug side effects."""
        correlations = []
        symptom_lower = symptom.lower()

        # Common symptom synonyms
        symptom_synonyms = {
            "nausea": ["nausea", "sick", "queasy", "upset stomach"],
            "diarrhea": ["diarrhea", "loose stools", "bowel problems"],
            "headache": ["headache", "head pain", "migraine"],
            "dizziness": ["dizziness", "dizzy", "lightheaded", "vertigo"],
            "fatigue": ["fatigue", "tired", "exhausted", "weakness", "lethargy"],
            "muscle pain": ["muscle pain", "myalgia", "muscle ache", "body aches"],
            "cough": ["cough", "coughing", "dry cough"],
            "rash": ["rash", "skin rash", "hives", "itching"],
            "insomnia": ["insomnia", "can't sleep", "sleep problems"],
            "anxiety": ["anxiety", "anxious", "nervous", "panic"]
        }

        for effect in side_effects:
            effect_name_lower = effect["name"].lower()

            # Check for direct or synonym match
            match_score = 0
            matched = False

            # Direct name match
            if effect_name_lower in symptom_lower or symptom_lower in effect_name_lower:
                match_score = 85
                matched = True
            else:
                # Check synonyms
                for canonical, synonyms in symptom_synonyms.items():
                    if any(s in symptom_lower for s in synonyms):
                        if any(s in effect_name_lower for s in synonyms) or canonical in effect_name_lower:
                            match_score = 75
                            matched = True
                            break

            if matched:
                # Adjust score based on probability
                prob_factor = min(1.0, effect.get("probability", 5) / 20)
                adjusted_score = int(match_score * (0.7 + 0.3 * prob_factor))

                # Analyze time consistency
                time_consistency = self._analyze_time_consistency(
                    effect.get("time_pattern", "immediate"),
                    duration
                )

                correlations.append(SymptomCorrelation(
                    symptom=symptom,
                    drug=drug.title(),
                    correlation_score=adjusted_score,
                    explanation=f"{effect['name']} is a {effect.get('likelihood', 'common').replace('_', ' ')} side effect "
                               f"occurring in ~{effect.get('probability', 5)}% of patients. {effect.get('mechanism', '')}",
                    time_consistency=time_consistency
                ))

        # Sort by correlation score
        correlations.sort(key=lambda x: x.correlation_score, reverse=True)
        return correlations[:3]  # Return top 3 correlations

    def _analyze_time_consistency(self, time_pattern: str, duration: Optional[str]) -> str:
        """Analyze if symptom timing is consistent with expected pattern."""
        if not duration:
            return "Unable to assess - duration not provided"

        duration_lower = duration.lower()

        # Parse approximate duration
        is_recent = any(term in duration_lower for term in ["day", "today", "yesterday", "1 week", "few days"])
        is_intermediate = any(term in duration_lower for term in ["week", "2 week", "1 month"])
        is_long_term = any(term in duration_lower for term in ["month", "year"])

        if time_pattern == "immediate":
            if is_recent:
                return "Timing consistent - immediate effects typically appear within days of starting"
            else:
                return "Timing less typical - immediate effects usually appear early in treatment"
        elif time_pattern == "delayed":
            if is_intermediate:
                return "Timing consistent - delayed effects typically appear after 1-4 weeks"
            elif is_recent:
                return "Possibly too early - delayed effects usually need more time to develop"
            else:
                return "Timing plausible - delayed effects can persist with continued use"
        elif time_pattern == "cumulative":
            if is_long_term:
                return "Timing consistent - cumulative effects develop with prolonged use"
            else:
                return "May be too early - cumulative effects typically need months to years"
        else:  # dose_dependent
            return "May correlate with recent dose changes - review dosing history"

    def _calculate_severity_score(
        self,
        side_effects: list[dict],
        symptom: Optional[str],
        correlations: list[SymptomCorrelation]
    ) -> int:
        """Calculate overall severity score for the drug."""
        base_score = 30  # Base severity for any medication

        # Factor in serious effects
        serious_count = sum(1 for e in side_effects if e.get("severity") in ["severe", "critical"])
        base_score += serious_count * 10

        # Factor in correlations
        if correlations:
            max_correlation = max(c.correlation_score for c in correlations)
            # Find the severity of the matched effect
            for effect in side_effects:
                if any(effect["name"].lower() in c.explanation.lower() for c in correlations):
                    if effect.get("severity") == "critical":
                        base_score += 30
                    elif effect.get("severity") == "severe":
                        base_score += 20
                    elif effect.get("severity") == "moderate":
                        base_score += 10
                    break

        return min(100, base_score)

    def _calculate_confidence(self, drug_known: bool, has_symptom: bool, has_correlations: bool) -> int:
        """Calculate analysis confidence."""
        if not drug_known:
            return 25

        confidence = 80
        if has_symptom and has_correlations:
            confidence = 90
        elif has_symptom and not has_correlations:
            confidence = 70  # Symptom may not be drug-related

        return confidence

    def _calculate_urgency(self, correlations: list[SymptomCorrelation], symptom: Optional[str]) -> int:
        """Calculate urgency score based on findings."""
        if not symptom:
            return 20  # No specific urgency without symptoms

        if not correlations:
            return 30  # Symptom may not be drug-related

        max_score = max(c.correlation_score for c in correlations)
        return min(70, int(max_score * 0.8))

    def _build_sections(
        self,
        drug: str,
        drug_info: dict,
        expected_effects: list[SideEffect],
        serious_events: list[SideEffect],
        correlations: list[SymptomCorrelation]
    ) -> list[UISection]:
        """Build UI sections for response."""
        sections = []

        # Symptom correlation section (if applicable)
        if correlations:
            correlation_items = []
            for corr in correlations:
                correlation_items.append(SectionItem(
                    label=f"Match: {corr.symptom}",
                    value=corr.explanation,
                    score=corr.correlation_score,
                    badge=f"{corr.correlation_score}% MATCH",
                    severity="moderate" if corr.correlation_score > 70 else "mild",
                    explanation=corr.time_consistency
                ))

            sections.append(UISection(
                title="Symptom-Drug Correlation",
                icon="git-compare",
                items=correlation_items,
                priority=0
            ))

        # Common side effects section
        common_items = []
        for effect in expected_effects[:6]:
            likelihood_badge = effect.likelihood.value.replace("_", " ").upper()
            common_items.append(SectionItem(
                label=effect.name,
                value=effect.management,
                probability=effect.probability_percent,
                badge=likelihood_badge,
                severity=effect.severity.value,
                explanation=effect.mechanism
            ))

        sections.append(UISection(
            title="Expected Side Effects",
            icon="pill",
            items=common_items,
            expandable=True,
            priority=1
        ))

        # Serious adverse events section
        if serious_events:
            serious_items = []
            for event in serious_events:
                serious_items.append(SectionItem(
                    label=event.name,
                    value=event.management,
                    probability=event.probability_percent,
                    badge=event.severity.value.upper(),
                    severity=event.severity.value,
                    explanation=event.mechanism
                ))

            sections.append(UISection(
                title="Serious Adverse Events",
                icon="alert-octagon",
                items=serious_items,
                expandable=True,
                priority=2
            ))

        # Time pattern section
        pattern_counts = {}
        for effect in expected_effects:
            pattern = effect.time_pattern.value.replace("_", " ").title()
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        pattern_items = [
            SectionItem(
                label=pattern,
                value=f"{count} side effect{'s' if count > 1 else ''} with this timing",
                explanation=self._get_pattern_explanation(pattern)
            )
            for pattern, count in pattern_counts.items()
        ]

        sections.append(UISection(
            title="Time Patterns",
            icon="clock",
            items=pattern_items,
            priority=3
        ))

        return sections

    def _get_pattern_explanation(self, pattern: str) -> str:
        """Get explanation for time pattern."""
        explanations = {
            "Immediate": "Occurs within hours to days of starting medication",
            "Delayed": "Develops after 1-4 weeks of continuous use",
            "Dose Dependent": "More likely at higher doses; may resolve with reduction",
            "Cumulative": "Builds up over months to years of use"
        }
        return explanations.get(pattern, "")

    def _generate_red_flags(
        self,
        drug_info: dict,
        symptom: Optional[str],
        correlations: list[SymptomCorrelation]
    ) -> list[RedFlag]:
        """Generate red flags for serious concerns."""
        red_flags = []

        # Check for correlation with serious effects
        for effect in drug_info.get("side_effects", []):
            if effect.get("severity") in ["severe", "critical"]:
                # Check if user symptom matches
                if symptom and any(
                    symptom.lower() in corr.explanation.lower()
                    for corr in correlations
                ):
                    if effect["name"].lower() in correlations[0].explanation.lower() if correlations else False:
                        red_flags.append(RedFlag(
                            flag=f"Possible {effect['name']}",
                            explanation=f"Your symptom '{symptom}' may indicate this serious adverse event",
                            action=effect.get("management", "Seek immediate medical attention"),
                            severity=SeverityLevel.SEVERE if effect.get("severity") == "severe" else SeverityLevel.CRITICAL
                        ))

        # Add when to seek help as potential red flags
        for warning in drug_info.get("when_to_seek_help", [])[:3]:
            red_flags.append(RedFlag(
                flag=warning,
                explanation="This symptom requires prompt medical evaluation",
                action="Contact healthcare provider immediately",
                severity=SeverityLevel.MODERATE
            ))

        return red_flags[:5]

    def _generate_recommendations(
        self,
        drug_info: dict,
        request: SideEffectRequest,
        correlations: list[SymptomCorrelation]
    ) -> list[Recommendation]:
        """Generate actionable recommendations."""
        recommendations = []

        # If symptom correlates strongly
        if correlations and correlations[0].correlation_score > 70:
            recommendations.append(Recommendation(
                title="Discuss Symptom with Provider",
                description=f"Your symptom '{request.symptom}' appears related to {request.drug}. "
                           "Discuss with prescriber for possible dose adjustment or alternative.",
                priority="high",
                category="follow-up",
                timeframe="Within this week"
            ))

        # General monitoring
        recommendations.append(Recommendation(
            title="Monitor for Side Effects",
            description=f"Track any new symptoms when starting {request.drug}. Most common effects "
                       "appear within the first 2-4 weeks and often improve with continued use.",
            priority="medium",
            category="monitoring",
            timeframe="Ongoing"
        ))

        # Specific to drug class
        drug_class = drug_info.get("class", "")
        if drug_class == "statin":
            recommendations.append(Recommendation(
                title="Report Muscle Symptoms",
                description="Report unexplained muscle pain, tenderness, or weakness to your provider promptly.",
                priority="medium",
                category="monitoring",
                timeframe="Ongoing"
            ))
        elif drug_class == "ace_inhibitor":
            recommendations.append(Recommendation(
                title="Monitor for Cough",
                description="A persistent dry cough is common with ACE inhibitors. If bothersome, discuss switching to an ARB.",
                priority="low",
                category="monitoring",
                timeframe="First 1-2 months"
            ))
        elif drug_class == "nsaid":
            recommendations.append(Recommendation(
                title="Limit Duration of Use",
                description="Use lowest effective dose for shortest duration. Take with food to reduce GI side effects.",
                priority="medium",
                category="medication",
                timeframe="Apply now"
            ))

        return recommendations

    def _format_alternatives(self, alternatives: list[dict]) -> list[Alternative]:
        """Format alternative suggestions."""
        return [
            Alternative(
                name=alt["name"],
                reason=alt["reason"],
                advantages=[alt["reason"]],
                considerations=["Discuss with prescriber before switching", "May have different side effect profile"]
            )
            for alt in alternatives
        ]

    def _generate_summary(
        self,
        request: SideEffectRequest,
        drug_info: dict,
        correlations: list[SymptomCorrelation]
    ) -> str:
        """Generate concise summary."""
        drug = request.drug.title()
        total_effects = len(drug_info.get("side_effects", []))

        if request.symptom and correlations:
            best_match = correlations[0]
            return (f"Analysis of {drug} identified {total_effects} known side effects. "
                   f"Your symptom '{request.symptom}' shows {best_match.correlation_score}% correlation "
                   f"with documented effects. {best_match.time_consistency}")
        else:
            common_count = sum(1 for e in drug_info.get("side_effects", [])
                              if e.get("likelihood") in ["very_common", "common"])
            return (f"Analysis of {drug} identified {total_effects} known side effects, "
                   f"including {common_count} common effects. Most are mild and manageable.")

    def _create_unknown_drug_response(self, request: SideEffectRequest) -> SideEffectResponse:
        """Create response for unknown drug."""
        return SideEffectResponse(
            summary=f"'{request.drug}' not found in database. Unable to provide comprehensive side effect analysis.",
            scores=ScoreSet(confidence=20, severity=50),
            drug_analyzed=request.drug,
            user_symptom=request.symptom,
            expected_effects=[],
            serious_adverse_events=[],
            symptom_correlations=[],
            when_to_seek_help=[
                "Consult prescriber or pharmacist for side effect information",
                "Report any unusual symptoms to your healthcare provider"
            ],
            sections=[
                UISection(
                    title="Drug Not Found",
                    icon="help-circle",
                    items=[
                        SectionItem(
                            label="Unknown Medication",
                            value="This drug is not in our database. Verify spelling or consult a pharmacist.",
                            severity="moderate"
                        )
                    ],
                    priority=0
                )
            ],
            red_flags=[],
            recommendations=[
                Recommendation(
                    title="Consult Pharmacist",
                    description="Your pharmacist can provide detailed side effect information for this medication.",
                    priority="high",
                    category="follow-up",
                    timeframe="As soon as possible"
                )
            ],
            alternatives=[],
            metadata={
                "drug_found": False,
                "analysis_version": "2.0"
            },
            timestamp=datetime.utcnow()
        )


# Singleton instance
_analyzer_instance: Optional[SideEffectAnalyzer] = None


async def get_side_effect_analyzer() -> SideEffectAnalyzer:
    """Get or create SideEffectAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SideEffectAnalyzer()
    return _analyzer_instance
