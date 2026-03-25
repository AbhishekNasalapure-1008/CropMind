"""
CropMind — Fertilizer Recommendation Engine
Maps nutrient deficiency predictions to actionable fertilizer recommendations.
"""

# ─────────────────────────────────────────────
# FERTILIZER DATABASE
# ─────────────────────────────────────────────
FERTILIZER_DB = {
    "nitrogen": {
        "low": {
            "fertilizer": "Urea (46-0-0)",
            "rate": "25 kg/ha",
            "method": "Foliar spray (2% solution)",
            "urgency": "Monitor",
            "reason_template": (
                "Light nitrogen deficiency detected. A foliar spray provides quick uptake "
                "without disturbing soil structure. Monitor over the next 7–10 days."
            ),
        },
        "moderate": {
            "fertilizer": "Urea (46-0-0)",
            "rate": "45 kg/ha",
            "method": "Soil incorporation (broadcast + light irrigation)",
            "urgency": "Within 1 Week",
            "reason_template": (
                "Moderate nitrogen deficiency is limiting photosynthesis and growth. Soil "
                "incorporation ensures steady release. Apply before the next rainfall event "
                "for best uptake."
            ),
        },
        "severe": {
            "fertilizer": "Ammonium Nitrate (34-0-0)",
            "rate": "60 kg/ha",
            "method": "Split application (50% now, 50% in 10 days)",
            "urgency": "Immediate",
            "reason_template": (
                "Severe nitrogen deficiency detected — crop yield is at risk. Split "
                "application prevents nitrogen burn while maximising uptake. Consider soil "
                "testing to understand long-term N availability."
            ),
        },
    },
    "phosphorus": {
        "low": {
            "fertilizer": "SSP (0-16-0)",
            "rate": "30 kg/ha",
            "method": "Band placement near root zone",
            "urgency": "Monitor",
            "reason_template": (
                "Mild phosphorus deficiency. Band placement near roots maximises uptake "
                "efficiency since phosphorus is relatively immobile in soil."
            ),
        },
        "moderate": {
            "fertilizer": "DAP (18-46-0)",
            "rate": "50 kg/ha",
            "method": "Soil incorporation before irrigation",
            "urgency": "Within 1 Week",
            "reason_template": (
                "Moderate phosphorus deficiency is impacting root development and energy "
                "transfer. DAP provides both nitrogen and phosphorus for balanced recovery. "
                "Incorporate before irrigation for best dissolution."
            ),
        },
        "severe": {
            "fertilizer": "TSP (0-46-0)",
            "rate": "70 kg/ha",
            "method": "Soil application + foliar supplement (0.5% KH₂PO₄)",
            "urgency": "Immediate",
            "reason_template": (
                "Severe phosphorus deficiency detected — critical for root and reproductive "
                "development. Combine soil application with a foliar supplement for faster "
                "recovery. Repeat foliar spray after 7 days."
            ),
        },
    },
    "potassium": {
        "low": {
            "fertilizer": "MOP (0-0-60)",
            "rate": "25 kg/ha",
            "method": "Broadcast + light incorporation",
            "urgency": "Monitor",
            "reason_template": (
                "Low potassium deficiency detected. Potassium supports water regulation and "
                "disease resistance. Light broadcast application maintains soil levels."
            ),
        },
        "moderate": {
            "fertilizer": "SOP (0-0-50)",
            "rate": "40 kg/ha",
            "method": "Soil incorporation (avoid direct seed/root contact)",
            "urgency": "Within 1 Week",
            "reason_template": (
                "Moderate potassium deficiency is reducing crop stress tolerance. SOP is "
                "preferred for chloride-sensitive crops. Apply mid-season for best uptake."
            ),
        },
        "severe": {
            "fertilizer": "Potassium Nitrate (13-0-46)",
            "rate": "55 kg/ha",
            "method": "Fertigation or foliar spray (2% solution)",
            "urgency": "Immediate",
            "reason_template": (
                "Severe potassium deficiency detected — leaf scorch and premature senescence "
                "are likely. Potassium nitrate via fertigation provides the fastest corrective "
                "response. Follow up with soil test after 2 weeks."
            ),
        },
    },
}

# ─────────────────────────────────────────────
# AGRONOMIC INSIGHTS
# ─────────────────────────────────────────────
AGRONOMIC_INSIGHTS = {
    "nitrogen": {
        "rice":      "Nitrogen is the most limiting nutrient in flooded rice systems. Deficiency causes yellowing (chlorosis) beginning from older leaves. In waterlogged conditions, N leaches rapidly, making split applications essential.",
        "wheat":     "Nitrogen drives wheat tillering and grain protein content. Deficiency at vegetative stage significantly reduces yield potential. Top-dressing with urea before flag leaf emergence is critical.",
        "maize":     "Maize is a heavy nitrogen feeder. Deficiency causes V-shaped yellowing from leaf tip toward the midrib on lower leaves. Side-dress application at V6 stage delivers maximum ROI.",
        "tomato":    "Nitrogen deficiency in tomato shows as pale green leaves and stunted growth. It directly reduces fruit set and size. Fertigation allows precise nitrogen management.",
        "potato":    "Nitrogen drives potato canopy development and tuber bulking. Deficiency causes premature senescence and reduced tuber count. Avoid excessive N which promotes foliage over tubers.",
        "default":   "Nitrogen is the primary driver of vegetative growth and chlorophyll synthesis. Deficiency causes yellowing of older leaves first (mobile nutrient). Apply nitrogen sources as per soil test recommendations.",
    },
    "phosphorus": {
        "soybean":   "Phosphorus is critical for soybean nodulation and nitrogen fixation. Deficiency causes dark green leaves with purple/reddish stems due to anthocyanin accumulation.",
        "wheat":     "Phosphorus supports wheat root development and tiller initiation. Deficiency at seedling stage can have lasting yield impacts. Band application at sowing is most efficient.",
        "potato":    "Phosphorus is essential for potato tuber initiation. Deficiency causes poor root development and delayed maturity. Band placement at planting maximises uptake in cold or acidic soils.",
        "default":   "Phosphorus drives root development, energy transfer (ATP), and reproductive growth. Deficiency causes purple/reddish discoloration on older leaves and stems. Phosphorus is immobile in soil — band placement near roots maximises efficiency.",
    },
    "potassium": {
        "tomato":    "Potassium is critical for tomato fruit quality, sugar accumulation, and disease resistance. Deficiency causes marginal leaf scorch and blossom-end rot (often confused with calcium deficiency).",
        "sugarcane": "Potassium supports sugarcane stalk strength, sugar translocation, and lodging resistance. Deficiency causes leaf tip and margin scorch starting from lower leaves.",
        "cotton":    "Potassium deficiency in cotton ('cotton leaf roll') causes bronzing and premature leaf shedding. It reduces fiber quality and boll retention. SOP is preferred to avoid chloride toxicity.",
        "default":   "Potassium regulates water uptake, enzyme activation, and stress tolerance. Deficiency causes marginal leaf scorch (necrosis at leaf edges), starting with older leaves. It improves crop quality metrics like sugar, starch, and protein content.",
    },
}

SYMPTOM_DESCRIPTIONS = {
    "nitrogen":   ["yellowing of lower/older leaves (chlorosis)", "stunted growth and reduced tillering", "pale green to yellow leaf color spreading upward", "thin, weak stems", "early leaf drop"],
    "phosphorus": ["purple or reddish leaf undersides and stems", "dark green leaves that appear dull", "delayed maturity and poor root development", "small, erect leaves", "reduced flower and fruit set"],
    "potassium":  ["marginal leaf scorch (brown, crispy edges)", "interveinal chlorosis on older leaves", "weak stems prone to lodging", "poor fruit/seed quality", "increased susceptibility to drought and disease"],
}

PREVENTIVE_TIPS = {
    "nitrogen": [
        "Conduct annual soil testing to track organic nitrogen reserves.",
        "Incorporate crop residues and green manures to build soil N over time.",
        "Use split nitrogen applications to reduce leaching losses.",
    ],
    "phosphorus": [
        "Maintain soil pH between 6.0–7.0 for optimal phosphorus availability.",
        "Incorporate phosphorus into the soil before planting for best root uptake.",
        "Use mycorrhizal inoculants to enhance natural phosphorus uptake by roots.",
    ],
    "potassium": [
        "Avoid excessive irrigation that leaches potassium from sandy soils.",
        "Incorporate potassium-rich organic matter (wood ash, compost) regularly.",
        "Monitor soil K levels annually and apply maintenance doses to high-demand crops.",
    ],
}

SOIL_NOTES = {
    "sandy":  "Sandy soil has low cation exchange capacity — split applications and slow-release fertilizers are strongly recommended to reduce losses.",
    "clay":   "Clay soil retains nutrients well but may have drainage issues — ensure proper soil aeration before application.",
    "loamy":  "Loamy soil provides ideal conditions for nutrient uptake. Standard rates apply.",
    "silty":  "Silty soil can compact easily — avoid heavy machinery after application to prevent structure damage.",
    "peaty":  "Peaty soil has high organic matter but can be acidic, reducing nutrient availability. Check pH before application.",
    "chalky": "Chalky (alkaline) soil can lock up phosphorus and micronutrients. Acidifying amendments may improve overall nutrient availability.",
}


def _severity_from_score(score: float) -> str:
    """Convert probability score to severity level."""
    if score >= 0.70:
        return "severe"
    elif score >= 0.45:
        return "moderate"
    elif score >= 0.25:
        return "low"
    return "none"


def get_recommendation(
    deficiencies: dict,
    severity: str,
    soil_type: str,
    crop_type: str
) -> dict:
    """
    Build complete recommendation package.

    Args:
        deficiencies: {"nitrogen": float, "phosphorus": float, "potassium": float}
        severity: overall severity string from model_inference
        soil_type: lowercase string
        crop_type: lowercase string

    Returns:
        {
            "recommendations": [...],
            "agronomic_insight": str,
            "symptoms_matched": [...],
            "preventive_tips": [...]
        }
    """
    soil_type = soil_type.lower().strip()
    crop_type = crop_type.lower().strip()

    # Determine top deficiency
    primary = max(deficiencies, key=deficiencies.get)
    primary_score = deficiencies[primary]

    # Collect all nutrients above threshold
    active_deficiencies = {
        k: v for k, v in deficiencies.items() if v >= 0.25
    }
    # Sort by severity
    sorted_deficiencies = sorted(active_deficiencies.items(), key=lambda x: x[1], reverse=True)

    recommendations = []
    for nutrient, score in sorted_deficiencies[:2]:  # top 2 recommendations
        sev = _severity_from_score(score)
        if sev == "none":
            continue
        rec = FERTILIZER_DB.get(nutrient, {}).get(sev)
        if not rec:
            continue

        soil_note = SOIL_NOTES.get(soil_type, "")
        reason = rec["reason_template"]
        if soil_note:
            reason += f" Note: {soil_note}"

        recommendations.append({
            "nutrient": nutrient,
            "fertilizer": rec["fertilizer"],
            "rate": rec["rate"],
            "method": rec["method"],
            "urgency": rec["urgency"],
            "reason": reason,
        })

    if not recommendations:
        recommendations.append({
            "nutrient": "general",
            "fertilizer": "Balanced NPK (10-10-10)",
            "rate": "30 kg/ha",
            "method": "Broadcast before irrigation",
            "urgency": "Monitor",
            "reason": "No significant individual deficiency detected. A balanced NPK maintenance application is recommended to sustain crop health.",
        })

    # Agronomic insight for primary deficiency
    insight_map = AGRONOMIC_INSIGHTS.get(primary, AGRONOMIC_INSIGHTS.get("nitrogen"))
    agronomic_insight = insight_map.get(crop_type, insight_map.get("default", ""))

    # Symptoms
    symptoms = SYMPTOM_DESCRIPTIONS.get(primary, [])

    # Preventive tips
    tips = PREVENTIVE_TIPS.get(primary, [])

    return {
        "recommendations": recommendations,
        "agronomic_insight": agronomic_insight,
        "symptoms_matched": symptoms,
        "preventive_tips": tips,
    }
