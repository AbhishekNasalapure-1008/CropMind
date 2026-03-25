"""
CropMind — Context-Aware Adjustment Engine
Applies agronomic context (soil, climate, crop, growth stage) to adjust model predictions.
"""


# Multiplicative adjustment factors for each nutrient based on context
SOIL_ADJUSTMENTS = {
    "sandy":   {"nitrogen": 1.05, "phosphorus": 1.10, "potassium": 1.12},
    "clay":    {"nitrogen": 1.00, "phosphorus": 1.02, "potassium": 0.92},
    "loamy":   {"nitrogen": 1.00, "phosphorus": 1.00, "potassium": 1.00},
    "silty":   {"nitrogen": 1.02, "phosphorus": 1.03, "potassium": 1.02},
    "peaty":   {"nitrogen": 0.95, "phosphorus": 1.06, "potassium": 1.08},
    "chalky":  {"nitrogen": 1.03, "phosphorus": 1.15, "potassium": 1.04},
}

CLIMATE_ADJUSTMENTS = {
    "tropical":     {"nitrogen": 1.12, "phosphorus": 1.00, "potassium": 1.02},
    "subtropical":  {"nitrogen": 1.06, "phosphorus": 1.02, "potassium": 1.04},
    "arid":         {"nitrogen": 1.02, "phosphorus": 1.03, "potassium": 1.10},
    "semi-arid":    {"nitrogen": 1.03, "phosphorus": 1.04, "potassium": 1.12},
    "temperate":    {"nitrogen": 1.00, "phosphorus": 1.00, "potassium": 1.00},
    "continental":  {"nitrogen": 1.01, "phosphorus": 1.01, "potassium": 1.01},
}

CROP_ADJUSTMENTS = {
    "rice":      {"nitrogen": 1.12, "phosphorus": 1.02, "potassium": 1.00},
    "wheat":     {"nitrogen": 1.08, "phosphorus": 1.05, "potassium": 1.02},
    "maize":     {"nitrogen": 1.15, "phosphorus": 1.03, "potassium": 1.04},
    "sugarcane": {"nitrogen": 1.10, "phosphorus": 1.02, "potassium": 1.08},
    "cotton":    {"nitrogen": 1.06, "phosphorus": 1.06, "potassium": 1.06},
    "tomato":    {"nitrogen": 1.05, "phosphorus": 1.05, "potassium": 1.10},
    "potato":    {"nitrogen": 1.04, "phosphorus": 1.08, "potassium": 1.12},
    "soybean":   {"nitrogen": 0.90, "phosphorus": 1.12, "potassium": 1.02},
    "other":     {"nitrogen": 1.00, "phosphorus": 1.00, "potassium": 1.00},
}

GROWTH_STAGE_ADJUSTMENTS = {
    "seedling":   {"nitrogen": 1.10, "phosphorus": 1.08, "potassium": 1.00},
    "vegetative": {"nitrogen": 1.18, "phosphorus": 1.03, "potassium": 1.00},
    "flowering":  {"nitrogen": 1.02, "phosphorus": 1.18, "potassium": 1.05},
    "fruiting":   {"nitrogen": 1.00, "phosphorus": 1.08, "potassium": 1.18},
    "maturity":   {"nitrogen": 0.95, "phosphorus": 1.02, "potassium": 1.10},
}


def adjust_for_context(
    predictions: dict,
    crop_type: str,
    soil_type: str,
    climate_zone: str,
    growth_stage: str
) -> dict:
    """
    Apply multiplicative adjustments to model predictions based on agronomic context.

    Args:
        predictions: {"nitrogen": float, "phosphorus": float, "potassium": float}
                     with values in [0, 1].
        crop_type, soil_type, climate_zone, growth_stage: lowercase strings.

    Returns:
        Adjusted and re-normalized predictions dict.
    """
    crop_type = crop_type.lower().strip()
    soil_type = soil_type.lower().strip()
    climate_zone = climate_zone.lower().strip().replace("_", "-")
    growth_stage = growth_stage.lower().strip()

    soil_adj   = SOIL_ADJUSTMENTS.get(soil_type, SOIL_ADJUSTMENTS["loamy"])
    climate_adj = CLIMATE_ADJUSTMENTS.get(climate_zone, CLIMATE_ADJUSTMENTS["temperate"])
    crop_adj   = CROP_ADJUSTMENTS.get(crop_type, CROP_ADJUSTMENTS["other"])
    stage_adj  = GROWTH_STAGE_ADJUSTMENTS.get(growth_stage, GROWTH_STAGE_ADJUSTMENTS["vegetative"])

    adjusted = {}
    for nutrient in ("nitrogen", "phosphorus", "potassium"):
        raw = predictions.get(nutrient, 0.0)
        factor = (
            soil_adj[nutrient] *
            climate_adj[nutrient] *
            crop_adj[nutrient] *
            stage_adj[nutrient]
        )
        # Cap combined factor so context fine-tunes rather than overrides
        factor = min(factor, 1.15)
        adjusted[nutrient] = min(raw * factor, 1.0)

    # Re-normalize so values stay meaningful (cap individual at 1.0, keep relative scale)
    total = sum(adjusted.values())
    if total > 0:
        scale = min(total, 3.0) / total  # don't inflate beyond sum of 3.0
        adjusted = {k: round(min(v * scale, 1.0), 4) for k, v in adjusted.items()}

    return adjusted
