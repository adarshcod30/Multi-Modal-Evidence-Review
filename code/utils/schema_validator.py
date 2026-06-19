"""
Schema validator — validates output rows against allowed values.
Silently corrects invalid values using safe fallbacks.
"""


ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ALLOWED_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
}

ALLOWED_OBJECT_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender",
        "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid",
        "corner", "port", "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal",
        "label", "contents", "item", "unknown",
    },
}

ALLOWED_SEVERITY = {"none", "low", "medium", "high", "unknown"}

ALLOWED_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

ALLOWED_BOOL_STRINGS = {"true", "false"}


def validate(row: dict, claim_object: str, valid_image_ids: list) -> dict:
    """Validate and fix all output fields in a row dict. Returns corrected row."""
    # claim_status
    if row.get("claim_status", "") not in ALLOWED_CLAIM_STATUS:
        row["claim_status"] = "not_enough_information"

    # issue_type
    if row.get("issue_type", "") not in ALLOWED_ISSUE_TYPES:
        row["issue_type"] = "unknown"

    # object_part
    allowed_parts = ALLOWED_OBJECT_PARTS.get(claim_object.lower(), set())
    if row.get("object_part", "") not in allowed_parts:
        row["object_part"] = "unknown"

    # severity
    if row.get("severity", "") not in ALLOWED_SEVERITY:
        row["severity"] = "unknown"

    # evidence_standard_met
    val = str(row.get("evidence_standard_met", "")).lower()
    if val not in ALLOWED_BOOL_STRINGS:
        row["evidence_standard_met"] = "false"
    else:
        row["evidence_standard_met"] = val

    # valid_image
    val = str(row.get("valid_image", "")).lower()
    if val not in ALLOWED_BOOL_STRINGS:
        row["valid_image"] = "true"
    else:
        row["valid_image"] = val

    # risk_flags
    raw_flags = str(row.get("risk_flags", "none"))
    flags = [f.strip() for f in raw_flags.split(";") if f.strip()]
    valid_flags = [f for f in flags if f in ALLOWED_RISK_FLAGS]
    # Remove 'none' if other flags exist
    if len(valid_flags) > 1 and "none" in valid_flags:
        valid_flags = [f for f in valid_flags if f != "none"]
    if not valid_flags:
        valid_flags = ["none"]
    row["risk_flags"] = ";".join(sorted(set(valid_flags)))

    # supporting_image_ids
    raw_ids = str(row.get("supporting_image_ids", "none"))
    if raw_ids.lower() == "none":
        row["supporting_image_ids"] = "none"
    else:
        ids = [i.strip() for i in raw_ids.split(";") if i.strip()]
        valid_ids = [i for i in ids if i in valid_image_ids]
        if not valid_ids:
            row["supporting_image_ids"] = "none"
        else:
            row["supporting_image_ids"] = ";".join(valid_ids)

    # Ensure string fields are not empty
    if not row.get("evidence_standard_met_reason"):
        row["evidence_standard_met_reason"] = "Unable to determine evidence sufficiency"
    if not row.get("claim_status_justification"):
        row["claim_status_justification"] = "Unable to provide justification"

    return row
