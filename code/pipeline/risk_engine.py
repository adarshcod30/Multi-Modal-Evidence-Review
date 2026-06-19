"""
Stage 3 — Risk Engine
Pure Python, zero API calls.
Aggregates risk flags from Stage 1, Stage 2, and user history.
"""


def compute_risk(stage1_output: dict, stage2_output: list, user_history: dict) -> dict:
    """
    Stage 3: Compute risk flags and valid_image from all signals.
    
    Args:
        stage1_output: Output from Stage 1 (claim extraction)
        stage2_output: List of per-image analysis dicts from Stage 2
        user_history: User history dict for this user
    
    Returns:
        Dict with risk_flags (list), valid_image (bool), history_risk_applied (bool)
    """
    flags = set()

    # === From Stage 1 (conversation) ===
    if stage1_output.get("adversarial_text_detected", False):
        flags.add("text_instruction_present")

    # Detect aggressive/threatening language from the extracted claim or conversation
    # (We check for escalation patterns in the original claim text)
    # This is a heuristic check — the Stage 1 LLM already parsed the conversation

    # === From Stage 2 (images) ===
    any_claimed_part_visible = False
    all_wrong_object = True
    any_quality_issues = False

    for img_result in stage2_output:
        # Collect image quality flags
        quality_flags = img_result.get("image_quality_flags", [])
        if isinstance(quality_flags, str):
            quality_flags = [quality_flags]
        for qf in quality_flags:
            if qf and qf != "none":
                flags.add(qf)
                any_quality_issues = True

        # Text in image detection
        if img_result.get("text_in_image_detected", False):
            flags.add("text_instruction_present")

        # Wrong object detection
        if img_result.get("wrong_object_in_image", False):
            flags.add("wrong_object")
        else:
            all_wrong_object = False

        # Non-original image detection
        if img_result.get("is_non_original", False):
            flags.add("non_original_image")

        # Track if any image shows the claimed part
        if img_result.get("claimed_part_visible", False):
            any_claimed_part_visible = True

    # If no images at all, all_wrong_object stays True by default — handle edge case
    if not stage2_output:
        all_wrong_object = False

    # No image shows the claimed part
    if not any_claimed_part_visible and stage2_output:
        flags.add("damage_not_visible")

    # Detect claim mismatch: images show different damage/part than claimed
    has_mismatch = False
    for img_result in stage2_output:
        if img_result.get("object_visible", False) and not img_result.get("matches_claim", True):
            has_mismatch = True
            break
    if has_mismatch:
        flags.add("claim_mismatch")

    # === From User History ===
    history_flags_str = user_history.get("history_flags", "none")
    history_flags = [f.strip() for f in history_flags_str.split(";") if f.strip() and f.strip() != "none"]
    history_risk_applied = False

    if "user_history_risk" in history_flags:
        flags.add("user_history_risk")
        history_risk_applied = True

    if "manual_review_required" in history_flags:
        flags.add("manual_review_required")

    # === Cascade Rules ===
    if "user_history_risk" in flags and any_quality_issues:
        flags.add("manual_review_required")

    if "non_original_image" in flags:
        flags.add("manual_review_required")

    if "text_instruction_present" in flags:
        flags.add("manual_review_required")

    if "wrong_object" in flags:
        flags.add("manual_review_required")

    if "claim_mismatch" in flags:
        flags.add("manual_review_required")

    # === valid_image Logic ===
    any_non_original = any(img.get("is_non_original", False) for img in stage2_output)
    valid_image = True

    if any_non_original:
        valid_image = False
    elif all_wrong_object and stage2_output:
        valid_image = False

    # === Finalize Flags ===
    # Deduplicate and sort
    flags_list = sorted(flags)

    # Remove 'none' if there are actual flags
    if flags_list and "none" in flags_list and len(flags_list) > 1:
        flags_list = [f for f in flags_list if f != "none"]

    # If empty, set to ["none"]
    if not flags_list:
        flags_list = ["none"]

    return {
        "risk_flags": flags_list,
        "valid_image": valid_image,
        "history_risk_applied": history_risk_applied,
    }
