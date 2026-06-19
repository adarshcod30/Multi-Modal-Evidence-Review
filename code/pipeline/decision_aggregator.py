"""
Stage 4 — Decision Aggregator
Combines all signals from Stages 1-3 into final verdict using an LLM call.
Produces the 8 output-only columns (risk_flags and valid_image come from Stage 3).
"""

import json
import time


def _build_system_prompt(claim_object: str) -> str:
    """Build the system prompt for the decision aggregator."""

    # Object-part allowed values
    part_map = {
        "car": "front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, fender, quarter_panel, body, unknown",
        "laptop": "screen, keyboard, trackpad, hinge, lid, corner, port, base, body, unknown",
        "package": "box, package_corner, package_side, seal, label, contents, item, unknown",
    }
    allowed_parts = part_map.get(claim_object, "unknown")

    return f"""You are a damage claim decision aggregator. Based on the analysis of the conversation, images, and risk signals, you must produce the final verdict.

DECISION RULES — images are the PRIMARY source of truth:

1. claim_status:
   - "supported": images clearly show the claimed damage on the claimed part
   - "contradicted": images show the claimed part but with no such damage, OR images show clearly different damage than claimed
   - "not_enough_information": no image adequately shows the claimed part, OR image quality is too poor to assess

2. evidence_standard_met: "true" if at least one image was sufficient to evaluate the claim (even if the verdict is "contradicted"). "false" ONLY if no image showed enough of the relevant part to make any judgment.

3. supporting_image_ids: For "supported" — images showing the damage. For "contradicted" — images showing the part without damage or with different damage. For "not_enough_information" — use "none". List only IDs that exist in the claim's image list.

4. claim_status_justification: Concise, grounded in visual evidence. Mention specific image IDs when helpful. NEVER make user history the primary reason for a verdict.

5. issue_type: What is ACTUALLY visible in images — not what was claimed. Use "unknown" if nothing assessable. Use "none" if the part is visible and undamaged. Allowed: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown

6. object_part: What was actually examined. Allowed values for {claim_object}: {allowed_parts}

7. severity: none = no damage visible, low = minor surface marks, medium = clear damage, high = severe/broken/missing, unknown = cannot assess

8. Multi-damage claims (is_multi_damage = true): Address both claimed parts in the justification. Use the part with strongest image evidence for object_part. Include all relevant images in supporting_image_ids.

User history NEVER overrides clear visual evidence. History only adds risk flags, not the verdict.

Respond ONLY with valid JSON. No markdown fences, no preamble. Output the JSON object directly:
{{
  "evidence_standard_met": "true" or "false",
  "evidence_standard_met_reason": "short sentence explaining why",
  "issue_type": "one of the allowed issue_type values",
  "object_part": "one of the allowed object_part values for {claim_object}",
  "claim_status": "supported or contradicted or not_enough_information",
  "claim_status_justification": "concise image-grounded explanation",
  "supporting_image_ids": "img_1;img_2 or none",
  "severity": "none or low or medium or high or unknown"
}}"""


def decide(client, stage1_output: dict, stage2_output: list, stage3_output: dict,
           claim_object: str, user_id: str, image_paths_str: str) -> dict:
    """
    Stage 4: Combine all signals into final verdict.
    
    Args:
        client: Google GenAI client instance
        stage1_output: Output from Stage 1
        stage2_output: List of per-image analysis dicts from Stage 2
        stage3_output: Output from Stage 3
        claim_object: One of 'car', 'laptop', 'package'
        user_id: User ID string
        image_paths_str: Original image_paths string from CSV
    
    Returns:
        Dict with the 8 decision output fields
    """
    system_prompt = _build_system_prompt(claim_object)

    # Build per-image summaries for context
    image_summaries = []
    for img in stage2_output:
        summary = (
            f"Image {img['image_id']}: "
            f"object_visible={img.get('object_visible', False)}, "
            f"claimed_part_visible={img.get('claimed_part_visible', False)}, "
            f"damage_visible={img.get('damage_visible', False)}, "
            f"visible_issue={img.get('visible_issue_type', 'unknown')}, "
            f"visible_part={img.get('visible_object_part', 'unknown')}, "
            f"matches_claim={img.get('matches_claim', False)}, "
            f"quality_flags={img.get('image_quality_flags', [])}, "
            f"text_detected={img.get('text_in_image_detected', False)}, "
            f"text_content=\"{img.get('text_in_image_content', '')}\", "
            f"wrong_object={img.get('wrong_object_in_image', False)}, "
            f"non_original={img.get('is_non_original', False)}, "
            f"evidence_met={img.get('evidence_standard_met', False)}, "
            f"note=\"{img.get('image_analysis_note', '')}\", "
            f"relevance={img.get('relevance_score', 0.0)}"
        )
        image_summaries.append(summary)

    user_message = f"""Claim Details:
- User ID: {user_id}
- Claim Object: {claim_object}
- Extracted Claim: {stage1_output.get('extracted_claim', '')}
- Claimed Part: {stage1_output.get('claimed_part', 'unknown')}
- Claimed Issue: {stage1_output.get('claimed_issue', 'unknown')}
- Is Multi-Damage: {stage1_output.get('is_multi_damage', False)}
- Adversarial Text Detected: {stage1_output.get('adversarial_text_detected', False)}
- Distractor Present: {stage1_output.get('distractor_present', False)}

Image Analysis Results:
{chr(10).join(image_summaries)}

Risk Assessment:
- Risk Flags: {';'.join(stage3_output.get('risk_flags', ['none']))}
- Valid Image: {stage3_output.get('valid_image', True)}
- History Risk Applied: {stage3_output.get('history_risk_applied', False)}

Available image IDs: {', '.join([img['image_id'] for img in stage2_output])}

Based on all the above, produce the final verdict as JSON."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=user_message,
                config={
                    "system_instruction": system_prompt,
                    "max_output_tokens": 1000,
                }
            )

            text = response.text
            text = text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)

            # Ensure all required keys exist
            defaults = _default_decision()
            for key, default_val in defaults.items():
                if key not in result:
                    result[key] = default_val

            # Normalize string booleans
            for field in ["evidence_standard_met"]:
                result[field] = str(result[field]).lower()

            return result

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"  [Stage 4] JSON parse failed after retries")
            return _default_decision()

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"  [Stage 4] API error (attempt {attempt+1}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue
            print(f"  [Stage 4] Failed after retries: {e}")
            return _default_decision()


def _default_decision() -> dict:
    """Return safe fallback decision values."""
    return {
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "Unable to determine evidence sufficiency",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Unable to provide justification due to processing error",
        "supporting_image_ids": "none",
        "severity": "unknown",
    }
