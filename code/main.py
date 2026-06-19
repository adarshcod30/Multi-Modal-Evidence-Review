"""
Multi-Modal Evidence Review System — Main Pipeline Entry Point

Processes all claims in dataset/claims.csv through a 4-stage pipeline:
  Stage 1: Claim Extraction (LLM text-only)
  Stage 2: VLM Image Analysis (per-image vision calls)
  Stage 3: Risk Engine (pure Python)
  Stage 4: Decision Aggregation (LLM text-only)

Writes output.csv to the repository root.
"""

import csv
import json
import os
import sys
import time

# Ensure we can import from the code directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from google import genai

from utils.csv_loader import (
    load_claims,
    load_user_history,
    load_evidence_requirements,
    get_relevant_requirements,
    get_default_user_history,
)
from utils.image_utils import parse_image_paths, get_image_id
from utils.schema_validator import validate
from pipeline.claim_extractor import extract as extract_claim
from pipeline.image_analyzer import analyze_images
from pipeline.risk_engine import compute_risk
from pipeline.decision_aggregator import decide


# Output CSV column order
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


def get_fallback_row(row: dict) -> dict:
    """Return a safe fallback output row when processing fails."""
    return {
        "user_id": row.get("user_id", ""),
        "image_paths": row.get("image_paths", ""),
        "user_claim": row.get("user_claim", ""),
        "claim_object": row.get("claim_object", ""),
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "Processing error occurred",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Unable to process claim due to an error",
        "supporting_image_ids": "none",
        "valid_image": "true",
        "severity": "unknown",
    }


def process_claim(client, row: dict, user_history: dict, requirements: list,
                  dataset_root: str, claim_index: int, total_claims: int) -> dict:
    """Process a single claim through the full 4-stage pipeline."""
    user_id = row.get("user_id", "unknown")
    image_paths_str = row.get("image_paths", "")
    user_claim = row.get("user_claim", "")
    claim_object = row.get("claim_object", "unknown")

    print(f"Processing claim {claim_index + 1}/{total_claims}: ({user_id})")

    # Look up user history
    user_hist = user_history.get(user_id, get_default_user_history())

    # Select relevant evidence requirements
    relevant_reqs = get_relevant_requirements(requirements, claim_object)

    # Parse image paths and get valid image IDs
    image_paths = parse_image_paths(image_paths_str)
    valid_image_ids = [get_image_id(p) for p in image_paths]

    # Stage 1 — Claim Extraction
    print(f"  Stage 1: Extracting claim...")
    stage1_output = extract_claim(client, user_claim, claim_object)
    print(f"  -> Extracted: {stage1_output.get('extracted_claim', '')[:80]}...")

    # Stage 2 — VLM Image Analysis
    print(f"  Stage 2: Analyzing {len(image_paths)} image(s)...")
    stage2_output = analyze_images(
        client, image_paths, stage1_output, claim_object, relevant_reqs, dataset_root
    )
    for img_result in stage2_output:
        print(f"    -> {img_result['image_id']}: "
              f"matches={img_result.get('matches_claim', False)}, "
              f"damage={img_result.get('damage_visible', False)}")

    # Stage 3 — Risk Engine
    print(f"  Stage 3: Computing risk flags...")
    stage3_output = compute_risk(stage1_output, stage2_output, user_hist)
    print(f"  -> Flags: {stage3_output['risk_flags']}, valid_image: {stage3_output['valid_image']}")

    # Stage 4 — Decision Aggregation
    print(f"  Stage 4: Aggregating decision...")
    stage4_output = decide(
        client, stage1_output, stage2_output, stage3_output,
        claim_object, user_id, image_paths_str
    )
    print(f"  -> Status: {stage4_output.get('claim_status', 'unknown')}, "
          f"Severity: {stage4_output.get('severity', 'unknown')}")

    # Assemble output row
    output_row = {
        # Pass-through columns
        "user_id": user_id,
        "image_paths": image_paths_str,
        "user_claim": user_claim,
        "claim_object": claim_object,
        # Stage 4 outputs
        "evidence_standard_met": stage4_output.get("evidence_standard_met", "false"),
        "evidence_standard_met_reason": stage4_output.get("evidence_standard_met_reason", ""),
        "issue_type": stage4_output.get("issue_type", "unknown"),
        "object_part": stage4_output.get("object_part", "unknown"),
        "claim_status": stage4_output.get("claim_status", "not_enough_information"),
        "claim_status_justification": stage4_output.get("claim_status_justification", ""),
        "supporting_image_ids": stage4_output.get("supporting_image_ids", "none"),
        "severity": stage4_output.get("severity", "unknown"),
        # Stage 3 outputs
        "risk_flags": ";".join(stage3_output.get("risk_flags", ["none"])),
        "valid_image": str(stage3_output.get("valid_image", True)).lower(),
    }

    # Validate all fields
    output_row = validate(output_row, claim_object, valid_image_ids)

    return output_row


def main():
    """Main entry point — process all claims and write output.csv."""
    start_time = time.time()

    # Determine repo root (parent of code/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    # Load .env from repo root
    env_path = os.path.join(repo_root, ".env")
    load_dotenv(env_path)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found. Place it in .env file at the repo root.")
        sys.exit(1)

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)
    print("Gemini client initialized successfully.")

    # Load datasets
    dataset_root = os.path.join(repo_root, "dataset")
    claims = load_claims(os.path.join(dataset_root, "claims.csv"))
    user_history = load_user_history(os.path.join(dataset_root, "user_history.csv"))
    requirements = load_evidence_requirements(
        os.path.join(dataset_root, "evidence_requirements.csv")
    )

    print(f"\nLoaded {len(claims)} claims, {len(user_history)} user profiles, "
          f"{len(requirements)} evidence requirements.\n")

    # Process all claims
    results = []
    total_api_calls = 0
    status_counts = {"supported": 0, "contradicted": 0, "not_enough_information": 0}

    for i, row in enumerate(claims):
        try:
            output_row = process_claim(
                client, row, user_history, requirements, dataset_root, i, len(claims)
            )
            results.append(output_row)

            # Count API calls: 1 (stage1) + N images (stage2) + 1 (stage4)
            n_images = len(parse_image_paths(row.get("image_paths", "")))
            total_api_calls += 2 + n_images

            # Track status
            status = output_row.get("claim_status", "not_enough_information")
            if status in status_counts:
                status_counts[status] += 1

        except Exception as e:
            print(f"  ERROR processing claim {i+1}: {e}")
            fallback = get_fallback_row(row)
            # Validate fallback too
            valid_ids = [get_image_id(p) for p in parse_image_paths(row.get("image_paths", ""))]
            fallback = validate(fallback, row.get("claim_object", ""), valid_ids)
            results.append(fallback)
            status_counts["not_enough_information"] += 1
            total_api_calls += 1  # At least one failed call

        # Rate limit sleep
        if i < len(claims) - 1:
            time.sleep(4.5)

        print()

    # Write output.csv to repo root
    output_path = os.path.join(repo_root, "output.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for result_row in results:
            writer.writerow(result_row)

    elapsed = time.time() - start_time

    print("=" * 60)
    print(f"COMPLETE — output.csv written to {output_path}")
    print(f"Total claims processed: {len(results)}")
    print(f"  Supported:              {status_counts['supported']}")
    print(f"  Contradicted:           {status_counts['contradicted']}")
    print(f"  Not enough information: {status_counts['not_enough_information']}")
    print(f"Total API calls made:     ~{total_api_calls}")
    print(f"Approximate runtime:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
