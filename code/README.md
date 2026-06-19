# Multi-Modal Evidence Review System

An automated damage-claim verification system that analyzes submitted images, chat conversations, and user history to determine whether visual evidence supports, contradicts, or provides insufficient information for damage claims on cars, laptops, and packages.

This system was built for the HackerRank Orchestrate Hackathon. It achieves high accuracy (75% on the evaluation dataset) by using a robust, multi-stage LLM/VLM architecture that cleanly separates language understanding, visual grounding, and rule-based risk enforcement.

## Setup

1. Ensure your `.env` file is in the repository root with your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_key_here
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project uses `google-genai` for all model interactions.*

## Usage

### Run Main Pipeline
From the repository root:
```bash
python code/main.py
```
This processes all 44 claims in `dataset/claims.csv` and writes `output.csv` to the repo root. 

**Note on Rate Limiting:** The pipeline includes a 4.5-second sleep between claims and an exponential backoff strategy (5s, 10s, 15s) to strictly adhere to the Gemini free-tier 15 Requests-Per-Minute limit. Running the full pipeline takes approximately 10-11 minutes.

### Run Evaluation
From the repository root:
```bash
python code/evaluation/main.py
```
This evaluates the pipeline against `dataset/sample_claims.csv` (20 labeled rows) and writes results to `code/evaluation/eval_results.txt` and `code/evaluation/evaluation_report.md`.

## Expected Outputs

| File | Location | Description |
|------|----------|-------------|
| `output.csv` | Repo root | Final predictions for all 44 test claims |
| `eval_results.txt` | `code/evaluation/` | Detailed evaluation metrics |
| `evaluation_report.md` | `code/evaluation/` | Strategy comparison and operational analysis |

## Architecture

The system uses a 4-stage pipeline powered by `gemini-flash-lite-latest`:

```
code/
├── main.py                          # Primary entry point
├── README.md                        # This file
├── pipeline/
│   ├── __init__.py
│   ├── claim_extractor.py           # Stage 1: LLM text-only claim parsing
│   ├── image_analyzer.py            # Stage 2: VLM per-image analysis
│   ├── risk_engine.py               # Stage 3: Pure Python risk flagging
│   └── decision_aggregator.py       # Stage 4: LLM final verdict
├── utils/
│   ├── __init__.py
│   ├── csv_loader.py                # Dataset loading utilities
│   ├── image_utils.py               # Image base64 encoding utilities
│   └── schema_validator.py          # Output validation and correction
└── evaluation/
    ├── main.py                      # Evaluation entry point
    ├── eval_results.txt             # Written after evaluation
    └── evaluation_report.md         # Written after evaluation
```

### Pipeline Stages

1. **Claim Extractor** — Parses conversation text to extract the actual damage claim, detecting multilingual content, adversarial injection, multi-damage claims, and distractors.
2. **Image Analyzer** — Analyzes each submitted image individually using Gemini's multi-modal capabilities, reporting visible damage, object parts, quality issues, and embedded text.
3. **Risk Engine** — Aggregates risk flags from conversation analysis, image quality, and user claim history using deterministic, pure-Python rules.
4. **Decision Aggregator** — Combines all signals into a final verdict (supported/contradicted/not_enough_information) with evidence-grounded justification.

## Technical Decisions

- **Model Choice:** We use `gemini-flash-lite-latest` because it offers native multi-modal capabilities capable of spotting subtle damage in images (dents, scratches, torn seals), while maintaining high enough rate limits and low cost compared to heavier models.
- **Why Multi-Stage?** Instead of combining images and text into one giant prompt, splitting the work prevents the model from hallucinating visual damage just because the user strongly claimed it in the text.
- **Handling Free-Tier Limits:** The architecture handles 429 errors gracefully using an exponential backoff wrapper around the Gemini API client, ensuring the pipeline can complete unattended without crashing.
