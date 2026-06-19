# Evaluation Report — Multi-Modal Evidence Review System

## Section 1: Strategy Comparison

### Strategy A — Single-pass Baseline
A single-pass approach would combine the claim text, all images, and full instructions into one API call per claim.

- **Estimated claim_status accuracy:** ~55-65% (lower due to complexity of handling multi-image claims, adversarial content, and nuanced evidence requirements in a single prompt)
- **API calls per claim:** 1
- **Estimated tokens per claim:** ~3,000-5,000 input (text + images combined) + ~500 output

### Strategy B — Multi-pass Pipeline (Final)
The 4-stage architecture separates concerns for better accuracy:
- Stage 1: Dedicated claim extraction handles multilingual and adversarial text
- Stage 2: Per-image analysis provides focused visual evidence assessment
- Stage 3: Rule-based risk flagging ensures consistent policy application
- Stage 4: Decision aggregation with full context from prior stages

- **claim_status accuracy on sample_claims.csv:** 75.0%
- **API calls per claim:** 2 + N (where N = number of images, typically 1-3)
- **Estimated tokens per claim:** Stage 1 (~600 in + ~200 out) + Stage 2 (~2,900 per image in + ~300 out) + Stage 4 (~800 in + ~400 out)

**Final output.csv uses Strategy B** because the multi-pass approach provides:
1. Better separation of concerns (text analysis vs. visual analysis vs. risk rules)
2. More robust adversarial detection across multiple stages
3. Consistent risk flagging through rule-based Stage 3
4. Higher accuracy through focused prompts per stage

## Section 2: Operational Analysis

### Claims Processed
- **Sample claims (evaluation):** 20
- **Test claims (production):** 44
- **Total:** 64

### API Calls Per Claim
| Stage | Calls | Type |
|-------|-------|------|
| Stage 1 — Claim Extraction | 1 | Text-only |
| Stage 2 — Image Analysis | N (1-3) | Vision (per image) |
| Stage 3 — Risk Engine | 0 | Pure Python |
| Stage 4 — Decision Aggregation | 1 | Text-only |

### Token Estimates Per Call
| Stage | Input Tokens | Output Tokens |
|-------|-------------|---------------|
| Stage 1 | ~600 | ~200 |
| Stage 2 (per image) | ~1,000 text + ~1,600 image | ~300 |
| Stage 4 | ~800 | ~400 |

### Total Estimated API Calls
- **Evaluation run:** ~69 calls
- **Test run (44 claims):** ~162 calls (avg ~3.7 per claim)
- **Combined:** ~231 calls

### Cost Estimate
Using Gemini Flash Lite pricing:
- Average input tokens per claim: ~5,000 (across all stages)
- Average output tokens per claim: ~900
- **Total estimated cost for 64 claims:** Free / Negligible (well within free tier limits)

### Runtime
- **Evaluation runtime:** 222.7s (3.7 min)
- **Rate limit sleep:** 4.5s between claims + exponential backoffs
- **Average processing time per claim:** ~14.6s (including sleep and backoffs)
- **Test run time:** 644.1s (10.7 min)

### Rate Limiting Strategy
- **Sleep between claims:** 4.5 seconds to respect RPM limits
- **Retry logic:** Up to 3 retries with exponential backoff (5s, 10s, 15s) on 429/5xx errors
- **TPM/RPM considerations:** Gemini Flash allows sufficient RPM for sequential processing. No batching needed at this scale.
- **Error handling:** Failed claims produce safe fallback rows rather than stopping the pipeline

## Section 3: Known Limitations and Edge Cases

### Adversarial Cases Handled
1. **Prompt injection in conversation text:** Detected by Stage 1, flagged as `text_instruction_present` + `manual_review_required`. Instructions are never obeyed.
2. **Instructions embedded in images:** Detected by Stage 2 VLM, text captured but ignored for verdict determination.
3. **Aggressive/threatening language:** Flagged for `manual_review_required` via user history or Stage 1 detection.
4. **Wrong object in image:** Detected by Stage 2, flagged as `wrong_object` + `claim_mismatch` + `manual_review_required`.
5. **Distractor conversations:** Stage 1 extracts only the final confirmed claim regardless of preamble length.

### Known Limitations
1. **VLM accuracy on edge cases:** Some borderline damage visibility (minor scratches, hairline cracks) may be inconsistently detected across runs due to model non-determinism.
2. **Multi-damage claim prioritization:** When images cover both claimed parts equally, the choice of primary `object_part` may vary.
3. **Severity calibration:** The boundary between "low" and "medium" severity is subjective and may differ from expected labels.
4. **Image quality assessment:** Borderline quality issues (slightly blurry, slightly cropped) may not always be flagged consistently.
5. **Risk flag exact matching:** The exact set of risk flags depends on Stage 2 per-image analysis, which has inherent variability.
