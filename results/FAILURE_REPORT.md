# P5 failure report: streaming / variable-length scenario embeddings

**Verdict: killed (parked) at Experiment 0. Survived Exp −1, but the instability turned out to be near-tie reshuffling that trivial fixes remove or that doesn't affect retrieval.**

## What was tested
Whether fixed-window scenario embeddings give unstable nearest neighbours when the same event is embedded with different window lengths (4→64 steps), which would motivate a streaming / variable-length embedding.
Setup: frozen SigLIP ViT-B/16 (open_clip, webli) stand-in encoder, CPU, 200 RefAV val events from 132 logs, 2 Hz, k=10, seed 0. The intended P1 encoder does not exist yet.

## Kill rules (from PLAN.md)
- Exp −1 (H1): neighbour overlap@10 across 4→64-step windows > 0.7 → stable, stop.
- Exp 0: if a trivial control (normalisation, fixed-stride chunking, query-length matching) fixes the instability → one-line finding, stop.

## Numbers (from `results/e_minus_1/results.json` and `results/e0/results.json`)
**Exp −1:** overlap@10 (4 vs 64 steps) = 0.517 < 0.7 → survived. But 79% of events were clipped at 32 steps and 100% at 64 (effective mean length 32.1); mean cosine of the same event at 4 vs 64 steps = 0.980; random-per-step baseline overlap 0.094; chance 0.050.

**Exp 0 (L ≤ 32, controls):**
| Quantity | Value |
|---|---|
| Overlap(4,32), all 200 events | 0.565 |
| Overlap(4,32), unclipped subset (n=42) | 0.731 |
| Overlap(4,32), expanded unclipped pool (n=230) | 0.616 |
| Unclipped n=42 with corpus-mean / per-length-mean subtraction | 0.724 / 0.757 |
| Expanded n=230 with corpus-mean / per-length-mean subtraction | 0.652 / 0.656 |
| Mean gap between 10th and 11th neighbour similarity (all events) | 0.0011–0.0013 (mean 10th-neighbour cos 0.92–0.94) |
| Query-length matching, diag − off-diag (pool of 1054): R@1 / R@10 / mAP | +0.0010 / −0.0005 / +0.0004 |
| Query-length matched retrieval, absolute R@10 (pool) | 0.046–0.062 |
| Event-to-log mAP, fixed-window mean-pool vs 4 s chunks max | 0.069–0.071 vs 0.070–0.071 |
| Length-only negative control, event-to-log mAP | 0.051–0.056 |

Script flags: "TRIVIAL FIX WINS: normalisation overlap(4,32) > 0.7" (on the n=42 subset) and "SMALL N".

## Why it was killed
The low overlap was mostly clipping plus near-ties: the 10th and 11th neighbours differ by about 0.001 in cosine, so tiny changes reshuffle them without meaning. On unclipped events overlap was already > 0.7, matching query length had no effect on retrieval, chunking tied mean-pooling, and stand-in retrieval was close to chance (and only modestly above a length-only control). No motivating failure was shown. The idea also depends on the unbuilt P1 encoder. To revisit: rerun `src/exp0.py` with the P1 encoder.
Caveat: the "trivial fix wins" flag rests on n=42; on the n=230 pool overlap stayed 0.62–0.66.

## Lesson
Check neighbour margins and clipping before reading overlap as instability, and don't build a motivating-failure test on a near-chance stand-in encoder.
