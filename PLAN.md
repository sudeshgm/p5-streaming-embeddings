# WORKING RESEARCH NOTE — P5

## Streaming, variable-length scenario embeddings

**Blocked on P1.** May be absorbed into P1 as an ablation — decide at Experiment −1.

---

## 1. Claims

- **C1.** Fixed-window scenario embeddings have unstable ANN neighbourhoods under window-length
  change.
- **C2.** Sliced-Wasserstein pooling over the temporal token set, plus contextualised encoding,
  is stable under length.
- **C3.** A monotonic evidence score supports online triggering with a measurable latency/accuracy
  trade-off.

## 2. Method components, each borrowed and each currently untested temporally

| Component | Source | Why it fits |
|---|---|---|
| **Attention Sliced-Wasserstein pooling** | OmniRet (Huynh et al., CVPR 2026) | Preserves a distribution of token features instead of a mean. Introduced for static multi-modal token sets; a variable-length temporal token set is the more natural application, and nobody has tried it. |
| **Contextualised window encoding** | *Seeing without Pixels* (Xue et al., CVPR 2026) | Extend the window by a randomly split margin, encode the whole thing, pool only the original span. Invented to fight the low information density of short trajectories — exactly the problem here. |
| **Causal temporal reduction** | StreamingTOM (Chen et al., CVPR 2026) | Per-frame token budget under strict causality; the only streaming-compatible compression in the proceedings. |
| **Matryoshka nesting** | Franca (Venkataramanan et al., CVPR 2026) | Lets the streaming index use short prefixes for the fast path. |

The paper's honest framing: none of these is new, and the contribution is the demonstration that
a variable-length retrievable embedding is constructible and that fixed-window embeddings are
measurably broken without it.

## 3. Hypotheses

| # | Hypothesis | Falsified if |
|---|---|---|
| H1 | Fixed-window embeddings are length-unstable. | Neighbour overlap@10 across 4→64-step windows > 0.7. **Exp −1 kill criterion.** |
| H2 | SW pooling beats mean pooling on length stability. | Δ overlap@10 < 0.1. |
| H3 | SW pooling does not cost retrieval accuracy at fixed length. | > 3 points Recall@1 worse than mean pooling at the native window. |
| H4 | The evidence score is monotone in evidence. | Score is non-monotone on more than 30 % of positive clips. |
| H5 | Online triggering reaches offline accuracy with bounded latency. | Requires > 80 % of the scenario duration to reach 90 % of offline accuracy. |

## 4. Experiments, kill-first

### 4.1 Experiment −1 — three days
Neighbour-overlap@10 across window lengths using the P1 encoder.
**Stop rule:** overlap already high → no instability, no paper. Fold the result into P1 as a
one-paragraph robustness note.

### 4.2 Experiment 0 — controls, one week
1. **Trivial length normalisation.** Does simply L2-normalising and/or subtracting the corpus mean
   fix the instability? If yes, publish the one-liner and stop. This is the same trivial-baseline
   trap flagged in `../../vjepa_temporal/` and in `../lhf1_vjepa_spectral_audit/`; it is the most
   common way projects of this shape die.
2. **Fixed-stride chunking baseline.** Chop every log into fixed 4 s chunks, embed each
   independently, retrieve by max-over-chunks. This is what a competent engineer would do without
   reading any papers, and it may be sufficient. If it is, say so — that is a useful finding, and
   it should be the baseline in every table.
3. **Query-length matching.** Much apparent instability is just query/target length mismatch.
   Control for it before claiming a representational problem.

Control 2 is the real competitor. Do not write a streaming encoder before beating it.

### 4.3 Experiment 1 — pooling comparison (C2)
Mean, attentive, max, and Sliced-Wasserstein pooling; length stability and native-length accuracy.
Report both, because a pooling that is stable and worse is not progress.

### 4.4 Experiment 2 — online triggering (C3)
Rolling encoder over full logs; per-timestep evidence score. Evaluate as online scenario detection
on RefAV, which annotates precise start and end timestamps — this is why the evaluation is possible
at all, and it is worth stating.

Metrics: HOTA-Temporal for the offline equivalent, plus **detection latency** (timesteps after
scenario onset before firing) and throughput (clips/s). A trigger that is accurate but fires at
the end of the scenario is useless.

### 4.5 Negative controls
- **Fixed-stride chunking** (Exp 0 control 2) in every table.
- **Oracle window.** Give the model the ground-truth scenario span. This is the ceiling; report
  everything against it.
- **Length-only predictor.** Predict relevance from clip length alone, to confirm the index is not
  simply preferring long logs.

## 5. Risk register

| Risk | Reality | Mitigation |
|---|---|---|
| No instability to fix | Quite possible. | Exp −1, three days, then fold into P1. |
| Trivial normalisation fixes it | Very possible. | Exp 0 control 1, before any encoder work. |
| Chunking baseline is sufficient | Likely for short logs; less likely for long nuPlan logs. | Evaluate on both; the long-log regime is where the claim lives. |
| SW pooling implementation cost | Slicing adds compute per query. | Measure and report it; a pooling that doubles query cost defeats the purpose. |
| Depends entirely on P1 | Yes. | Do not start early. |

## 6. Failure modes

1. Exp −1 shows stability → fold into P1, no separate paper.
2. Trivial normalisation fixes it → one-line finding, stop.
3. Chunking matches the streaming encoder → publish chunking as the recommended practice; still
   useful, much smaller.
4. Evidence score is non-monotone → drop C3, keep C1 and C2.

## 7. Week plan

| Week | Work | Stop rule |
|---|---|---|
| 0 (3 days) | Exp −1 neighbour-overlap across window lengths. | Stable → fold into P1, stop. |
| 1 | Exp 0 controls: normalisation, fixed-stride chunking, query-length matching. | Trivial fix wins → one-liner, stop. |
| 2–3 | Exp 1 pooling comparison. | SW pooling costs accuracy → report and re-scope. |
| 4–5 | Exp 2 online triggering on RefAV; latency/accuracy frontier. | — |
| 6 | Writing. | — |

## 8. Title directions

- *An Index You Can Stream Into*
- *How Long Is a Scenario? Length-Stable Embeddings for Driving Logs*

## 9. Non-goals

A new video-LLM. Real-time on-vehicle deployment claims. Beating StreamingTOM on VideoMME — the
comparison is about embeddings, not about video QA.
