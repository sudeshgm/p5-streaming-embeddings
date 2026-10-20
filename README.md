# P5 — Streaming and variable-length scenario embeddings

**Status:** blocked on P1 · **Priority:** absorbable into P1 as an ablation · **Budget:** 4–6 weeks

---

## The claim

Of the sixteen video papers deep-read from CVPR 2026, **not one produces a streaming or
variable-length embedding**. Every one embeds a fixed frame budget with uniform sampling; only
DIG selects frames adaptively from content, and even it operates offline over a complete video.
Two papers handle streaming *inference* (StreamingTOM, AdaSpark) but neither emits an embedding
you can put in an index.

Driving logs are 15 s (Argoverse 2) to hours (nuPlan). A scenario-retrieval system that requires
a fixed window cannot answer "find this, wherever it happens, in any log".

## Three claims

- **C1.** Fixed-window embeddings have unstable ANN neighbourhoods under window-length change,
  and this instability is measurable and large.
- **C2.** A rolling encoder emitting a per-timestep embedding, pooled with **Sliced-Wasserstein
  pooling** rather than a mean, is stable under window length. OmniRet (CVPR 2026) introduces
  attention Sliced-Wasserstein pooling precisely because "the entire single-vector paradigm faces
  theoretical limitations" — and applies it only to static token sets, never temporally, where
  pooling a *distribution* over time is the more natural fit.
- **C3.** A monotonic "scenario evidence" score enables online triggering: the system fires as
  soon as the scenario is unambiguous, which is what a fleet-side trigger actually needs.

## Experiment −1 — three days

Take the P1 encoder. Embed the same event with windows of 4, 8, 16, 32, 64 timesteps. Measure
**neighbour overlap@10** in the index across window lengths.

**Stop rule.** If overlap is already high, fixed-window embeddings are length-robust, there is no
instability to fix, and the paper has no motivating failure. Stop.

## Relationship to P1

If the instability is real but modest, this becomes P1's "variable-length" ablation section rather
than a separate paper. Decide after Experiment −1; do not build a second system prematurely.

---

See [PLAN.md](PLAN.md).

## Stand-in encoder (Experiment −1)
Until the P1 encoder exists, Exp −1 uses a **frozen per-frame SigLIP-B/16** (`open_clip` `ViT-B-16-SigLIP`, `webli` weights, 224², 768-d) run on CPU.
One embedding per timestep at **2 Hz** (the RefAV evaluation timestamps; nearest `ring_front_center` camera frame).
Window embeddings = mean-pool of per-frame embeddings, L2-normalised. Fallback (unused): DINOv2-S.

Data: RefAV val annotations (HF `CainanD/RefAV`) + Argoverse 2 Sensor val (`s3://argoverse/datasets/av2/sensor/val/`, `--no-sign-request`), ring_front_center only.
Pipeline: `src/download_frames.py` → `src/encode.py` → `src/exp_minus1.py`. Results in `results/e_minus_1/`.
