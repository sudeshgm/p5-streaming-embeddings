# Internship project — Searching recordings of any length

**Duration** 3 months · **Hardware** one 16 GB GPU · **Data** open-source datasets only

> **Sequencing note.** This project is easier if a situation-search system already exists to build
> on. If it does not, week 1 builds a simple baseline version instead — this is planned for and the
> schedule below already includes it.

---

## 1. What this project is about

Almost every method that turns a video into a searchable vector assumes a fixed number of frames.
Pick sixteen, or thirty-two, spread them evenly across the clip, and summarise. That works when
all your clips are five seconds long. Driving recordings are not: some are fifteen seconds, some
are twenty minutes, and the event you care about might occupy two seconds somewhere in the middle.

Two things go wrong. First, if you summarise a whole long recording into one vector, a brief event
is diluted into insignificance. Second, and less obviously, the *same* event can end up at
different places in the search space depending on how much surrounding footage was included —
which means search results depend on how you happened to cut the video.

You will measure how badly this happens, and build a version that does not have the problem: a
rolling summary that produces a vector as the recording plays, together with a running score for
how much evidence has accumulated so far.

## 2. Why it matters

Archives contain recordings of wildly different lengths, and nobody wants search results that
depend on where the file happened to be cut. A representation that behaves consistently, and can
be computed as data arrives rather than only afterwards, is more useful in practice.

## 3. What you should already know

**Needed**
- PyTorch, including custom training loops.
- Sequence models — transformers or recurrent networks — and how pooling over time works.
- Retrieval evaluation.

**Useful, not required**
- Vector search libraries.
- Any streaming or online-processing experience.

**Not needed**
- Driving domain expertise.

## 4. What you will build

**Part of the project**
- A measurement of how much search results shift when the same event is embedded with different
  amounts of surrounding footage.
- A rolling encoder producing a vector per time step rather than one per clip.
- A comparison of ways to pool over time, including one that keeps the *distribution* of features
  over time rather than just their average.
- A running evidence score, and a measurement of how quickly it can trigger after an event begins.

**Not part of the project**
- Building a video language model.
- Any claim about real-time in-vehicle deployment.
- Improving accuracy on standard fixed-length video benchmarks.

## 5. Getting started — week 0

**Reading**
1. *Efficient and High-Fidelity Omni Modality Retrieval* — specifically the pooling method that
   retains a distribution rather than an average, and why single-vector summaries are limited.
2. *Seeing without Pixels* — the trick of encoding a wider window than you actually score, to give
   short segments enough context.
3. *StreamingTOM* — how one recent method handles video arriving progressively.
4. Documentation for the datasets you will use, particularly one with precise start and end times
   for events.

**Setup:** a driving or general video dataset with event start/end annotations, plus a longer-form
dataset for the length experiments.

**First hands-on task (day 4–5):** take one event and embed it with windows of 4, 8, 16, 32 and 64
time steps. Look at the ten nearest neighbours in each case. How much do those ten change? That
single measurement tells you how large the problem is and shapes the rest of the project.

## 6. Milestones

| | Week | Target | Done when |
|---|---|---|---|
| **M1** | 3 | Instability measured across window lengths, with the simple fixes ruled in or out. | You can say how much of the instability survives after trying the obvious cheap remedies. |
| **M2** | 6 | Rolling encoder producing per-time-step vectors, compared against fixed-window summaries. | Both stability and plain accuracy reported — a representation that is stable but worse is not progress. |
| **M3** | 9 | Pooling comparison, including the distribution-preserving method. | Every pooling method measured on both stability and accuracy, with its computational cost. |
| **M4** | 12 | Running evidence score and trigger-delay measurement. Report and talk done. | Accuracy plotted against how long after an event starts the system fires. |

## 7. Week-by-week

| Week | Focus |
|---|---|
| 0 | Reading; first neighbour-stability measurement. |
| 1 | Either connect to the existing search system, or build a simple baseline encoder to develop against. |
| 2 | **Try the boring fixes first.** Does normalising the vectors, or subtracting the average across the archive, remove most of the instability? If so, that is the answer and the rest of the project changes shape — better to find out in week 2 than week 10. |
| 3 | **Build the obvious competitor:** cut every recording into fixed four-second pieces, embed each independently, and search by taking the best-matching piece. This is what a sensible engineer would do without reading any papers, and it may well be good enough. It belongs in every comparison table from here on. **M1** |
| 4–5 | Rolling encoder. Encode a wider window than you score, so short events get context without being diluted. |
| 6 | Compare against fixed windows and against the chunking competitor. **M2** |
| 7–8 | Pooling comparison: average, attention-weighted, maximum, and the distribution-preserving method. Measure the extra cost of each — one that doubles query time defeats the purpose. |
| 9 | Consolidate; stability and accuracy together. **M3** |
| 10 | Running evidence score over full recordings; evaluate against precise event start and end times. |
| 11 | Trigger delay against accuracy. A system that is accurate but only fires once the event has ended is not useful, so measure both. |
| 12 | Report, presentation, handover. **M4** |

Keep two reference points in every table: an *oracle* that is given the exact event boundaries
(the ceiling), and a check that the system is not simply favouring longer recordings.

## 8. What you will deliver

1. **Rolling encoder and evaluation code.**
2. **The stability measurements** — how much do search results depend on how the video was cut?
3. **Pooling comparison**, with costs.
4. **Report, 10–15 pages**, with a recommendation on how to index recordings of mixed length.
5. **Presentation, about 20 minutes**, to the team.

## 9. Working rhythm

- Weekly one-to-one, 45 minutes.
- Short written status every Friday.
- Demo to the wider team at M2 and M4.
- If you are stuck for more than two days, say so.

## 10. If something does not work

| Situation | Plan |
|---|---|
| Results turn out already stable across window lengths. | Good to know, and quick to establish. Shift to the running evidence score and trigger delay, which stand on their own. Decide by week 2. |
| Simple normalisation fixes it. | Then that is the finding. Report it clearly and move the effort to the streaming part. |
| The fixed-chunk competitor matches the rolling encoder. | Recommend chunking. A clear, well-evidenced "the simple thing is enough" is a genuinely useful outcome, especially for people about to build this. |
| The evidence score is erratic. | Drop that component; the stability and pooling results are a complete project by themselves. |

## 11. What you will take away

A real understanding of how design choices in pooling and windowing affect what a search system
can find — an issue that shows up in audio, sensor data and text as much as in video. Experience
with online processing, where you cannot look ahead. And practice benchmarking against the obvious
simple alternative, which is the comparison reviewers and colleagues always ask for.
