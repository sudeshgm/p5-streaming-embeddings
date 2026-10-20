"""Experiment -1 (+ Exp 0 controls): neighbour overlap@10 across window lengths for fixed-window mean-pooled embeddings.
Usage: python src/exp_minus1.py [DATA_DIR] [OUT_DIR]"""
import sys, json, time, numpy as np, pandas as pd, faiss
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1] if len(sys.argv) > 1 else '/workspace/p5data'
O = sys.argv[2] if len(sys.argv) > 2 else 'results/e_minus_1'
LENS = [4, 8, 16, 32, 64]; K = 10; N_EVENTS = 200; MAX_PER_LOG = 2; SEED = 0; CHUNK = 8  # 8 steps = 4 s @ 2 Hz
t0 = time.time(); rng = np.random.default_rng(SEED)
F = np.load(f'{D}/features/siglip_b16.npy'); idx = pd.read_csv(f'{D}/features/index.csv')
logs = {l: (g.index.values, g.timestamp_ns.values) for l, g in idx.groupby('log_id', sort=True)}

# ---- events: contiguous runs of 2 Hz steps containing a REFERRED_OBJECT for a (log, prompt) ----
ref = pd.read_parquet(f'{D}/referred_ts.parquet'); cand = []
for (l, p), g in ref.groupby(['log_id', 'prompt']):
    rows, ts = logs[l]
    steps = np.unique([int(np.argmin(np.abs(ts - t))) for t in g.timestamp_ns.values])
    runs = np.split(steps, np.where(np.diff(steps) > 2)[0] + 1)  # allow 1-step gaps
    for r in runs:
        cand.append(dict(log_id=l, prompt=p, start_step=int(r[0]), end_step=int(r[-1]), start_ns=int(ts[r[0]]),
                         end_ns=int(ts[r[-1]]), centre=int(round((r[0] + r[-1]) / 2)), n_log_steps=len(ts)))
cand = pd.DataFrame(cand)
# pick up to MAX_PER_LOG events per log, distinct prompts, round-robin, then cap at N_EVENTS
cand = cand.sample(frac=1, random_state=SEED).drop_duplicates(['log_id', 'prompt'])
ev = cand.groupby('log_id').head(MAX_PER_LOG).sample(frac=1, random_state=SEED).head(N_EVENTS).reset_index(drop=True)
N = len(ev)

def windows(e, L):
    n = e.n_log_steps; s = e.centre - L // 2; a, b = max(0, s), min(n, s + L)
    return a, b, (a != s) or (b != s + L)

def l2(x): return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)

def pooled(feat):  # -> {L: (N,d)}, clipping table
    E, clip = {}, []
    for L in LENS:
        M = []
        for i, e in ev.iterrows():
            rows, _ = logs[e.log_id]; a, b, c = windows(e, L); M.append(feat[rows[a:b]].mean(0)); clip.append((i, L, a, b, b - a, c))
        E[L] = np.stack(M)
    return E, clip

def knn_from_sim(S):
    S = S.copy(); np.fill_diagonal(S, -np.inf); return np.argsort(-S, axis=1)[:, :K]

def knn_faiss(X):
    X = l2(X.astype(np.float32)); ix = faiss.IndexFlatIP(X.shape[1]); ix.add(X)
    _, I = ix.search(X, K + 1); return np.stack([[j for j in r if j != q][:K] for q, r in enumerate(I)])

def overlap_matrix(NB):
    M = np.zeros((5, 5))
    for i, a in enumerate(LENS):
        for j, b in enumerate(LENS):
            M[i, j] = np.mean([len(set(NB[a][q]) & set(NB[b][q])) / K for q in range(N)])
    return M

def summarize(M): off = M[~np.eye(5, dtype=bool)]; return dict(matrix=M.round(4).tolist(), mean_offdiag=float(off.mean()), overlap_4_64=float(M[0, 4]))

def run_mean(feat, centre=False):
    E, clip = pooled(feat)
    if centre: E = {L: l2(E[L]) - l2(E[L]).mean(0, keepdims=True) for L in LENS}  # corpus-mean subtraction per length
    return overlap_matrix({L: knn_faiss(E[L]) for L in LENS}), clip, E

def run_chunks(feat):  # 4 s fixed-stride chunks inside each window; event similarity = max over chunk pairs
    NB = {}
    for L in LENS:
        C = []
        for _, e in ev.iterrows():
            rows, _ = logs[e.log_id]; a, b, _ = windows(e, L)
            C.append(l2(np.stack([feat[rows[s:min(s + CHUNK, b)]].mean(0) for s in range(a, b, CHUNK)])))
        owner = np.concatenate([[i] * len(c) for i, c in enumerate(C)]); A = np.concatenate(C)
        S_all = A @ A.T; S = np.full((N, N), -np.inf)
        for i in range(N):
            S[i] = np.maximum.reduceat(S_all[owner == i].max(0), np.r_[0, np.cumsum([len(c) for c in C])[:-1]])
        NB[L] = knn_from_sim(S)
    return overlap_matrix(NB)

res = {}
M, clip, E = run_mean(F); res['siglip_mean'] = summarize(M)
res['ctrl_a_corpus_mean_sub'] = summarize(run_mean(F, centre=True)[0])
res['ctrl_b_4s_chunks_maxpool'] = summarize(run_chunks(F))
Rf = rng.standard_normal(F.shape).astype(np.float32)
res['random_per_timestep_baseline'] = summarize(run_mean(Rf)[0])
res['random_per_timestep_ctrl_a'] = summarize(run_mean(Rf, centre=True)[0])
res['random_per_timestep_ctrl_b'] = summarize(run_chunks(Rf))
res['chance_iid_event_embeddings'] = K / (N - 1)
# confound diagnostics: fraction of top-10 neighbours from same log; cos(4,64) per event
lg = np.asarray(ev.log_id.astype(str).tolist()); nb = {L: knn_faiss(E[L]) for L in LENS}
res['same_log_neighbour_frac'] = {L: float(np.mean(lg[nb[L]] == lg[:, None])) for L in LENS}
res['mean_cos_4_vs_64_same_event'] = float(np.mean(np.sum(l2(E[4]) * l2(E[64]), 1)))
cl = pd.DataFrame(clip, columns=['event', 'L', 'a', 'b', 'eff_len', 'clipped'])
res['clipping'] = {int(L): dict(frac_clipped=float(g.clipped.mean()), mean_eff_len=float(g.eff_len.mean())) for L, g in cl.groupby('L')}
H = res['siglip_mean']['overlap_4_64']
res['H1'] = dict(headline_overlap_4_64=H, kill_threshold=0.7, killed=bool(H > 0.7),
                 verdict='KILL (length-stable; fold into P1)' if H > 0.7 else 'SURVIVES (length-unstable; proceed)')
res['meta'] = dict(n_events=N, n_logs=int(ev.log_id.nunique()), n_prompts=int(ev.prompt.nunique()), encoder='open_clip ViT-B-16-SigLIP webli, frozen, CPU',
                   rate_hz=2, lens=LENS, k=K, chunk_steps=CHUNK, seed=SEED, median_log_steps=float(ev.n_log_steps.median()), runtime_s=round(time.time() - t0, 1))
import os; os.makedirs(O, exist_ok=True)
json.dump(res, open(f'{O}/results.json', 'w'), indent=2); ev.to_csv(f'{O}/events.csv', index=False); cl.to_csv(f'{O}/clipping.csv', index=False)
rows = [dict(condition=k, len_a=a, len_b=b, overlap10=v['matrix'][i][j]) for k, v in res.items() if isinstance(v, dict) and 'matrix' in v
        for i, a in enumerate(LENS) for j, b in enumerate(LENS)]
pd.DataFrame(rows).to_csv(f'{O}/overlap_matrices.csv', index=False)
conds = ['siglip_mean', 'ctrl_a_corpus_mean_sub', 'ctrl_b_4s_chunks_maxpool', 'random_per_timestep_baseline']
fig, ax = plt.subplots(1, 4, figsize=(18, 4.5))
for a, c in zip(ax, conds):
    m = np.array(res[c]['matrix']); im = a.imshow(m, vmin=0, vmax=1, cmap='viridis'); a.set_title(f"{c}\nmean off-diag={res[c]['mean_offdiag']:.3f}", fontsize=9)
    a.set_xticks(range(5), LENS); a.set_yticks(range(5), LENS)
    for i in range(5):
        for j in range(5): a.text(j, i, f'{m[i,j]:.2f}', ha='center', va='center', color='w', fontsize=8)
fig.colorbar(im, ax=ax); fig.suptitle(f'Exp -1 overlap@10 (N={N} events) | H1 overlap(4,64)={H:.3f} vs 0.7 -> {res["H1"]["verdict"]}')
plt.savefig(f'{O}/overlap_matrices.png', dpi=120, bbox_inches='tight')
for c in conds + ['random_per_timestep_ctrl_a', 'random_per_timestep_ctrl_b']: print(c, res[c]['overlap_4_64'], res[c]['mean_offdiag'])
print(json.dumps({k: res[k] for k in ['H1', 'meta', 'clipping', 'same_log_neighbour_frac', 'chance_iid_event_embeddings']}, indent=1))
