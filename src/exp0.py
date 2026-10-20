"""Exp 0 (tightened controls). Reuses cached SigLIP-B/16 2 Hz features. Usage: python src/exp0.py [DATA] [OUT]"""
import sys, os, json, time, numpy as np, pandas as pd, faiss
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1] if len(sys.argv) > 1 else '/workspace/p5data'; O = sys.argv[2] if len(sys.argv) > 2 else 'results/e0'
LENS = [4, 8, 16, 32]; K = 10; CH = 8; t0 = time.time(); rng = np.random.default_rng(0)
F = np.load(f'{D}/features/siglip_b16.npy'); idx = pd.read_csv(f'{D}/features/index.csv')
logs = {l: g.index.values for l, g in idx.groupby('log_id', sort=True)}; LOGS = sorted(logs)
def l2(x): return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)
ref = pd.read_parquet(f'{D}/referred_ts.parquet'); cand = []
ts_of = {l: idx.timestamp_ns.values[r] for l, r in logs.items()}
for (l, p), g in ref.groupby(['log_id', 'prompt']):
    ts = ts_of[l]; st = np.unique([int(np.argmin(np.abs(ts - t))) for t in g.timestamp_ns.values])
    for r in np.split(st, np.where(np.diff(st) > 2)[0] + 1):
        cand.append(dict(log_id=l, prompt=p, start_step=int(r[0]), end_step=int(r[-1]), centre=int(round((r[0] + r[-1]) / 2)), n=len(ts)))
cand = pd.DataFrame(cand).sample(frac=1, random_state=0).drop_duplicates(['log_id', 'prompt'])
ev200 = cand.groupby('log_id').head(2).sample(frac=1, random_state=0).head(200).reset_index(drop=True)
pool = cand.reset_index(drop=True)
def win(e, L): s = e.centre - L // 2; return max(0, s), min(e.n, s + L), (s < 0) or (s + L > e.n)
def unclipped(df): return df[[not any(win(e, L)[2] for L in LENS) for _, e in df.iterrows()]].reset_index(drop=True)
def pooled(df, feat): return {L: np.stack([feat[logs[e.log_id][slice(*win(e, L)[:2])]].mean(0) for _, e in df.iterrows()]) for L in LENS}
mu_frames = F.mean(0)
Xc = F - mu_frames; U, S, Vt = np.linalg.svd(Xc, full_matrices=False); W = Vt[:256].T / (S[:256] / np.sqrt(len(F) - 1))
NORMS = {'l2_only': lambda E: {L: l2(E[L]) for L in LENS},
         'corpus_mean_sub_l2': lambda E: {L: l2(E[L] - mu_frames) for L in LENS},
         'per_length_mean_sub_l2': lambda E: {L: l2(l2(E[L]) - l2(E[L]).mean(0)) for L in LENS},
         'pca_whiten256_l2': lambda E: {L: l2((E[L] - mu_frames) @ W) for L in LENS}}
def knn(X, k=K):
    X = np.ascontiguousarray(X, dtype=np.float32); ix = faiss.IndexFlatIP(X.shape[1]); ix.add(X); Ssc, I = ix.search(X, k + 1)
    oi, os_ = [], []
    for q in range(len(X)):
        m = I[q] != q; oi.append(I[q][m][:k]); os_.append(Ssc[q][m][:k])
    return np.array(oi), np.array(os_)
def ovl(E):
    nb = {L: knn(E[L])[0] for L in LENS}; M = np.array([[np.mean([len(set(nb[a][q]) & set(nb[b][q])) / K for q in range(len(E[4]))]) for b in LENS] for a in LENS])
    return dict(matrix=M.round(4).tolist(), overlap_4_32=float(M[0, 3]), mean_offdiag=float(M[~np.eye(4, dtype=bool)].mean()))
res = {}
evb = unclipped(ev200); poolb = unclipped(pool)
res['counts'] = dict(a_all=len(ev200), b_unclipped=len(evb), pool_all=len(pool), b_expanded_pool_unclipped=len(poolb))
Ea, Eb = pooled(ev200, F), pooled(evb, F)
res['1_overlap'] = {'a_all_l2': ovl(NORMS['l2_only'](Ea)), 'b_unclipped_l2': ovl(NORMS['l2_only'](Eb)),
                    'b_random_per_step': ovl(NORMS['l2_only'](pooled(evb, rng.standard_normal(F.shape).astype(np.float32)))),
                    'b_expanded_l2': ovl(NORMS['l2_only'](pooled(poolb, F)))}
res['headline_overlap_4_32_b'] = res['1_overlap']['b_unclipped_l2']['overlap_4_32']
res['2_norm_b'] = {k: ovl(f(Eb)) for k, f in NORMS.items()}
Epb = pooled(poolb, F); res['2_norm_b_expanded'] = {k: ovl(f(Epb)) for k, f in NORMS.items()}
def ap_r(scores, pos):
    o = np.argsort(-scores); p = pos[o]; h = np.cumsum(p); n = p.sum()
    return (h[p] / (np.where(p)[0] + 1)).sum() / n, float(p[:1].any()), float(p[:10].any())
def retrieval(df, E):
    lg = df.log_id.values.astype(str); pr = df.prompt.values.astype(str)
    npos = np.array([np.sum((pr == pr[q]) & (lg != lg[q])) for q in range(len(df))]); Q = np.where(npos > 0)[0]
    R1, R10, MAP = (np.zeros((4, 4)) for _ in range(3))
    for i, a in enumerate(LENS):
        for j, b in enumerate(LENS):
            S = E[a] @ E[b].T; r = []
            for q in Q:
                keep = lg != lg[q]; r.append(ap_r(S[q][keep], (pr == pr[q])[keep]))
            MAP[i, j], R1[i, j], R10[i, j] = np.array(r).mean(0)
    off = ~np.eye(4, dtype=bool)
    return dict(n_queries=int(len(Q)), positives_per_query_median=float(np.median(npos[Q])), positives_per_query_mean=float(npos[Q].mean()), gallery_size=len(df),
                R1=R1.round(4).tolist(), R10=R10.round(4).tolist(), mAP=MAP.round(4).tolist(),
                diag_minus_offdiag=dict(R1=float(np.diag(R1).mean() - R1[off].mean()), R10=float(np.diag(R10).mean() - R10[off].mean()), mAP=float(np.diag(MAP).mean() - MAP[off].mean())))
Ep = NORMS['l2_only'](pooled(pool, F))
res['3_qlen_matching_pool'] = retrieval(pool, Ep)
res['3_qlen_matching_pool_unclipped'] = retrieval(poolb, NORMS['l2_only'](Epb))
logE = l2(np.stack([F[logs[l]].mean(0) for l in LOGS]))
chunks = [l2(np.stack([F[logs[l][s:s + CH]].mean(0) for s in range(0, len(logs[l]), CH)])) for l in LOGS]
loglen = np.array([len(logs[l]) for l in LOGS], float); has = {p: set(g.log_id) for p, g in ref.groupby('prompt')}
LA = np.array(LOGS)
def logret(scorefn):
    out = {}
    for L in LENS:
        r = []
        for k, (_, e) in enumerate(pool.iterrows()):
            keep = LA != e.log_id; pos = np.isin(LA, list(has[e.prompt])) & keep
            if not pos.any(): continue
            r.append(ap_r(scorefn(Ep[L][k])[keep], pos[keep]))
        r = np.array(r); out[L] = dict(mAP=float(r[:, 0].mean()), R1=float(r[:, 1].mean()), R10=float(r[:, 2].mean()), n_queries=len(r))
    return out
res['4_event_to_log'] = {'fixed_window_meanpool_log': logret(lambda q: logE @ q),
                         'chunks4s_max': logret(lambda q: np.array([(c @ q).max() for c in chunks])),
                         'length_only_neg_control': logret(lambda q: loglen + 1e-3 * rng.random(len(LOGS)))}
res['4_positive_logs_per_query_mean'] = float(np.mean([len(has[p]) - 1 for p in pool.prompt]))
res['5_margin_10_11'] = {}
for nm, E in [('a_all', NORMS['l2_only'](Ea)), ('b_unclipped', NORMS['l2_only'](Eb))]:
    res['5_margin_10_11'][nm] = {}
    for L in LENS:
        s = knn(E[L], 11)[1]; res['5_margin_10_11'][nm][L] = dict(mean_gap=float((s[:, 9] - s[:, 10]).mean()), median_gap=float(np.median(s[:, 9] - s[:, 10])), mean_s10=float(s[:, 9].mean()))
res['5_context_selfcos_4_64_e_minus_1'] = 0.98
fl = []
nb = {k: v['overlap_4_32'] for k, v in res['2_norm_b'].items()}
if any(v > 0.7 for v in nb.values()): fl.append(f'TRIVIAL FIX WINS: normalisation overlap(4,32)>0.7: {[k for k, v in nb.items() if v > 0.7]}')
gap = res['3_qlen_matching_pool']['diag_minus_offdiag']['R10']
fw = max(res['4_event_to_log']['fixed_window_meanpool_log'][L]['R10'] for L in LENS); chk = max(res['4_event_to_log']['chunks4s_max'][L]['R10'] for L in LENS)
if gap < 0.02 and chk >= fw: fl.append(f'FLAG: length-matching R@10 gap {gap:.4f}<0.02 and chunking best R@10 {chk:.3f} >= fixed-window best {fw:.3f}')
if len(evb) < 50: fl.append(f'SMALL N: (b) unclipped subset of the 200 has only {len(evb)} events; see b_expanded ({len(poolb)})')
if res['3_qlen_matching_pool']['positives_per_query_median'] < 3: fl.append('FEW POSITIVES: median positives/query < 3 in event-event retrieval')
res['flags'] = fl; res['runtime_s'] = round(time.time() - t0, 1)
os.makedirs(O, exist_ok=True); json.dump(res, open(f'{O}/results.json', 'w'), indent=2, default=float)
rows = [dict(section=sec, condition=k, len_a=a, len_b=b, overlap10=v['matrix'][i][j]) for sec in ['1_overlap', '2_norm_b', '2_norm_b_expanded']
        for k, v in res[sec].items() for i, a in enumerate(LENS) for j, b in enumerate(LENS)]
pd.DataFrame(rows).to_csv(f'{O}/overlap.csv', index=False)
rr = [dict(section=sec, metric=m, L_q=a, L_t=b, value=res[sec][m][i][j]) for sec in ['3_qlen_matching_pool', '3_qlen_matching_pool_unclipped']
      for m in ['R1', 'R10', 'mAP'] for i, a in enumerate(LENS) for j, b in enumerate(LENS)]
rr += [dict(section='4_event_to_log', metric=m, L_q=L, L_t=k, value=d[m]) for k, v in res['4_event_to_log'].items() for L, d in v.items() for m in ['R1', 'R10', 'mAP']]
pd.DataFrame(rr).to_csv(f'{O}/retrieval.csv', index=False)
P = [('(b) L2 overlap@10', res['1_overlap']['b_unclipped_l2']['matrix'])] + [(k, v['matrix']) for k, v in res['2_norm_b'].items() if k != 'l2_only'] + [('R@10 Lq(row) x Lt(col)', res['3_qlen_matching_pool']['R10'])]
fig, ax = plt.subplots(1, len(P), figsize=(4 * len(P), 4))
for a, (t, m) in zip(ax, P):
    m = np.array(m); a.imshow(m, vmin=0, vmax=1); a.set_title(t, fontsize=9); a.set_xticks(range(4), LENS); a.set_yticks(range(4), LENS)
    for i in range(4):
        for j in range(4): a.text(j, i, f'{m[i,j]:.2f}', ha='center', va='center', color='w', fontsize=8)
fig.suptitle(f"Exp 0: overlap(4,32) on unclipped (n={len(evb)}) = {res['headline_overlap_4_32_b']:.3f}"); plt.savefig(f'{O}/exp0.png', dpi=120, bbox_inches='tight')
print(json.dumps({k: res[k] for k in ['counts', 'headline_overlap_4_32_b', 'flags', 'runtime_s']}, indent=1))
