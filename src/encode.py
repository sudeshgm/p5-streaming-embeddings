"""Frozen per-frame SigLIP-B/16 (open_clip ViT-B-16-SigLIP / webli) on CPU, one embedding per 2 Hz timestep."""
import sys, os, numpy as np, pandas as pd, torch, open_clip
from PIL import Image
D = sys.argv[1] if len(sys.argv) > 1 else '/workspace/p5data'
torch.set_num_threads(8)
m, _, pre = open_clip.create_model_and_transforms('ViT-B-16-SigLIP', pretrained='webli'); m.eval()
idx = pd.read_csv(f'{D}/frame_index.csv').sort_values(['log_id','timestamp_ns']).reset_index(drop=True)
os.makedirs(f'{D}/features', exist_ok=True)
out = f'{D}/features/siglip_b16.npy'
feats = np.zeros((len(idx), 768), np.float32); B = 32
with torch.no_grad():
    for s in range(0, len(idx), B):
        r = idx.iloc[s:s+B]
        x = torch.stack([pre(Image.open(f'{D}/frames/{l}/{t}.jpg').convert('RGB')) for l, t in zip(r.log_id, r.timestamp_ns)])
        feats[s:s+B] = m.encode_image(x).numpy()
        if s % 640 == 0: print(s, len(idx), flush=True)
np.save(out, feats); idx.to_csv(f'{D}/features/index.csv', index=False); print('saved', feats.shape)
