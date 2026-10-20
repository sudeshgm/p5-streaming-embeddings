"""Download ring_front_center frames for RefAV val logs at the 2 Hz RefAV eval timestamps (nearest camera frame)."""
import json, subprocess, os, sys, bisect
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else '/workspace/p5data'
ev = json.load(open('/workspace/refav/run/experiment_configs/eval_timestamps.json'))
logs = sorted(pd.read_parquet(f'{D}/referred_ts.parquet').log_id.unique())
S3 = 's3://argoverse/datasets/av2/sensor/val'
def do(log):
    out = f'{D}/frames/{log}'; os.makedirs(out, exist_ok=True)
    ls = subprocess.run(['aws','s3','ls','--no-sign-request',f'{S3}/{log}/sensors/cameras/ring_front_center/'],capture_output=True,text=True).stdout
    cams = sorted(int(l.split()[-1][:-4]) for l in ls.splitlines() if l.endswith('.jpg'))
    rows = []
    for t in ev[log]:
        i = bisect.bisect_left(cams, t); c = min(cams[max(0,i-1):i+1], key=lambda x: abs(x-t))
        f = f'{out}/{t}.jpg'
        if not os.path.exists(f):
            subprocess.run(['aws','s3','cp','--no-sign-request','--quiet',f'{S3}/{log}/sensors/cameras/ring_front_center/{c}.jpg',f])
        rows.append((log, t, c, abs(c-t)/1e6))
    return rows
with ThreadPoolExecutor(16) as ex: R = [r for rs in ex.map(do, logs) for r in rs]
pd.DataFrame(R, columns=['log_id','timestamp_ns','cam_ts','dt_ms']).to_csv(f'{D}/frame_index.csv', index=False)
print(len(R), 'frames for', len(logs), 'logs')
